"""Fixed 027 continuation and a same-state perceptual tail."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def run(work, initial, output):
    started = time.perf_counter()
    import numpy as np
    import torch
    from diff_gaussian_rasterization_faster import SparseGaussianAdam
    from .resources import CONFIG, VENDOR
    from .runtime import seed_all, write_json
    from .protocol import split_protocol
    from .sh_repair import repair
    from .io import load_ply, save_ply
    from .export import save_model_ply
    from .image_loading import load_training_images
    from .cameras import classic
    from .rendering import classic_imports, initialize, pipe
    from .optimization.training import TrainingSSIM, accumulate_statistics
    from .optimization.density import resolve_config, topology_due, assert_optimizer_topology
    from .optimization.absolute_gradient import GradientAccumulator
    from .optimization.fastgs_renderer import FastGSRenderer
    from .optimization.geometry_capacity import JointGeometry

    torch.set_num_threads(4)
    seed_all(CONFIG['seed'])
    sys.path[:0] = [str(VENDOR / 'resplat'), str(VENDOR / 'resplat/scripts')]
    from infer_colmap import load_colmap_scene
    protocol = split_protocol(json.loads((work / 'split.json').read_text()))
    data = load_colmap_scene(str(work), 'sparse/0', 'images')
    if data['image_names'] != protocol['names']:
        raise ValueError('Camera order and split order differ')
    cameras = dict(c2w=data['c2w'], K=data['intrinsics'])
    pivot = protocol['selected_sources'][len(protocol['selected_sources']) // 2]
    corrected = output / 'initial_corrected.ply'
    repair_info = repair(initial, corrected, cameras['c2w'][pivot, :3, :3])
    # This external SH repair is measured as B and never changes frozen sources.
    seed_all(CONFIG['seed'])
    cfg = resolve_config(CONFIG['density'])
    arrays = load_ply(corrected)
    if len(arrays['means']) > cfg['max_points']:
        raise ValueError('Initial point count exceeds the fixed 2M capacity')
    train_ids = protocol['candidate_pool']
    train = load_training_images(data['image_paths'], train_ids, protocol['shape_hw'], cfg['decode_workers']).cuda()
    cs = [classic(torch.from_numpy(cameras['c2w'][i]).cuda(),
                  torch.from_numpy(cameras['K'][i]).cuda(), *protocol['shape_hw']) for i in train_ids]
    centers = torch.from_numpy(cameras['c2w'][protocol['source_baseline'], :3, 3]).cuda()
    arrays['harmonics'][:, :, 1:] *= cfg['sh_decay']
    prepared = output / 'initial_training.ply'
    # Retain the proven covariance/factor conversion used during selection.
    save_ply(prepared, arrays)
    g = initialize(prepared, centers, CONFIG['optimizer'], sh_degree=3)
    classic_render, _, _, l1 = classic_imports()
    s = SimpleNamespace(g=g, config=cfg, protocol=protocol, O=train_ids,
        H=protocol['development'], cameras=cameras, train=train, cs=cs,
        extent=g.spatial_lr_scale, seed=CONFIG['seed'], backend='fastgs_abs',
        bg=torch.zeros(3, device='cuda'), pipeline=pipe(), abs_options=CONFIG['abs_options'])
    s.abs_accumulator = GradientAccumulator(g)
    s.geometry = JointGeometry(s, cfg['geometry_options'], cfg['relocation_options'])
    g.optimizer = SparseGaussianAdam(g.optimizer.param_groups, lr=0., eps=1e-15)
    renderer = FastGSRenderer(CONFIG['fastgs_mult'])
    ssim = TrainingSSIM()
    generator = torch.Generator().manual_seed(CONFIG['seed'])
    order, counts = [], np.zeros(len(train_ids), dtype=int)
    initial_count = len(g.get_xyz)
    torch.cuda.synchronize()
    setup_done = time.perf_counter()

    for step in range(1, CONFIG['base_steps'] + 1):
        if not order:
            order = torch.randperm(len(train_ids), generator=generator).tolist()
        j = order.pop()
        g.update_learning_rate(step)
        result = renderer(cs[j], g, s.pipeline, s.bg)
        loss = .8 * l1(result['render'], train[j]) + .2 * (1 - ssim(result['render'], train[j]))
        if step <= cfg['densify_stop']:
            s.geometry.before_backward(s, result, j, step)
        loss.backward()
        s.geometry.after_backward(s, result, j, step)
        with torch.no_grad():
            visible = result['radii'] > 0
            if step <= cfg['densify_stop']:
                accumulate_statistics(g, s.abs_accumulator, result['viewspace_points'], visible, result['radii'])
            g.optimizer.step(visible, len(g.get_xyz))
            g.optimizer.zero_grad(set_to_none=True)
            if topology_due(cfg, step):
                info = s.geometry.topology_update(s, step, cfg['max_points'], result)
                s.geometry.after_topology(s, info)
                assert_optimizer_topology(g)
            s.geometry.after_step(s, step)
        counts[j] += 1
        if step == 1 or step % 100 == 0:
            if not torch.isfinite(loss) or not all(torch.isfinite(p['params'][0]).all() for p in g.optimizer.param_groups):
                raise FloatingPointError(f'Non-finite parameters at step {step}')
        if step % 1000 == 0:
            print(f'B {step}/{CONFIG["base_steps"]}, points={len(g.get_xyz)}', flush=True)
    torch.cuda.synchronize()
    base_done = time.perf_counter()

    import lpips
    net = lpips.LPIPS(net='vgg').cuda().eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    tail = CONFIG['tail']
    g.update_learning_rate(CONFIG['base_steps'])
    for group in g.optimizer.param_groups:
        group['lr'] *= tail['lr_scale']
    optimizer = g.optimizer
    parameters = [group['params'][0] for group in optimizer.param_groups]
    moment_storage = {(i, key): value.data_ptr() for i, p in enumerate(parameters)
                      for key, value in optimizer.state[p].items() if torch.is_tensor(value)}
    point_count = len(g.get_xyz)
    torch.cuda.synchronize()
    tail_start = time.perf_counter()
    for _ in range(tail['steps']):
        if not order:
            order = torch.randperm(len(train_ids), generator=generator).tolist()
        j = order.pop()
        result = renderer(cs[j], g, s.pipeline, s.bg)
        pred, target = result['render'], train[j]
        loss = (tail['l1_weight'] * l1(pred, target) + tail['dssim_weight'] * (1 - ssim(pred, target))
                + tail['lpips_weight'] * net(pred[None], target[None], normalize=True).mean())
        loss.backward()
        optimizer.step(result['radii'] > 0, len(g.get_xyz))
        optimizer.zero_grad(set_to_none=True)
        counts[j] += 1
    torch.cuda.synchronize()
    tail_done = time.perf_counter()
    assert g.optimizer is optimizer and len(g.get_xyz) == point_count
    assert moment_storage == {(i, key): value.data_ptr() for i, p in enumerate(parameters)
                              for key, value in optimizer.state[p].items() if torch.is_tensor(value)}
    assert all(torch.isfinite(p).all() for p in parameters)
    save_model_ply(g, output / 'scene.ply')
    torch.cuda.synchronize()
    end = time.perf_counter()
    timing = dict(B_worker_seconds=end-started, export_completed_monotonic=end,
        setup_including_SH_repair_seconds=setup_done-started, base_optimization_seconds=base_done-setup_done,
        lpips_setup_seconds=tail_start-base_done, tail_optimization_seconds=tail_done-tail_start,
        final_export_seconds=end-tail_done, steps=int(counts.sum()), initial_points=initial_count,
        points=len(g.get_xyz), tail_preserved_momentum=True, tail_fixed_topology=True,
        training_indices=train_ids, test_indices=protocol['development'], sampling_counts=counts.tolist(),
        repair=repair_info)
    write_json(output / 'B-timing.json', timing)
    # Deliver PNGs with the same camera convention as training, using exported PLY.
    _, Model, _, _ = classic_imports()
    model = Model(3)
    model.load_ply(str(output / 'scene.ply'))
    from PIL import Image
    (output / 'rgb').mkdir()
    for index in protocol['development']:
        camera = classic(torch.from_numpy(cameras['c2w'][index]).cuda(),
                         torch.from_numpy(cameras['K'][index]).cuda(), *protocol['shape_hw'])
        with torch.no_grad():
            image = classic_render(camera, model, s.pipeline, s.bg)['render'].clamp(0, 1)
        Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')).save(output / 'rgb' / protocol['names'][index])
    torch.cuda.synchronize()
    write_json(output / 'render-timing.json', dict(seconds=time.perf_counter()-end,
        render_completed_monotonic=time.perf_counter(), ply_reloaded=True))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.work, args.initial, args.output)
