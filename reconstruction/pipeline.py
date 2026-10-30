"""Official A -> frozen ReSplat -> fixed 027+LPIPS300 -> delivery and metrics."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from .resources import ROOT, SPEC, CONFIG, VENDOR, BASELINE, CHECKPOINT, verify_vendor, verify_checkpoint
from .runtime import write_json


def gpu_environment(gpu=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    env.setdefault('OMP_NUM_THREADS', '4')
    env.setdefault('OPENBLAS_NUM_THREADS', '4')
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = subprocess.check_output(
            ['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    visible = env.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    uuid = subprocess.check_output(['nvidia-smi', '-i', visible, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    env['CUDA_VISIBLE_DEVICES'] = uuid
    return env, uuid


def stage(command, log, env):
    started = time.perf_counter()
    with log.open('w') as handle:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Stage failed ({result.returncode}); see {log}')
    return time.perf_counter()-started


def protocol_digest(work):
    digest = hashlib.sha256()
    for name in ('split.json', 'sparse/0/cameras.txt', 'sparse/0/images.txt'):
        digest.update(name.encode());digest.update((work / name).read_bytes())
    return digest.hexdigest()


def quality_score(metrics):
    clip = lambda x: max(0., min(1., x))
    return 20*clip((metrics['psnr']-15)/25) + 20*clip((metrics['ssim']-.7)/.25) + 40*clip((.25-metrics['lpips'])/.2)


def validate_reference(reference, uuid, digest):
    budget = reference.get('reference_training_seconds_median')
    if (reference.get('status') != 'complete' or reference.get('gpu_uuid') != uuid
            or reference.get('protocol_sha256') != digest
            or reference.get('reference_trainer') != 'fastgs'
            or reference.get('iterations') != 30000 or reference.get('mult') != .5
            or reference.get('commit') != SPEC['commit']
            or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0):
        raise ValueError('Reference must match scene, split, cameras, GPU and FastGS protocol')
    return budget


def apply_reference(report, reference):
    budget = validate_reference(reference, report['gpu_uuid'], report['protocol_sha256'])
    b = report['timing']['B_seconds']
    report['reference_seconds'] = budget
    report['reference'] = reference
    report['score'].update(time=20*max(0., min(1., (1-b/budget)/.9)), within_strict_B_budget=b < budget)
    report['score']['total'] = report['score']['quality']+report['score']['time'] if b < budget else None
    report['status'] = 'complete' if b < budget else 'budget_exceeded'
    return report


def save_report(output, report):
    write_json(output/'report.json', report)
    m, t, s = report['metrics']['mean'], report['timing'], report['score']
    lines = ['# Reconstruction report', '',
        '| PSNR | SSIM | LPIPS | B seconds | Full pipeline seconds | FastGS reference seconds |',
        '|---:|---:|---:|---:|---:|---:|',
        f"| {m['psnr']:.4f} | {m['ssim']:.6f} | {m['lpips']:.6f} | {t['B_seconds']:.3f} | {t['pipeline_including_render_seconds']:.3f} | {report['reference_seconds'] if report['reference_seconds'] else 'not measured'} |", '',
        f"Evaluation: {t['evaluation_seconds']:.3f}s (excluded from pipeline). Quality: {s['quality']:.3f}/80. Total: {s['total']}. Status: {report['status']}.",
        'B includes frozen wrapper postprocessing overhead. No formal acceptance claim. See report.json for all stage timings.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def main(argv=None, *, started=None):
    started = time.perf_counter() if started is None else started
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', help='Physical GPU index or UUID; otherwise respect CUDA_VISIBLE_DEVICES')
    p.add_argument('--reference', type=Path, help='Same-scene/same-GPU calibration.json from calibrate.py')
    args = p.parse_args(argv)
    scene, output = args.scene.resolve(), args.output.resolve()
    if not scene.is_dir():
        p.error('Scene directory does not exist')
    if output.exists():
        p.error('Output must be a new directory')
    before = verify_vendor()
    checkpoint_sha = verify_checkpoint()
    env, uuid = gpu_environment(args.gpu)
    output.mkdir(parents=True)
    write_json(output / 'config.json', CONFIG)
    report = dict(status='running', scene=str(scene), gpu_uuid=uuid, commit=SPEC['commit'],
        checkpoint_sha256=checkpoint_sha, configuration=CONFIG, formal_acceptance=False,
        reference_seconds=None, score=None)
    work = output / 'work'
    try:
        timing = {}
        print('A: official preprocessing and uniform 16-view selection', flush=True)
        timing['A_seconds'] = stage([sys.executable, BASELINE/'module_a/preprocess.py', '--scene_dir', scene,
            '--out', work, '--num_context', '16', '--group', 'auto'], output/'A.log', env)
        reference = None
        if args.reference:
            reference = json.loads(args.reference.read_text())
            validate_reference(reference, uuid, protocol_digest(work))
        print('ReSplat: frozen forward', flush=True)
        timing['model_process_seconds'] = stage([sys.executable, BASELINE/'run_resplat.py', '--work', work,
            '--out', output/'resplat', '--checkpoint', CHECKPOINT, '--num_refine', SPEC['num_refine']], output/'resplat.log', env)
        model = json.loads((output/'resplat/resplat.json').read_text())
        timing['model_load_seconds'] = model['model_load_seconds']
        timing['model_inference_seconds'] = model['inference_seconds']
        # Account for the frozen wrapper's output adaptation and incidental overhead
        # rather than hiding it in the neural forward. Conservative for strict B.
        overhead = max(0., timing['model_process_seconds']-model['model_load_seconds']-model['inference_seconds'])
        print('B: 027 / Sparse Adam / 5000 + LPIPS 300', flush=True)
        b_start = time.perf_counter()
        timing['B_worker_process_including_render_seconds'] = stage([sys.executable, '-m', 'reconstruction.optimize',
            '--work', work, '--initial', output/'resplat/resplat_world.ply', '--output', output], output/'B.log', env)
        b = json.loads((output/'B-timing.json').read_text())
        render = json.loads((output/'render-timing.json').read_text())
        timing['B_input_inclusive_seconds'] = b['export_completed_monotonic']-b_start
        timing['upstream_postprocess_and_overhead_seconds'] = overhead
        timing['B_seconds'] = timing['B_input_inclusive_seconds']+overhead
        timing['B_time_scope'] = 'worker launch through final PLY, plus frozen wrapper non-load/non-forward overhead; excludes final RGB rendering and metric evaluation'
        timing['render_seconds'] = render['seconds']
        timing['end_to_end_to_ply_seconds'] = b['export_completed_monotonic']-started
        timing['pipeline_including_render_seconds'] = render['render_completed_monotonic']-started
        print('Evaluation: official PSNR / SSIM / VGG-LPIPS', flush=True)
        timing['evaluation_seconds'] = stage([sys.executable, '-m', 'reconstruction.evaluate', '--work', work,
            '--output', output], output/'evaluation.log', env)
        metrics = json.loads((output/'metrics.json').read_text())
        quality = quality_score(metrics['mean'])
        scoring = dict(quality=quality, time=None, total=None, within_strict_B_budget=None,
                       convention='clip scene mean metrics to chart ranges; pending organizer aggregation details')
        unchanged = before == verify_vendor() and verify_checkpoint() == checkpoint_sha
        if not unchanged:
            raise RuntimeError('Frozen resources changed during reconstruction')
        report.update(status='complete' if scoring['within_strict_B_budget'] is not False else 'budget_exceeded',
            timing=timing, metrics=metrics, B=b, score=scoring, frozen_sources_unchanged=True,
            protocol_sha256=protocol_digest(work), artifacts={'ply':'scene.ply','rgb':'rgb','quality':'metrics.json'})
        if reference:
            apply_reference(report, reference)
        save_report(output, report)
        print(json.dumps(dict(output=str(output),metrics=metrics['mean'],timing=timing,score=scoring)), flush=True)
        return 0 if report['status']=='complete' else 2
    except Exception as error:
        report.update(status='failed', error=str(error))
        write_json(output/'report.json', report)
        raise
