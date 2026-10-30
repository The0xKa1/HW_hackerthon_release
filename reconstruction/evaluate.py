"""Official metrics on delivered RGB PNGs, after reconstruction completes."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time


def run(work, output):
    import torch
    from .resources import BASELINE, VENDOR
    from .runtime import write_json
    torch.set_num_threads(4)
    sys.path.insert(0, str(VENDOR / 'resplat'))
    spec = importlib.util.spec_from_file_location('official_metrics', BASELINE / 'evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = json.loads((work / 'split.json').read_text())['test']
    if sorted(p.name for p in (output / 'rgb').glob('*.png')) != sorted(names):
        raise ValueError('Delivered PNGs do not match the held-out split')
    values = {key: [] for key in ('psnr', 'ssim', 'lpips')}
    started = time.perf_counter()
    for begin in range(0, len(names), 4):
        selected = names[begin:begin+4]
        pred = module.load_batch(output / 'rgb', selected).cuda()
        gt = module.load_batch(work / 'images', selected).cuda()
        for key in values:
            values[key].append(getattr(module, 'compute_' + key)(gt, pred))
    values = {k: torch.cat(v) for k, v in values.items()}
    write_json(output / 'metrics.json', dict(num_views=len(names),
        mean={k: float(v.mean()) for k, v in values.items()},
        per_view=[dict(name=name, **{k: float(v[i]) for k, v in values.items()}) for i, name in enumerate(names)],
        metric_seconds=time.perf_counter()-started))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.work, a.output)
