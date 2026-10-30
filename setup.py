#!/usr/bin/env python3
"""Install the pinned runtime and external assets; requires a CUDA toolchain."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT/'resources.json').read_text())


def run(*args, cwd=ROOT):
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def checkout(url, commit, destination):
    if not destination.exists():
        run('git', 'clone', url, destination)
        run('git', '-C', destination, 'checkout', '--detach', commit)
    head = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(destination), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if head != commit or dirty:
        raise RuntimeError(f'Expected pristine {commit}: {destination}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets-only', action='store_true', help='Acquire/check sources and ReSplat weights without installing Python packages')
    a = p.parse_args()
    assets = ROOT/'assets';assets.mkdir(exist_ok=True)
    vendor = Path(os.environ.get('RESPLAT_VENDOR', assets/'hackathon_competition')).resolve()
    fastgs = Path(os.environ.get('FASTGS_SOURCE', assets/'FastGS')).resolve()
    faster = Path(os.environ.get('FASTERGS_SOURCE', assets/'FasterGS')).resolve()
    checkout(SPEC['repository'], SPEC['commit'], vendor)
    checkout(SPEC['fastgs_kernel_repository'], SPEC['fastgs_kernel_commit'], fastgs)
    checkout(SPEC['fastergs_repository'], SPEC['fastergs_commit'], faster)
    weight = Path(os.environ.get('RESPLAT_CHECKPOINT', assets/'weights'/SPEC['checkpoint'])).resolve()
    weight.parent.mkdir(parents=True, exist_ok=True)
    if not weight.exists():
        partial = weight.with_suffix('.download')
        urllib.request.urlretrieve(SPEC['checkpoint_source'], partial)
        with partial.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        if digest != SPEC['checkpoint_sha256']:
            raise RuntimeError('Downloaded checkpoint SHA256 mismatch')
        partial.replace(weight)
    with weight.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != SPEC['checkpoint_sha256']:
            raise RuntimeError('Checkpoint SHA256 mismatch')
    if a.assets_only:
        return
    pip = [sys.executable, '-m', 'pip', 'install']
    run(*pip, 'torch==2.7.0', 'torchvision==0.22.0', '--index-url', 'https://download.pytorch.org/whl/cu128')
    run(*pip, '-r', vendor/'resplat/requirements.txt', '-r', ROOT/'requirements.txt')
    run(*pip, '--no-build-isolation', 'gsplat==1.5.3')
    for source in [vendor/'resplat/src/model/encoder/pointops',
                   vendor/'baseline/3dgs/submodules/diff-gaussian-rasterization',
                   vendor/'baseline/3dgs/submodules/simple-knn',
                   vendor/'baseline/3dgs/submodules/fused-ssim',
                   fastgs/'submodules/diff-gaussian-rasterization_fastgs',
                   faster/'gs_submodules/diff-gaussian-rasterization_faster']:
        # Build in the source directory: pointops resolves CUDA paths relative to cwd.
        run(*pip, '--no-build-isolation', '--no-deps', '.', cwd=source)
    # Warm/download perceptual weights at installation, outside reconstruction timing.
    run(sys.executable, '-c', "import lpips; lpips.LPIPS(net='vgg')")
    print('Ready: python run_pipeline.py --scene <scene> --output <new-output>')


if __name__ == '__main__':
    main()
