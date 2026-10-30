"""Pinned external assets; no dependency on the development checkout."""
import json
import os
from pathlib import Path
import subprocess
from .runtime import sha256

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / 'resources.json').read_text())
CONFIG = json.loads((ROOT / 'config.json').read_text())
VENDOR = Path(os.environ.get('RESPLAT_VENDOR', ROOT / 'assets/hackathon_competition')).resolve()
BASELINE = VENDOR / 'baseline'
CHECKPOINT = Path(os.environ.get('RESPLAT_CHECKPOINT', ROOT / 'assets/weights' / SPEC['checkpoint'])).resolve()


def verify_vendor():
    head = subprocess.check_output(['git', '-C', str(VENDOR), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(VENDOR), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if head != SPEC['commit'] or dirty:
        raise ValueError('Frozen ReSplat repository must match the pristine pinned commit')
    names = subprocess.check_output(['git', '-C', str(VENDOR), 'ls-files', '-z']).decode().split('\0')
    return {name: sha256(VENDOR / name) for name in names if name}


def verify_checkpoint():
    digest = sha256(CHECKPOINT)
    if digest != SPEC['checkpoint_sha256']:
        raise ValueError('ReSplat checkpoint SHA256 mismatch')
    return digest
