"""Auditable resources, output guards and reproducibility metadata."""
from __future__ import annotations
import hashlib,json
from pathlib import Path

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def write_json(path: Path,data) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def seed_all(seed: int) -> None:
    import random,numpy as np,torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_float32_matmul_precision('high')
