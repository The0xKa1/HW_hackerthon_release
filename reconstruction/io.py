"""Validated world-frame Gaussian I/O with covariance factor conversion."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from . import _ply_codec as _codec

def validate(g: dict[str, np.ndarray]) -> None:
    """Require nonempty, finite world means/covariances/SH and bounded opacity."""
    n = len(g['means'])
    if n == 0 or g['means'].shape != (n, 3):
        raise ValueError('means must have nonempty shape (N,3)')
    for key, shape in [('covariances',(n,3,3)),('opacities',(n,))]:
        if g[key].shape != shape:
            raise ValueError(f'{key} must have shape {shape}')
    sh = g['harmonics']
    if sh.ndim != 3 or sh.shape[:2] != (n,3) or sh.shape[2] not in (1,4,9,16):
        raise ValueError('harmonics must be (N,3,(degree+1)^2), degree 0..3')
    for key in ['means','covariances','opacities','harmonics']:
        if not np.isfinite(g[key]).all():
            raise ValueError(f'nonfinite {key}')
    if np.any((g['opacities']<0)|(g['opacities']>1)):
        raise ValueError('opacity must be in [0,1]')
    if not np.allclose(g['covariances'], g['covariances'].transpose(0,2,1), atol=1e-7):
        raise ValueError('covariances must be symmetric')

def load_ply(path: str | Path) -> dict[str, np.ndarray]:
    """Read standard 3DGS PLY: wxyz, log scales, logits, channel-major SH."""
    g = _codec.load_standard(Path(path))
    validate(g)
    return g

def save_ply(path: str | Path, g: dict[str, np.ndarray]) -> dict:
    """Export using proven factors to preserve numerically near-planar axes."""
    from scipy.spatial.transform import Rotation
    validate(g)
    factors = None
    if 'scales' in g and 'rotations_wxyz' in g:
        factors = Rotation.from_quat(g['rotations_wxyz'][:,[1,2,3,0]]).as_matrix()*g['scales'][:,None,:]
    return _codec.export_world(Path(path), g['means'], g['covariances'], g['harmonics'], g['opacities'], world_factors=factors)

