# Shared numerical PLY codec; public validated I/O is provided by reconstruction.io.
"""External world-covariance -> standard 3DGS PLY adapter. No network changes."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from plyfile import PlyData, PlyElement


def covariance(scales, wxyz):
    rotation = Rotation.from_quat(np.asarray(wxyz)[:, [1, 2, 3, 0]]).as_matrix()
    return (rotation * np.asarray(scales)[:, None, :] ** 2) @ rotation.transpose(0, 2, 1)


def export_world(path, means, covars, harmonics, opacity, world_factors=None):
    """Keep coordinate frame and SH basis unchanged; factor the WORLD covariance.

    PLY stores wxyz, log scale and logit opacity. Endpoints are clipped only to
    float32 epsilon for finite logits; report the count, never silently prune.
    """
    arrays = [np.asarray(x) for x in (means, covars, harmonics, opacity)]
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("non-finite input")
    means, covars, harmonics, opacity = arrays
    n = len(means)
    if covars.shape != (n, 3, 3) or harmonics.shape[:2] != (n, 3) or opacity.shape != (n,):
        raise ValueError("shape mismatch")
    if np.any((opacity < 0) | (opacity > 1)):
        raise ValueError("opacity outside [0,1]")
    sym = (covars.astype(np.float64) + covars.transpose(0, 2, 1)) / 2
    values, vectors = np.linalg.eigh(sym)
    # Near-planar upstream scales reach 1e-6; float32 world covariance can lose
    # the 1e-12 eigenvalue. Permit ONLY bounded roundoff, report every repair.
    largest = values[:, -1]
    if np.any(largest <= 0) or np.any(values[:, 0] < -1e-6 * largest):
        raise ValueError("materially non-positive world covariance")
    if world_factors is None and np.any(values <= 0):
        raise ValueError("non-positive covariance without proven source factors")
    repair_count = 0
    repair_relative = 0.0
    if world_factors is not None:
        factor = np.asarray(world_factors, dtype=np.float64)
        if factor.shape != (n,3,3) or not np.isfinite(factor).all():
            raise ValueError("invalid world factors")
        factor_cov = factor @ factor.transpose(0,2,1)
        factor_relative = np.linalg.norm(factor_cov-sym, axis=(1,2)) / np.linalg.norm(sym, axis=(1,2))
        if factor_relative.max() > 1e-5:
            raise ValueError("source factors do not reproduce world covariance")
        vectors, scales, _ = np.linalg.svd(factor)
        if np.any(scales <= 0):
            raise ValueError("degenerate source factor; cannot write finite log scale")
        repair_count = 0
        repair_relative = float(factor_relative.max())
    else:
        scales = np.sqrt(values)
    # Orthogonal columns are axes; choose a proper rotation without changing cov.
    vectors[:, :, 0] *= np.where(np.linalg.det(vectors) < 0, -1, 1)[:, None]
    quat = Rotation.from_matrix(vectors).as_quat()[:, [3, 0, 1, 2]]
    eps = np.finfo(np.float32).eps
    alpha = np.clip(opacity, eps, 1-eps)
    names = (["x", "y", "z", "nx", "ny", "nz"] +
             [f"f_dc_{i}" for i in range(3)] +
             [f"f_rest_{i}" for i in range(3 * (harmonics.shape[2]-1))] +
             ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    data = np.concatenate([means, np.zeros_like(means), harmonics[:, :, 0],
                           harmonics[:, :, 1:].reshape(n, -1),
                           (np.log(alpha)-np.log1p(-alpha))[:, None],
                           np.log(scales), quat], axis=1).astype(np.float32)
    if not np.isfinite(data).all():
        raise ValueError("non-finite export")
    records = np.empty(n, dtype=[(x, '<f4') for x in names])
    for i, name in enumerate(names):
        records[name] = data[:, i]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(records, 'vertex')]).write(path)
    return {"points": n, "opacity_endpoint_clips": int(np.count_nonzero(alpha != opacity)),
            "frame_change": False, "sh_basis_change": False,
            "roundoff_repaired_points": repair_count, "repair_frobenius_relative_max": repair_relative,
            "factor_source_used": world_factors is not None,
            "repair_policy": "proven factor SVD, no eigenvalue clipping; factor-free path rejects all nonpositive eigenvalues; no points removed"}


def load_standard(path):
    a = PlyData.read(path)['vertex'].data
    def get(names):
        return np.stack([a[x] for x in names], axis=-1).astype(np.float32)
    rest = sorted([x for x in a.dtype.names if x.startswith('f_rest_')], key=lambda x: int(x[7:]))
    if len(rest) % 3:
        raise ValueError("invalid SH fields")
    dc = get([f'f_dc_{i}' for i in range(3)])[:, :, None]
    sh = np.concatenate([dc, get(rest).reshape(len(a), 3, -1)], axis=-1) if rest else dc
    scales = np.exp(get([f'scale_{i}' for i in range(3)]).astype(np.float64))
    quat = get([f'rot_{i}' for i in range(4)])
    from scipy.special import expit
    return dict(means=get(['x', 'y', 'z']), covariances=covariance(scales, quat).astype(np.float32),
                scales=scales.astype(np.float32), rotations_wxyz=quat, harmonics=sh,
                opacities=expit(np.asarray(a['opacity'], dtype=np.float64)).astype(np.float32))


