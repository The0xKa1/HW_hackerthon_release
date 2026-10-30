"""Byte-compatible classic GaussianModel PLY export with direct record packing.

Frozen vendor and model parameters are untouched. NumPy/plyfile are existing
runtime dependencies. This function includes file fsync, atomic replacement and
parent-directory fsync. Call it inside the caller's measured output interval.
"""
from __future__ import annotations
import os
from pathlib import Path
import tempfile


def records_from_model(g):
    """Pack raw GaussianModel attributes into native float32 PLY records.

    Field order, zero normals, channel-major DC/rest SH, raw opacity/logscale/
    quaternion and float32 casts match classic save_ply. No activation,
    normalization or model mutation is performed.
    """
    import numpy as np

    names = g.construct_list_of_attributes()
    xyz = g._xyz.detach().cpu().numpy()
    records = np.empty(len(xyz), dtype=[(name, "f4") for name in names])
    matrix = records.view(np.float32).reshape(len(xyz), len(names))
    matrix[:, :3] = xyz
    matrix[:, 3:6] = 0
    offset = 6
    for array in (
        g._features_dc.detach().cpu().numpy().transpose(0, 2, 1),
        g._features_rest.detach().cpu().numpy().transpose(0, 2, 1),
        g._opacity.detach().cpu().numpy(),
        g._scaling.detach().cpu().numpy(),
        g._rotation.detach().cpu().numpy(),
    ):
        width = int(np.prod(array.shape[1:]))
        # Fill an N,C,SH view without a contiguous transpose temporary.
        matrix[:, offset:offset + width].reshape(array.shape)[:] = array
        offset += width
    if offset != len(names):
        raise ValueError("Gaussian attribute/property count mismatch")
    return records


def save_model_ply(g, path):
    """Save the SAME raw GaussianModel attributes/header as classic save_ply.

    XYZ, zero normals, channel-major DC/rest SH, raw opacity, raw logscale and
    raw quaternion are written with the vendor's property names/native f4 dtype.
    No activation, normalization, precision policy or SH conversion is applied.
    Existing destination is replaced only after the complete temp file is synced.
    Like vendor save_ply, returns None. Does not invoke CUDA APIs directly; a CUDA
    model's ordinary detach/cpu transfers are synchronous and part of this call.
    """
    from plyfile import PlyData, PlyElement

    data = PlyData([PlyElement.describe(records_from_model(g), 'vertex')])
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + destination.name + '.', suffix='.tmp',
                                      dir=str(destination.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            data.write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        # A failed write leaves an existing destination intact. A failed directory
        # fsync can report an error after replace, but never exposes a partial PLY.
        if os.path.exists(temporary):
            os.unlink(temporary)
