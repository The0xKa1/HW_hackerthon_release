"""Correct the frozen exporter's SH1 rotation in B, without changing the model."""
import ast
import numpy as np
from plyfile import PlyData, PlyElement
from .resources import BASELINE


def repair(source, target, pivot_rotation):
    # Reuse the exact pinned export rotation, not an approximate reconstruction.
    scope = {'np': np}
    for node in ast.parse((BASELINE / 'run_resplat.py').read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in ('eval_sh_basis', 'sh_rotation_matrix'):
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'frozen_sh_rotation', 'exec'), scope)
    rotation = scope['sh_rotation_matrix'](pivot_rotation.astype(np.float64))[1:4, 1:4]
    sign = np.diag([1., -1., 1.])
    correction = (sign @ rotation @ sign) @ np.linalg.inv(rotation)
    original = PlyData.read(source)['vertex'].data
    result = original.copy()
    for channel in range(3):
        names = [f'f_rest_{channel * 15 + j}' for j in range(3)]
        fixed = np.stack([original[n] for n in names], axis=-1).astype(np.float64) @ correction.T
        for j, name in enumerate(names):
            result[name] = fixed[:, j]
    PlyData([PlyElement.describe(result, 'vertex')]).write(target)
    return {'matrix': correction.tolist(), 'changed': 'SH degree 1 only', 'uses_test_RGB': False}
