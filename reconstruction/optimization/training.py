"""Training statistics and fused SSIM for continuation.

Do not re-export this module from CPU entry points: it imports torch. Nothing
here selects a configuration, patches vendor code, or installs an operator.
Statistics consume the FastGS signed and absolute gradient channels.

Keep the existing LR/render/loss/backward/hooks -> statistics -> Adam step ->
zero_grad -> topology order and the operator/densify_stop gate. Statistics
replace BOTH the max-radii update and GradientAccumulator.add, exactly once.
The caller still resets/resizes buffers after topology and owns distinct-view
counts. SH layout/degree, parameter identities, sampling and renderer settings
are untouched. Construct a TrainingSSIM inside timed B setup.
"""
from __future__ import annotations

import torch

__all__ = [
    "accumulate_statistics",
    "TrainingSSIM",
]


@torch.no_grad()
def accumulate_statistics(g, accumulator, proxy, visible, radii):
    """Update four statistics buffers without dynamic indexing or scalar reads.

    Buffers must match the current point count: max_radii2D/radii/visible are
    (N,), signed/absolute accumulators and denom are (N, 1). The proxy provides
    true signed/absolute NDC gradients with at least two columns; missing
    absolute gradients have no signed-gradient fallback. Reset, means and
    topology behavior remain with the original accumulator.

    where preserves invisible rows even when their radii/gradients contain
    NaNs. Original proxy NDC conversion and torch.norm are retained. This
    updates neither gradients used by Adam nor SH/geometry parameters.
    """
    mask = visible[:, None]
    g.max_radii2D.copy_(torch.where(
        visible, torch.maximum(g.max_radii2D, radii.float()), g.max_radii2D))
    signed, absolute = proxy.grad, proxy.absgrad
    signed_norm = torch.norm(signed[:, :2], dim=-1, keepdim=True)
    absolute_norm = torch.norm(absolute[:, :2], dim=-1, keepdim=True)
    g.xyz_gradient_accum.copy_(torch.where(
        mask, g.xyz_gradient_accum + signed_norm, g.xyz_gradient_accum))
    g.denom.copy_(torch.where(mask, g.denom + 1, g.denom))
    accumulator.absolute.copy_(torch.where(
        mask, accumulator.absolute + absolute_norm, accumulator.absolute))


class TrainingSSIM:
    """Fused 11x11 Gaussian SSIM with classic zero-padding and RGB reduction.

    The CUDA kernel replaces per-target moment caches. Training stays float32;
    target RGB is frozen, and only prediction gradients are needed. Official
    evaluation remains the organizer's independent metric implementation.
    """

    implementation = 'fused_ssim_same_float32'

    def __init__(self):
        from fused_ssim import fused_ssim
        self._fused_ssim = fused_ssim

    def __call__(self, img1, img2, window_size=11, size_average=True):
        if window_size != 11 or not size_average:
            raise ValueError('Competition training uses averaged 11x11 SSIM')
        if img2.requires_grad:
            raise ValueError('Training target RGB must be frozen')
        if img1.ndim != 3 or img1.shape != img2.shape or img1.shape[0] != 3:
            raise ValueError('Expected matching RGB CHW tensors')
        if (not img1.is_cuda or img1.device != img2.device or
                img1.dtype != torch.float32 or img2.dtype != torch.float32):
            raise ValueError('Fused training SSIM requires float32 tensors on one CUDA device')
        return self._fused_ssim(img1.unsqueeze(0), img2.unsqueeze(0), padding='same')

