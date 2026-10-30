"""Signed-gradient cloning and absolute-gradient splitting with fixed quotas.

Training statistics come from FastGS. The gsplat camera adapter and settings
are used only for expected-depth evidence in the early opacity repair.
"""
from __future__ import annotations




def camera_matrices(camera):
    """Invert classic camera construction, preserving off-center intrinsics.

    Classic samples integer pixel centers, gsplat samples integer+0.5. The
    gsplat principal point is therefore original cx/cy +0.5. This aligns
    Gaussian-to-pixel offsets; it does not establish backend equivalence.
    """
    import torch
    view = camera.world_view_transform.T.contiguous()
    projection = torch.linalg.solve(camera.world_view_transform, camera.full_proj_transform).T
    w, h = camera.image_width, camera.image_height
    K = torch.zeros((3, 3), dtype=view.dtype, device=view.device)
    K[0, 0] = projection[0, 0] * w / 2.
    K[1, 1] = projection[1, 1] * h / 2.
    K[0, 2] = (projection[0, 2] + 1) * w / 2.
    K[1, 2] = (projection[1, 2] + 1) * h / 2.
    K[2, 2] = 1
    return view[None], K[None]








class GradientAccumulator:
    """Average per-view vector norms; denominator counts projected-valid views.

    Repeated training visits count as separate optimization exposures. Projection
    validity is not alpha-composited visibility. Invisible views are excluded.
    """
    def __init__(self, g):
        self.reset(g)

    def reset(self, g):
        import torch
        self.absolute = torch.zeros((len(g.get_xyz), 1), device=g.get_xyz.device)


    def means(self, g):
        import torch
        signed = torch.nan_to_num(g.xyz_gradient_accum / g.denom, nan=0.)[:, 0]
        absolute = torch.nan_to_num(self.absolute / g.denom, nan=0.)[:, 0]
        return signed, absolute


def select_parents(signed, absolute, denom, scales, extent, percent_dense,
                   slots, births_per_event, split_fraction):
    """Allocate clone/split budgets using signed clone and absolute split scores."""
    import torch
    seen = denom.reshape(-1) > 0
    finite = torch.isfinite(signed) & torch.isfinite(absolute)
    large = scales.max(dim=1).values > percent_dense * extent
    small_ids = torch.nonzero(seen & finite & ~large & (signed > 0)).flatten()
    split_score = absolute
    large_ids = torch.nonzero(seen & finite & large & (split_score > 0)).flatten()
    total = min(max(0, slots), births_per_event)
    split_request = int(total * split_fraction)
    clone_request = total - split_request
    clone_n, split_n = min(clone_request, len(small_ids)), min(split_request, len(large_ids))
    clones = small_ids[torch.topk(signed[small_ids], clone_n).indices] if clone_n else small_ids[:0]
    splits = large_ids[torch.topk(split_score[large_ids], split_n).indices] if split_n else large_ids[:0]
    return clones, splits, {'clones': clone_n, 'splits': split_n,
        'requested_clones': clone_request, 'requested_splits': split_request,
        'clone_candidates': len(small_ids), 'split_candidates': len(large_ids),
        'net_births': clone_n + split_n, 'new_child_points': clone_n + 2 * split_n,
        'unfilled_births': clone_request + split_request - clone_n - split_n,
        'count_control': 'fixed top-k category budgets'}


def topology_update(g, cfg, extent, radii, cap, accumulator,
                    births_per_event=10000, split_fraction=.5):
    import torch
    before = len(g.get_xyz)
    signed, absolute = accumulator.means(g)
    clones, splits, info = select_parents(signed, absolute, g.denom, g.get_scaling,
        extent, g.percent_dense, max(0, cap-before), births_per_event, split_fraction)
    clone_scores, split_scores = torch.zeros_like(signed), torch.zeros_like(signed)
    clone_scores[clones], split_scores[splits] = 1., 1.
    g.tmp_radii = radii
    # Untouched classic operators provide original clone/split and Adam surgery.
    g.densify_and_clone(clone_scores[:, None], .5, extent)
    g.densify_and_split(split_scores[:, None], .5, extent)
    prune = (g.get_opacity < cfg['alpha_prune']).squeeze(-1)
    info['pruned_low_alpha'] = int(prune.sum())
    g.prune_points(prune)
    g.tmp_radii = None
    accumulator.reset(g)
    if g.get_xyz.is_cuda:
        torch.cuda.empty_cache()
    info.update(before=before, after=len(g.get_xyz), quota_cap=cap,
                operator='abs-split', signed_mean=float(signed.mean()), absolute_mean=float(absolute.mean()),
                clone_score='signed', split_score='pixel-absolute')
    return info

DEPTH_SETTINGS = dict(packed=False, sparse_grad=False, absgrad=False, rasterize_mode="classic", eps2d=.3, radius_clip=0., near_plane=.01, far_plane=200., tile_size=16, render_mode="RGB+ED", camera_model="pinhole")
