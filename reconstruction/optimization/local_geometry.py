"""Bounded online RGB geometry evidence for continuation hooks.

Independent LPM/GaussianPro-inspired migration, not a paper reproduction.
All camera setup, ED proposals, patch matching, sparse compositing and opacity repair
execute inside B. No learned matcher, sparse anchors or persistent scene cache.
ED means gsplat's alpha-normalized expected Gaussian CENTER camera-z.
It is not RaDe's ray maximum, affine-plane depth or median depth.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F


DEFAULTS = {
    "mode": "opacity", "event_steps": (100, 300, 500),
    "patches": 48, "patch_radius": 2, "tile": 12,
    "depth_planes": 17, "depth_min_factor": .55, "depth_max_factor": 1.8,
    "min_ncc": .85, "min_margin": .04, "min_texture": .02,
    "depth_agreement": .06, "alternative_separation": .12,
    "min_parallax_deg": 1., "max_view_angle_deg": 45.,
    "min_rgb_error": .04, "proposal_min_alpha": .2,
    "foreground_gap": .15, "min_contribution": .05,
    "opacity_factor": .8, "max_opacity_edits": 128,
    "max_hook_seconds": 1.5,
}


def validate_options(options):
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown geometry options: {sorted(unknown)}")
    o = {**DEFAULTS, **options}
    if o["mode"] != "opacity":
        raise ValueError("Only opacity repair is supported")
    if (not isinstance(o["event_steps"], (list, tuple)) or
            not o["event_steps"] or
            any(type(x) is not int or x < 100 or x % 100
                for x in o["event_steps"]) or
            len(set(o["event_steps"])) != len(o["event_steps"])):
        raise ValueError("Use distinct positive topology events at multiples of 100")
    for key, low, high in (("patches", 1, 96), ("patch_radius", 1, 3),
                           ("tile", 6, 32), ("depth_planes", 5, 25),
                           ("max_opacity_edits", 0, 512)):
        if type(o[key]) is not int or not low <= o[key] <= high:
            raise ValueError(f"Invalid bounded integer option: {key}")
    for key in ("min_ncc", "min_margin", "min_texture", "depth_agreement",
                "alternative_separation", "min_rgb_error", "proposal_min_alpha",
                "foreground_gap", "min_contribution", "opacity_factor"):
        if not math.isfinite(o[key]) or not 0 < o[key] < 1:
            raise ValueError(f"Invalid fraction option: {key}")
    for key in ("max_hook_seconds",
                "min_parallax_deg", "max_view_angle_deg"):
        if not math.isfinite(o[key]) or o[key] <= 0:
            raise ValueError(f"Invalid positive option: {key}")
    if not .1 <= o["depth_min_factor"] < 1 < o["depth_max_factor"] <= 4:
        raise ValueError("Depth search factors must bracket one and be bounded")
    o["event_steps"] = tuple(sorted(o["event_steps"]))
    return o


def sample_pixels(image, xy):
    """CHW -> [...,C], coordinates are integer-centered image coordinates."""
    h, w = image.shape[-2:]
    grid = xy.reshape(1, -1, 1, 2).clone()
    grid[..., 0] = (grid[..., 0] + .5) * 2 / w - 1
    grid[..., 1] = (grid[..., 1] + .5) * 2 / h - 1
    out = F.grid_sample(image[None], grid, mode="bilinear",
                        padding_mode="zeros", align_corners=False)
    return out[0, :, :, 0].T.reshape(*xy.shape[:-1], image.shape[0])


def backproject(xy, depth, c2w, K):
    p = torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1)
    local = (p @ torch.linalg.inv(K).T) * depth[..., None]
    return local @ c2w[:3, :3].T + c2w[:3, 3]


def project(xyz, c2w, K):
    local = (xyz - c2w[:3, 3]) @ c2w[:3, :3]
    screen = local @ K.T
    z = local[..., 2]
    return screen[..., :2] / z[..., None].clamp_min(1e-8), z


def patch_offsets(radius, device, dtype):
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    return torch.stack((xx.flatten(), yy.flatten()), dim=-1)


def zncc(reference, warped, min_texture):
    """Mean-subtracted RGB patch NCC; reject untextured patches per channel."""
    # [..., pixels, channels]: remove each channel's spatial mean, so a flat
    # colored patch cannot pass merely because its R/G/B values differ.
    a = reference - reference.mean(dim=-2, keepdim=True)
    b = warped - warped.mean(dim=-2, keepdim=True)
    aa, bb = a.square().mean((-2, -1)), b.square().mean((-2, -1))
    similarity = (a * b).mean((-2, -1)) / (aa * bb).sqrt().clamp_min(1e-12)
    valid = (aa >= min_texture ** 2) & (bb >= min_texture ** 2)
    return similarity.clamp(-1, 1), valid


@torch.no_grad()
def match_planes(reference, sources, ref_c2w, ref_K, src_c2w, src_K,
                 xy, depths, options):
    """Independent two-view RGB evidence, with a shared candidate depth grid.

    Each hypothesis is a reference-camera fronto-parallel plane. Patches warp by
    backprojection/reprojection (including differing intrinsics). Source depth
    maps are deliberately absent; this is NOT GaussianPro geometric cycle-check.
    """
    o = options
    count, planes = depths.shape
    if len(sources) != 2:
        raise ValueError("Exactly two distinct support images are required")
    offsets = patch_offsets(o["patch_radius"], xy.device, xy.dtype)
    patch = xy[:, None, :] + offsets[None]
    ref_patch = sample_pixels(reference, patch)
    world = backproject(patch[:, None].expand(-1, planes, -1, -1),
                        depths[:, :, None].expand(-1, -1, len(offsets)),
                        ref_c2w, ref_K)
    center_world = backproject(xy[:, None].expand(-1, planes, -1),
                               depths, ref_c2w, ref_K)
    ray_ref = F.normalize(center_world - ref_c2w[:3, 3], dim=-1)
    scores, validities = [], []
    for k in range(2):
        uv, z = project(world, src_c2w[k], src_K[k])
        h, w = sources[k].shape[-2:]
        inside = ((uv[..., 0] >= 0) & (uv[..., 0] <= w - 1) &
                  (uv[..., 1] >= 0) & (uv[..., 1] <= h - 1) & (z > .01)).all(-1)
        ref_h, ref_w = reference.shape[-2:]
        ref_inside = ((patch[..., 0] >= 0) & (patch[..., 0] <= ref_w - 1) &
                      (patch[..., 1] >= 0) & (patch[..., 1] <= ref_h - 1)).all(-1)
        score, texture = zncc(ref_patch[:, None], sample_pixels(sources[k], uv),
                              o["min_texture"])
        ray_src = F.normalize(center_world - src_c2w[k, :3, 3], dim=-1)
        parallax = (ray_ref * ray_src).sum(-1)
        valid = (inside & ref_inside[:, None] & texture &
                 (parallax < math.cos(math.radians(o["min_parallax_deg"]))) &
                 torch.isfinite(depths) & (depths > .01))
        scores.append(torch.where(valid, score, score.new_full((), -2.)))
        validities.append(valid)
    score = torch.stack(scores)  # two, patches, depth hypotheses
    per_best, per_idx = score.max(-1)
    row = torch.arange(count, device=xy.device)
    chosen = score.mean(0).argmax(-1)
    best_depth = depths[row, chosen]
    individual_depth = torch.stack([depths[row, per_idx[k]] for k in range(2)])
    agreement = ((individual_depth - best_depth).abs() /
                 best_depth.clamp_min(1e-8)).max(0).values <= o["depth_agreement"]
    chosen_score = score[:, row, chosen]
    # Adjacent depth samples are correlated; uniqueness excludes a local depth
    # neighborhood. Require a VALID remote alternative in BOTH views. Otherwise
    # an out-of-bounds plane could manufacture a spurious "unique" match.
    remote = ((depths - best_depth[:, None]).abs() /
              best_depth[:, None].clamp_min(1e-8)) >= o["alternative_separation"]
    remote_valid = torch.stack(validities) & remote[None]
    other = score.masked_fill(~remote_valid, -2.).max(-1).values
    margin = (chosen_score - other).min(0).values
    valid = (agreement & (chosen_score.min(0).values >= o["min_ncc"]) &
             (margin >= o["min_margin"]) & remote_valid.any(-1).all(0) &
             torch.isfinite(best_depth))
    return {"depth": best_depth, "valid": valid, "margin": margin,
            "ncc": chosen_score.min(0).values,
            "world": center_world[row, chosen],
            "individual_depth": individual_depth, "scores": score}


@torch.no_grad()
def select_error_patches(rgb, target, options):
    o = options
    error = (rgb.detach() - target).abs().mean(0)
    h, w = error.shape
    r = o["patch_radius"] + 1
    score = F.avg_pool2d(error[None, None], 2 * r + 1, 1, r)[0, 0]
    score[:r] = score[-r:] = -1
    score[:, :r] = score[:, -r:] = -1
    vals, ids = F.max_pool2d(score[None, None], o["tile"], o["tile"],
                           return_indices=True)
    vals, ids = vals.flatten(), ids.flatten()
    n = min(o["patches"], len(ids))
    selected = vals.topk(n).indices
    ids, vals = ids[selected], vals[selected]
    ok = vals >= o["min_rgb_error"]
    ids, vals = ids[ok], vals[ok]
    return torch.stack((ids % w, ids // w), -1).to(rgb.dtype), vals


@torch.no_grad()
def foreground_contribution(meta, opacity, xy, matched_depth, gap=.15,
                            min_weight=.05, chunk=8):
    """Sparse alpha compositing from current gsplat metadata, detached.

    Integer image centers are +.5 in gsplat. We sort by center z and integrate
    all valid splats with alpha cutoff 1/255 and transmittance cutoff 1e-4.
    This is a sparse compositing reconstruction, not native per-Gaussian renderer
    instrumentation: tile culling/float accumulation can differ at boundaries.
    Projection validity alone never triggers a point edit.
    """
    means, depths, conics = (meta[k][0].detach() for k in ("means2d", "depths", "conics"))
    radii = meta["radii"][0].detach()
    n = len(depths)
    finite = (torch.isfinite(depths) & torch.isfinite(means).all(-1) &
              torch.isfinite(conics).all(-1))
    valid = (radii > 0).all(-1) if radii.ndim == 2 else radii > 0
    order = torch.argsort(depths.masked_fill(~finite, float("inf")))
    means, z, conics = means[order], depths[order], conics[order]
    radii, valid = radii[order], (valid & finite & (depths > .01))[order]
    op = opacity.detach().reshape(-1)[order].clamp(0, .999)
    strength = op.new_zeros(n)
    for start in range(0, len(xy), chunk):
        pix = xy[start:start + chunk] + .5
        d = pix[:, None] - means[None]
        q = (conics[None, :, 0] * d[..., 0].square() +
             2 * conics[None, :, 1] * d[..., 0] * d[..., 1] +
             conics[None, :, 2] * d[..., 1].square())
        alpha = (op[None] * torch.exp(-.5 * q.clamp_min(0))).clamp(max=.999)
        bounds = (d.abs() <= (radii if radii.ndim == 2 else radii[:, None])[None]).all(-1)
        alpha = torch.where(valid[None] & bounds & (q >= 0) & torch.isfinite(q) &
                            (alpha >= 1 / 255.), alpha, torch.zeros_like(alpha))
        post = torch.cumprod(1 - alpha, dim=-1)
        before = torch.cat((torch.ones_like(post[:, :1]), post[:, :-1]), -1)
        # gsplat discards the splat that would cross the termination threshold.
        weight = torch.where(post > 1e-4, before * alpha, torch.zeros_like(alpha))
        foreground = z[None] < matched_depth[start:start + chunk, None] * (1 - gap)
        contribution = torch.where(foreground & (weight >= min_weight), weight, 0.)
        strength = torch.maximum(strength, contribution.max(0).values)
    result = strength.new_zeros(n)
    result[order] = strength
    return result


@torch.no_grad()
def attenuate_opacity(g, ids, factor):
    """Keep parameters registered; reset only the edited opacity Adam rows."""
    from reconstruction.optimization.density import zero_moments
    if len(ids):
        alpha = (g.get_opacity[ids] * factor).clamp(1e-6, 1 - 1e-6)
        g._opacity[ids] = torch.logit(alpha)
        zero_moments(g.optimizer, g._opacity, ids)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class GeometryHook:
    def __init__(self, state, **options):
        t = time.perf_counter()
        self.options = validate_options(options)
        self.events, self.pending, self.applied = [], None, set()
        self.device = state.g.get_xyz.device
        self.dtype = state.g.get_xyz.dtype
        self.seed = int(getattr(state, "seed", 0))
        # Access camera entries in O only; array positions are global image IDs.
        ids = list(state.O)
        self.c2w = torch.as_tensor(state.cameras["c2w"][ids],
                                   device=self.device, dtype=self.dtype)
        self.K = torch.as_tensor(state.cameras["K"][ids],
                                 device=self.device, dtype=self.dtype).clone()
        h, w = state.train.shape[-2:]
        self.K[:, 0] *= w
        self.K[:, 1] *= h
        if len(ids) != len(state.train) or self.c2w.shape[1:] != (4, 4):
            raise ValueError("Training pool/camera indexing mismatch")
        centers, forwards = self.c2w[:, :3, 3], self.c2w[:, :3, 2]
        distance = torch.cdist(centers, centers)
        dot = F.normalize(forwards, dim=-1) @ F.normalize(forwards, dim=-1).T
        extent = float(state.extent)
        eligible = ((distance > .002 * extent) & (distance < .8 * extent) &
                    (dot > math.cos(math.radians(self.options["max_view_angle_deg"]))))
        # Favor useful small baselines, not coincident cameras.
        rank = (distance / extent - .05).abs().masked_fill(~eligible, float("inf"))
        k = min(2, len(ids))
        values, self.neighbors = rank.topk(k, largest=False)
        self.neighbor_valid = torch.isfinite(values).all(-1) & (k == 2)
        _synchronize(self.device)
        self.setup_seconds = time.perf_counter() - t
        self.work_seconds = self.setup_seconds

    @torch.no_grad()
    def _expected_depth(self, s, j, *, metadata_target=None):
        from gsplat.rendering import rasterization
        from reconstruction.optimization.absolute_gradient import camera_matrices
        view, K = camera_matrices(s.cs[j])
        from reconstruction.optimization.absolute_gradient import DEPTH_SETTINGS
        settings = dict(DEPTH_SETTINGS)
        settings.update(render_mode="RGB+ED", absgrad=False)
        rgbd, alpha, meta = rasterization(
            means=s.g.get_xyz, quats=s.g._rotation, scales=s.g.get_scaling,
            opacities=s.g.get_opacity[:, 0], colors=s.g.get_features,
            sh_degree=s.g.active_sh_degree, viewmats=view, Ks=K,
            width=s.train.shape[-1], height=s.train.shape[-2],
            backgrounds=s.bg[None], **settings)
        if metadata_target is not None:
            metadata_target["meta"] = meta
        return rgbd[0, :, :, -1], alpha[0, :, :, 0]

    @torch.no_grad()
    def before_backward(self, s, result, j, step):
        o = self.options
        if step not in o["event_steps"] or step in self.applied:
            return None
        self.applied.add(step)
        event = {"step": step, "reference_pool_local": int(j),
                 "reference_training_index": int(s.O[j]), "accepted_matches": 0,
                 "births": 0, "opacity_edits": 0, "status": "started"}
        self.events.append(event)
        if self.work_seconds >= o["max_hook_seconds"]:
            event["status"] = "work_budget_skip"
            return None
        if not bool(self.neighbor_valid[j]):
            event["status"] = "no_two_pose_neighbors"
            return None
        if getattr(s, "backend", "gsplat_abs") not in ("gsplat_abs", "fastgs_abs"):
            raise ValueError("geometry hook requires a supported ABS backend")
        _synchronize(self.device)
        start = time.perf_counter()
        try:
            if getattr(s, "backend", "gsplat_abs") == "fastgs_abs":
                # Reuse projection metadata from the existing depth pass.
                result = dict(result)
                event['projection_evidence_backend'] = 'gsplat_abs_depth_pass'
            self._prepare(s, result, j, event)
        finally:
            _synchronize(self.device)
            event["evidence_seconds"] = time.perf_counter() - start
            self.work_seconds += event["evidence_seconds"]
        return None

    def _prepare(self, s, result, j, event):
        o = self.options
        xy, errors = select_error_patches(result["render"], s.train[j], o)
        event["patches_attempted"] = len(xy)
        if not len(xy):
            event["status"] = "no_high_error_patches"
            return
        if getattr(s, "backend", "gsplat_abs") == "fastgs_abs":
            depth, alpha = self._expected_depth(s, j, metadata_target=result)
        else:
            depth, alpha = self._expected_depth(s, j)
        center_depth = sample_pixels(depth[None], xy)[:, 0]
        center_alpha = sample_pixels(alpha[None], xy)[:, 0]
        offsets = xy.new_tensor([[0, 0], [-4, 0], [4, 0], [0, -4], [0, 4]])
        local_depth = sample_pixels(depth[None], xy[:, None] + offsets)[:, :, 0]
        local_alpha = sample_pixels(alpha[None], xy[:, None] + offsets)[:, :, 0]
        factors = torch.linspace(math.log(o["depth_min_factor"]),
                                 math.log(o["depth_max_factor"]),
                                 o["depth_planes"], device=xy.device).exp()
        proposal = torch.cat((center_depth[:, None] * factors, local_depth), -1)
        valid_center = ((center_depth > .01) & torch.isfinite(center_depth) &
                        (center_alpha >= o["proposal_min_alpha"]))
        # Invalid neighboring ED values cannot become accepted geometry evidence.
        proposal[:, o["depth_planes"]:] = proposal[:, o["depth_planes"]:].masked_fill(
            local_alpha < o["proposal_min_alpha"], float("nan"))
        # NaNs are replaced before grid_sample; invalid proposals stay negative
        # and are rejected by match_planes' positive-depth gate.
        proposal = torch.nan_to_num(proposal, nan=-1., posinf=-1., neginf=-1.)
        src = self.neighbors[j]
        event["support_training_indices"] = [int(s.O[int(k)]) for k in src]
        match = match_planes(s.train[j], s.train[src], self.c2w[j], self.K[j],
                             self.c2w[src], self.K[src], xy, proposal, o)
        valid = match["valid"] & valid_center
        event["accepted_matches"] = int(valid.sum())
        if not bool(valid.any()):
            event["status"] = "no_confident_multiview_match"
            return
        xy, d = xy[valid], match["depth"][valid]
        xyz = match["world"][valid]
        change = (d - center_depth[valid]).abs() / d.clamp_min(1e-8)
        event["mean_ncc"] = float(match["ncc"][valid].mean())
        event["mean_relative_ed_difference"] = float(change.mean())
        pending = {"event": event, "children": None, "opacity_ids": None,
                   "positive_gradient_ids": None}
        if o["max_opacity_edits"]:
            strength = foreground_contribution(result["meta"], s.g.get_opacity, xy, d,
                o["foreground_gap"], o["min_contribution"])
            ids = torch.nonzero(strength > 0).flatten()
            n = min(len(ids), o["max_opacity_edits"])
            ids = ids[strength[ids].topk(n).indices] if n else ids[:0]
            pending["opacity_ids"] = ids
            event["foreground_contributors"] = len(ids)
        event["status"] = "evidence_ready"
        self.pending = pending

    @torch.no_grad()
    def after_backward(self, s, result, j, step):
        del result, j
        p = self.pending
        if p is None or p["event"]["step"] != step or p["opacity_ids"] is None:
            return
        ids = p["opacity_ids"]
        grad = s.g._opacity.grad
        # Positive derivative: a small opacity decrease locally reduces CURRENT
        # training objective (RGB for these standalone candidates; other hooks
        # may add losses in a root-composed run). Not global geometry proof.
        if grad is None:
            p["positive_gradient_ids"] = ids[:0]
        else:
            val = grad.detach().reshape(-1)[ids]
            p["positive_gradient_ids"] = ids[torch.isfinite(val) & (val > 0)]

    @torch.no_grad()
    def topology_update(self, s, step, cap, result):
        p = self.pending
        if p is None or p["event"]["step"] != step:
            return None
        _synchronize(self.device)
        start = time.perf_counter()
        event, o = p["event"], self.options
        ids = p["positive_gradient_ids"]
        if ids is not None:
            attenuate_opacity(s.g, ids, o["opacity_factor"])
            event["opacity_edits"] = len(ids)
        event.update(status="applied", topology_handled=False)
        _synchronize(self.device)
        elapsed = time.perf_counter() - start
        event["management_seconds_including_baseline_topology"] = elapsed
        self.work_seconds += elapsed
        self.pending = None
        return None

    def after_topology(self, s, info):
        del s, info
        # If the root did not invoke topology_update, do not reuse stale point IDs.
        if self.pending is not None:
            self.pending["event"]["status"] = "hook_topology_interface_not_called"
        self.pending = None



