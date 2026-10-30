"""Opacity-guided MCMC relocation followed by absolute-gradient splitting."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from reconstruction.optimization.density import (
    PARAMETERS, assert_optimizer_topology, zero_moments,
)


@dataclass(frozen=True)
class Options:
    dead_opacity: float = .005
    relocate_fraction: float = .02
    max_copies: int = 16

    def validate(self):
        if type(self.max_copies) is not int or not 2 <= self.max_copies <= 32:
            raise ValueError('max_copies must be an integer in [2,32]')
        if not 0 < self.dead_opacity < 1:
            raise ValueError('dead_opacity must lie in (0,1)')
        if not 0 <= self.relocate_fraction <= .05:
            raise ValueError('relocate_fraction must lie in [0,.05]')


def relocation_parameters(opacity, scales, copies, *, max_copies=16):
    """Eq.9 of 3DGS-MCMC, with a single-sum binomial identity.

    D = sum(k=1..N) (-1)^(k+1) C(N,k) a'^k/sqrt(k), using
    sum(i=k..N) C(i-1,k-1) = C(N,k). Calculations use float64;
    log1p/expm1 avoid cancellation at small alpha. No alpha floor: clamping
    children to .005 would break center preservation. Callers reject such moves.
    This is device-agnostic PyTorch, not the CUDA-only gsplat relocation helper.
    """
    opacity = opacity.reshape(-1)
    copies = copies.reshape(-1)
    if scales.shape != (len(opacity), 3) or copies.shape != opacity.shape:
        raise ValueError('Expected opacity[N], scales[N,3], copies[N]')
    if not 2 <= max_copies <= 32:
        raise ValueError('max_copies must be in [2,32]')
    if not bool(((copies >= 1) & (copies <= max_copies) &
                 (copies == copies.round())).all()):
        raise ValueError('Invalid copy multiplicity; clipping is forbidden')
    if not bool((torch.isfinite(opacity) & (opacity > 0) & (opacity < 1)).all()):
        raise ValueError('Opacity must be finite and strictly between 0 and 1')
    if not bool((torch.isfinite(scales) & (scales > 0)).all()):
        raise ValueError('Scales must be finite and positive')
    a = opacity.double()
    n = copies.to(dtype=torch.float64)
    child = -torch.expm1(torch.log1p(-a) / n)
    # Fixed small loop: no per-k GPU synchronization or host array round trip.
    term, denominator = n * child, n * child
    for k in range(2, max_copies + 1):
        term = term * (n - k + 1).clamp_min(0) / k * (-child)
        denominator = denominator + term / math.sqrt(k)
    out_scale = scales.double() * (a / denominator)[:, None]
    return child.to(opacity.dtype), out_scale.to(scales.dtype)


@torch.no_grad()
def relocate_slots(g, donors, targets, *, alpha_floor, max_copies):
    """Constant-size relocation with exact multiplicity and Adam row reset.

    Target and donor sets must be disjoint. Oversubscribed or too-low child
    alpha groups are rejected as a whole, rather than silently changing N or
    clamping alpha. Unchanged Adam rows, Parameter identities and steps survive.
    """
    if donors.shape != targets.shape or donors.ndim != 1:
        raise ValueError('Donor and target index arrays must be equal-length 1D')
    empty = donors[:0]
    if not len(donors):
        return {'relocated': 0, 'rejected_copies': 0, 'rejected_alpha': 0}, empty, empty
    if len(torch.unique(donors)) != len(donors):
        raise ValueError('Each donor slot may be moved once')
    if bool(torch.isin(donors, targets).any()):
        raise ValueError('Donor and target sets must be disjoint')
    n = len(g.get_xyz)
    if bool(((donors < 0) | (donors >= n) | (targets < 0) | (targets >= n)).any()):
        raise ValueError('Point index out of bounds')
    parents, inverse, times = torch.unique(targets, return_inverse=True, return_counts=True)
    copies = times + 1
    eligible = copies <= max_copies
    rejected_copies = int((~eligible[inverse]).sum())
    alpha = g.get_opacity[parents, 0]
    # Exactly saturated sigmoid has no finite log(1-alpha); exclude that target.
    eligible &= torch.isfinite(alpha) & (alpha > 0) & (alpha < 1)
    ids = torch.nonzero(eligible).flatten()
    new_alpha = torch.zeros_like(alpha)
    new_scales = torch.zeros_like(g._scaling[parents])
    if len(ids):
        a, scale = relocation_parameters(alpha[ids], g.get_scaling[parents[ids]],
                                         copies[ids], max_copies=max_copies)
        new_alpha[ids], new_scales[ids] = a, scale
    accepted = eligible & (new_alpha >= alpha_floor)
    rejected_alpha = int((eligible[inverse] & ~accepted[inverse]).sum())
    keep = accepted[inverse]
    donors, targets = donors[keep], targets[keep]
    selected = parents[accepted]
    if len(donors):
        g._opacity[selected, 0] = torch.logit(new_alpha[accepted])
        g._scaling[selected] = torch.log(new_scales[accepted])
        changed = torch.cat((selected, donors))
        for attr in PARAMETERS.values():
            p = getattr(g, attr)
            p[donors] = p[targets].clone()
            zero_moments(g.optimizer, p, changed)
        for name in ('xyz_gradient_accum', 'denom', 'max_radii2D'):
            getattr(g, name)[changed] = 0
    else:
        changed = empty
    return {'relocated': len(donors), 'targets': len(selected),
            'rejected_copies': rejected_copies, 'rejected_alpha': rejected_alpha,
            'rejected_nonfinite_or_saturated': int((~torch.isfinite(alpha) | (alpha <= 0) | (alpha >= 1))[inverse].sum()),
            'multiplicity_max_accepted': int(copies[accepted].max()) if len(selected) else 0,
            'moments': 'zero target and incoming rows; retain unrelated rows and Adam step'}, changed, donors


class AllocationHook:
    def __init__(self, state, **options):
        self.options = Options(**options)
        self.options.validate()
        if getattr(state, 'abs_accumulator', None) is None:
            raise ValueError('Allocation requires an absolute-gradient accumulator')
        self.device = state.g.get_xyz.device
        self.rng = torch.Generator(device=self.device).manual_seed(int(state.seed) + 60517)
        self.initial_points = len(state.g.get_xyz)
        self.events = []
        self.host_seconds = {'topology': 0.}

    @torch.no_grad()
    def topology_update(self, s, step, cap, result):
        start = time.perf_counter()
        o, g = self.options, s.g
        before = len(g.get_xyz)
        if before > cap:
            raise ValueError('Incoming point count exceeds topology cap')
        alpha = g.get_opacity[:, 0].detach()
        dead = torch.nonzero(alpha <= o.dead_opacity).flatten()
        limit = min(len(dead), int(before * o.relocate_fraction))
        donors = dead[torch.topk(alpha[dead], limit, largest=False).indices] if limit else dead[:0]
        alive = torch.nonzero((alpha > o.dead_opacity) & (alpha < 1) & torch.isfinite(alpha)).flatten()
        info = {'relocated': 0, 'donors_available': len(dead), 'donors_requested': limit}
        changed = donors[:0]
        if limit and len(alive):
            weights = alpha[alive]
            targets = alive[torch.multinomial(weights, limit, replacement=True, generator=self.rng)]
            moved, changed, _ = relocate_slots(g, donors, targets,
                alpha_floor=max(o.dead_opacity, s.config['alpha_prune']) * 1.001,
                max_copies=o.max_copies)
            info.update(moved)
            if s.abs_accumulator is not None:
                s.abs_accumulator.absolute[changed] = 0
        # Stale old-position radii must not be carried onto relocated points.
        radii = result['radii']
        if len(changed):
            radii = radii.clone()
            radii[changed] = 0
        from reconstruction.optimization.absolute_gradient import topology_update
        topo = topology_update(g, s.config, s.extent, radii, cap, s.abs_accumulator,
            s.abs_options['births_per_event'], s.abs_options['split_fraction'])
        topo['operator'] = 'relocation+abs-split'
        assert_optimizer_topology(g)
        if len(g.get_xyz) > cap:
            raise AssertionError('Point cap exceeded')
        event = {'step': int(step), **info, 'before': before, 'after': len(g.get_xyz)}
        self.events.append(event)
        topo['allocation'] = event
        self.host_seconds['topology'] += time.perf_counter() - start
        return topo

    def after_topology(self, s, info):
        pass




