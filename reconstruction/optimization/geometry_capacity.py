"""Geometry repair followed by one capacity manager, with explicit row ordering."""
from operator import index
import time

import torch

from reconstruction.optimization.local_geometry import GeometryHook, validate_options
from reconstruction.optimization.allocation import AllocationHook
from reconstruction.optimization.density import assert_optimizer_topology


class JointGeometry:
    def __init__(self, state, geometry_options=None, relocation_options=None):
        options = validate_options(geometry_options or {})
        # Vendor clone/split/prune does not resize AMSGrad's maximum moment.
        if any(group.get("amsgrad", False) or
               "max_exp_avg_sq" in state.g.optimizer.state.get(group["params"][0], {})
               for group in state.g.optimizer.param_groups):
            raise ValueError("Allocation abs topology does not support AMSGrad")
        self.geometry = GeometryHook(state, **options)
        self.manager = AllocationHook(state, **(relocation_options or {}))
        self.events = []

    def before_backward(self, s, result, j, step):
        self.geometry.before_backward(s, result, j, step)
        fn = getattr(self.manager, "before_backward", None)
        return fn(s, result, j, step) if fn else None

    def after_backward(self, s, result, j, step):
        self.geometry.after_backward(s, result, j, step)
        fn = getattr(self.manager, "after_backward", None)
        if fn:
            fn(s, result, j, step)

    @torch.no_grad()
    def topology_update(self, s, step, cap, result):
        cap = index(cap)
        before = len(s.g.get_xyz)
        if before > cap:
            raise ValueError("Incoming point count exceeds topology cap")
        if len(result["radii"]) != before:
            raise ValueError("Radii must describe the pre-topology population")
        # Opacity IDs refer to the pre-topology population. Apply them before
        # either relocation or split/prune can change their meaning.
        pending = self.geometry.pending
        event = None
        if pending is not None and pending["event"]["step"] == step:
            event = pending["event"]
        elif pending is not None:
            pending["event"]["status"] = "stale_geometry_step"
            self.geometry.pending = None
        start = time.perf_counter()
        handled = self.geometry.topology_update(s, step, cap, result)
        if handled is not None:
            raise AssertionError("Geometry must not run baseline topology here")
        info = self.manager.topology_update(s, step, cap, result)
        if info is None:
            raise AssertionError("Capacity manager must handle every event")
        if len(s.g.get_xyz) > cap:
            raise AssertionError("Capacity manager exceeded cap")
        assert_optimizer_topology(s.g)
        info.update(after=len(s.g.get_xyz), quota_cap=cap,
                    manager_quota_cap=cap,
                    geometry_surface_births=0,
                    geometry_opacity_edits=event["opacity_edits"] if event else 0)
        if event is not None:
            event.update(topology_handled=True)
        # Complete cleanup against the FINAL population before returning. The
        # engine's subsequent after_topology callback is safe to repeat.
        self.after_topology(s, info)
        self.events.append({
            "step": step, "surface_births": 0,
            "opacity_edits": info["geometry_opacity_edits"],
            "host_seconds_including_manager": time.perf_counter() - start,
        })
        return info

    def after_topology(self, s, info):
        self.geometry.after_topology(s, info)
        self.manager.after_topology(s, info)

    def after_step(self, s, step):
        fn = getattr(self.manager, "after_step", None)
        if fn:
            fn(s, step)



