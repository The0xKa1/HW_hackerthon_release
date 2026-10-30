"""B-only adapter to the pinned author's FastGS extension.

The extension returns true per-pixel absolute screen derivatives in columns
2:4 of its Nx4 dummy-means gradient; columns 0:2 are signed derivatives.
Both already use NDC units. No absolute-of-reduced-gradient fallback exists.
"""
from __future__ import annotations

import math


class FastGSScreenProxy:
    def __init__(self, means2d):
        self.means2d = means2d

    def _gradient(self):
        value = self.means2d.grad
        if value is None or value.ndim != 2 or value.shape[1] != 4:
            raise RuntimeError('True FastGS Nx4 signed/ABS screen gradient missing')
        return value

    @property
    def grad(self):
        return self._gradient()[:, :2]

    @property
    def absgrad(self):
        return self._gradient()[:, 2:4]


class FastGSRenderer:
    """Keep classic model/Adam, using only FastGS rasterization.

    mult=1 preserves the extension's full alpha>=1/255 tile ellipse; mult=.5
    clips its squared extent. Neither changes the image dimensions. FastGS
    native radii validity includes near clipping at z=.2 and opacity-aware
    tile overlap, so it is not identical to gsplat projected validity.
    """
    def __init__(self, mult=1.0):
        if isinstance(mult, bool) or mult not in (1.0, 0.5):
            raise ValueError('FastGS mult must be 1.0 or 0.5')
        self.mult = float(mult)
        self.metric_maps = {}
        self.settings = {
            'renderer': 'diff_gaussian_rasterization_fastgs', 'mult': self.mult,
            'absgrad': 'true per-pixel absolute in Nx4 gradient columns 2:4',
            'screen_gradient_coordinates': 'NDC', 'eps2d': .3,
            'near_plane': .2, 'far_plane': None,
            'visibility': 'native radii>0: near-valid and opacity-aware tile overlap',
            'get_flag': False,
        }

    def __call__(self, camera, g, pipeline, background):
        import torch
        from diff_gaussian_rasterization_fastgs import (
            GaussianRasterizationSettings, GaussianRasterizer,
        )
        if pipeline.compute_cov3D_python or pipeline.convert_SHs_python:
            raise ValueError('FastGS B adapter requires native covariance and SH evaluation')
        xyz = g.get_xyz
        means2d = torch.zeros((len(xyz), 4), dtype=xyz.dtype,
                              device=xyz.device, requires_grad=True)
        key = (camera.image_height, camera.image_width, xyz.device)
        if key not in self.metric_maps:
            self.metric_maps[key] = torch.zeros(camera.image_height * camera.image_width,
                                                dtype=torch.int32, device=xyz.device)
        settings = GaussianRasterizationSettings(
            image_height=int(camera.image_height), image_width=int(camera.image_width),
            tanfovx=math.tan(camera.FoVx * .5), tanfovy=math.tan(camera.FoVy * .5),
            bg=background, scale_modifier=1.0,
            viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
            sh_degree=g.active_sh_degree, campos=camera.camera_center,
            mult=self.mult, prefiltered=False, debug=pipeline.debug,
            get_flag=False, metric_map=self.metric_maps[key],
        )
        rgb, radii, metric_counts = GaussianRasterizer(raster_settings=settings)(
            means3D=xyz, means2D=means2d,
            dc=g._features_dc, shs=g._features_rest, colors_precomp=None,
            opacities=g.get_opacity, scales=g.get_scaling,
            rotations=g.get_rotation, cov3D_precomp=None,
        )
        return {'render': rgb, 'radii': radii,
                'viewspace_points': FastGSScreenProxy(means2d),
                'visibility_filter': radii > 0, 'accum_metric_counts': metric_counts}
