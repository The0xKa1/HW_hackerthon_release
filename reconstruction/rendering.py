"""GPU render adapters, without modifying upstream renderers."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np

def classic_imports():
    from .resources import VENDOR
    backend = VENDOR / 'baseline/3dgs'
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from utils.loss_utils import ssim,l1_loss
    return render,GaussianModel,ssim,l1_loss

def initialize(ply: Path,centers,rn: dict,*,sh_degree=None):
    """Complete classic 3DGS training state after loading a standard PLY."""
    import torch
    _,GaussianModel,_,_=classic_imports()
    from .io import load_ply
    degree=int(np.sqrt(load_ply(ply)['harmonics'].shape[-1]))-1 if sh_degree is None else sh_degree
    if type(degree) is not int or not 0<=degree<=3:raise ValueError('Invalid SH degree')
    g=GaussianModel(degree);g.load_ply(str(ply));n=len(g.get_xyz)
    g.spatial_lr_scale=float(torch.linalg.vector_norm(centers-centers.mean(0),dim=1).max())*1.1
    if g.spatial_lr_scale<=0:raise ValueError('degenerate camera extent')
    g.max_radii2D=torch.zeros(n,device='cuda');g.pretrained_exposures=None;g.exposure_mapping={str(i):i for i in range(len(centers))};g._exposure=torch.nn.Parameter(torch.eye(3,4,device='cuda')[None].repeat(len(centers),1,1),requires_grad=False)
    args=dict(rn);args.update(percent_dense=.01,iterations=rn.get('max_steps',10000),exposure_lr_init=.01,exposure_lr_final=.001,exposure_lr_delay_steps=0,exposure_lr_delay_mult=0.)
    g.training_setup(SimpleNamespace(**args));return g

def pipe():return SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,debug=False,antialiasing=False)
