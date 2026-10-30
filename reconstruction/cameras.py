"""Camera contracts: c2w OpenCV axes and normalized pinhole intrinsics."""
from __future__ import annotations
from types import SimpleNamespace
import math
import numpy as np

def validate(c2w: np.ndarray, K: np.ndarray) -> None:
    if c2w.shape != (4,4) or K.shape != (3,3) or not np.isfinite(c2w).all() or not np.isfinite(K).all():
        raise ValueError('camera requires finite c2w(4,4), K(3,3)')
    if not np.allclose(c2w[3],[0,0,0,1]) or not np.allclose(c2w[:3,:3].T@c2w[:3,:3],np.eye(3),atol=1e-4) or np.linalg.det(c2w[:3,:3])<0:
        raise ValueError('c2w must be a rigid proper transform')
    if K[0,0]<=0 or K[1,1]<=0 or not np.allclose(K[2],[0,0,1]):
        raise ValueError('invalid pinhole intrinsics')


def classic(c2w, K, h: int,w: int,near: float=.01,far: float=200):
    """Build the half-pixel projection convention, including off-center intrinsics."""
    import torch
    if h<=0 or w<=0 or not 0<near<far:raise ValueError('invalid image size or clipping planes')
    device=c2w.device;c=c2w.detach().cpu().double();k=K.detach().cpu().double()
    validate(c.numpy(),k.numpy());inv=torch.linalg.inv(c)
    fx,fy=k[0,0]*w,k[1,1]*h;cx,cy=k[0,2]*w,k[1,2]*h
    p=torch.zeros((4,4),dtype=torch.float64);p[0,0]=2*fx/w;p[1,1]=2*fy/h;p[0,2]=(2*cx+1)/w-1;p[1,2]=(2*cy+1)/h-1;p[2,2]=far/(far-near);p[2,3]=-far*near/(far-near);p[3,2]=1
    return SimpleNamespace(FoVx=2*math.atan(w/(2*float(fx))),FoVy=2*math.atan(h/(2*float(fy))),image_height=h,image_width=w,world_view_transform=inv.T.contiguous().float().to(device),full_proj_transform=(p@inv).T.contiguous().float().to(device),camera_center=c[:3,3].float().to(device))
