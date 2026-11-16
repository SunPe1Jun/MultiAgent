"""Disjoint perspective faces over a fixed angular attention volume.

Each face uses one real RGB source per time bin. No patchwork atlas is needed
for model inputs; missing rays remain unknown and all source hashes are kept.
"""
from __future__ import annotations
import math
import numpy as np
from PIL import Image
from .spherical_attention import (spherical_grid,angles_to_directions,solid_angle_weights,
                                 _camera_coordinates,_fov_mask)

VERSION='native_perspective_attention_faces_v1'
FACES=[('FRONT',[0,0,1],[1,0,0],[0,1,0]),('RIGHT',[1,0,0],[0,0,-1],[0,1,0]),
       ('BACK',[0,0,-1],[-1,0,0],[0,1,0]),('LEFT',[-1,0,0],[0,0,1],[0,1,0]),
       ('UP',[0,1,0],[1,0,0],[0,0,-1]),('DOWN',[0,-1,0],[1,0,0],[0,0,1])]


def cube_owner(directions):
    """Return one face per ray. Even exact seams have deterministic ownership."""
    d=np.asarray(directions,dtype=float)
    if d.shape[-1]!=3 or not np.all(np.isfinite(d)) or np.any(np.linalg.norm(d,axis=-1)<1e-10):
        raise ValueError('Finite nonzero directions required')
    scores=np.stack([np.sum(d*np.array(f[1]),axis=-1) for f in FACES],axis=-1)
    return np.argmax(scores,axis=-1)


def face_grid(face,size=224):
    if not 0<=face<len(FACES) or size<2:raise ValueError('Invalid face/size')
    _,forward,right,up=FACES[face]
    v,u=np.meshgrid(1-2*(np.arange(size)+.5)/size,2*(np.arange(size)+.5)/size-1,indexing='ij')
    rays=np.array(forward)+u[...,None]*np.array(right)+v[...,None]*np.array(up)
    rays/=np.linalg.norm(rays,axis=-1,keepdims=True)
    return rays


def sphere_indices(rays,width,height):
    yaw=np.arctan2(rays[...,0],rays[...,2]);pitch=np.arcsin(np.clip(rays[...,1],-1,1))
    x=np.floor((yaw+math.pi)/(2*math.pi)*width).astype(int)%width
    y=np.clip(np.floor((math.pi/2-pitch)/math.pi*height).astype(int),0,height-1)
    return x,y


def select_faces(camera_coverage,count=3):
    """Select windows solely from recorded camera coverage, never gaze heat."""
    if camera_coverage.ndim==3:camera_coverage=camera_coverage.sum(axis=0)
    h,w=camera_coverage.shape;yg,pg=spherical_grid(w,h)
    owner=cube_owner(angles_to_directions(yg,pg));area=solid_angle_weights(w,h)
    areas=np.array([float(area[(owner==i)&(camera_coverage>0)].sum()) for i in range(6)])
    order=np.argsort(-areas,kind='stable')[:count]
    return order.tolist(),{'face_recorded_solid_angle_sr':areas.tolist(),
        'selection':'largest recorded camera footprint by solid angle; gaze maps not used',
        'selected_solid_angle_sr':float(areas[order].sum())}


def face_mass_audit(maps,selected):
    _,h,w=maps.shape;yg,pg=spherical_grid(w,h);owner=cube_owner(angles_to_directions(yg,pg))
    total=maps.sum(axis=(1,2));mass=np.array([[float(maps[k][owner==face].sum()) for face in selected] for k in range(3)])
    return {'selected_face_duration_seconds':mass.tolist(),'total_duration_seconds':total.tolist(),
            'retained_duration_fraction':np.divide(mass.sum(axis=1),total,out=np.zeros_like(total),where=total>0).tolist(),
            'scope':'three model windows only; full six-face sphere remains stored'}


def sample_face_density(maps,face,size=224):
    _,h,w=maps.shape;rays=face_grid(face,size);x,y=sphere_indices(rays,w,h)
    density=maps/solid_angle_weights(w,h)[None,...]
    # A sample near a discrete grid seam belongs to the face of the sampled ray.
    yg,pg=spherical_grid(w,h);owner=cube_owner(angles_to_directions(yg,pg))
    result=density[:,y,x].copy();result[:,owner[y,x]!=face]=0
    return result.astype(np.float32)


def sample_native_face(frame,canonical_pose,face,size=224):
    rays=face_grid(face,size)
    with Image.open(frame['source_path']) as im:source=np.asarray(im.convert('RGB'))
    h,w=source.shape[:2];pose=frame['geometry']['reference_pose']
    mask=_fov_mask(rays,canonical_pose,pose,(w,h))
    if mask is None:raise ValueError('Frame pose missing')
    local=_camera_coordinates(rays,canonical_pose,pose);z=np.maximum(local[...,2],1e-12)
    f=h/(2*math.tan(math.radians(pose['fov_vertical_degrees'])/2))
    x=np.clip(np.rint(w/2+f*local[...,0]/z).astype(int),0,w-1)
    y=np.clip(np.rint(h/2-f*local[...,1]/z).astype(int),0,h-1)
    result=np.full((size,size,3),22,np.uint8);result[mask]=source[y[mask],x[mask]]
    return Image.fromarray(result),mask


def choose_native_source(frames,canonical_pose,face,window,offset,size=80):
    """Use a real source not later than bin end; prefer in-bin visual coverage.

    No source RGB from a later phase is silently substituted. A previous source
    is used only when no frame falls inside the bin, and this is explicit.
    """
    lo,hi=map(float,window)
    available=[f for f in frames if f['pts']-offset<=hi+1e-6]
    in_bin=[f for f in available if f['pts']-offset>=lo-1e-6]
    candidates=in_bin or available
    if not candidates:raise ValueError('No supported source before bin end')
    scored=[]
    for frame in candidates:
        _,mask=sample_native_face(frame,canonical_pose,face,size)
        score=float(mask.mean())
        scored.append((score,-abs((frame['pts']-offset)-(lo+hi)/2),frame))
    best=max(scored,key=lambda r:r[:2])
    return best[2],{'used_in_bin_frame':bool(in_bin),'recorded_face_fraction':best[0],
                    'selection':'largest geometric face coverage, midpoint distance breaks ties',
                    'source_telemetry_seconds':best[2]['pts']-offset,'window':window}


def transform_density(tiles,condition):
    """Tiles are [time, region, height, width]. Controls preserve RGB."""
    if condition=='TRUE_HEAT':return tiles.copy()
    if condition=='REVERSE_TIME':return tiles[::-1].copy()
    if condition=='SHIFT_SPACE':return np.roll(tiles,1,axis=1)
    if condition=='NO_HEAT':return np.zeros_like(tiles)
    raise ValueError('Unknown heat control')


def observation_for_selected_face(observations,canonical_pose,face,face_density):
    """Pick evidence only inside the model-selected face; never replace the face."""
    from scipy.spatial.transform import Rotation
    ref=Rotation.from_quat(canonical_pose['rotation_xyzw'])
    candidates=[o for o in observations if int(cube_owner(ref.inv().apply(o['world_ray'])))==face]
    if not candidates:return None
    y,x=np.unravel_index(np.argmax(face_density),face_density.shape)
    ray=face_grid(face,face_density.shape[0])[y,x]
    return max(candidates,key=lambda o:float(np.dot(ref.inv().apply(o['world_ray']),ray)))
