"""Fixed-reference angular attention, recorded-view coverage and RGB provenance.

The sphere unifies orientations; it does not solve translation/parallax, depth,
occlusion or object identity. Its RGB atlas is an angular lookup into recorded
frames, not a reconstructed panorama or newly generated scene content.
"""
from __future__ import annotations
import math
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from .attention_volume import support_intervals

VERSION='fixed_reference_spherical_attention_v2'
COLORS=np.array([[50,150,255],[255,200,40],[255,90,55]],dtype=np.float32)


def _finite_vec(value):
    try:vector=np.asarray(value,dtype=float)
    except (TypeError,ValueError):return None
    return vector if vector.shape==(3,) and np.all(np.isfinite(vector)) else None


def world_ray(record):
    point=_finite_vec(record.get('gaze_point_world'));origin=_finite_vec(record.get('camera_position'))
    vector=point-origin if point is not None and origin is not None else _finite_vec(record.get('gaze_vector_world'))
    if vector is None:return None
    norm=float(np.linalg.norm(vector))
    return vector/norm if math.isfinite(norm) and norm>1e-8 else None


def pose_rotation(pose):
    try:return Rotation.from_quat(np.asarray(pose['rotation_xyzw'],dtype=float))
    except (KeyError,TypeError,ValueError):return None


def direction_to_angles(direction):
    x,y,z=map(float,direction)
    return math.atan2(x,z),math.atan2(y,math.hypot(x,z))


def angles_to_directions(yaw,pitch):
    cos_pitch=np.cos(pitch)
    return np.stack([cos_pitch*np.sin(yaw),np.sin(pitch),cos_pitch*np.cos(yaw)],axis=-1)


def spherical_grid(width=360,height=180):
    if width<4 or height<2:raise ValueError('Spherical grid too small')
    yaw=-math.pi+(np.arange(width,dtype=float)+.5)*2*math.pi/width
    pitch=math.pi/2-(np.arange(height,dtype=float)+.5)*math.pi/height
    return np.meshgrid(yaw,pitch)


def solid_angle_weights(width,height):
    edges=np.linspace(math.pi/2,-math.pi/2,height+1)
    return np.repeat(((np.sin(edges[:-1])-np.sin(edges[1:]))*2*math.pi/width)[:,None],width,axis=1)


def _gaussian_on_sphere(yaw_grid,pitch_grid,yaw,pitch,sigma):
    cosine=np.sin(pitch_grid)*math.sin(pitch)+np.cos(pitch_grid)*math.cos(pitch)*np.cos(yaw_grid-yaw)
    angle=np.arccos(np.clip(cosine,-1.,1.))
    values=np.exp(-.5*(angle/sigma)**2)*solid_angle_weights(yaw_grid.shape[1],yaw_grid.shape[0])
    total=float(values.sum())
    return (values/total).astype(np.float32) if total>0 else values.astype(np.float32)


def _camera_coordinates(canonical_directions,canonical_pose,camera_pose):
    ref=pose_rotation(canonical_pose);camera=pose_rotation(camera_pose)
    if ref is None or camera is None:return None
    return camera.inv().apply(ref.apply(canonical_directions.reshape(-1,3))).reshape(canonical_directions.shape)


def _fov_mask(canonical_directions,canonical_pose,camera_pose,image_size):
    local=_camera_coordinates(canonical_directions,canonical_pose,camera_pose)
    if local is None:return None
    try:vfov=float(camera_pose['fov_vertical_degrees']);width,height=map(float,image_size)
    except (KeyError,TypeError,ValueError):return None
    if not math.isfinite(vfov) or not 0<vfov<180 or min(width,height)<=0:return None
    z=local[...,2];tan_y=math.tan(math.radians(vfov)/2);tan_x=width/height*tan_y
    return (z>1e-7)&(np.abs(local[...,0]/np.maximum(z,1e-12))<=tan_x)&(np.abs(local[...,1]/np.maximum(z,1e-12))<=tan_y)


def build_spherical_volume(records,*,query_time,canonical_pose,image_size,clock_supported,
                           current_pose=None,history_seconds=2.,bins=3,grid_size=(360,180),sigma_degrees=2.):
    width,height=map(int,grid_size)
    if bins!=3 or min(width,height)<2 or not (math.isfinite(query_time) and history_seconds>0 and 0<sigma_degrees<90):
        raise ValueError('Invalid spherical volume configuration')
    ref=pose_rotation(canonical_pose)
    if ref is None:raise ValueError('Canonical rotation required')
    yg,pg=spherical_grid(width,height);directions=angles_to_directions(yg,pg)
    maps=np.zeros((bins,height,width),np.float32);coverage=np.zeros_like(maps)
    intervals=[[query_time-history_seconds+history_seconds*i/bins,query_time-history_seconds+history_seconds*(i+1)/bins] for i in range(bins)]
    base={'version':VERSION,'query_time_seconds':query_time,'history_seconds':history_seconds,
          'grid_size':[width,height],'bins':[{'window':w,'duration_seconds':0.,'observations':[]} for w in intervals]}
    if not clock_supported:return maps,coverage,{**base,'status':'RGB_FALLBACK'}
    # View footprints exist even when eye tracking is invalid. They are separate from gaze mass.
    camera_rows=[dict(row,validity='valid') for row in records]
    for item in support_intervals(camera_rows,query_time,history_seconds):
        row=item['record'];pose={'rotation_xyzw':row.get('camera_rotation_xyzw'),'fov_vertical_degrees':row.get('camera_fov_degrees')}
        mask=_fov_mask(directions,canonical_pose,pose,image_size)
        if mask is None:continue
        for i,(lo,hi) in enumerate(intervals):
            duration=max(0.,min(hi,item['support_interval'][1])-max(lo,item['support_interval'][0]))
            coverage[i][mask]+=duration
    excluded=Counter();groups=[[] for _ in range(bins)]
    for item in support_intervals(records,query_time,history_seconds):
        row=item['record'];ray=world_ray(row)
        if ray is None:excluded['missing_world_ray']+=1;continue
        local=ref.inv().apply(ray);yaw,pitch=direction_to_angles(local)
        kernel=_gaussian_on_sphere(yg,pg,yaw,pitch,math.radians(sigma_degrees))
        for i,(lo,hi) in enumerate(intervals):
            duration=max(0.,min(hi,item['support_interval'][1])-max(lo,item['support_interval'][0]))
            if duration<=0:continue
            maps[i]+=kernel*duration
            groups[i].append({'source_index':row['source_index'],'gaze_time_seconds':row['telemetry_time_seconds'],
                'support_interval':item['support_interval'],'duration_seconds':duration,'world_ray':ray.tolist(),
                'gaze_point_world':row.get('gaze_point_world'),'canonical_yaw_degrees':math.degrees(yaw),
                'canonical_pitch_degrees':math.degrees(pitch)})
    bstats=[]
    for i,window in enumerate(intervals):
        duration=float(maps[i].sum())
        bstats.append({'index_old_to_new':i,'window':window,'duration_seconds':duration,
            'coverage_solid_angle_time_sr_seconds':float((coverage[i]*solid_angle_weights(width,height)).sum()),
            'observations':groups[i]})
    current_mask=_fov_mask(directions,canonical_pose,current_pose or canonical_pose,image_size)
    total=float(maps.sum());current=float(maps[:,current_mask].sum()) if current_mask is not None else None
    union=coverage.sum(axis=0)>0
    return maps,coverage,{**base,'status':'SPHERICAL_VOLUME','bins':bstats,'excluded':dict(excluded),
        'sigma_degrees':sigma_degrees,'total_duration_seconds':total,
        'current_fov_duration_seconds':current,'current_fov_fraction':current/total if total>0 and current is not None else None,
        'union_view_solid_angle_sr':float(solid_angle_weights(width,height)[union].sum()),
        'canonical_frame':'fixed reference orientation; never recentered between queries',
        'translation_compensated':False,'visibility':'unknown_frustum_only','object_identity':'not established',
        'semantics':'cell mass is supported seconds; camera coverage is separate; angular map is not a reconstructed scene'}


def render_spherical_debug(maps,coverage,stats,size=(900,450)):
    total=maps.sum(axis=0);h,w=total.shape
    # Display density per solid angle; mass integration still uses per-cell seconds.
    density=maps/solid_angle_weights(w,h)[None,...];sum_density=density.sum(axis=0)
    base=np.full((h,w,3),22.,np.float32);seen=coverage.sum(axis=0)>0;base[seen]=[65,65,65]
    peak=float(sum_density.max())
    if peak>0:
        strength=min(1.,float(total.sum())/.25)
        alpha=np.sqrt(sum_density/peak)[...,None]*.9*strength
        color=np.einsum('khw,kc->hwc',density/(sum_density[None,...]+1e-20),COLORS)
        base=base*(1-alpha)+color*alpha
    image=Image.fromarray(base.clip(0,255).astype('uint8')).resize(size,Image.Resampling.NEAREST)
    draw=ImageDraw.Draw(image);width,height=size
    for txt,y in [('pitch +90',8),('pitch 0',height//2),('pitch -90',height-18)]:draw.text((8,y),txt,fill='white')
    draw.text((width//2-32,8),'yaw 0',fill='white');draw.text((width-80,height-18),'yaw +180',fill='white')
    return image


def angular_rgb_atlas(frames,canonical_pose,*,grid_size=(720,360)):
    """Rotation-only ray lookup with latest-source index at each atlas pixel."""
    width,height=grid_size;yg,pg=spherical_grid(width,height);directions=angles_to_directions(yg,pg)
    rgb=np.full((height,width,3),22,np.uint8);owner=np.full((height,width),-1,np.int32)
    for index,frame in enumerate(frames):
        with Image.open(frame['source_path']) as im:image=np.asarray(im.convert('RGB'))
        h,w=image.shape[:2];pose=frame['geometry']['reference_pose']
        mask=_fov_mask(directions,canonical_pose,pose,(w,h))
        if mask is None:continue
        local=_camera_coordinates(directions,canonical_pose,pose);z=np.maximum(local[...,2],1e-12)
        f=h/(2*math.tan(math.radians(pose['fov_vertical_degrees'])/2))
        x=np.clip(np.rint(w/2+f*local[...,0]/z).astype(int),0,w-1)
        y=np.clip(np.rint(h/2-f*local[...,1]/z).astype(int),0,h-1)
        rgb[mask]=image[y[mask],x[mask]];owner[mask]=index
    return Image.fromarray(rgb),owner


def overlay_on_atlas(atlas,maps):
    w,h=atlas.size
    resized=np.stack([np.asarray(Image.fromarray(plane).resize((w,h),Image.Resampling.BILINEAR)) for plane in maps])
    density=resized/solid_angle_weights(w,h)[None,...];total=density.sum(axis=0);peak=float(total.max())
    if peak<=0:return atlas.copy()
    strength=min(1.,float(maps.sum())/.25)
    alpha=(np.sqrt(total/peak)*.55*strength)[...,None]
    color=np.einsum('khw,kc->hwc',density/(total[None,...]+1e-20),COLORS)
    source=np.asarray(atlas,dtype=np.float32)
    return Image.fromarray((source*(1-alpha)+color*alpha).clip(0,255).astype('uint8'))


def render_cube_faces(atlas,maps,*,face_size=256):
    """Six 90-degree perspective windows into the same stored angular sphere.

    This display avoids the severe stretching at the poles of a longitude/latitude
    rectangle. No new RGB is generated; each face samples the recorded atlas.
    """
    faces=[('FRONT',[0,0,1],[1,0,0],[0,1,0]),('RIGHT',[1,0,0],[0,0,-1],[0,1,0]),
           ('BACK',[0,0,-1],[-1,0,0],[0,1,0]),('LEFT',[-1,0,0],[0,0,1],[0,1,0]),
           ('UP',[0,1,0],[1,0,0],[0,0,-1]),('DOWN',[0,-1,0],[1,0,0],[0,0,1])]
    source=np.asarray(atlas);ah,aw=source.shape[:2];mh,mw=maps.shape[1:]
    density=maps/solid_angle_weights(mw,mh)[None,...];total=density.sum(axis=0);peak=float(total.max())
    canvas=Image.new('RGB',(face_size*3,(face_size+24)*2),'#16212c');draw=ImageDraw.Draw(canvas)
    v,u=np.meshgrid(1-2*(np.arange(face_size)+.5)/face_size,2*(np.arange(face_size)+.5)/face_size-1,indexing='ij')
    for index,(name,forward,right,up) in enumerate(faces):
        rays=np.array(forward)+u[...,None]*np.array(right)+v[...,None]*np.array(up)
        rays/=np.linalg.norm(rays,axis=-1,keepdims=True)
        yaw=np.arctan2(rays[...,0],rays[...,2]);pitch=np.arcsin(np.clip(rays[...,1],-1,1))
        ax=np.floor((yaw+math.pi)/(2*math.pi)*aw).astype(int)%aw
        ay=np.clip(np.floor((math.pi/2-pitch)/math.pi*ah).astype(int),0,ah-1)
        dx=np.floor((yaw+math.pi)/(2*math.pi)*mw).astype(int)%mw
        dy=np.clip(np.floor((math.pi/2-pitch)/math.pi*mh).astype(int),0,mh-1)
        rgb=source[ay,ax].astype(float)
        values=density[:,dy,dx];sum_density=values.sum(axis=0)
        if peak>0:
            a=(np.sqrt(sum_density/peak)*.55*min(1.,float(maps.sum())/.25))[...,None]
            color=np.einsum('khw,kc->hwc',values/(sum_density[None,...]+1e-20),COLORS)
            rgb=rgb*(1-a)+color*a
        x=index%3*face_size;y=index//3*(face_size+24)
        draw.text((x+8,y+6),name+' | 90 deg',fill='white')
        canvas.paste(Image.fromarray(rgb.clip(0,255).astype('uint8')),(x,y+24))
    return canvas
