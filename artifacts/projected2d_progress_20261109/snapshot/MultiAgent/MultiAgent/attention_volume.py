"""Causal, mass-conserving attention volumes and RGB-preserving presentations.

A volume is measured gaze duration per pixel and time interval, not a target
probability. Spatial normalization is display-only and never hides sparse support.
"""
from __future__ import annotations

import math
from collections import Counter
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion,label

from .gaze_evidence import project_gaze
from .scripted_attention import valid_point,crop_at

VERSION='causal_attention_volume_v2'
COLORS=np.array([[50,150,255],[255,200,40],[255,90,55]],dtype=np.uint8)


def support_intervals(records,query,history=2.,max_gap=.25):
    """Capped Voronoi support, clipped to a past-only window.

    Invalid samples cut neighbors' intervals and contribute no duration. Long
    gaps stay empty; there is no synthetic minimum duration for an isolated hit.
    """
    if not (math.isfinite(query) and history>0 and max_gap>0):raise ValueError('Invalid time window')
    ordered=sorted(records,key=lambda r:(r['telemetry_time_seconds'],r['source_index']))
    times=np.array([r['telemetry_time_seconds'] for r in ordered],dtype=float)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times)<=0):raise ValueError('Strict finite gaze times required')
    output=[]
    for i,row in enumerate(ordered):
        t=float(times[i])
        if not query-history<=t<=query:continue
        # Do not inspect a future sample to determine current support.
        before=float(times[i-1]) if i>0 else t-max_gap
        after=float(times[i+1]) if i+1<len(times) and times[i+1]<=query else t+max_gap
        left=max(query-history,t-min(t-before,max_gap)/2)
        right=min(query,t+min(after-t,max_gap)/2)
        if right<=left or row.get('validity')!='valid':continue
        output.append({'record':row,'support_interval':[left,right],'weight_seconds':right-left})
    return output


def _kernel(shape,point,sigma):
    height,width=shape;x,y=point
    radius=max(2,int(math.ceil(3*sigma)))
    x0,x1=max(0,int(x)-radius),min(width,int(x)+radius+1)
    y0,y1=max(0,int(y)-radius),min(height,int(y)+radius+1)
    yy,xx=np.ogrid[y0:y1,x0:x1]
    k=np.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma*sigma)).astype(np.float32)
    # Each sample deposits its duration, including at an image edge.
    k/=k.sum()
    return (slice(y0,y1),slice(x0,x1)),k


def build_volume(records,*,query,pose,image_size,clock_supported,history=2.,bins=3,sigma_degrees=2.):
    width,height=image_size
    if min(image_size)<=0 or bins!=3 or not history>0:raise ValueError('Positive size/window, three bins required')
    maps=np.zeros((bins,height,width),np.float32)
    intervals=[[query-history+history*i/bins,query-history+history*(i+1)/bins] for i in range(bins)]
    groups=[[] for _ in range(bins)];excluded=Counter()
    if not clock_supported or pose is None:
        return maps,{'version':VERSION,'status':'RGB_FALLBACK','clock_supported':False,
            'query':query,'history_seconds':history,'bins':[{'window':b,'duration':0.,'observations':[]} for b in intervals],
            'semantics':'No supported clock; do not construct attention.'}
    if abs(float(pose['telemetry_time_seconds'])-query)>1e-6:raise ValueError('Query and camera clock disagree')
    focal=height/(2*math.tan(math.radians(float(pose['fov_vertical_degrees']))/2))
    sigma=max(1.,focal*math.tan(math.radians(sigma_degrees)))
    for item in support_intervals(records,query,history):
        row=item['record'];projection=project_gaze(row,pose,width,height)
        if projection['status']!='projected_frustum_only':excluded[projection['status']]+=1;continue
        point=projection['pixel_xy']
        if not valid_point(point,image_size):raise ValueError('Invalid supported projection')
        area,kernel=_kernel((height,width),point,sigma)
        for index,(lo,hi) in enumerate(intervals):
            duration=max(0.,min(hi,item['support_interval'][1])-max(lo,item['support_interval'][0]))
            if duration<=0:continue
            maps[(index,*area)]+=kernel*duration
            groups[index].append({'source_index':row['source_index'],'gaze_time_seconds':row['telemetry_time_seconds'],
                'support_interval':item['support_interval'],'weight_seconds':duration,'pixel_xy':point,
                'gaze_point_world':row.get('gaze_point_world'),'projection_source':projection['projection_source']})
    stats=[]
    for index,observations in enumerate(groups):
        duration=sum(r['weight_seconds'] for r in observations)
        stats.append({'index_old_to_new':index,'window':intervals[index],
            'duration':duration,'coverage':duration/(history/bins),
            'effective_samples':len(observations),'display_support':min(1.,duration/.25),
            'observations':observations,'modes':extract_modes(maps[index],observations,image_size)})
    return maps,{'version':VERSION,'status':'SUPPORTED_VOLUME','clock_supported':True,'query':query,
        'history_seconds':history,'sigma_degrees':sigma_degrees,'sigma_pixels':sigma,
        'bins':stats,'excluded_status_counts':dict(excluded),
        'total_frustum_duration':sum(b['duration'] for b in stats),
        'visibility':'unknown_frustum_only','dynamic_correspondence':'unknown',
        'semantics':'mass is supported duration in seconds; normalized shape is not target confidence'}


def mass_region(density,fraction=.8):
    if not 0<fraction<=1:raise ValueError('Mass fraction must be in (0,1]')
    total=float(density.sum())
    if total<=0:return np.zeros_like(density,dtype=bool)
    values=np.sort(density.ravel())[::-1]
    index=min(len(values)-1,int(np.searchsorted(np.cumsum(values,dtype=float),total*fraction)))
    return (density>=values[index]) & (density>0)


def extract_modes(density,observations,image_size,max_modes=2):
    if not observations or float(density.sum())<=0:return []
    regions,n=label(mass_region(density,.85))
    total=float(density.sum());modes=[]
    for component in range(1,n+1):
        region=regions==component;mass=float(density[region].sum())
        if mass/total<.08:continue
        yy,xx=np.nonzero(region)
        px,py=float(np.average(xx,weights=density[region])),float(np.average(yy,weights=density[region]))
        # Every representative is an actual observation, never an invented midpoint.
        representative=min(observations,key=lambda r:(r['pixel_xy'][0]-px)**2+(r['pixel_xy'][1]-py)**2)
        modes.append({'observed_representative':representative,'mass_fraction':mass/total,
                      'bbox_pixels':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]})
    modes.sort(key=lambda r:-r['mass_fraction'])
    # Keep raw modes with explicit fractions. No object identity is assigned.
    return modes[:max_modes]


def render_volume(rgb,maps,stats,style):
    """Display shape with support gating, or expose the RGB through an aperture."""
    source=np.asarray(rgb.convert('RGB'),dtype=np.float32)
    if stats['status']!='SUPPORTED_VOLUME' or not np.any(maps):return rgb.copy()
    if maps.shape[1:]==source.shape[:2]:pass
    else:raise ValueError('Attention/image dimensions differ')
    if style=='CONTOURS':
        output=source.copy()
        for index,b in enumerate(stats['bins']):
            # Weak bins remain faint; normalizing spatial shape cannot create support.
            strength=b['display_support']
            for fraction,alpha in [(.85,.75),(.50,.95)]:
                region=mass_region(maps[index],fraction)
                boundary=region & ~binary_erosion(region,iterations=2)
                a=alpha*strength
                output[boundary]=output[boundary]*(1-a)+COLORS[index]*a
        return Image.fromarray(output.clip(0,255).astype('uint8'))
    if style=='SPOTLIGHT':
        focus=np.zeros(source.shape[:2],dtype=np.float32)
        for index,b in enumerate(stats['bins']):
            peak=float(maps[index].max())
            if peak>0:focus=np.maximum(focus,np.sqrt(maps[index]/peak)*b['display_support'])
        # No artificial colors on object centers; peripheral luminance is bounded.
        return Image.fromarray((source*(.55+.45*focus[...,None])).clip(0,255).astype('uint8'))
    if style=='SEPARATE_MAP':
        peak=np.maximum(maps.max(axis=(1,2)),1e-20)
        amplitudes=np.array([b['display_support'] for b in stats['bins']])
        normalized=np.sqrt(maps/peak[:,None,None])*amplitudes[:,None,None]
        color=np.einsum('khw,kc->hwc',normalized,COLORS.astype(float))
        return Image.fromarray(np.clip(color,0,255).astype('uint8'))
    raise ValueError(f'Unknown style: {style}')


def heat_guided_crop(rgb,point,maps,*,minimum=.18,maximum=.5,mass_fraction=.8):
    """Same queried point; local temporal density determines amount of context.

    Use the newest bin with support. Remote modes are excluded by the maximum
    aperture before calculating weighted distance quantiles. No center is moved.
    """
    if not valid_point(point,rgb.size):raise ValueError('Valid query point required')
    newest=next((plane for plane in maps[::-1] if plane.sum()>0),None)
    if newest is None:return crop_at(rgb,point,minimum)
    yy,xx=np.indices(newest.shape);distance=np.hypot(xx-point[0],yy-point[1])
    limit=min(rgb.size)*maximum/2
    eligible=(distance<=limit)&(newest>0)
    if not np.any(eligible):return crop_at(rgb,point,minimum)
    order=np.argsort(distance[eligible]);d=distance[eligible][order];w=newest[eligible][order]
    i=min(len(d)-1,int(np.searchsorted(np.cumsum(w,dtype=float),float(w.sum())*mass_fraction)))
    ratio=float(np.clip(2*d[i]/min(rgb.size),minimum,maximum))
    view,meta=crop_at(rgb,point,ratio)
    meta.update(rule='local recent-bin 80% duration radius, center fixed',mass_fraction=mass_fraction)
    return view,meta
