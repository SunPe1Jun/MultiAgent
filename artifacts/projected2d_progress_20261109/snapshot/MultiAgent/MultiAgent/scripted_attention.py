"""Deterministic spatial emphasis and evidence cards, independent of dwell amplitude.

A spatial mask expresses a rendering aperture, never confidence or target identity.
Clock/measurement validity must be checked before enabling the aperture.
"""
from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw

VERSION = 'scripted_spatial_aperture_v1'


def valid_point(point, size):
    try:
        x, y = map(float, point)
        return math.isfinite(x) and math.isfinite(y) and 0 <= x < size[0] and 0 <= y < size[1]
    except (TypeError, ValueError):
        return False


def spatial_foveation(image, point, *, clock_supported, vertical_fov,
                     inner_degrees=8., outer_degrees=18., clear_floor=.55):
    source = image.convert('RGB')
    if not clock_supported or not valid_point(point, source.size):
        return source.copy(), {'status': 'RGB_FALLBACK', 'aperture_enabled': False}
    if not (math.isfinite(vertical_fov) and 0 < vertical_fov < 180
            and 0 <= inner_degrees < outer_degrees < 90 and 0 <= clear_floor <= 1):
        raise ValueError('Invalid aperture/FOV parameters')
    focal = source.height / (2*math.tan(math.radians(vertical_fov)/2))
    yy, xx = np.indices((source.height, source.width), dtype=np.float32)
    radius_degrees = np.degrees(np.arctan(np.hypot(xx-point[0], yy-point[1])/focal))
    fraction = np.clip((radius_degrees-inner_degrees)/(outer_degrees-inner_degrees), 0., 1.)
    mask = 1.-fraction*fraction*(3.-2.*fraction)
    levels = []
    for factor in (2, 4, 8):
        small = source.resize((max(1,source.width//factor), max(1,source.height//factor)), Image.Resampling.BOX)
        levels.append(np.asarray(small.resize(source.size, Image.Resampling.BICUBIC), dtype=np.float32))
    position = (1.-mask)*(len(levels)-1)
    low = np.floor(position).astype(np.intp)
    high = np.minimum(low+1,len(levels)-1)
    mix = (position-low)[...,None]
    indices_y,indices_x=np.indices(mask.shape)
    stack=np.stack(levels)
    blurred=stack[low,indices_y,indices_x]*(1.-mix)+stack[high,indices_y,indices_x]*mix
    original_weight=clear_floor+(1.-clear_floor)*mask
    original=np.asarray(source,dtype=np.float32)
    pixels=np.clip(original*original_weight[...,None]+blurred*(1.-original_weight[...,None]),0,255).astype(np.uint8)
    return Image.fromarray(pixels), {'status':'SPATIAL_APERTURE', 'aperture_enabled':True,
        'version':VERSION,'inner_degrees':inner_degrees,'outer_degrees':outer_degrees,
        'clear_floor':clear_floor,'inner_region_fraction':float(np.mean(mask>=1.)),
        'minimum_original_weight':float(original_weight.min()),
        'maximum_original_weight':float(original_weight.max()),
        'mask_semantics':'rendering aperture, not gaze confidence or target probability',
        'evidence_amplitude_used':False}


def crop_at(image, point, ratio=.35):
    if not valid_point(point, image.size) or not 0 < ratio <= 1:
        raise ValueError('Valid region point and crop ratio required')
    width,height=image.size
    side=max(2,int(round(min(width,height)*ratio)))
    left,top=int(round(point[0]-side/2)),int(round(point[1]-side/2))
    bbox=[left,top,left+side,top+side]
    view=image.crop(bbox)
    overlap=max(0,min(width,bbox[2])-max(0,left))*max(0,min(height,bbox[3])-max(0,top))
    return view, {'bbox_source_pixels':bbox,'padding_fraction':1-overlap/(side*side),
                  'point_in_view':[point[0]-left,point[1]-top],'ratio':ratio}


def draw_indicator(image, xy, radius=9):
    """Four corners leave the queried center pixels unobscured."""
    draw=ImageDraw.Draw(image)
    x,y=xy
    for sx in (-1,1):
        for sy in (-1,1):
            corner=(x+sx*radius,y+sy*radius)
            for width,color in [(4,'black'),(2,'white')]:
                draw.line((corner,(corner[0]-sx*4,corner[1])),fill=color,width=width)
                draw.line((corner,(corner[0],corner[1]-sy*4)),fill=color,width=width)
    return image


def evidence_card(context, view, point_in_context, point_in_view, *, frame_id, pts):
    width,height=448,448
    left=context.resize((width,height),Image.Resampling.LANCZOS)
    right=view.resize((width,height),Image.Resampling.LANCZOS)
    a=[point_in_context[0]/context.width*width,point_in_context[1]/context.height*height]
    b=[point_in_view[0]/view.width*width,point_in_view[1]/view.height*height]
    draw_indicator(left,a)
    draw_indicator(right,b)
    canvas=Image.new('RGB',(width*2,height+32),'#17212b')
    canvas.paste(left,(0,32));canvas.paste(right,(width,32))
    draw=ImageDraw.Draw(canvas)
    draw.text((8,8),f'{frame_id} | PTS {pts:.3f}s | CONTEXT',fill='white')
    draw.text((width+8,8),'REGION VIEW',fill='white')
    return canvas


def summarize_projected_history(projected, *, query_time, image_size,
                                history_seconds=2., bins=3, grid_size=4):
    """Numerical history in ONE frame's camera; no color interpretation required.

    Duration fractions describe measured frustum-supported gaze, not confidence,
    visibility or object identity. Representatives are actual observed pixels.
    Callers must use the same aligned frame camera for all supplied projections.
    """
    if not (math.isfinite(query_time) and math.isfinite(history_seconds) and history_seconds > 0
            and bins > 0 and grid_size > 0 and min(image_size) > 0):
        raise ValueError('Finite query, positive window/bins/grid/image required')
    start=query_time-history_seconds
    groups=[[] for _ in range(bins)]
    excluded={}
    for row in projected:
        status=row.get('status')
        if status!='projected_frustum_only':
            excluded[status or 'missing_status']=excluded.get(status or 'missing_status',0)+1
            continue
        timestamp=float(row['gaze_time_seconds']);weight=float(row['weight_seconds'])
        if not math.isfinite(timestamp) or not math.isfinite(weight) or weight < 0:
            raise ValueError('Invalid history observation time/weight')
        if timestamp < start-1e-6 or timestamp > query_time+1e-6:
            raise ValueError('History observation outside past-only window')
        point=row.get('pixel_xy')
        if not valid_point(point,image_size):
            raise ValueError('Frustum-supported history has invalid pixel')
        index=max(0,min(bins-1,int((timestamp-start)/history_seconds*bins)))
        groups[index].append({'source_index':row['source_index'],
            'gaze_time_seconds':timestamp,'weight_seconds':weight,'pixel_xy':list(map(float,point))})
    summaries=[]
    for index,items in enumerate(groups):
        duration=sum(r['weight_seconds'] for r in items)
        grid=np.zeros((grid_size,grid_size),dtype=np.float64)
        for r in items:
            x,y=r['pixel_xy']
            grid[min(grid_size-1,int(y/image_size[1]*grid_size)),
                 min(grid_size-1,int(x/image_size[0]*grid_size))]+=r['weight_seconds']
        representative=None;dispersion=None
        if duration > 0:
            points=np.array([r['pixel_xy'] for r in items],dtype=np.float64)
            weights=np.array([r['weight_seconds'] for r in items],dtype=np.float64)
            mean=np.average(points,axis=0,weights=weights)
            # Weighted squared-distance medoid: retain an actual measurement,
            # rather than inventing a midpoint between potentially distinct objects.
            positive=np.flatnonzero(weights > 0)
            chosen=int(positive[np.argmin(np.sum((points[positive]-mean)**2,axis=1))])
            representative=items[chosen].copy()
            dispersion=float(np.sqrt(np.average(np.sum((points-mean)**2,axis=1),weights=weights)))
        summaries.append({'bin_index_old_to_new':index,
            'window_telemetry_seconds':[start+history_seconds*index/bins,start+history_seconds*(index+1)/bins],
            'window_end_inclusive':index==bins-1,'projected_sample_count':len(items),
            'frustum_supported_duration_seconds':duration,
            'cell_duration_seconds':grid.tolist(),
            'cell_duration_fraction':(grid/duration).tolist() if duration>0 else None,
            'representative_observed_point':representative,'spatial_rms_pixels':dispersion,
            'observations':items})
    return {'schema':'scripted_past_gaze_bins_v1','query_telemetry_seconds':query_time,
        'history_seconds':history_seconds,'image_size':list(image_size),'grid_size':grid_size,
        'bins_old_to_new':summaries,'excluded_projection_status_counts':excluded,
        'visibility':'unknown_frustum_only','dynamic_correspondence':'unknown',
        'semantics':'measured duration distribution; not confidence, object identity or calibrated probability'}
