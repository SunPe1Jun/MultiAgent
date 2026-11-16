"""Gaze-driven historical RGB proposals that survive leaving the current view.

These are engineering attention segments, not physiological fixation labels or
object tracks. Semantics must be read from the selected historical RGB itself.
"""
from __future__ import annotations
import math
from collections import Counter
import numpy as np
from .attention_volume import support_intervals
from .gaze_evidence import project_gaze

VERSION='native_attention_memory_proposals_v1'


def build_memory_proposals(records,*,query,current_pose,image_size,clock_supported,
                           history=2.,split_degrees=8.,minimum_duration=.12):
    if not 0<split_degrees<180 or minimum_duration<0:raise ValueError('Invalid segmentation parameters')
    if not clock_supported:
        return {'version':VERSION,'status':'RGB_FALLBACK','proposals':[],
                'reason':'No supported video/telemetry clock'}
    groups=[];excluded=Counter();previous=None
    for item in support_intervals(records,query,history):
        row=item['record'];position=row.get('camera_position');rotation=row.get('camera_rotation_xyzw')
        if position is None or rotation is None or row.get('camera_fov_degrees') is None:
            excluded['source_camera_unavailable']+=1;previous=None;continue
        pose={'position_world':position,'rotation_xyzw':rotation,'fov_vertical_degrees':row['camera_fov_degrees']}
        native=project_gaze(row,pose,*image_size)
        if native['status']!='projected_frustum_only':
            excluded['native_'+native['status']]+=1;previous=None;continue
        direction=(np.array(row['gaze_point_world'])-np.array(position) if row.get('gaze_point_world') is not None
                   else np.array(row['gaze_vector_world']))
        norm=float(np.linalg.norm(direction))
        if not math.isfinite(norm) or norm<1e-8:excluded['invalid_world_ray']+=1;previous=None;continue
        ray=direction/norm
        projected=project_gaze(row,current_pose,*image_size)
        observation={'source_index':row['source_index'],'gaze_time_seconds':row['telemetry_time_seconds'],
            'support_interval':item['support_interval'],'duration':item['weight_seconds'],
            'world_ray':ray.tolist(),'native_pixel_xy':native['pixel_xy'],
            'current_frustum_status':projected['status'],'current_pixel_xy':projected.get('pixel_xy')}
        if previous is None:
            groups.append([observation])
        else:
            anchor=np.array(groups[-1][0]['world_ray'])
            angle=math.degrees(math.acos(float(np.clip(np.dot(anchor,ray),-1,1))))
            gap=item['support_interval'][0]-previous['support_interval'][1]
            if gap>1e-6 or angle>split_degrees:groups.append([observation])
            else:groups[-1].append(observation)
        previous=observation
    proposals=[]
    for group in groups:
        rays=np.array([r['world_ray'] for r in group]);weights=np.array([r['duration'] for r in group])
        average=np.average(rays,axis=0,weights=weights)
        index=int(np.argmin(np.sum((rays-average)**2,axis=1)))
        duration=float(weights.sum())
        proposals.append({'proposal_id':f'P{len(proposals)+1:03d}',
            'window_telemetry_seconds':[group[0]['support_interval'][0],group[-1]['support_interval'][1]],
            'supported_duration_seconds':duration,'low_support':duration<minimum_duration,
            'observation_count':len(group),'representative':group[index],
            'outside_current_frustum_count':sum(r['current_frustum_status']!='projected_frustum_only' for r in group),
            'observations':group})
    return {'version':VERSION,'status':'NATIVE_MEMORY_PROPOSALS','query_telemetry_seconds':query,
        'history_seconds':history,'split_degrees':split_degrees,'minimum_duration':minimum_duration,
        'proposals':proposals,'excluded':dict(excluded),
        'native_supported_duration_seconds':sum(p['supported_duration_seconds'] for p in proposals),
        'semantics':'Engineering world-ray segments, not physiological fixation, target confidence or object identity. '
                    'Offscreen history is retained; current RGB reprojection is diagnostic only.'}
