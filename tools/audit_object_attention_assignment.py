#!/usr/bin/env python3
"""Distinguish broad-kernel mass loss from gaze centers without object proposals."""
import json
import math
import numpy as np
from scipy.ndimage import binary_dilation
from MultiAgent.attention_volume import _kernel
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.perspective_attention_study import read
from tools.run_object_attention_study import OUTPUT


def audit(output=OUTPUT):
    m=read(output/'manifest.json');rows=[]
    for event in m['events']:
        for frame in event['frames']:
            data=np.load(output/frame['npz_path']);masks=data['masks'];h,w=masks.shape[1:];union=masks.any(axis=0)
            boxes=np.zeros((h,w),bool)
            for obj in frame['objects']:
                x0,y0,x1,y1=obj['box'];boxes[max(0,int(y0)):min(h,math.ceil(y1)),max(0,int(x0)):min(w,math.ceil(x1))]=True
            expanded={radius:binary_dilation(union,iterations=radius) if radius else union for radius in [0,2,4,8,16]}
            observations=[o for b in frame['attention_stats']['bins'] for o in b['observations']]
            total=sum(o['weight_seconds'] for o in observations)
            row={'event':event['event'],'frame_id':frame['frame_id'],'total_seconds':total,'observation_count':len(observations),
                 'union_mask_area_fraction':float(union.mean()),'union_box_area_fraction':float(boxes.mean())}
            row['center_mask_seconds']={str(radius):0. for radius in expanded};row['center_box_seconds']=0.
            row['kernel_assigned_seconds']={str(sigma):0. for sigma in [.5,1.,2.,4.]}
            focal=h/(2*math.tan(math.radians(frame['geometry']['reference_pose']['fov_vertical_degrees'])/2))
            for obs in observations:
                x,y=obs['pixel_xy'];px=min(w-1,max(0,int(round(x))));py=min(h-1,max(0,int(round(y))));duration=obs['weight_seconds']
                for radius,mask in expanded.items():
                    if mask[py,px]:row['center_mask_seconds'][str(radius)]+=duration
                if boxes[py,px]:row['center_box_seconds']+=duration
                for angle in [.5,1.,2.,4.]:
                    area,kernel=_kernel((h,w),obs['pixel_xy'],max(1.,focal*math.tan(math.radians(angle))))
                    row['kernel_assigned_seconds'][str(angle)]+=float(kernel[union[area]].sum())*duration
            rows.append(row)
    total=sum(r['total_seconds'] for r in rows)
    summary={'total_seconds_over_overlapping_queries':total,
        'duration_weighted_center_in_mask_fraction':{str(radius):sum(r['center_mask_seconds'][str(radius)] for r in rows)/total for radius in [0,2,4,8,16]},
        'duration_weighted_center_in_box_fraction':sum(r['center_box_seconds'] for r in rows)/total,
        'kernel_sigma_degrees_to_assigned_fraction':{str(angle):sum(r['kernel_assigned_seconds'][str(angle)] for r in rows)/total for angle in [.5,1.,2.,4.]}}
    save(output/'assignment_sensitivity.json',{'summary':summary,'rows':rows,'manifest_sha256':sha(output/'manifest.json'),
        'scope':'Post-protocol diagnostic. No formal image/prompt/response changed. Mask hit and box hit do not prove target identity.'})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':audit()
