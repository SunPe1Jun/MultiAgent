#!/usr/bin/env python3
"""Separate missing visual proposals from history-reprojection effects."""
from collections import Counter
import json
import numpy as np
from MultiAgent.attention_volume import build_volume
from MultiAgent.context import load_contexts
from MultiAgent.gaze_evidence import gaze_records
from MultiAgent.object_attention import mask_attention_links
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.perspective_attention_study import read
from MultiAgent.spherical_attention_study import AUDIT
from tools.run_object_attention_study import OUTPUT


def run(output=OUTPUT):
    m=read(output/'manifest.json');selection=read(AUDIT/'selection.json');keys={e['event'] for e in m['events']}
    contexts=load_contexts({'samples':[r for r in selection['samples'] if f"{r['scene_key']}/{r['row_index']}" in keys]},load_ground_truth=False)
    contexts={f'{c.scene_key}/{c.row_index}':c for c in contexts};rows=[]
    for event in m['events']:
        records=gaze_records(contexts[event['event']])
        for frame in event['frames']:
            masks=np.load(output/frame['npz_path'])['masks'];h,w=masks.shape[1:];g=frame['geometry']
            for history in [.25,.5,1.,2.]:
                maps,stats=build_volume(records,query=g['telemetry_time_seconds'],pose=g['reference_pose'],image_size=(w,h),
                    clock_supported=g['clock_available'],history=history,bins=3,sigma_degrees=2.)
                links,_,_=mask_attention_links(maps,masks);total=sum(links['total_seconds']);assigned=sum(links['assigned_seconds'])
                rows.append({'event':event['event'],'frame_id':frame['frame_id'],'history_seconds':history,
                    'total_supported_frustum_seconds':total,'assigned_seconds':assigned,'assigned_fraction':assigned/total if total else None,
                    'mask_union_fraction':float(masks.any(axis=0).mean()) if len(masks) else 0.,'objects':len(masks)})
    summary=[]
    for history in [.25,.5,1.,2.]:
        r=[r for r in rows if r['history_seconds']==history];total=sum(r['total_supported_frustum_seconds'] for r in r);assigned=sum(r['assigned_seconds'] for r in r)
        summary.append({'history_seconds':history,'assigned_fraction':assigned/total,'assigned_seconds':assigned,'total_seconds':total,
                        'frames_with_support':sum(r['total_supported_frustum_seconds']>0 for r in r)})
    result={'manifest_sha256':sha(output/'manifest.json'),'summary':summary,'rows':rows,
        'scope':'Same instruction-conditioned candidate masks and native query pose; no GT. Duration totals overlap across queries.'}
    save(output/'gaze_support_diagnostic.json',result);print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
