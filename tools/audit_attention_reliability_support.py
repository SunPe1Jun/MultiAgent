#!/usr/bin/env python3
"""Audit whether each selected real object pair actually carries gaze evidence.

Checks labels/attention correspondence as a diagnostic; never treats synthetic
or center-derived targets as verified object GT. Gates are engineering values.
"""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from MultiAgent.attention_reliability import pair_attention
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.perspective_attention_study import read
from tools.run_attention_reliability_study import OUTPUT


def audit(output=OUTPUT):
    m=read(output/'manifest.json');rows=[]
    for case in m['cases']:
        if case['family']!='real_center_proxy':continue
        data=np.load(output/case['pair_npz_path']);mass,residual,links=pair_attention(data['raw_maps'],data['masks'],'TRUE_HEAT')
        total=mass.sum(0);supported=float(total.sum());all_seconds=float(data['raw_maps'].sum())
        row={'case_id':case['id'],'event':case['event'],'target_physical_index':0,'mass_seconds_by_phase_object':mass.tolist(),
            'target_assigned_seconds':float(total[0]),'alternative_assigned_seconds':float(total[1]),
            'pair_assigned_seconds':supported,'all_frustum_seconds':all_seconds,
            'pair_fraction_of_frustum_support':supported/all_seconds if all_seconds>0 else None,
            'target_share_within_pair':float(total[0]/supported) if supported>0 else None,
            'support_bin':'at_least_250ms' if supported>=.25 else '10_to_250ms' if supported>=.01 else 'below_10ms',
            'gaze_argmax_agrees_weak_target':bool(total[0]>total[1]) if supported>=.01 else None,
            'known_label_is_verified_object_gt':False}
        rows.append(row)
    summary={'selected_real_cases':len(rows),'interactions':len({r['event'] for r in rows}),'by_support':{}}
    for group in ['below_10ms','10_to_250ms','at_least_250ms']:
        rr=[r for r in rows if r['support_bin']==group]
        summary['by_support'][group]={'cases':len(rr),'interactions':len({r['event'] for r in rr}),
            'gaze_argmax_agrees_weak_target':sum(r['gaze_argmax_agrees_weak_target'] is True for r in rr)}
    result={'manifest_sha256':sha(output/'manifest.json'),'summary':summary,'rows':rows,
        'scope':'Annotation-conditioned selected pairs. 250ms gate was set in frozen renderer; 10ms stratum is diagnostic. No image/prompt/model change, no target-GT claim.'}
    save(output/'support_audit.json',result);print(json.dumps(summary,indent=2))


if __name__=='__main__':audit()
