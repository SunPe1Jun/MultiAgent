#!/usr/bin/env python3
"""Descriptive paired comparison on the same eight events, no new inference."""
import json
from pathlib import Path
from MultiAgent.attention_reliability import cluster_paired_interval
from MultiAgent.heatmap_temporal_pilot import save
from tools.run_full_interaction_referring import OUTPUT as ROOT,read


def f1(a,b):
    p=set(a);g=set(b);tp=len(p&g);return 2*tp/(len(p)+len(g)) if len(p)+len(g) else 0.


def build():
    a=read(ROOT/'grounding_v2/analysis.json');b=read(ROOT/'grounding_v3/analysis.json')
    first={r['id']:r for r in a['events']};second={r['id']:r for r in b['events']}
    common=sorted(set(first)&set(second));rows=[]
    for event in common:
        x,y=first[event],second[event]
        assert set(x['gt'])==set(y['gt'])
        per={}
        for cond in ('NO_ADDED_HEAT','TRUE_HEAT','SPATIAL_MIRROR','TIME_REVERSE'):
            per[cond]={'v2_f1':f1(x['predictions'][cond],x['gt']),'v3_f1':f1(y['predictions'][cond],y['gt']),
                       'v2_exact':x['metrics'][cond]['exact'],'v3_exact':y['metrics'][cond]['exact'],
                       'v2_prediction':x['predictions'][cond],'v3_prediction':y['predictions'][cond]}
        rows.append({'event':event,'gt':x['gt'],'conditions':per})
    result={'same_old_interactions':len(rows),'adaptive_input_changes':['coordinate origin instruction','shared instruction-only mention hints','camera support availability hints'],
        'inference_reused_only_when_exact_prompt_and_image_hash_match':True,
        'conditions':{c:{'v3_minus_v2_f1':float(sum(r['conditions'][c]['v3_f1']-r['conditions'][c]['v2_f1'] for r in rows)/len(rows)),
            'paired_interaction_interval':cluster_paired_interval([{'cluster':r['event'],'baseline':r['conditions'][c]['v2_f1'],'treatment':r['conditions'][c]['v3_f1']} for r in rows]),
            'v2_exact':sum(r['conditions'][c]['v2_exact'] for r in rows),
            'v3_exact':sum(r['conditions'][c]['v3_exact'] for r in rows)} for c in ('NO_ADDED_HEAT','TRUE_HEAT','SPATIAL_MIRROR','TIME_REVERSE')},
        'events':rows,'gt_used_only_for_posthoc_scoring':True,'independent_validation':False}
    save(ROOT/'same8_v2_v3_comparison.json',result);print(json.dumps({k:v for k,v in result.items() if k!='events'},ensure_ascii=False,indent=2))


if __name__=='__main__':build()
