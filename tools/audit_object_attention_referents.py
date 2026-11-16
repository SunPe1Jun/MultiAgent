#!/usr/bin/env python3
"""Post-inference, event-referent-only center-projection diagnostic.

These centers have unverified pixel calibration/visibility. This is a weak
geometric retention proxy, never segmentation, object identity or final F1 GT.
"""
from collections import Counter
import json
import re
import numpy as np
from scipy.ndimage import binary_dilation
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.object_attention import canonical_category
from MultiAgent.perspective_attention_study import read
from MultiAgent.spherical_attention_study import AUDIT
from tools.run_object_attention_study import OUTPUT,CONDITIONS


def family(name):
    value=canonical_category(re.sub(r'\d+$','',name).replace('_',' '))
    return {'table':'desk','warehouse':'building'}.get(value,value)


def audit(output=OUTPUT):
    m=read(output/'manifest.json');responses={p.stem:read(p) for p in (output/'responses').glob('*.json')}
    missing=[r['id'] for r in m['requests'] if r['id'] not in responses]
    if missing:raise ValueError(f'Freeze all inference responses before reading referent audit; missing {len(missing)}')
    eligible=[];offscreen=[];objects=[];audit_sources=[]
    for event in m['events']:
        path=AUDIT/'events'/event['event']/'audit.json';source=read(path)
        assert source['projection_scope']=='event_referents_only'
        audit_sources.append({'path':str(path),'sha256':sha(path)})
        original={r['frame_id']:r for r in source['frames']}
        for frame in event['frames']:
            row=original[frame['frame_id']];assert row['source_sha256']==frame['source_sha256']
            data=np.load(output/frame['npz_path']);masks=data['masks'];h,w=masks.shape[1:]
            dilated=np.stack([binary_dilation(mask,iterations=8) for mask in masks]) if len(masks) else masks
            retained={}
            for c in CONDITIONS:
                calls=[r for r in m['requests'] if r['event_id']==event['id'] and r['frame_id']==frame['frame_id'] and r['condition']==c]
                all_valid=all(responses[r['id']]['schema_valid'] for r in calls)
                if all_valid:
                    selected=set(v for r in calls for v in responses[r['id']]['parsed']['selected'])
                    uncertain=set(v for r in calls for v in responses[r['id']]['parsed']['uncertain'])
                    retained[c]={'selected':selected,'with_uncertain':selected|uncertain,'all_pages_valid':True}
                else:retained[c]={'selected':set(),'with_uncertain':set(),'all_pages_valid':False}
            refs=[]
            for reference in row['referent_projections']:
                base={'event':event['event'],'event_id':event['id'],'split':event['split'],'frame_id':frame['frame_id'],
                    'reference_name':reference['referent_name'],'reference_category':family(reference['referent_name']),
                    'in_frustum':reference['in_frustum'],'visibility':'unknown','status':reference['status']}
                if not reference['in_frustum']:
                    offscreen.append(base);continue
                x,y=reference['pixel_xy_unclipped'];px=min(w-1,max(0,int(round(x))));py=min(h-1,max(0,int(round(y))))
                box_ids=[];mask_ids=[];tolerant=[]
                for i,obj in enumerate(frame['objects']):
                    if family(obj['category'])!=base['reference_category']:continue
                    x0,y0,x1,y1=obj['box']
                    if x0<=x<=x1 and y0<=y<=y1:box_ids.append(obj['id'])
                    if masks[i,py,px]:mask_ids.append(obj['id'])
                    if dilated[i,py,px]:tolerant.append(obj['id'])
                base.update(point_xy=[x,y],box_matching_ids=box_ids,mask_matching_ids=mask_ids,mask_8px_matching_ids=tolerant,
                    proposal_covered=bool(box_ids),mask_covered=bool(mask_ids),ambiguous_boxes=len(box_ids)>1)
                base['conditions']={c:{'schema_valid':r['all_pages_valid'],
                    'selected_hits_center':bool(set(box_ids)&r['selected']),
                    'selected_or_uncertain_hits_center':bool(set(box_ids)&r['with_uncertain'])} for c,r in retained.items()}
                eligible.append(base);refs.append(base)
            for i,obj in enumerate(frame['objects']):
                objects.append({'event':event['event'],'frame_id':frame['frame_id'],'id':obj['id'],'category':obj['category'],
                    'matches_any_projected_reference_center':any(obj['id'] in r['box_matching_ids'] for r in refs),
                    'conditions':{c:{'selected':obj['id'] in r['selected'],'with_uncertain':obj['id'] in r['with_uncertain']} for c,r in retained.items()}})
    summary={}
    for split in ['development','check','all']:
        refs=[r for r in eligible if split=='all' or r['split']==split];covered=[r for r in refs if r['proposal_covered']]
        summary[split]={'in_frustum_reference_frames':len(refs),'proposal_box_center_covered':len(covered),
            'proposal_mask_center_covered':sum(r['mask_covered'] for r in refs),'ambiguous_box_centers':sum(r['ambiguous_boxes'] for r in refs),
            'by_condition':{}}
        for c in CONDITIONS:
            rr=[r for r in refs if r['conditions'][c]['schema_valid']]
            summary[split]['by_condition'][c]={'valid_reference_frames':len(rr),
                'selected_hits':sum(r['conditions'][c]['selected_hits_center'] for r in rr),
                'selected_or_uncertain_hits':sum(r['conditions'][c]['selected_or_uncertain_hits_center'] for r in rr),
                'proposal_covered_centers_lost_after_selection':sum(r['proposal_covered'] and not r['conditions'][c]['selected_or_uncertain_hits_center'] for r in rr)}
    for c in CONDITIONS:
        summary['all']['by_condition'][c].update(selected_objects=sum(r['conditions'][c]['selected'] for r in objects),
            retained_objects=sum(r['conditions'][c]['with_uncertain'] for r in objects),
            retained_without_matching_reference_center=sum(r['conditions'][c]['with_uncertain'] and not r['matches_any_projected_reference_center'] for r in objects))
    save(output/'referent_center_audit.json',{'summary':summary,'eligible_reference_frames':eligible,'offscreen_reference_frames':offscreen,
        'objects':objects,'audit_sources':audit_sources,'manifest_sha256':sha(output/'manifest.json'),'script_sha256':sha(__file__),
        'projections_scope':'only existing event referents, reused after inference. No all-anchor projection.',
        'limitations':'Weak center containment proxy, unverified calibration/visibility, not pixel/object/referent accuracy. Unmatched proposals are not necessarily false positives.'})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':audit()
