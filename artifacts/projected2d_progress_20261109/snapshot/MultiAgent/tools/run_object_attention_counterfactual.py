#!/usr/bin/env python3
"""Same-category object-attention permutation, graphical input only.

Changes only mass assigned to visually proposed same-category instances. This
is a diagnostic control; it never relabels object identity or target GT.
"""
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.perspective_attention_study import read
from MultiAgent.qwen_backend import QwenBackend
from tools.run_object_attention_study import OUTPUT,render_card


def shifted_links(objects,links):
    result=copy.deepcopy(links);mass=np.array(links['mass_seconds_by_phase_object']);shifted=mass.copy();groups=defaultdict(list)
    for i,obj in enumerate(objects):groups[obj['category']].append(i)
    for group in groups.values():
        if len(group)>1:shifted[:,group]=np.roll(mass[:,group],1,axis=1)
    result['mass_seconds_by_phase_object']=shifted.tolist()
    return result,float(np.abs(mass-shifted).sum())


def prepare(parent):
    root=parent/'counterfactual';root.mkdir(exist_ok=True)
    if (root/'manifest.json').exists():raise ValueError('Frozen counterfactual already exists')
    m=read(parent/'manifest.json');requests=[];changed=[]
    for event in m['events']:
        for frame in event['frames']:
            links,delta=shifted_links(frame['objects'],frame['links'])
            if delta<.02:continue
            changed.append({'event':event['event'],'frame_id':frame['frame_id'],'total_l1_mass_change_seconds':delta})
            data=np.load(parent/frame['npz_path'])
            with Image.open(frame['source_path']) as im:rgb=im.convert('RGB')
            for r in m['requests']:
                if r['event_id']!=event['id'] or r['frame_id']!=frame['frame_id'] or r['condition']!='OBJECT_HEAT':continue
                rid=r['id'].replace('OBJECT_HEAT','SAME_CATEGORY_SHIFT');path=root/'images'/f'{rid}.png';path.parent.mkdir(exist_ok=True)
                render_card(rgb,frame['objects'],data['masks'],data['maps'],links,r['indices'],'OBJECT_HEAT').save(path)
                requests.append({**r,'id':rid,'condition':'SAME_CATEGORY_SHIFT','path':str(path.relative_to(root)),
                    'sha256':sha(path),'paired_parent_id':r['id']})
    save(root/'manifest.json',{'parent_manifest_sha256':sha(parent/'manifest.json'),'requests':requests,'changed_frames':changed,
        'regression_only':True,'selection_rule':'Keep frames with >=20ms total L1 mass change under within-category cyclic permutation. No response or GT used.',
        'scope':'Same RGB, proposals, IDs, prompts; only graphical object-linked heat moves among same-category instances. Unassigned gaze unchanged.',
        'code_sha256':sha(__file__)})
    save(root/'protocol_lock.json',{'manifest_sha256':sha(root/'manifest.json'),'before_model':True})
    print('CONTROL_PREPARED',len(changed),'frames',len(requests),'calls',flush=True)


def run(parent):
    root=parent/'counterfactual';m=read(root/'manifest.json');assert sha(parent/'manifest.json')==m['parent_manifest_sha256']
    assert sha(root/'manifest.json')==read(root/'protocol_lock.json')['manifest_sha256']
    backend=QwenBackend(root/'model_cache',model='/ai/data/Qwen3-VL-8B-Instruct',local_only=True,
        namespace='object_attention_counterfactual_v1',trace_visual_inputs=True)
    rows=m['requests'][:];random.Random(20261106).shuffle(rows)
    for r in rows:
        dest=root/'responses'/f"{r['id']}.json"
        if dest.exists():continue
        assert sha(root/r['path'])==r['sha256'];print('INFER',r['id'],flush=True)
        parsed=backend.ask(r['id'],'select_objects',r['prompt'],[root/r['path']],max_new_tokens=160);raw=read(backend.requests[-1])
        valid=set(parsed)=={'selected','uncertain'} and all(isinstance(parsed[k],list) for k in parsed)
        if valid:
            ids=parsed['selected']+parsed['uncertain'];valid=all(isinstance(i,str) and i in r['available_ids'] for i in ids) and len(set(ids))==len(ids)
        save(dest,{'id':r['id'],'paired_parent_id':r['paired_parent_id'],'parsed':parsed,'schema_valid':valid,
            'request_path':backend.requests[-1],'processor_audit':raw['processor_audit'],
            'vision_verified':raw['visual_tensor_trace']['input_consumption_verified']})
        print('RESULT',parsed,flush=True)
    report(parent)


def report(parent):
    root=parent/'counterfactual';m=read(root/'manifest.json');pairs=[]
    for r in m['requests']:
        control=root/'responses'/f"{r['id']}.json";true=parent/'responses'/f"{r['paired_parent_id']}.json"
        if not control.exists() or not true.exists():continue
        a=read(true);b=read(control)
        if not a['schema_valid'] or not b['schema_valid']:continue
        pairs.append({'event':r['event'],'frame_id':r['frame_id'],'page':r['page'],'parent_id':r['paired_parent_id'],
            'control_id':r['id'],'selected_changed':set(a['parsed']['selected'])!=set(b['parsed']['selected']),
            'retained_changed':set(a['parsed']['selected']+a['parsed']['uncertain'])!=set(b['parsed']['selected']+b['parsed']['uncertain'])})
    save(root/'analysis.json',{'planned_calls':len(m['requests']),'completed_calls':len(list((root/'responses').glob('*.json'))),
        'valid_paired_pages':len(pairs),'selected_set_changes':sum(r['selected_changed'] for r in pairs),
        'retained_set_changes':sum(r['retained_changed'] for r in pairs),'rows':pairs,
        'scope':'Output sensitivity to a deliberately wrong graphical attention/object association, not target accuracy.'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','report']);p.add_argument('--parent',type=Path,default=OUTPUT)
    a=p.parse_args();{'prepare':prepare,'run':run,'report':report}[a.action](a.parent.resolve())
