#!/usr/bin/env python3
"""Adaptive full-scene followup; whole-composite coordinate contract and mentions."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
from MultiAgent.agents import ReferentialAgent
from MultiAgent.context import load_contexts
from MultiAgent.full_frame_grounding_v3 import VERSION,grounding_prompt,validate_composite_grounding
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.qwen_backend import QwenBackend
import tools.run_full_frame_grounding as base
from tools.run_full_interaction_referring import ROOT,read

OUTPUT=base.PARENT/'grounding_v3'


def prepare(output):
    if (output/'manifest.json').exists():raise ValueError('Never overwrite frozen followup')
    parent=read(base.OUTPUT/'manifest.json')
    # One median-row interaction per subscene; no annotation or response used.
    chosen=[]
    for scene in sorted({e['scene_key'] for e in parent['events']}):
        group=sorted([e for e in parent['events'] if e['scene_key']==scene],key=lambda e:e['row_index'])
        chosen.append(group[len(group)//2])
    events=json.loads(json.dumps(chosen));contexts=load_contexts({'samples':events},load_ground_truth=False)
    backend=QwenBackend(output/'language_cache',model=parent['model'],local_only=True,namespace=VERSION+'_language')
    parser=ReferentialAgent(backend);requests=[];language_records=[]
    for event,context in zip(events,contexts):
        print('LANGUAGE',event['event'],flush=True);parsed=parser.run(context)
        hints=[{k:m.get(k) for k in ('mention_id','text','span','object_type','role','kind','count','coreference_id')}
               for m in parsed['mentions']]
        event['mention_hints']=hints
        language_path=output/'language'/f"{event['id']}.json";save(language_path,parsed)
        event['language_parse_path']=str(language_path.relative_to(output));event['language_parse_sha256']=sha(language_path)
        event['prompt']=grounding_prompt(event)
        for frame in event['frames']:
            for image in frame['images'].values():image['path']=os.path.relpath(base.OUTPUT/image['path'],output)
        fingerprints={}
        for condition in parent['conditions']:
            hashes=[f['images'][condition]['sha256'] for f in event['frames']]
            fingerprint=hashlib.sha256(json.dumps([event['prompt'],hashes],sort_keys=True).encode()).hexdigest()
            rid=event['id']+'_'+condition;canonical=fingerprints.setdefault(fingerprint,rid)
            requests.append({'id':rid,'event_id':event['id'],'condition':condition,
                'canonical_request_id':canonical,'model_input_fingerprint':fingerprint})
    names=['MultiAgent/full_frame_grounding_v3.py','tools/run_full_frame_grounding_v3.py',
        'MultiAgent/full_frame_grounding.py','tools/run_full_frame_grounding.py',
        'MultiAgent/agents.py','MultiAgent/mention_mapping.py','MultiAgent/qwen_backend.py','MultiAgent/qwen_video_compat.py']
    manifest={**parent,'version':VERSION,'events':events,'requests':requests,
        'visual_parent_manifest_sha256':sha(base.OUTPUT/'manifest.json'),
        'planned_unique_model_calls':len({r['canonical_request_id'] for r in requests}),
        'planned_language_interactions':len(events),'language_model_call_records':backend.call_records,
        'source_sha256':{n:sha(ROOT/n) for n in names},
        'reason':'GT-free inspection of V2 choices found G001 concentration and unstable RGB-subpanel coordinate interpretation. Fix whole-image 0..1000 coordinates, expose camera readout support, add shared instruction-only mention hints. No GT metrics were read to select cases or prompts.',
        'selection':'Median row per subscene from fixed24, independently of GT/heat/model outputs; 8 old interactions, adaptive regression.',
        'coordinate_contract':'Model whole displayed image 0..1000; script converts point into RGB-panel-local normalized coordinates.',
        'validation_policy':'Individually reject malformed/nonliteral/unknown-category/unsupported-view/outside-panel point rows; keep separately valid rows, never repair or fabricate targets.',
        'default_pipeline_changed':False,'unseen_validation':False,'adaptive_regression_only':True}
    save(output/'manifest.json',manifest)
    save(output/'protocol_lock.json',{'manifest_sha256':sha(output/'manifest.json'),'before_any_visual_inference':True,
        'before_any_inference':True,'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'shared_language_parses_frozen':True,'gt_loaded':False,'primary_metrics':['Exact-set','micro-F1']})
    for name in names:
        dest=output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    save(output/'language_calls.json',{'calls':backend.call_records,'gt_loaded':False,
        'shared_all_heat_conditions':True,'scope':'Language-only mention extraction, not visual selection.'})
    print('FROZEN',len(events),'interactions',len(requests),'condition results',manifest['planned_unique_model_calls'],'visual calls',len(backend.call_records),'language calls',flush=True)
    del parser,backend
    import gc,torch
    gc.collect();torch.cuda.empty_cache()


def infer(output):
    base.VERSION=VERSION
    base.validate_grounding=validate_composite_grounding
    base.infer(output)


def evaluate(output):
    base.evaluate(output)
    m=read(output/'manifest.json');analysis=read(output/'analysis.json')
    analysis['language_model_calls']=len(m['language_model_call_records'])
    analysis['coordinate_contract']='whole composite 0..1000, script converts to local RGB'
    analysis['model_task']='Instruction-only shared mention parse, full-field visual referent localization, fixed GT-free scene-ID decoder.'
    analysis['mention_hint_coverage']={}
    for condition in m['conditions']:
        rr=[read(output/'responses'/f"{e['id']}_{condition}.json") for e in m['events']]
        analysis['mention_hint_coverage'][condition]={'unaccounted_hint_phrases':sum(len(r['grounding_validation'].get('unaccounted_mention_hints',[])) for r in rr),
            'expected_hint_phrases':sum(len(e['mention_hints']) for e in m['events'])}
    save(output/'analysis.json',analysis)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','infer','evaluate']);p.add_argument('--output',type=Path,default=OUTPUT)
    a=p.parse_args();{'prepare':prepare,'infer':infer,'evaluate':evaluate}[a.action](a.output.resolve())
