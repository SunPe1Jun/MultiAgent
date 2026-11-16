#!/usr/bin/env python3
"""Recheck explicit per-referent frame comparison with an aligned JSON example."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import sys

from MultiAgent.projected2d_attention_study import save_json, sha_file
from MultiAgent.projected2d_main_image import normalize
from MultiAgent.projected2d_routed_rgb import MODEL, ROOT
from tools.audit_referent_frame_choice import prompt_for as prior_prompt, code_lock

SOURCE=ROOT/'.local-sync/referent_frame_choice_audit_20261109'
OUTPUT=ROOT/'.local-sync/referent_frame_checks_v2_20261109'


def prompt_for(event):
    prompt=prior_prompt(event,'PER_REFERENT_COMPARISON')
    old='"predictions":[{"frame_id":"one_of_the_supplied_frames","point_xy":[x,y]}],"evidence":"..."}]}'
    new='"predictions":[{"frame_id":"one_of_the_supplied_frames","point_xy":[x,y]}],"evidence":"...","frame_checks":[{"frame_id":"one_supplied_frame_id","observation":"specific visible evidence or not visible"}]}]}'
    if prompt.count(old)!=1:raise ValueError('Prior JSON schema changed')
    return prompt.replace(old,new)


def verify(out):
    for name,expected in json.loads((out/'protocol_lock.json').read_text())['files'].items():
        if sha_file(Path(name))!=expected:raise ValueError('Locked input changed: '+name)


def prepare(out):
    if (out/'input_manifest.json').exists():raise ValueError('Refuse to overwrite frozen probe')
    code_lock(SOURCE)
    original=json.loads((SOURCE/'input_manifest.json').read_text())
    out.mkdir(parents=True,exist_ok=True)
    m={'version':'frame_comparison_schema_v2','events':original['events'],
       'changes':'Same images, mentions, action contexts and prompt as PER_REFERENT_COMPARISON; corrected JSON example includes frame_checks.',
       'sample_status':'Same 10 previously seen development interactions; not held out.',
       'source_sha256':sha_file(SOURCE/'input_manifest.json')}
    save_json(out/'input_manifest.json',m)
    paths=[Path(__file__).resolve(),SOURCE/'protocol_lock.json',SOURCE/'input_manifest.json',out/'input_manifest.json',
      ROOT/'tools/audit_referent_frame_choice.py',ROOT/'MultiAgent/projected2d_main_image.py',
      ROOT/'MultiAgent/qwen_backend.py',ROOT/'tools/run_full_frame_grounding_v4.py',MODEL/'config.json']
    save_json(out/'protocol_lock.json',{'files':{str(p):sha_file(p) for p in paths}})
    print('FROZEN_CHECK_V2',len(m['events']),'events',flush=True)


def infer(out):
    from MultiAgent.qwen_backend import QwenBackend
    from tools.run_full_frame_grounding_v4 import clean_visual_trace
    verify(out);m=json.loads((out/'input_manifest.json').read_text());backend=QwenBackend(out/'model_cache',str(MODEL),local_only=True,namespace='frame_checks_v2',trace_visual_inputs=True)
    for i,e in enumerate(m['events'],1):
        path=out/'responses'/f"{e['id']}.json";panels=e['panels'];images=[p['inputs']['MAIN_RGB_768'] for p in panels]
        captions=[f"Frame {p['frame_id']} | video_time={p['requested_video_time']:.3f} seconds" for p in panels]
        prompt=prompt_for(e);fingerprint=hashlib.sha256(json.dumps([prompt,captions,[x['sha256'] for x in images]]).encode()).hexdigest()
        if path.exists():
            if json.loads(path.read_text())['fingerprint']!=fingerprint:raise ValueError('Response fingerprint mismatch')
            continue
        for image in images:
            if sha_file(Path(image['path']))!=image['sha256']:raise ValueError('RGB changed')
        req=None
        try:
            raw=backend.ask(e['id'],'FRAME_CHECK_V2',prompt,[x['path'] for x in images],max_new_tokens=2304,image_captions=captions)
            req=Path(backend.requests[-1]);record=json.loads(req.read_text());parsed,errors,conv=normalize(raw,e)
            checks={}; supplied={p['frame_id'] for p in panels}
            for row in raw.get('mentions',[]):
                if not isinstance(row,dict):continue
                entries=row.get('frame_checks',[])
                if not isinstance(entries,list):entries=[]
                ids=[x.get('frame_id') for x in entries if isinstance(x,dict)]
                checks[row.get('mention_id')]={'entries':entries,'all_frames_named':len(ids)==len(supplied) and set(ids)==supplied}
            if raw.get('_parse_warning'):errors.append(raw['_parse_warning'])
            r={'status':'COMPLETE' if not errors else 'PARSE_WITH_WARNINGS','parsed':parsed,'warnings':errors,
               'frame_checks':checks,'conversions':conv,'raw_response':record['response'],'request_path':str(req),
               'processor_audit':record.get('processor_audit'),
               'vision_verified':bool((record.get('visual_tensor_trace') or {}).get('input_consumption_verified'))}
            if not r['vision_verified']:raise ValueError('Vision input verification failed')
        except Exception as exc:
            r={'status':'FAILED','parsed':{'mentions':[]},'error':f'{type(exc).__name__}: {exc}'}
            if backend.requests:req=Path(backend.requests[-1])
        finally:
            if req:clean_visual_trace(str(req))
        r.update(event=e['id'],fingerprint=fingerprint,prompt=prompt,input_images=images,image_captions=captions)
        save_json(path,r);print('FRAME_CHECK_V2',i,'/',len(m['events']),e['id'],r['status'],flush=True)


def report(out):
    verify(out);m=json.loads((out/'input_manifest.json').read_text());counts=Counter();paths=[];frames=Counter();detail=[]
    for e in m['events']:
        path=out/'responses'/f"{e['id']}.json"
        if not path.is_file():raise ValueError('Incomplete probe')
        paths.append(path);r=json.loads(path.read_text());counts[r['status']]+=1;counts['vision_verified']+=r.get('vision_verified',False)
        for mention in e['mentions']:
            counts['mentions']+=1;counts['all_frames_named']+=r.get('frame_checks',{}).get(mention['mention_id'],{}).get('all_frames_named',False)
        for row in r['parsed']['mentions']:
            for pred in row['predictions']:frames[pred['frame_id']]+=1
        detail.append({'event':e['id'],'status':r['status'],'checks':r.get('frame_checks',{}),'mentions':r['parsed']['mentions']})
    save_json(out/'predictions_frozen.json',{'responses':{str(p):sha_file(p) for p in paths}})
    save_json(out/'summary.json',{'counts':dict(counts),'frame_choices':dict(frames),'events':detail,
       'caveat':'Compliance with requested comparison fields does not establish the observations are correct.'})
    print('REPORTED_V2',dict(counts),dict(frames),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','infer','report']);p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    {'prepare':prepare,'infer':infer,'report':report}[a.stage](a.output.resolve())
if __name__=='__main__':main()
