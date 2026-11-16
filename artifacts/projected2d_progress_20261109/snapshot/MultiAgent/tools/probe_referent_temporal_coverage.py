#!/usr/bin/env python3
"""Probe 2D localization with mention-specific temporal RGB evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from MultiAgent.projected2d_attention_study import save_json, sha_file
from MultiAgent.projected2d_routed_rgb import MODEL, ROOT
from MultiAgent.projected2d_main_image import normalize

PRIOR = ROOT / '.local-sync/referent_rgb_expansion32_20261109_repaired/manifest.json'
SOURCE = ROOT / '.local-sync/projected2d_main_rgb32_20261109/input_manifest.json'
OUTPUT = ROOT / '.local-sync/referent_temporal_coverage_probe_20261109'


def prompt_for(event):
    task = event['task']
    return '''Locate ONE instruction referent in the supplied original interaction frames.
Each image is a full RGB frame identified by the text immediately before it, including its real
source frame ID and video time. The complete instruction is provided to resolve meaning and
relationships. Focus on the exact current expression and its action context. The time range is a
rough clause-order hypothesis, not a measured speech time; it is not a label and can be wrong.
Compare all supplied images for the correct object, visibility, interaction and identity. You may
choose any of them, or return ambiguous/not_visible if the visual evidence is insufficient. Other
referents are selected independently, and several referents may legitimately select the same frame.
For a component locate the indicated whole parent. Select a real source frame ID and point_xy=[x,y]
in the entire RGB image, on a 0..1000 scale from top-left. Do not infer target identity from order.
Return only JSON with one row:
{"mentions":[{"mention_id":"...","status":"localized|ambiguous|not_visible",
"predictions":[{"frame_id":"actual_supplied_source_frame_id","point_xy":[x,y]}],"evidence":"..."}]}.
''' + json.dumps({'instruction': event['instruction'], 'current_mention': task['mention'],
                  'action_context': task['action_context'], 'soft_phase_range': task['soft_phase_range'],
                  'source_frames': [{'frame_id': x['frame_id'], 'video_time_seconds': x['time_seconds']}
                                    for x in task['frames']]}, ensure_ascii=False)


def verify(out):
    lock = json.loads((out/'protocol_lock.json').read_text())
    for name, digest in lock['files'].items():
        if sha_file(Path(name)) != digest:
            raise ValueError('Study input changed: ' + name)


def prepare(out):
    if (out/'input_manifest.json').exists():
        raise ValueError('Refuse to overwrite frozen study')
    main = json.loads(SOURCE.read_text())
    prior = {e['id']: e for e in json.loads(PRIOR.read_text())['events']}
    events = []
    for event in main['events']:
        if event['id'] not in {'scene2_482', 'scene2_412'}:
            continue
        old = prior[event['id']]
        for mention in event['mentions']:
            spec = old['temporal_search']['mentions'][mention['mention_id']]
            ids = set(spec['primary_frame_ids'] + spec['context_frame_ids'])
            frames = [f for f in old['frames'] if f['frame_id'] in ids]
            if not frames:
                raise ValueError('Mention has no existing full-event frames')
            for frame in frames:
                if sha_file(Path(frame['source_path'])) != frame['source_sha256']:
                    raise ValueError('Existing RGB frame changed')
            # Preserve separate original RGB image files, only send target-free time metadata.
            events.append({'id': event['id'], 'scene_key': event['scene_key'], 'row_index': event['row_index'],
                'event_id': event['event_id'], 'instruction': event['instruction'],
                'task': {'mention': mention, 'action_context': spec['action_context'],
                    'soft_phase_range': spec['primary_range'],
                    'frames': [{'frame_id': f['frame_id'], 'time_seconds': f['time_seconds'],
                                'source_path': f['source_path'], 'source_sha256': f['source_sha256']}
                               for f in frames]}})
    if len(events) != 10:
        raise ValueError('Expected ten mention occurrences in two events')
    out.mkdir(parents=True, exist_ok=True)
    m = {'version': 'referent_temporal_coverage_probe_v1', 'events': events,
        'selection': 'scene2_412 and scene2_482 chosen for multi-clause phase coverage; old development samples',
        'evaluation_scope': 'Existing original 2D evaluator only scores P1/P2/P3. Earlier V-frame outputs cannot be given official numerical scores without independent 2D labels.',
        'frame_selection': 'Previously frozen temporal_focus primary/context frame IDs; no GT, new model or new frame proposals.',
        'source_768_note': 'Existing full-event frames are 672x672 original-RGB-derived JPEGs, not the 768 original-panel images.',
        'annotation_used': False}
    save_json(out/'input_manifest.json', m)
    deps = [Path(__file__).resolve(), out/'input_manifest.json', SOURCE, PRIOR,
            ROOT/'MultiAgent/qwen_backend.py', ROOT/'MultiAgent/projected2d_main_image.py',
            ROOT/'tools/run_full_frame_grounding_v4.py', MODEL/'config.json']
    save_json(out/'protocol_lock.json', {'files': {str(p): sha_file(p) for p in deps}})
    print('FROZEN', len(events), 'mention calls,', sum(len(e['task']['frames']) for e in events), 'image instances', flush=True)


def infer(out):
    from MultiAgent.qwen_backend import QwenBackend
    from tools.run_full_frame_grounding_v4 import clean_visual_trace
    verify(out);m=json.loads((out/'input_manifest.json').read_text())
    backend=QwenBackend(out/'model_cache',str(MODEL),local_only=True,namespace='referent_temporal_coverage_v1',trace_visual_inputs=True)
    for i,event in enumerate(m['events'],1):
        task=event['task'];mid=task['mention']['mention_id'];target=out/'responses'/f"{event['id']}_{mid}.json"
        images=task['frames'];captions=[f"Frame {f['frame_id']} | video_time={f['time_seconds']:.3f} seconds" for f in images]
        prompt=prompt_for(event);fingerprint=hashlib.sha256(json.dumps([prompt,captions,[f['source_sha256'] for f in images]]).encode()).hexdigest()
        if target.exists():
            if json.loads(target.read_text())['fingerprint']!=fingerprint:raise ValueError('Response fingerprint mismatch')
            continue
        for f in images:
            if sha_file(Path(f['source_path']))!=f['source_sha256']:raise ValueError('Input RGB changed')
        req=None
        try:
            raw=backend.ask(event['id']+'/'+mid,'MENTION_TIME_RGB',prompt,[f['source_path'] for f in images],
                            max_new_tokens=1024,image_captions=captions)
            req=Path(backend.requests[-1]);request=json.loads(req.read_text())
            normalize_event={'mentions':[task['mention']],'panels':[{'frame_id':f['frame_id']} for f in images]}
            parsed,errors,coordinates=normalize(raw,normalize_event)
            if raw.get('_parse_warning'):errors.append(raw['_parse_warning'])
            response={'status':'COMPLETE' if not errors else 'PARSE_WITH_WARNINGS','parsed':parsed,'warnings':errors,
                'coordinates':coordinates,'raw_response':request['response'],'request_path':str(req),
                'vision_verified':bool((request.get('visual_tensor_trace') or {}).get('input_consumption_verified')),
                'processor_audit':request.get('processor_audit')}
            if not response['vision_verified']:raise ValueError('Vision input was not verified')
        except Exception as exc:
            response={'status':'FAILED','parsed':{'mentions':[]},'error':f'{type(exc).__name__}: {exc}'}
            if backend.requests:req=Path(backend.requests[-1])
        finally:
            if req:clean_visual_trace(str(req))
        response.update(event=event['id'],mention_id=mid,fingerprint=fingerprint,prompt=prompt,
                        frames=images,image_captions=captions)
        save_json(target,response)
        print('TEMPORAL_PROBE',i,'/',len(m['events']),event['id'],mid,response['status'],flush=True)


def report(out):
    verify(out);m=json.loads((out/'input_manifest.json').read_text());paths=[];rows=[]
    for e in m['events']:
        mid=e['task']['mention']['mention_id'];p=out/'responses'/f"{e['id']}_{mid}.json"
        if not p.is_file():raise ValueError('Missing response')
        paths.append(p);r=json.loads(p.read_text());pred=r['parsed'].get('mentions',[])
        vals=[x for row in pred for x in row['predictions']]
        lo,hi=e['task']['soft_phase_range'];times={x['frame_id']:x['time_seconds'] for x in e['task']['frames']}
        rows.append({'event':e['id'],'mention_id':mid,'mention':e['task']['mention']['text'],
          'action_context':e['task']['action_context'],'soft_phase_range':[lo,hi],
          'supplied_frames':[{'frame_id':f['frame_id'],'time_seconds':f['time_seconds']} for f in e['task']['frames']],
          'status':r['status'],'predictions':[dict(p,time_seconds=times[p['frame_id']],
                  within_soft_phase=lo<=times[p['frame_id']]<=hi) for p in vals],
          'evidence':pred[0]['evidence'] if pred else '', 'vision_verified':r.get('vision_verified',False)})
    save_json(out/'predictions_frozen.json',{'responses':{str(p):sha_file(p) for p in paths}})
    save_json(out/'summary.json',{'mentions':rows,'mentions_count':len(rows),
      'localized':sum(bool(r['predictions']) for r in rows),'vision_verified':sum(r['vision_verified'] for r in rows),
      'interpretation':'Frame coverage is verifiable, but phase hypotheses are not speech timestamps. No matched 2D GT available for these V frames, so choices are not claimed correct.'})
    print('REPORTED',len(rows),'mentions',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','infer','report']);p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    {'prepare':prepare,'infer':infer,'report':report}[a.stage](a.output.resolve())


if __name__=='__main__':main()
