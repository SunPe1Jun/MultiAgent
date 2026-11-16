#!/usr/bin/env python3
"""Full global visual grounding, then fixed GT-free scene-ID decoding."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
from MultiAgent.full_frame_grounding import VERSION,grounding_prompt,validate_grounding,decode_grounding,MAX_DISTANCE,MIN_MARGIN
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.qwen_backend import QwenBackend
from tools.run_full_interaction_referring import OUTPUT as PARENT,ROOT,read,evaluate as evaluate_final_sets

OUTPUT=PARENT/'grounding_v2'


def prepare(output):
    if (output/'manifest.json').exists():raise ValueError('Never overwrite frozen grounding inputs')
    parent=read(PARENT/'manifest.json');events=json.loads(json.dumps(parent['events']));requests=[]
    for event in events:
        event['prompt']=grounding_prompt(event)
        fingerprints={}
        for frame in event['frames']:
            for image in frame['images'].values():image['path']=os.path.relpath(PARENT/image['path'],output)
        for condition in parent['conditions']:
            hashes=[f['images'][condition]['sha256'] for f in event['frames']]
            fingerprint=hashlib.sha256(json.dumps([event['prompt'],hashes],sort_keys=True).encode()).hexdigest()
            rid=event['id']+'_'+condition
            canonical=fingerprints.setdefault(fingerprint,rid)
            requests.append({'id':rid,'event_id':event['id'],'condition':condition,
                'canonical_request_id':canonical,'model_input_fingerprint':fingerprint})
    names=['MultiAgent/full_frame_grounding.py','tools/run_full_frame_grounding.py',
           'MultiAgent/qwen_backend.py','MultiAgent/qwen_video_compat.py']
    manifest={**parent,'version':VERSION,'events':events,'requests':requests,
        'parent_manifest_sha256':sha(PARENT/'manifest.json'),'max_new_tokens':1000,
        'planned_unique_model_calls':len({r['canonical_request_id'] for r in requests}),
        'model_task':'Choose all referred visible objects from global RGB/video as category+normalized pixel points, no anchor IDs requested.',
        'decoder':{'all_type_compatible_scene_candidates':True,'uses_event_gt':False,
            'uses_gaze_to_choose_identity':False,'max_normalized_distance':MAX_DISTANCE,'minimum_runner_up_margin':MIN_MARGIN,
            'public_candidate_centers_projected_internally_only':True,'all_anchor_projection_audit_retained':False,
            'same_decoder_for_all_conditions':True,'geometry_unverified':True},
        'evaluation_policy':'Model visual choice followed by identical approximate scene geometry decoder; original final-set evaluation after all outputs frozen. Decoder failures reported separately.',
        'source_sha256':{n:sha(ROOT/n) for n in names},
        'adaptive_regression_only':True,
        'reason':'Direct canonical-ID reasoning produced nine completed empty-set outputs before expansion stopped. This followup removes arbitrary-ID reasoning from the model, retaining whole-frame visual referent selection.'}
    save(output/'manifest.json',manifest)
    save(output/'protocol_lock.json',{'manifest_sha256':sha(output/'manifest.json'),'before_any_inference':True,
        'frozen_utc':datetime.now(timezone.utc).isoformat(),'primary_metrics':['Exact-set','micro-F1'],
        'no_gt_annotation_reads_before_frozen_predictions':True})
    for name in names:
        dest=output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    print('FROZEN',len(events),'interactions',len(requests),'condition results',manifest['planned_unique_model_calls'],'unique model calls',flush=True)


def infer(output,limit=None):
    m=read(output/'manifest.json');assert sha(output/'manifest.json')==read(output/'protocol_lock.json')['manifest_sha256']
    backend=QwenBackend(output/'model_cache',model=m['model'],local_only=True,namespace=VERSION,
        trace_visual_inputs=True,video_max_pixels=m['video_max_pixels'])
    events={e['id']:e for e in m['events']}
    canonical={r['id']:r for r in m['requests'] if r['id']==r['canonical_request_id']}
    aliases={rid:[r for r in m['requests'] if r['canonical_request_id']==rid] for rid in canonical}
    schedule=list(canonical.values());random.Random(20261106).shuffle(schedule);completed=0
    for req in schedule:
        if all((output/'responses'/f"{r['id']}.json").exists() for r in aliases[req['id']]):continue
        event=events[req['event_id']];condition=req['condition'];print('INFER',req['id'],flush=True);start=time.monotonic()
        paths=[output/f['images'][condition]['path'] for f in event['frames']]
        for path,frame in zip(paths,event['frames']):assert sha(path)==frame['images'][condition]['sha256']
        metadata=[{k:f[k] for k in ('frame_id','source_frame_id','source_frame_index','time_seconds')} for f in event['frames']]
        parsed=backend.ask_video(req['id'],'global_visual_grounding',event['prompt'],paths,metadata,
            max_new_tokens=m['max_new_tokens'],video_path=event['video_path'],
            video_time_range=[event['frames'][0]['time_seconds'],event['frames'][-1]['time_seconds']])
        raw=read(backend.requests[-1]);audit=raw['processor_audit'];checked=validate_grounding(parsed,event)
        decoded=decode_grounding(checked['referents'],event)
        validation={'schema_valid':checked['schema_valid'],'errors':checked['errors'],
            'selected_object_names':decoded['selected_object_names'] if checked['schema_valid'] else []}
        for alias in aliases[req['id']]:
            save(output/'responses'/f"{alias['id']}.json",{**alias,'event':event['event'],
                'scene_key':event['scene_key'],'row_index':event['row_index'],'parsed':parsed,
                'grounding_validation':checked,'validation':validation,'decoder':decoded,
                'request_path':backend.requests[-1],'processor_audit':audit,
                'vision_verified':audit['visual_tensor_trace']['input_consumption_verified'],
                'seconds':time.monotonic()-start,'exact_input_reused':alias['id']!=req['id']})
        print('RESULT',req['id'],'schema',checked['schema_valid'],'visual_objects',len(checked['referents']),
            'decoded',validation['selected_object_names'],'seconds',round(time.monotonic()-start,2),flush=True)
        save(output/'progress.json',{'completed_condition_results':len(list((output/'responses').glob('*.json'))),
            'planned_condition_results':len(m['requests']),'planned_unique_model_calls':len(canonical),
            'last':req['id'],'gt_loaded':False})
        completed+=1
        if limit and completed>=limit:break


def evaluate(output):
    evaluate_final_sets(output)
    m=read(output/'manifest.json');analysis=read(output/'analysis.json')
    results=[read(output/'responses'/f"{r['id']}.json") for r in m['requests']]
    analysis['completed_condition_results']=len(results)
    analysis['completed_model_calls']=len({r['request_path'] for r in results})
    analysis['model_task']='Full-field visual referent localization; fixed GT-free public-scene anchor decoder.'
    analysis['decoder_limits']='Approximate anchor-center association with unknown visibility/dynamic identity; final ID score measures model+decoder, not pure model canonical-ID reasoning.'
    analysis['grounding_and_decoder']={}
    from collections import Counter
    for condition in m['conditions']:
        rr=[r for r in results if r['condition']==condition]
        analysis['grounding_and_decoder'][condition]={
            'interactions':len(rr),'raw_visual_referents':sum(len(r['parsed'].get('referents',[])) for r in rr if isinstance(r['parsed'].get('referents'),list)),
            'valid_visual_referents':sum(len(r['grounding_validation']['referents']) for r in rr),
            'unresolved_mentions':sum(len(r['parsed'].get('unresolved',[])) for r in rr if isinstance(r['parsed'].get('unresolved'),list)),
            'decoded_referent_rows':sum(sum(d['selected_candidate'] is not None for d in r['decoder']['decisions']) for r in rr),
            'decoder_reasons':dict(Counter(d['reason'] for r in rr for d in r['decoder']['decisions'])),
            'rejected_visual_rows':sum(len(r['grounding_validation'].get('rejected_referents',[])) for r in rr),
            'rejected_unresolved_rows':sum(len(r['grounding_validation'].get('rejected_unresolved',[])) for r in rr),
            'nonempty_visual_interactions':sum(bool(r['grounding_validation']['referents']) for r in rr),
            'nonempty_decoded_interactions':sum(bool(r['validation']['selected_object_names']) for r in rr)}
    save(output/'analysis.json',analysis)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','infer','evaluate']);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--limit',type=int)
    a=p.parse_args();{'prepare':prepare,'infer':lambda o:infer(o,a.limit),'evaluate':evaluate}[a.action](a.output.resolve())
