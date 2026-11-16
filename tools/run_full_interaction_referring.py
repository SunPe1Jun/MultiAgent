#!/usr/bin/env python3
"""Frozen full-interaction referent selection with post-inference original evaluation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from MultiAgent.attention_volume import build_volume
from MultiAgent.baseline import module, scene_paths
from MultiAgent.camera_geometry import aligned_frame_geometry
from MultiAgent.context import load_contexts
from MultiAgent.full_interaction_referring import (
    VERSION, CONDITIONS, referring_prompt, render_full_frame, validate_prediction, paired_set_metrics)
from MultiAgent.gaze_evidence import gaze_records
from MultiAgent.heatmap_temporal_pilot import save, sha
from MultiAgent.local_video_visual import qwen_temporal_groups
from MultiAgent.qwen_backend import QwenBackend

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.local-sync/projection_expansion24_20261105'
OUTPUT = ROOT / '.local-sync/full_interaction_referring_20261106'
MODEL = '/ai/data/Qwen3-VL-8B-Instruct'
VIDEO_MAX_PIXELS = 23_330_816


def read(path): return json.loads(Path(path).read_text())


def camera_for_prompt(geometry):
    if not geometry['clock_available']: return None
    camera = geometry['reference_pose']
    rotation = Rotation.from_quat(camera['rotation_xyzw'])
    rounded = lambda x: [round(float(v), 4) for v in x]
    return {'position_world': rounded(camera['position_world']),
            'forward_world': rounded(rotation.apply([0, 0, 1])),
            'up_world': rounded(rotation.apply([0, 1, 0])),
            'vertical_fov_degrees': round(float(camera['fov_vertical_degrees']), 3)}


def prepare(output):
    if (output/'manifest.json').exists(): raise ValueError('Preserve frozen inputs; never overwrite manifest')
    output.mkdir(parents=True, exist_ok=True)
    selection = read(SOURCE/'selection.json')
    # Strip previous study tags: only scene/row/event addresses select contexts.
    samples = [{k:s[k] for k in ('scene_key','row_index','event_id')} for s in selection['samples']]
    contexts = load_contexts({'samples':samples}, load_ground_truth=False)
    events, requests = [], []
    for ctx in contexts:
        assert ctx.provenance['ground_truth_loaded'] is False
        key=f'{ctx.scene_key}/{ctx.row_index}';event_id=f'{ctx.scene_key}_{ctx.row_index}'
        pose_path=SOURCE/'pose_alignment'/ctx.scene_key/str(ctx.row_index)/'frame_pose_contract.json'
        contract=read(pose_path)
        candidate_table=[{'candidate_id':c.candidate_id,'aliases':list(c.aliases),
                          'position_world':[c.x_world,c.y_world,c.z_world]} for c in ctx.candidates]
        # Stable nonsemantic order, identical across conditions and all frames.
        random.Random('full-scene-candidates:'+ctx.scene_key).shuffle(candidate_table)
        frames=[];directory=output/'events'/ctx.scene_key/str(ctx.row_index)
        for source in contract['frames']:
            path=Path(source['source_path']);assert sha(path)==source['source_sha256']
            geometry=aligned_frame_geometry(source,sha(path),source['frame_time']['actual_container_pts_seconds'],0.)
            with Image.open(path) as im:rgb=im.convert('RGB')
            maps,stats=build_volume(gaze_records(ctx),query=geometry['telemetry_time_seconds'],
                pose=geometry['reference_pose'],image_size=rgb.size,
                clock_supported=geometry['clock_available'],history=2.,bins=3,sigma_degrees=2.)
            frame_dir=directory/source['frame_id'];frame_dir.mkdir(parents=True,exist_ok=True)
            array_path=frame_dir/'attention.npz';np.savez_compressed(array_path,maps=maps)
            images={}
            for condition in CONDITIONS:
                image_path=frame_dir/f'{condition}.png'
                image=render_full_frame(rgb,maps,condition);image.save(image_path)
                images[condition]={'path':str(image_path.relative_to(output)),'sha256':sha(image_path),'size':list(image.size)}
            frames.append({'frame_id':source['frame_id'],'source_frame_id':f"frame_{source['frame_time']['source_frame_index']:06d}",
                'source_frame_index':source['frame_time']['source_frame_index'],
                'time_seconds':geometry['video_time_seconds'],'telemetry_time_seconds':geometry['telemetry_time_seconds'],
                'source_path':str(path),'source_sha256':sha(path),'source_size':list(rgb.size),
                'geometry':geometry,'prompt_camera':camera_for_prompt(geometry),
                'maps_path':str(array_path.relative_to(output)),'maps_sha256':sha(array_path),
                'attention_stats':stats,'images':images})
        assert len(frames)==12 and all(a['time_seconds']<b['time_seconds'] for a,b in zip(frames,frames[1:]))
        prompt=referring_prompt(ctx.text,candidate_table,frames)
        event={'id':event_id,'event':key,'scene_key':ctx.scene_key,'row_index':ctx.row_index,
            'event_id':ctx.event_id,'instruction':ctx.text,'scene_candidate_table':candidate_table,
            'video_path':ctx.video_path,'video_sha256':sha(ctx.video_path),
            'telemetry_path':ctx.json_path,'telemetry_sha256':sha(ctx.json_path),
            'anchor_table':str(scene_paths(ctx.scene_key)['anchor']),'anchor_table_sha256':sha(scene_paths(ctx.scene_key)['anchor']),
            'pose_contract_path':str(pose_path),'pose_contract_sha256':sha(pose_path),
            'clock_available_frames':sum(f['geometry']['clock_available'] for f in frames),
            'clock_status':contract['clock'],'frames':frames,
            'prompt':prompt,'qwen_temporal_groups':qwen_temporal_groups(frames),
            'gt_loaded':False,'anchors_projected':False,'object_crops':False,'answer_conditioned_candidates':False}
        events.append(event)
        for condition in CONDITIONS:
            requests.append({'id':event_id+'_'+condition,'event_id':event_id,'condition':condition})
        print('PREPARED',key,'full_frames',len(frames),'heat_supported_frames',event['clock_available_frames'],flush=True)
    source_files=['MultiAgent/full_interaction_referring.py','tools/run_full_interaction_referring.py']
    manifest={'version':VERSION,'events':events,'requests':requests,'conditions':list(CONDITIONS),
        'selection_source':str(SOURCE/'selection.json'),'selection_source_sha256':sha(SOURCE/'selection.json'),
        'sample_selection':'All 24 preexisting fixed interactions, no filtering on GT, model outputs, heat support or object proposals. Old regression cohort; prior source manifest includes historical selection tags.',
        'model':MODEL,'max_new_tokens':900,'video_max_pixels':VIDEO_MAX_PIXELS,
        'layout':'One unmodified full RGB panel plus three registered full-field attention panels, no object crops or projected anchors.',
        'controls':'Actual gaze, no added gaze, horizontal reflection of actual heat, reverse three within-frame age bins; common RGB/pose/candidates/prompts/budget.',
        'gt_loaded':False,'anchors_projected':False,'default_pipeline_changed':False,
        'unseen_validation':False,'evaluation_policy':'Only after all model responses frozen: original scene referent GT, original evaluator; primary Exact-set and micro-F1, helpful/harmful paired interactions.',
        'known_limits':['RGB has recorder marks in every condition; no-added-heat is not gaze-free.',
            'The 12 sampled complete frames cover each existing interaction window, not every decoded video frame.',
            'Source RGB is already 672-square resized by parent extraction; no original-pixel losslessness claim.',
            'Numeric common camera pose and full scene world candidate table support ID mapping; no gaze numbers in prompt.',
            'Fixed source PTS are paired by Qwen into six temporal patches; not exact spoken mention alignment.',
            'Current-frame history projection is approximate with unknown occlusion/dynamic correspondence.'],
        'source_sha256':{f:sha(ROOT/f) for f in source_files}}
    save(output/'manifest.json',manifest)
    save(output/'protocol_lock.json',{'manifest_sha256':sha(output/'manifest.json'),
        'before_any_inference':True,'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'primary_metrics':['Exact-set','micro-F1'],'gt_read_policy':'separate evaluate action only after all predictions frozen'})
    for name in source_files:
        dest=output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    print('FROZEN',len(events),'interactions',len(requests),'model requests',flush=True)


def infer(output,limit=None):
    m=read(output/'manifest.json');assert sha(output/'manifest.json')==read(output/'protocol_lock.json')['manifest_sha256']
    backend=QwenBackend(output/'model_cache',model=m['model'],local_only=True,
        namespace=VERSION,trace_visual_inputs=True,video_max_pixels=m['video_max_pixels'])
    events={e['id']:e for e in m['events']};schedule=m['requests'][:];random.Random(20261106).shuffle(schedule)
    completed=0
    for req in schedule:
        dest=output/'responses'/f"{req['id']}.json"
        if dest.exists():continue
        e=events[req['event_id']];condition=req['condition']
        paths=[output/f['images'][condition]['path'] for f in e['frames']]
        for path,f in zip(paths,e['frames']):assert sha(path)==f['images'][condition]['sha256']
        metadata=[{k:f[k] for k in ('frame_id','source_frame_id','source_frame_index','time_seconds')} for f in e['frames']]
        start=time.monotonic();print('INFER',req['id'],flush=True)
        parsed=backend.ask_video(req['id'],'full_referent_set',e['prompt'],paths,metadata,
            max_new_tokens=m['max_new_tokens'],video_path=e['video_path'],
            video_time_range=[e['frames'][0]['time_seconds'],e['frames'][-1]['time_seconds']])
        raw=read(backend.requests[-1]);audit=raw['processor_audit']
        check=validate_prediction(parsed,[c['candidate_id'] for c in e['scene_candidate_table']],
            {g['view_id'] for g in e['qwen_temporal_groups']},e['instruction'])
        save(dest,{**req,'event':e['event'],'scene_key':e['scene_key'],'row_index':e['row_index'],
            'parsed':parsed,'validation':check,'request_path':backend.requests[-1],
            'processor_audit':audit,'vision_verified':audit['visual_tensor_trace']['input_consumption_verified'],
            'seconds':time.monotonic()-start})
        print('RESULT',req['id'],check['schema_valid'],check['selected_object_names'],
              'tokens',audit['expanded_video_token_count'],'seconds',round(time.monotonic()-start,2),flush=True)
        completed+=1
        save(output/'progress.json',{'completed':len(list((output/'responses').glob('*.json'))),
            'planned':len(m['requests']),'last':req['id'],'gt_loaded':False})
        if limit and completed>=limit:break


def evaluate(output):
    m=read(output/'manifest.json');assert sha(output/'manifest.json')==read(output/'protocol_lock.json')['manifest_sha256']
    predictions={r['id']:read(output/'responses'/f"{r['id']}.json") for r in m['requests']}
    # Freeze all responses BEFORE importing/reading scene GT. The evaluator is
    # the first code in this tool which opens annotation files.
    save(output/'predictions_frozen.json',{'manifest_sha256':sha(output/'manifest.json'),
        'before_gt_evaluation':True,'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'response_sha256':{k:sha(output/'responses'/f'{k}.json') for k in predictions}})
    evaluator=module('eval');gt_by_scene={};all_gt={}
    for scene in sorted({e['scene_key'] for e in m['events']}):
        aliases,_=evaluator.load_anchor_alias_map(scene_paths(scene)['anchor'])
        gt_by_scene[scene]=(aliases,evaluator.load_gt_rows(scene_paths(scene)['gt'],aliases))
    result={'completed_model_calls':len(predictions),'conditions':{},'events':[],
        'source_manifest_sha256':sha(output/'manifest.json'),'gt_loaded_only_after_frozen_predictions':True,
        'primary_scope':'Final referent sets on 24 old fixed interactions. No GT-conditioned candidate filtering or crops.',
        'unseen_validation':False,'all_schema_valid':all(r['validation']['schema_valid'] for r in predictions.values()),
        'all_vision_verified':all(r['vision_verified'] for r in predictions.values()),
        'expanded_video_tokens':sorted({r['processor_audit']['expanded_video_token_count'] for r in predictions.values()}),
        'processor_effective_frame_sizes':sorted({tuple(r['processor_audit']['processor_effective_frame_size_px']) for r in predictions.values()})}
    for condition in CONDITIONS:
        records=[];original_rows=[];original_gt=[]
        for event in m['events']:
            r=predictions[event['id']+'_'+condition]
            pred=r['validation']['selected_object_names']
            dest=output/'predictions'/condition/event['scene_key']/f"row_{event['row_index']}.json"
            save(dest,{'row_index':event['row_index'],'event_id':event['event_id'],
                'parsed_response':{'selected_object_names':pred},
                'response_status':'ok' if r['validation']['schema_valid'] else 'schema_failure',
                'used_storyboard_fallback':False})
            aliases,gt_rows=gt_by_scene[event['scene_key']];gt=gt_rows[event['row_index']]
            all_gt[event['id']]=gt['mapped_referents']
            records.append({'id':event['id'],'gt':gt['mapped_referents'],'prediction':pred})
            original_rows.extend(evaluator.build_eval_rows([dest],gt_rows,aliases));original_gt.append(gt)
        summary=evaluator.summarize(original_rows,original_gt)
        save(output/'evaluation'/condition/'original_summary.json',summary)
        evaluator.write_eval_csv(output/'evaluation'/condition/'evaluation.csv',original_rows)
        result['conditions'][condition]={'set_metrics':paired_set_metrics([r for r in records if r['gt']]),
            'original_evaluator':summary,'schema_failures':sum(not predictions[e['id']+'_'+condition]['validation']['schema_valid'] for e in m['events'])}
    for event in m['events']:
        gt=set(all_gt[event['id']]);choices={c:predictions[event['id']+'_'+c]['validation']['selected_object_names'] for c in CONDITIONS}
        metrics={c:{'exact':set(p)==gt,'tp':len(set(p)&gt),'fp':len(set(p)-gt),'fn':len(gt-set(p)),
                    'f1':2*len(set(p)&gt)/(len(p)+len(gt)) if len(p)+len(gt) else 0.} for c,p in choices.items()}
        result['events'].append({'id':event['id'],'event':event['event'],'instruction':event['instruction'],
            'gt':sorted(gt),'predictions':choices,'metrics':metrics,'heat_supported_frames':event['clock_available_frames']})
    from MultiAgent.attention_reliability import cluster_paired_interval
    contrasts={}
    for control in ('NO_ADDED_HEAT','SPATIAL_MIRROR','TIME_REVERSE'):
        rows=result['events']
        contrasts[control]={metric:{**cluster_paired_interval([{'cluster':r['id'],'baseline':float(r['metrics'][control][metric]),
                'treatment':float(r['metrics']['TRUE_HEAT'][metric])} for r in rows]),
            'helpful_interactions':sum(r['metrics']['TRUE_HEAT'][metric]>r['metrics'][control][metric] for r in rows),
            'harmful_interactions':sum(r['metrics']['TRUE_HEAT'][metric]<r['metrics'][control][metric] for r in rows)} for metric in ('exact','f1')}
    result['paired_contrasts_true_heat']=contrasts
    result['support_strata']={}
    for label,subset in [('some_supported_heat',[r for r in result['events'] if r['heat_supported_frames']>0]),
                         ('rgb_fallback_only',[r for r in result['events'] if r['heat_supported_frames']==0])]:
        result['support_strata'][label]={c:paired_set_metrics([{'gt':r['gt'],'prediction':r['predictions'][c]} for r in subset]) for c in CONDITIONS}
    save(output/'analysis.json',result)
    print(json.dumps({c:d['set_metrics'] for c,d in result['conditions'].items()},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','infer','evaluate'])
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--limit',type=int)
    args=parser.parse_args();{'prepare':prepare,'infer':lambda p:infer(p,args.limit),'evaluate':evaluate}[args.action](args.output.resolve())
