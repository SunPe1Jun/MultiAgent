#!/usr/bin/env python3
"""Controlled visual iteration around the original Experiment 2 wrapper."""
from __future__ import annotations
import argparse
from argparse import Namespace
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from MultiAgent.projected2d_attention_study import CSV_COLUMNS, load_module, read_csv, save_json, sha_file, write_csv
from MultiAgent.projected2d_exact_set import exact_counts

ROOT=Path(__file__).resolve().parents[1]
TRI=ROOT.parent/'TriModal-Referring'
MODEL=ROOT.parent/'Qwen3-VL-8B-Instruct'
BASE=ROOT/'.local-sync/2d_original_wrapper_recovery_20261109'
FOCUS=ROOT/'.local-sync/referent_temporal_prompt_v3_20261109/input_manifest.json'
MAPS=ROOT/'.local-sync/2d_heatmap_foveation_recovery_20261109'
OUTPUT=ROOT/'.local-sync/2d_original_visual_iteration_20261109'
CONFIGS={
 'FOV_WIDE': {'kind':'paired','foveation':True,'crop':'original','floor':.80,'dilation':12.},
 'HEAT_CROP_FOV': {'kind':'paired','foveation':True,'crop':'heat','floor':.80,'dilation':12.},
 'HEAT_CROP': {'kind':'paired','foveation':False,'crop':'heat','floor':1.,'dilation':12.},
 'SINGLE_HD': {'kind':'full','foveation':False,'crop':'none','floor':1.,'dilation':12.,'edge':1120},
 'SINGLE_HD_FOV': {'kind':'full','foveation':True,'crop':'none','floor':.80,'dilation':12.,'edge':1120},
}


def visual():
 return load_module('original2d_visual_iteration',TRI/'exam2/run_qwen3vl_2d_point_grounding.py')


def check(out):
 d=json.loads((out/'protocol_lock.json').read_text())
 for p,h in d['files'].items():
  if sha_file(Path(p))!=h:raise ValueError('Frozen dependency changed: '+p)
 return d


def heat_bbox(meta,maps,source_size):
 """Gaze-only recent-support envelope with current point retained; no target inputs."""
 width,height=source_size;density=maps[-1].astype(np.float64)
 if density.sum()<.1:return None
 # Use recent density mass; 90% envelope has a safety margin for gaze noise.
 yy,xx=np.indices(density.shape);order=np.argsort(density.ravel())[::-1]
 mass=np.cumsum(density.ravel()[order]);idx=min(len(order)-1,int(np.searchsorted(mass,.9*mass[-1])))
 selected=density>=density.ravel()[order[idx]]
 sy,sx=np.nonzero(selected)
 if not len(sx):return None
 cx=float((density*xx).sum()/density.sum())/density.shape[1]
 cy=float((density*yy).sum()/density.sum())/density.shape[0]
 u=float(meta.get('gaze_u_norm') or cx);v=float(meta.get('gaze_v_norm') or cy)
 xlo=min(sx.min()/density.shape[1],u);xhi=max((sx.max()+1)/density.shape[1],u)
 ylo=min(sy.min()/density.shape[0],v);yhi=max((sy.max()+1)/density.shape[0],v)
 side=float(np.clip(max(xhi-xlo,yhi-ylo)*1.35,.22,.45))*min(width,height)
 side=max(32,min(min(width,height),round(side)))
 # Emphasize measured current gaze, while retaining recent local gaze uncertainty.
 cx=(.5*cx+.5*u)*width;cy=(.5*cy+.5*v)*height
 left=max(0,min(width-side,round(cx-side/2)));top=max(0,min(height-side,round(cy-side/2)))
 return [left,top,left+side,top+side]


def prepare(out,condition,scope):
 cfg=CONFIGS[condition];v=visual()
 target=out/condition
 if (target/'input_manifest.json').exists():raise ValueError('Refuse overwrite frozen condition')
 focus=json.loads(FOCUS.read_text());ids={e['id'] for e in focus['events']}
 old=json.loads((ROOT/'.local-sync/projected2d_main_rgb32_20261109/input_manifest.json').read_text())
 selected=[e for e in old['events'] if (scope=='all' or (e['id'] in ids if scope=='dev' else e['id'] not in ids))]
 allrows=read_csv(BASE/'manifest.csv');keys={(e['scene_key'],e['row_index']) for e in selected}
 rows=[r for r in allrows if (r['scene'],int(r['row_index'])) in keys]
 target.mkdir(parents=True,exist_ok=True);write_csv(target/'manifest.csv',rows,list(rows[0]))
 grouped=v.group_manifest(rows);events=[];audit=[]
 oldmap={ (p['event'],p['frame_id']):p for p in json.loads((MAPS/'input_image_audit.json').read_text())['panels']}
 for e in selected:
  key=(e['scene_key'],e['row_index']);raw=BASE/'responses'/f"{key[0]}_row_{key[1]}.json"
  if not raw.exists():raise ValueError('Original baseline response not complete: '+str(raw))
  baseline=json.loads(raw.read_text());metas=[];images=[]
  for source in baseline['panel_meta']:
   fid=source['panel_id'];src=Path(source['source_frame_path']);meta=dict(source)
   img=Image.open(src).convert('RGB');mapspec=oldmap.get((e['id'],fid))
   if mapspec is None:raise ValueError('Missing gaze heatmap for '+e['id']+'/'+fid)
   maps=np.load(mapspec['heatmap_path'])['maps'];supported=bool(mapspec['clock_supported'])
   folder=target/'inputs'/e['id']/fid;folder.mkdir(parents=True,exist_ok=True)
   result=img;fovmeta={'status':'UNMODIFIED_RGB'}
   if cfg['kind']=='full':
    edge=cfg['edge'];result=img.copy();result.thumbnail((edge,edge),Image.Resampling.LANCZOS)
   if cfg['foveation']:
    from MultiAgent.foveation_attention import foveate_from_history
    # Resize density conservatively to the display resolution, preserving duration.
    layers=[]
    for plane in maps:
     resized=np.array(Image.fromarray(plane).resize(result.size,Image.Resampling.BILINEAR),dtype=np.float32,copy=True)
     total=float(plane.sum());resized*=total/max(float(resized.sum()),1e-20);layers.append(resized)
    result,fovmeta=foveate_from_history(result,np.stack(layers),clock_supported=supported,
     vertical_fov_degrees=mapspec['fov_degrees'],min_rgb_weight=cfg['floor'],dilation_degrees=cfg['dilation'])
   if cfg['kind']=='full':
    dest=folder/'full.png';result.save(dest)
    meta.update(image_role='full',frame_path=str(dest));meta.pop('crop_bbox_norm',None)
   else:
    # Always form local details from the original source, never a 512px intermediary.
    bbox=heat_bbox(source,maps,img.size) if cfg['crop']=='heat' and supported else None
    if bbox is None:
     crop=Path(source['crop_path']);bboxnorm=v.parse_bbox_norm(source['crop_bbox_norm'])
    else:
     crop=folder/'crop.jpg';img.crop(bbox).resize((768,768),Image.Resampling.LANCZOS).save(crop,quality=92)
     bboxnorm=(bbox[0]/img.width,bbox[1]/img.height,bbox[2]/img.width,bbox[3]/img.height)
    fullpath=folder/'full.png';result.save(fullpath)
    dest=folder/'paired.jpg'
    v.create_paired_full_crop_image(fullpath,crop,dest,fid,bboxnorm,768,True)
    meta.update(frame_path=str(dest),crop_path=str(crop),crop_bbox_norm=','.join(f'{z:.4f}' for z in bboxnorm))
   metas.append(meta);images.append({'path':str(dest),'sha256':sha_file(dest)})
   audit.append({'event':e['id'],'frame_id':fid,'foveation':fovmeta,'heatmap':mapspec['heatmap_path'],
    'heatmap_sha256':sha_file(Path(mapspec['heatmap_path'])),'crop_bbox_norm':meta.get('crop_bbox_norm'),
    'source_sha256':sha_file(src),'source_size':list(img.size),'input':str(dest)})
   img.close()
  prompt=v.build_prompt(grouped[key],metas,'expected_count')
  events.append({'id':e['id'],'scene':key[0],'row_index':key[1],'event_id':e['event_id'],
    'instruction':e['instruction'],'images':images,'panel_meta':metas,'prompt':prompt,
    'coordinate_policy':'paired_canvas_map' if cfg['kind']=='paired' else 'none'})
 save_json(target/'input_manifest.json',{'condition':condition,'config':cfg,'scope':scope,'events':events,
  'protocol':'Original wrapper, system/user prompt, expected_count, 768 output tokens, BF16, panels, evaluator. Only visual rendering and necessary crop metadata change.',
  'gt_disclosure':'GT-derived expected_count retained equally in original baseline and interventions; no target identity/location used in rendering.',
  'sample':'Development events previously seen. Confirmation scope excludes dev10 but remains previously seen; not independent holdout.'})
 save_json(target/'image_audit.json',{'panels':audit})
 deps=[Path(__file__).resolve(),target/'input_manifest.json',target/'manifest.csv',target/'image_audit.json',
  TRI/'exam2/run_qwen3vl_2d_point_grounding.py',TRI/'scripts/grounding/run_qwen3vl_local_keyframe_grounding.py',
  TRI/'exam2/evaluate_2d_point_grounding.py',ROOT/'MultiAgent/foveation_attention.py',ROOT/'MultiAgent/visual_input_trace.py',MODEL/'config.json']
 save_json(target/'protocol_lock.json',{'files':{str(p):sha_file(p) for p in deps}})
 print('PREPARED',condition,len(events),'events',len(audit),'panels',flush=True)


def infer(out,condition):
 target=out/condition;check(target);v=visual();m=json.loads((target/'input_manifest.json').read_text())
 q=v.load_qwen_module(TRI)
 runner=q.LocalQwen3VLRunner(str(MODEL),'bfloat16',False,768,True,'image',1,None,None,'debug');runner.load()
 from MultiAgent.visual_input_trace import VisualInputTrace
 from tools.run_full_frame_grounding_v4 import clean_visual_trace
 for i,e in enumerate(m['events'],1):
  path=target/'responses'/(e['id']+'.json')
  fingerprint=hashlib.sha256(json.dumps([e['prompt'],[x['sha256'] for x in e['images']],e['coordinate_policy']]).encode()).hexdigest()
  if path.exists():
   if json.loads(path.read_text())['fingerprint']!=fingerprint:raise ValueError('Cached fingerprint mismatch')
   continue
  for x in e['images']:
   if sha_file(Path(x['path']))!=x['sha256']:raise ValueError('Input changed')
  images=[Image.open(x['path']).convert('RGB') for x in e['images']]
  inputs=v.build_model_inputs(runner.processor,images,e['prompt'],e['panel_meta'],'none')
  for image in images:image.close()
  inputs=q.move_batch_to_device(inputs,runner.device)
  trace=VisualInputTrace(runner,inputs,path,kind='image');started=time.monotonic()
  try:
   with runner.runtime.torch.inference_mode():
    generated=runner.model.generate(**inputs,max_new_tokens=768,do_sample=False,use_cache=True)
  finally:data=trace.close()
  if not data['input_consumption_verified']:raise ValueError('Vision consumption failed')
  raw=q.decode_response(runner.processor,q.trim_generated_ids(inputs,generated)).strip()
  parsed=v.extract_json(raw);mapped=v.map_prediction_coordinates(parsed,e['panel_meta'],e['coordinate_policy'])
  save_json(path,{'event':e['id'],'fingerprint':fingerprint,'raw_response':raw,'raw_parsed':parsed,'parsed':mapped,
   'status':'COMPLETE' if parsed is not None else 'PARSE_FAILED','vision_verified':True,'visual_tensor_trace':data,
   'elapsed_seconds':time.monotonic()-started,'generated_tokens':int(generated.shape[-1]-inputs['input_ids'].shape[-1])})
  clean_visual_trace(str(path))
  print('VISUAL',condition,i,'/',len(m['events']),e['id'],'points',len((mapped or {}).get('referents',[])),flush=True)


def evaluate(out,condition):
 target=out/condition;check(target);m=json.loads((target/'input_manifest.json').read_text());rows=[];frames=Counter();paths=[]
 for e in m['events']:
  p=target/'responses'/(e['id']+'.json');r=json.loads(p.read_text());paths.append(p)
  pred=r.get('parsed') or {'referents':[]}
  frames.update(x.get('panel_id') for x in pred.get('referents',[]))
  rows.append({'scene':e['scene'],'row_index':e['row_index'],'event_id':e['event_id'],'instruction':e['instruction'],
   'model_input_image':';'.join(x['path'] for x in e['images']),'model_raw_output':r['raw_response'],'parsed_json':json.dumps(pred),
   'parse_ok':str(r['status']=='COMPLETE'),'prediction_count':len(pred['referents']),'error_message':''})
 save_json(target/'predictions_frozen.json',{'responses':{str(p):sha_file(p) for p in paths}})
 write_csv(target/'predictions.csv',rows,CSV_COLUMNS)
 evaldir=target/'evaluation';subprocess.run([sys.executable,str(TRI/'exam2/evaluate_2d_point_grounding.py'),
  '--manifest',str(target/'manifest.csv'),'--pred_csv',str(target/'predictions.csv'),'--output_dir',str(evaldir),'--coordinate_mode','panel'],check=True,capture_output=True,text=True)
 s=json.loads((evaldir/'2d_eval_summary.json').read_text());exact=exact_counts(read_csv(evaldir/'2d_eval_detail.csv'))
 save_json(target/'summary.json',{'metrics':s,'exact':exact,'frames':dict(frames)})
 print('SCORED',condition,s['events'],'point',s['point_100_f1'],'joint',s['joint_100_f1'],'TP',s['joint_tp_100'],'exact',exact['joint_exact_100'],flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','infer','evaluate']);p.add_argument('--condition',choices=list(CONFIGS),required=True);p.add_argument('--scope',choices=['dev','confirm','all'],default='dev');p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
 if a.stage=='prepare':prepare(a.output.resolve(),a.condition,a.scope)
 else:{'infer':infer,'evaluate':evaluate}[a.stage](a.output.resolve(),a.condition)
if __name__=='__main__':main()
