#!/usr/bin/env python3
"""Evaluate only events whose pixels or prompt changed under each visual intervention."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from MultiAgent.projected2d_attention_study import read_csv,write_csv,save_json
from MultiAgent.projected2d_exact_set import exact_counts
from tools.iterate_original2d_visual import ROOT,TRI,BASE

OUT=ROOT/'.local-sync/2d_original_visual_results32_20261109'
DIFF=json.loads((OUT/'input_differences_by_event.json').read_text())
MANIFEST=read_csv(BASE/'manifest.csv')
CONDITIONS={
 'HEAT_CROP':OUT/'HEAT_CROP.csv',
 'FIXED_GAZE22':OUT/'FIXED_GAZE22.csv',
 'MIRRORED_HEAT_CROP':OUT/'MIRRORED_HEAT_CROP.csv',
 'HEAT_CROP_FOV':OUT/'HEAT_CROP_FOV.csv',
 'FOV95':OUT/'FOV95.csv'}
NATIVE=BASE/'predictions.csv'
OLD=ROOT/'.local-sync/projected2d_baseline_audit_20261109/original_qwen8_same32_replay_predictions.csv'
DEV={ (e['scene_key'],int(e['row_index'])) for e in json.loads((ROOT/'.local-sync/referent_frame_choice_audit_20261109/input_manifest.json').read_text())['events'] }


def score(name,rows,keys):
 selected=[r for r in rows if (r['scene'],int(r['row_index'])) in keys]
 if len(selected)!=len(keys):raise ValueError(f'{name} has {len(selected)} rows for {len(keys)} keys')
 pred=OUT/(name+'.csv');write_csv(pred,selected,list(selected[0]))
 manifest=[r for r in MANIFEST if (r['scene'],int(r['row_index'])) in keys]
 man=OUT/(name+'_manifest.csv');write_csv(man,manifest,list(manifest[0]))
 target=OUT/name
 subprocess.run([sys.executable,str(TRI/'exam2/evaluate_2d_point_grounding.py'),'--manifest',str(man),'--pred_csv',str(pred),'--output_dir',str(target),'--coordinate_mode','panel'],check=True,capture_output=True,text=True)
 summary=json.loads((target/'2d_eval_summary.json').read_text());detail=read_csv(target/'2d_eval_detail.csv')
 return {'metrics':summary,'exact':exact_counts(detail)},detail


def metric_array(rows):
 d={(r['scene'],int(r['row_index'])):r for r in rows}
 keys=sorted(d);a=np.array([[int(d[k]['joint_tp_100']),int(d[k]['joint_fp_100']),int(d[k]['joint_fn_100'])] for k in keys],float)
 return keys,a


def f1(a):return 2*a[...,0]/np.maximum(2*a[...,0]+a[...,1]+a[...,2],1)


def bootstrap(candidate,control,keys,seed):
 a={ (r['scene'],int(r['row_index'])):r for r in candidate};b={ (r['scene'],int(r['row_index'])):r for r in control}
 aa=np.array([[int(a[k]['joint_tp_100']),int(a[k]['joint_fp_100']),int(a[k]['joint_fn_100'])] for k in keys],float)
 bb=np.array([[int(b[k]['joint_tp_100']),int(b[k]['joint_fp_100']),int(b[k]['joint_fn_100'])] for k in keys],float)
 rng=np.random.default_rng(seed);ix=rng.integers(0,len(keys),(50000,len(keys)))
 return {'delta':float(f1(aa.sum(0))-f1(bb.sum(0))),'ci95':np.quantile(f1(aa[ix].sum(1))-f1(bb[ix].sum(1)),[.025,.975]).tolist()}


def main():
 native=read_csv(NATIVE);old=read_csv(OLD);results={}
 for condition,path in CONDITIONS.items():
  changes=DIFF[condition]
  event_keys={f"{r['scene']}_{int(r['row_index'])}":(r['scene'],int(r['row_index'])) for r in MANIFEST}
  keys={event_keys[r['event']] for r in changes if r['changed_input']}
  if len(keys)!=23:raise ValueError(f'Expected 23 actually modified events for {condition}, got {len(keys)}')
  candidate=read_csv(path);cur,_=score(condition+'_MODIFIED_VS_NATIVE',candidate,keys)
  base,_=score(condition+'_NATIVE_ON_MODIFIED',native,keys)
  rel,_=score(condition+'_RELEASED_OLD_ON_MODIFIED',old,keys)
  dev=keys&DEV;confirm=keys-DEV
  splits={}
  for part,subset in [('dev_modified',dev),('confirm_modified',confirm)]:
   if not subset:continue
   cs,_=score(condition+'_'+part,candidate,subset);ns,_=score(condition+'_native_'+part,native,subset);os,_=score(condition+'_released_'+part,old,subset)
   splits[part]={'n':len(subset),'candidate_joint100':cs['metrics']['joint_100_f1'],'native_joint100':ns['metrics']['joint_100_f1'],
                 'released_joint100':os['metrics']['joint_100_f1'],'candidate_tp':cs['metrics']['joint_tp_100'],
                 'native_tp':ns['metrics']['joint_tp_100'],'released_tp':os['metrics']['joint_tp_100']}
  results[condition]={'actually_modified_events':len(keys),'unchanged_events':9,
    'candidate':cur,'native_wrapper_same_modified':base,'released_old_same_modified':rel,
    'candidate_vs_native_bootstrap':bootstrap(score(condition+'_bootstrap_candidate',candidate,keys)[1],score(condition+'_bootstrap_native',native,keys)[1],keys,20261109),
    'candidate_vs_released_bootstrap':bootstrap(score(condition+'_bootstrap_candidate_old',candidate,keys)[1],score(condition+'_bootstrap_released',old,keys)[1],keys,20261110),
    'modified_event_splits':splits,
    'unchanged_raw_responses_identical_to_native':sum(not r['changed_input'] and r['same_raw_output'] for r in changes)}
  c=cur['metrics'];n=base['metrics'];o=rel['metrics']
  print(condition,'modified',len(keys),'Joint',round(c['joint_100_f1'],4),'native',round(n['joint_100_f1'],4),'old',round(o['joint_100_f1'],4),flush=True)
 save_json(OUT/'treatment_coverage_evaluation.json',{'method':'Per-condition scores restricted to the 23 interactions where paired image hashes or prompt differ from native original wrapper. The remaining 9 reproduce identical response with identical images and prompt.',
  'warning':'These 23 interactions are selected by input-change status, not GT outcome. All 32 event intention-to-treat scores are also retained separately.',
  'conditions':results})

if __name__=='__main__':main()
