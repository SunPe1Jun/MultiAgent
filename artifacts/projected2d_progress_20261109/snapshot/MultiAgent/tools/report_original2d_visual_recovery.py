#!/usr/bin/env python3
"""Summarize original-wrapper recovery and controlled visual development trials."""
import json
from pathlib import Path
import subprocess
import sys
from collections import Counter
import numpy as np
from MultiAgent.projected2d_attention_study import read_csv,write_csv,save_json,sha_file
from MultiAgent.projected2d_exact_set import exact_counts
from tools.iterate_original2d_visual import check,ROOT,TRI,BASE

OUT=ROOT/'.local-sync/2d_original_visual_results32_20261109'
DIR=ROOT/'.local-sync'
CONFIGS={
 'HEAT_CROP':[DIR/'2d_original_visual_fastdev_20261109/HEAT_CROP',DIR/'2d_original_visual_confirm22_20261109/HEAT_CROP'],
 'HEAT_CROP_FOV':[DIR/'2d_original_visual_iteration_20261109/HEAT_CROP_FOV',DIR/'2d_original_visual_confirm22_20261109/HEAT_CROP_FOV'],
 'MIRRORED_HEAT_CROP':[DIR/'2d_original_visual_mirror10_20261109/HEAT_CROP',DIR/'2d_original_visual_mirror22_20261109/HEAT_CROP'],
 'FOV95':[DIR/'2d_original_visual_followup10_20261109/HEAT_CROP_FOV95',DIR/'2d_original_visual_confirm22_20261109/HEAT_CROP_FOV95'],
 'FIXED_GAZE22':[DIR/'2d_original_visual_followup10_20261109/FIXED_GAZE22',DIR/'2d_original_visual_fixed22_confirm22_20261109/HEAT_CROP'],
}
DEVONLY={name:DIR/'2d_original_visual_iteration_20261109'/name for name in ('FOV_WIDE','SINGLE_HD','SINGLE_HD_FOV')}


def evaluate(name,rows,manifest):
 path=OUT/(name+'.csv');write_csv(path,rows,list(rows[0]));target=OUT/name
 subprocess.run([sys.executable,str(TRI/'exam2/evaluate_2d_point_grounding.py'),'--manifest',str(manifest),'--pred_csv',str(path),'--output_dir',str(target),'--coordinate_mode','panel'],check=True,capture_output=True,text=True)
 metrics=json.loads((target/'2d_eval_summary.json').read_text());detail=read_csv(target/'2d_eval_detail.csv')
 return {'metrics':metrics,'exact':exact_counts(detail)},detail


def verify_root(root):
 check(root);f=json.loads((root/'predictions_frozen.json').read_text())['responses']
 for p,h in f.items():
  if sha_file(Path(p))!=h:raise ValueError('Prediction hash changed: '+p)
 v=json.loads((root/'processor_reconstruction_verification.json').read_text())
 assert v['calls_verified']==len(f)
 return len(f)


def main():
 OUT.mkdir(parents=True,exist_ok=True)
 baseline={'metrics':json.loads((BASE/'evaluation_all/2d_eval_summary.json').read_text()),'exact':exact_counts(read_csv(BASE/'evaluation_all/2d_eval_detail.csv'))}
 f=json.loads((BASE/'predictions_frozen.json').read_text())['responses'];assert len(f)==32
 for p,h in f.items():assert sha_file(Path(p))==h
 results={'NATIVE_ORIGINAL_WRAPPER':baseline};details={};roots=[]
 devkeys={(r['scene'],int(r['row_index'])) for r in read_csv(DIR/'referent_frame_choice_audit_20261109/rebuilt_manifest_all.csv')}
 for name,dirs in CONFIGS.items():
  complete=[d for d in dirs if (d/'summary.json').exists()]
  if not complete:continue
  rows=[];scores=[]
  for d in complete:
   count=verify_root(d);roots.append((d,count));rows.extend(read_csv(d/'predictions.csv'))
   sc=json.loads((d/'summary.json').read_text());scores.append({'root':str(d),'scope':json.loads((d/'input_manifest.json').read_text())['scope'],**sc})
  if len(complete)!=2:
   results[name+'_DEV_ONLY']={'subsets':scores,'interpretation':'No 32-event confirmation completed.'};continue
  assert len(rows)==32 and len({(r['scene'],r['row_index']) for r in rows})==32
  res,det=evaluate(name,rows,BASE/'manifest.csv');res['subsets']=scores;results[name]=res;details[name]=det
 for name,d in DEVONLY.items():
  verify_root(d);roots.append((d,10));results[name+'_DEV_ONLY']=json.loads((d/'summary.json').read_text())
 old=read_csv(DIR/'projected2d_baseline_audit_20261109/original_qwen8_same32_replay_predictions.csv')
 results['RELEASED_OLD_REFERENCE'],olddetail=evaluate('RELEASED_OLD_REFERENCE',old,BASE/'manifest.csv')
 treatment_path=OUT/'treatment_coverage_evaluation.json'
 if treatment_path.is_file():
  treatment=read_csv(DIR/'2d_original_visual_results32_20261109/input_differences_by_event.json') if False else json.loads((DIR/'2d_original_visual_results32_20261109/input_differences_by_event.json').read_text())
  treatment_metrics=json.loads(treatment_path.read_text())
  for name,value in treatment_metrics['conditions'].items():
   actual=results.get(name)
   if actual is not None:
    results[name]['actually_modified_event_analysis']=value
  summaries=treatment
  noops={name:[r['event'] for r in rows if not r['changed_input']] for name,rows in summaries.items()}
  cache=json.loads((DIR/'2d_original_gaze_cache32_20261109/input_image_audit.json').read_text())
  cache_by_event={}
  for panel in cache['panels']:
   cache_by_event.setdefault(panel['event'],[]).append(panel)
  no_op_reasons={event:{'panels':len(cache_by_event.get(event,[])),
    'recent_support_seconds':[sum(x['phase_support_seconds']) if x['phase_support_seconds'] else 0. for x in cache_by_event.get(event,[])],
    'all_heatmaps_empty':all(sha_file(Path(x['heatmap_path']))==x['heatmap_sha256'] and
      float(np.load(x['heatmap_path'])['maps'].sum())==0 for x in cache_by_event.get(event,[]))}
    for event in noops.get('HEAT_CROP',[])}
  summary_extra={'treatment_coverage':treatment_metrics,
    'events_where_visual_input_changed':23,'events_with_exact_native_input_prompt_and_response':9,
    'no_change_events_by_condition':noops,'no_change_reasons_for_heat_crop':no_op_reasons,
    'interpretation':'All-32 results are intention-to-treat and include nine exact RGB fallback events. Treatment-on estimate is separately reported for the 23 events whose image hashes or prompt geometry differed from the native wrapper. Selection of development and confirmation events predates this run; neither set is a prospective holdout.'}
 else:summary_extra={}
 key=lambda x:(x['scene'],int(x['row_index']))
 baserows=read_csv(BASE/'evaluation_all/2d_eval_detail.csv')
 def array(rows):return np.array([[int(r['joint_tp_100']),int(r['joint_fp_100']),int(r['joint_fn_100'])] for r in sorted(rows,key=key)],float)
 def f1(z):return 2*z[...,0]/np.maximum(2*z[...,0]+z[...,1]+z[...,2],1)
 rng=np.random.default_rng(20261109);indices=rng.integers(0,32,size=(20000,32));uncertainty={}
 for name,det in details.items():
  current=array(det);uncertainty[name]={}
  for ref,rows in [('NATIVE_ORIGINAL_WRAPPER',baserows),('RELEASED_OLD_REFERENCE',olddetail)]:
   other=array(rows);delta=f1(current[indices].sum(1))-f1(other[indices].sum(1))
   uncertainty[name][ref]={'difference_joint100':float(f1(current.sum(0))-f1(other.sum(0))),
    'paired_bootstrap95':np.quantile(delta,[.025,.975]).tolist()}
 native=json.loads((BASE/'run_config.json').read_text())
 total=32+sum(n for _,n in roots)
 summary={'date':'2026-11-09','sample':'Same 32 previously seen exploratory events; dev10 and disjoint confirmation22. Confirmation22 is not independent holdout and was used in earlier experiments.',
  'native_original_config':native,'conditions':results,'paired_uncertainty':uncertainty,
  'model_calls_this_recovery':total,'verified_new_visual_calls':sum(n for _,n in roots),
  'native_baseline_provenance':'Original wrapper used natively, source and input images hashed; no retrospective vision-forward trace claimed for those 32 native baseline calls.',
  'protocol_disclosure':{'expected_count':'Original GT-derived physical referent count retained in baseline and every visual intervention; no target identities or GT points in image processing.',
   'images':'Best scoring branch keeps original paired full image plus gaze-derived source-resolution detail. Single main-image branch remains a separately scored ablation and failed on dev10.',
   'referents':'Original physical-referent output for baseline comparison; previously accepted per-mention protocol remains archived separately and is not being counted as these gains.',
   'evaluation':'Original Experiment 2 evaluator, frozen original panels and source-normalized coordinates. No change to GT or matching.'},
  'causal_limits':['Multiple candidates screened on dev10; exploratory selection.',
   'Mirror and scale controls separate attention-location effects from changing detail magnification.',
   'Improved old-subset numbers do not establish improvement on the full 4000-event baseline.',
   'No claim heatmap/foveation is effective solely because a combined score improves. Compare components.'],
  'roots_verified':[{'root':str(p),'count':n} for p,n in roots],**summary_extra}
 save_json(OUT/'completion_audit.json',summary)
 for name,value in results.items():
  if 'metrics' in value:
   s=value['metrics'];print(name,s['events'],'Point100',round(s['point_100_f1'],4),'Joint100',round(s['joint_100_f1'],4),'Exact',value['exact']['joint_exact_100'])
 print('CALLS',total,'VISUAL_VERIFIED',sum(n for _,n in roots))

if __name__=='__main__':main()
