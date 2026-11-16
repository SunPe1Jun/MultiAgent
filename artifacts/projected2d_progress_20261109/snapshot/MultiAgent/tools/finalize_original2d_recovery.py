#!/usr/bin/env python3
"""Authoritative report with actual treatment coverage and corrected crop controls."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from MultiAgent.projected2d_attention_study import read_csv,write_csv,save_json,sha_file
from MultiAgent.projected2d_exact_set import exact_counts
from tools.iterate_original2d_visual import ROOT,TRI,BASE,check

DATA=ROOT/'.local-sync'
OUT=DATA/'2d_original_visual_results32_20261109'
PAIRS={
 'HEAT_CROP':(DATA/'2d_original_visual_fastdev_20261109/HEAT_CROP',DATA/'2d_original_visual_confirm22_20261109/HEAT_CROP'),
 'HEAT_CROP_FOV80':(DATA/'2d_original_visual_iteration_20261109/HEAT_CROP_FOV',DATA/'2d_original_visual_confirm22_20261109/HEAT_CROP_FOV'),
 'HEAT_CROP_FOV95':(DATA/'2d_original_visual_followup10_20261109/HEAT_CROP_FOV95',DATA/'2d_original_visual_confirm22_20261109/HEAT_CROP_FOV95'),
 'MIRROR_HEAT_CROP':(DATA/'2d_original_visual_mirror10_20261109/HEAT_CROP',DATA/'2d_original_visual_mirror22_20261109/HEAT_CROP'),
 'FIXED22_CORRECTED':(DATA/'2d_original_visual_followup10_20261109/FIXED_GAZE22',DATA/'2d_original_real_fixed22_confirm22_20261109/FIXED_CURRENT_GAZE'),
 'FIXED18':(DATA/'2d_original_fixed18_dev_v2_20261109/FIXED_GAZE18',DATA/'2d_original_fixed18_confirm_v2_20261109/FIXED_GAZE18'),
 'FIXED27':(DATA/'2d_original_fixed27_dev_v2_20261109/FIXED_GAZE27',DATA/'2d_original_fixed27_confirm_v2_20261109/FIXED_GAZE27')}
DEVONLY={name:DATA/'2d_original_visual_iteration_20261109'/name for name in ['FOV_WIDE','SINGLE_HD','SINGLE_HD_FOV']}
DEVONLY['FIXED30']=DATA/'2d_original_fixed30_dev_20261109/FIXED_GAZE30'
ALLOWED_PREP_DRIFT={'iterate_original2d_followup.py','prepare_original2d_confirmation.py'}


def key(row):return row['scene'],int(row['row_index'])
def jread(path):return json.loads(path.read_text())


def verify(root):
    frozen=jread(root/'predictions_frozen.json')['responses'];lock=jread(root/'protocol_lock.json')['files'];drift=[]
    for filename,expected in lock.items():
        actual=sha_file(Path(filename))
        if actual!=expected:
            if Path(filename).name not in ALLOWED_PREP_DRIFT:raise ValueError('Dependency changed: '+filename)
            drift.append(filename)
    for filename,digest in frozen.items():
        if sha_file(Path(filename))!=digest:raise ValueError('Frozen response changed: '+filename)
    proof=root/'recorded_input_verification.json'
    if not proof.is_file():proof=root/'processor_reconstruction_verification.json'
    if not proof.is_file():raise ValueError('Missing independently reconstructed input proof: '+str(proof))
    audit=jread(proof)
    if audit['calls_verified']!=len(frozen):raise ValueError('Incomplete input reconstruction')
    return {'root':str(root),'calls':len(frozen),'source_lock_fully_matches':not drift,
        'historical_preparation_helper_drift':drift,'input_tensor_reconstruction_verified':True}


def score(label,rows,keys,manifest):
    selected=[row for row in rows if key(row) in keys]
    if len(selected)!=len(keys):raise ValueError(f'{label}: {len(selected)} rows for {len(keys)} events')
    target=OUT/'verified_final'/label;target.mkdir(parents=True,exist_ok=True)
    pred=target/'predictions.csv';man=target/'manifest.csv'
    write_csv(pred,selected,list(selected[0]));chosen=[r for r in manifest if key(r) in keys]
    write_csv(man,chosen,list(chosen[0]))
    evaluation=target/'evaluation'
    subprocess.run([sys.executable,str(TRI/'exam2/evaluate_2d_point_grounding.py'),
        '--manifest',str(man),'--pred_csv',str(pred),'--output_dir',str(evaluation),
        '--coordinate_mode','panel'],check=True,capture_output=True,text=True)
    metrics=jread(evaluation/'2d_eval_summary.json');detail=read_csv(evaluation/'2d_eval_detail.csv')
    return {'metrics':metrics,'exact':exact_counts(detail),'detail_path':str(evaluation/'2d_eval_detail.csv')}


def bootstrap(candidate,baseline,keys):
    a={key(r):r for r in read_csv(Path(candidate['detail_path']))};b={key(r):r for r in read_csv(Path(baseline['detail_path']))}
    order=sorted(keys)
    def pack(items):return np.asarray([[int(items[k]['joint_tp_100']),int(items[k]['joint_fp_100']),int(items[k]['joint_fn_100'])] for k in order],float)
    aa=pack(a);bb=pack(b)
    def f(x):return 2*x[...,0]/np.maximum(2*x[...,0]+x[...,1]+x[...,2],1)
    ix=np.random.default_rng(20261109).integers(0,len(order),(20000,len(order)))
    delta=f(aa[ix].sum(axis=1))-f(bb[ix].sum(axis=1))
    return {'delta':float(f(aa.sum(axis=0))-f(bb.sum(axis=0))),
        'paired_bootstrap95':np.quantile(delta,[.025,.975]).tolist(),
        'limitation':'Exploratory, postselection, previously seen interactions. Interval is descriptive and does not establish prospective gain.'}


def main():
    manifest=read_csv(BASE/'manifest.csv');keys={key(r) for r in manifest}
    if len(keys)!=32:raise ValueError('Expected 32 original interactions')
    dev={key(r) for r in read_csv(DATA/'referent_frame_choice_audit_20261109/rebuilt_manifest_all.csv')}
    native=read_csv(BASE/'predictions.csv');old=read_csv(DATA/'projected2d_baseline_audit_20261109/original_qwen8_same32_replay_predictions.csv')
    check(BASE)
    for filename,digest in jread(BASE/'predictions_frozen.json')['responses'].items():
        if sha_file(Path(filename))!=digest:raise ValueError('Native baseline changed')
    native_events={}
    for path in (BASE/'responses').glob('*.json'):
        e=jread(path);native_events[(e['scene'],int(e['row_index']))]=e
    base_score=score('NATIVE_BASELINE_ALL32',native,keys,manifest)
    old_score=score('OLD_RELEASED_REFERENCE_ALL32',old,keys,manifest)
    result={'native_baseline':base_score,'old_released_reference':old_score,'conditions':{},'dev_only':{}}
    roots_verified=[];geometry=[]
    for condition,pair in PAIRS.items():
        rows=[];changed=set();detail=[]
        for root in pair:
            roots_verified.append(verify(root));m=jread(root/'input_manifest.json')
            rows.extend(read_csv(root/'predictions.csv'))
            for e in m['events']:
                k=(e['scene'],int(e['row_index']));previous=native_events[k]
                oldhash=[sha_file(Path(p)) for p in previous['model_input_images']]
                newhash=[x['sha256'] for x in e['images']]
                for image in e['images']:
                    if sha_file(Path(image['path']))!=image['sha256']:raise ValueError('Input image changed')
                did_change=oldhash!=newhash or e['prompt']!=previous['prompt_text']
                if did_change:changed.add(k)
                response=jread(root/'responses'/(e['id']+'.json'))
                detail.append({'event':e['id'],'changed':did_change,'same_images':oldhash==newhash,
                    'same_prompt':e['prompt']==previous['prompt_text'],
                    'same_raw_response':response['raw_response']==previous['model_raw_output']})
                if condition=='FIXED22_CORRECTED':
                    # Independently verify named fixed-scale treatment in every supported frame.
                    for p in e['panel_meta']:
                        box=[float(x) for x in p['crop_bbox_norm'].split(',')]
                        width=box[2]-box[0]
                        if abs(width-.35)>.0001 and abs(width-.22)>.0006:
                            raise ValueError('Incorrect fixed22 treatment: '+e['id'])
                        geometry.append({'event':e['id'],'frame':p['panel_id'],'box':box,'fixed22_or_original_fallback':True})
        if len(rows)!=32 or {key(r) for r in rows}!=keys:raise ValueError('Pair does not cover all32')
        scores=score(condition+'_ALL32',rows,keys,manifest)
        scores['dev10']=score(condition+'_DEV10',rows,dev,manifest)
        scores['confirm22']=score(condition+'_CONFIRM22',rows,keys-dev,manifest)
        scores['modified']=score(condition+'_MODIFIED',rows,changed,manifest)
        scores['native_same_modified']=score(condition+'_NATIVE_SAME_MODIFIED',native,changed,manifest)
        scores['old_same_modified']=score(condition+'_OLD_SAME_MODIFIED',old,changed,manifest)
        scores['input_coverage']={'changed_events':len(changed),'unchanged_events':len(keys-changed),
            'unchanged_raw_responses_reproduced':sum(not d['changed'] and d['same_raw_response'] for d in detail),'events':detail}
        scores['versus_native_all32']=bootstrap(scores,base_score,keys)
        scores['versus_old_all32']=bootstrap(scores,old_score,keys)
        result['conditions'][condition]=scores
        metrics=scores['metrics'];print('VERIFIED_RESULT',condition,'Point',round(metrics['point_100_f1'],4),'Joint',round(metrics['joint_100_f1'],4),'modified',len(changed),flush=True)
    for condition,root in DEVONLY.items():
        roots_verified.append(verify(root));m=jread(root/'input_manifest.json')
        result['dev_only'][condition]=score(condition+'_DEVONLY',read_csv(root/'predictions.csv'),dev,manifest)
    # Count this wasted mislabeled control transparently; exclude it from effectiveness comparisons.
    bad=DATA/'2d_original_visual_fixed22_confirm22_20261109/HEAT_CROP'
    badproof=verify(bad);roots_verified.append(badproof)
    result.update(date='2026-11-09',sample='32 repeatedly used development interactions; dev10 and disjoint confirm22 remain previously seen. No independent holdout.',
        task_protocol='Restored original physical-referent output and GT-derived expected_count equally in baseline and all interventions. Original P panels, evaluator and paired_canvas_map. No GT identity or point in visual rendering.',
        best_observed='HEAT_CROP',foveation_conclusion='Both tested foveation strengths reduce the heat-crop gain; no combined heatmap+foveation benefit established.',
        visual_layout='Best branch uses original full RGB plus RGB local magnification. Single-HD and single-HD-foveation fail on dev10; these gains are not in the previous per-mention single-main-image protocol.',
        original_wrapper_reproduction='Native current wrapper was actually run for 32 events. Its old published predictions are not exactly reproduced; runtime differences remain unresolved.',
        rejected_mislabeled_control={'root':str(bad),'calls':badproof['calls'],'reason':'Called HEAT_CROP during preparation, despite directory/metadata labeling it fixed22. Excluded and replaced by explicit fixed-gaze control with verified boxes.'},
        source_lock_note='Some old CLI/preparation helpers were edited after frozen responses. This drift is disclosed per run. Immutable actual image, prompt, output and consumed processor tensors are reverified; full source-lock equivalence is not claimed for drifted runs.',
        calls_this_iteration=32+sum(r['calls'] for r in roots_verified),native_calls=32,
        visual_calls_reverified=sum(r['calls'] for r in roots_verified),roots_verified=roots_verified,
        fixed22_geometry=geometry,
        authoritative_supersedes=['completion_audit.json (earlier draft)','FIXED_GAZE22_summary.json (mislabeled confirmation)','heat_crop_paired_bootstrap.json (incorrect positional dev split; all32 intervals retain descriptive value)'])
    save_json(OUT/'FINAL_VERIFIED_REPORT.json',result)
    print('FINAL_REPORT',result['calls_this_iteration'],'model calls',result['visual_calls_reverified'],'visual requests verified',flush=True)

if __name__=='__main__':main()
