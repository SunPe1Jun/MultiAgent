#!/usr/bin/env python3
"""Rebuild complete-field inputs from telemetry, no referent or object GT read."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from MultiAgent.attention_volume import build_volume
from MultiAgent.camera_geometry import aligned_frame_geometry
from MultiAgent.context import load_contexts
from MultiAgent.full_interaction_referring import CONDITIONS,render_full_frame,control_maps,referring_prompt,validate_prediction
from MultiAgent.gaze_evidence import gaze_records
from MultiAgent.heatmap_temporal_pilot import save,sha
from tools.run_full_interaction_referring import OUTPUT,read,camera_for_prompt
from tools.verify_attention_region_study import Scripts


def verify(output, complete=False, jscheck_path=None):
    m=read(output/'manifest.json');assert read(output/'protocol_lock.json')['before_any_inference']
    assert sha(output/'manifest.json')==read(output/'protocol_lock.json')['manifest_sha256']
    for name,digest in m['source_sha256'].items():assert sha(output/'source'/name)==digest
    selection=read(m['selection_source']);assert sha(m['selection_source'])==m['selection_source_sha256']
    chosen=[{k:s[k] for k in ('scene_key','row_index','event_id')} for s in selection['samples']]
    # Prohibit annotation reads throughout context loading and reconstruction.
    def reject_annotations(event,args):
        if event=='open' and isinstance(args[0],str) and args[0].lower().endswith(('.xlsx','.xls')):
            raise RuntimeError('Object GT is forbidden in input verification: '+args[0])
    sys.addaudithook(reject_annotations)
    contexts=load_contexts({'samples':chosen},load_ground_truth=False)
    assert len(contexts)==len(m['events'])==24
    response_files={p.stem:read(p) for p in (output/'responses').glob('*.json')}
    if complete:assert set(response_files)=={r['id'] for r in m['requests']}
    image_count=0;failures=[];tokens=set();sizes=set();supported=0
    for event,ctx in zip(m['events'],contexts):
        assert not ctx.provenance['ground_truth_loaded'] and event['gt_loaded'] is False
        assert event['instruction']==ctx.text
        table={c['candidate_id']:c for c in event['scene_candidate_table']}
        assert set(table)=={c.candidate_id for c in ctx.candidates}
        for c in ctx.candidates:assert table[c.candidate_id]['position_world']==[c.x_world,c.y_world,c.z_world]
        assert sha(ctx.json_path)==event['telemetry_sha256'] and sha(ctx.video_path)==event['video_sha256']
        assert sha(event['anchor_table'])==event['anchor_table_sha256']
        assert sha(event['pose_contract_path'])==event['pose_contract_sha256']
        original=read(event['pose_contract_path'])['frames'];records=gaze_records(ctx)
        assert len(original)==len(event['frames'])==12
        for frame,base in zip(event['frames'],original):
            assert frame['source_path']==base['source_path'] and sha(frame['source_path'])==frame['source_sha256']==base['source_sha256']
            assert frame['source_frame_index']==base['frame_time']['source_frame_index']
            geometry=aligned_frame_geometry(base,sha(frame['source_path']),frame['time_seconds'],0.)
            assert geometry==frame['geometry'] and camera_for_prompt(geometry)==frame['prompt_camera']
            with Image.open(frame['source_path']) as image:rgb=image.convert('RGB')
            maps,stats=build_volume(records,query=geometry['telemetry_time_seconds'],pose=geometry['reference_pose'],
                image_size=rgb.size,clock_supported=geometry['clock_available'],history=2.,bins=3,sigma_degrees=2.)
            assert json.loads(json.dumps(stats))==frame['attention_stats'] and sha(output/frame['maps_path'])==frame['maps_sha256']
            with np.load(output/frame['maps_path']) as saved:np.testing.assert_array_equal(maps,saved['maps'])
            supported+=geometry['clock_available']
            for condition in CONDITIONS:
                view=render_full_frame(rgb,maps,condition);buffer=io.BytesIO();view.save(buffer,format='PNG')
                info=frame['images'][condition]
                assert hashlib.sha256(buffer.getvalue()).hexdigest()==info['sha256']==sha(output/info['path'])
                np.testing.assert_array_equal(np.asarray(view)[28:28+rgb.height,:rgb.width],np.asarray(rgb))
                if condition!='NO_ADDED_HEAT':
                    np.testing.assert_allclose(control_maps(maps,condition).sum(dtype=float),maps.sum(dtype=float),rtol=1e-7,atol=1e-8)
                image_count+=1
        assert event['prompt']==referring_prompt(event['instruction'],event['scene_candidate_table'],event['frames'])
        for condition in CONDITIONS:
            rid=event['id']+'_'+condition
            if rid not in response_files:continue
            response=response_files[rid];raw=read(response['request_path']);audit=raw['processor_audit']
            assert raw['prompt']==event['prompt'] and raw['max_new_tokens']==m['max_new_tokens']
            assert [im['sha256'] for im in raw['images']]==[f['images'][condition]['sha256'] for f in event['frames']]
            assert audit['input_frame_count']==12 and audit['qwen_temporal_patch_size']==2
            assert audit['qwen_template_timestamps_verified'] and audit['processor_sampling_disabled']
            assert audit['visual_tensor_trace']['input_consumption_verified'] and response['vision_verified']
            check=validate_prediction(raw['parsed'],list(table),{g['view_id'] for g in event['qwen_temporal_groups']},ctx.text)
            assert response['validation']==check
            if not check['schema_valid']:failures.append({'id':rid,'errors':check['errors']})
            tokens.add(audit['expanded_video_token_count']);sizes.add(tuple(audit['processor_effective_frame_size_px']))
    js=None
    if jscheck_path:
        sys.path.insert(0,str(jscheck_path));import quickjs
        dom="""class E{constructor(t='div'){this.tag=t;this.children=[];this.value=''}append(...xs){for(const x of xs){this.children.push(x);if(this.tag==='select'&&!this.value)this.value=String(x.value)}}replaceChildren(){this.children=[]}}const roots={};const document={querySelector(s){return roots[s]||(roots[s]=new E(['#event','#condition'].includes(s)?'select':s))},createElement(t){return new E(t)}};"""
        parser=Scripts();parser.feed((output/'index.html').read_text());ctxjs=quickjs.Context();ctxjs.eval(dom)
        for script in parser.scripts:ctxjs.eval(script)
        count=ctxjs.eval("let n=0;for(const e of data.events){q.value=e.id;for(const name of data.conditions){c.value=name;render();n+=document.querySelector('main').children.length}}n")
        assert count==1152;js={'cards_rendered':count,'full_browser':False}
    result={'interactions_verified':24,'all_full_scene_candidates_verified':True,'all_input_images_byte_reproduced':image_count,
        'full_rgb_panel_pixels_preserved':True,'heat_supported_frames':supported,'no_object_crops_or_anchor_projection':True,
        'annotation_reads_forbidden_in_rebuild':True,'model_calls_verified':len(response_files),
        'schema_failures':failures,'visual_tokens':sorted(tokens),'effective_frame_sizes':sorted(sizes),
        'all_model_vision_and_timestamps_verified':True,'viewer_logic':js,'unseen_validation':False}
    save(output/('verification.json' if complete else 'verification_pre.json'),result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--complete',action='store_true');p.add_argument('--jscheck-path',type=Path)
    a=p.parse_args();verify(a.output.resolve(),a.complete,a.jscheck_path)
