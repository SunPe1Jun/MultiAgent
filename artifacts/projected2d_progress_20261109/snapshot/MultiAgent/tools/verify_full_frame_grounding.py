#!/usr/bin/env python3
"""Verify frozen global choices, identical-input reuse and GT-free ID decoder."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from MultiAgent.full_frame_grounding import grounding_prompt,validate_grounding,decode_grounding
from MultiAgent.full_frame_grounding_v3 import grounding_prompt as grounding_prompt_v3,validate_composite_grounding
from MultiAgent.heatmap_temporal_pilot import save,sha
from tools.run_full_frame_grounding import OUTPUT,PARENT,read
from tools.verify_attention_region_study import Scripts


def verify(output,complete=False,jscheck_path=None):
    m=read(output/'manifest.json');parent=read(PARENT/'manifest.json')
    assert sha(PARENT/'manifest.json')==m['parent_manifest_sha256']
    assert sha(output/'manifest.json')==read(output/'protocol_lock.json')['manifest_sha256']
    for name,digest in m['source_sha256'].items():assert sha(output/'source'/name)==digest
    prior=read(PARENT/'verification_pre.json');assert prior['all_input_images_byte_reproduced']==1152
    assert prior['full_rgb_panel_pixels_preserved'] and prior['annotation_reads_forbidden_in_rebuild']
    rows={f.stem:read(f) for f in (output/'responses').glob('*.json')}
    if complete:assert set(rows)=={r['id'] for r in m['requests']}
    reqs={r['id']:r for r in m['requests']};events={e['id']:e for e in m['events']}
    parents={e['id']:e for e in parent['events']};checks=0;tokens=set();sizes=set();canonical_paths={};schema=[]
    for event in m['events']:
        base=parents[event['id']]
        assert event['scene_candidate_table']==base['scene_candidate_table']
        assert event['instruction']==base['instruction'] and event['gt_loaded'] is False
        expected_prompt=grounding_prompt_v3(event) if m['version']=='full_frame_grounding_v3_composite_coordinates' else grounding_prompt(event)
        assert event['prompt']==expected_prompt
        for frame,original in zip(event['frames'],base['frames']):
            assert frame['geometry']==original['geometry'] and frame['source_sha256']==original['source_sha256']
            for condition in m['conditions']:
                image=frame['images'][condition]
                assert image['sha256']==original['images'][condition]['sha256']==sha(output/image['path']);checks+=1
    def guard(event,args):
        if event=='open' and isinstance(args[0],str) and args[0].lower().endswith(('.xlsx','.xls')):
            raise RuntimeError('Annotations forbidden during decoder verification')
    sys.addaudithook(guard)
    for rid,response in rows.items():
        req=reqs[rid];event=events[req['event_id']];condition=req['condition']
        raw=read(response['request_path']);hashes=[f['images'][condition]['sha256'] for f in event['frames']]
        assert raw['prompt']==event['prompt'] and [v['sha256'] for v in raw['images']]==hashes
        fingerprint=hashlib.sha256(json.dumps([event['prompt'],hashes],sort_keys=True).encode()).hexdigest()
        assert req['model_input_fingerprint']==fingerprint
        canonical=reqs[req['canonical_request_id']]
        assert canonical['model_input_fingerprint']==fingerprint
        assert raw['max_new_tokens']==m['max_new_tokens']
        audit=raw['processor_audit'];assert audit['input_frame_count']==12
        assert audit['processor_sampling_disabled'] and not audit['processor_resampling_performed']
        assert audit['processor_temporal_padding_frames']==0 and audit['qwen_template_timestamps_verified']
        assert audit['visual_tensor_trace']['input_consumption_verified'] and response['vision_verified']
        assert audit['video_rope_compatibility']['vision_grid_and_pixels_unchanged']
        assert audit['video_rope_compatibility']['contract']=='qwen3vl_timestamp_video_rope_grid_split_v1'
        checked=(validate_composite_grounding(raw['parsed'],event)
                 if m['version']=='full_frame_grounding_v3_composite_coordinates'
                 else validate_grounding(raw['parsed'],event))
        assert checked==response['grounding_validation']
        decoded=decode_grounding(checked['referents'],event);assert decoded==response['decoder']
        # A fabricated GT value and candidate ordering cannot alter decoding.
        altered={**event,'gt':['deliberately_wrong_annotation'],'scene_candidate_table':list(reversed(event['scene_candidate_table']))}
        assert decode_grounding(checked['referents'],altered)==decoded
        assert response['validation']['selected_object_names']==decoded['selected_object_names']
        assert response['validation']['schema_valid']==checked['schema_valid']
        if not checked['schema_valid']:schema.append({'id':rid,'errors':checked['errors']})
        if req['canonical_request_id'] in canonical_paths:assert canonical_paths[req['canonical_request_id']]==response['request_path']
        canonical_paths[req['canonical_request_id']]=response['request_path']
        tokens.add(audit['expanded_video_token_count']);sizes.add(tuple(audit['processor_effective_frame_size_px']))
    audits=[]
    for path in (output/'model_cache').glob('*/call_audit.jsonl'):
        audits.extend(json.loads(line) for line in path.read_text().splitlines())
    if complete:
        actual=[a for a in audits if not a['cache_hit']]
        assert len(canonical_paths)==m['planned_unique_model_calls']==len(actual)
        assert not any(a.get('error') for a in audits)
        assert {a['request_path'] for a in audits}==set(canonical_paths.values())
        frozen=read(output/'predictions_frozen.json')
        assert frozen['before_gt_evaluation']
        for rid,digest in frozen['response_sha256'].items():assert sha(output/'responses'/f'{rid}.json')==digest
        guard_record=(output/'annotation_read_guard.txt').read_text();assert guard_record.startswith('Completed')
    js=None
    if jscheck_path:
        sys.path.insert(0,str(jscheck_path));import quickjs
        dom="""class E{constructor(t='div'){this.tag=t;this.children=[];this.value=''}append(...xs){for(const x of xs){this.children.push(x);if(this.tag==='select'&&!this.value)this.value=String(x.value)}}replaceChildren(){this.children=[]}}const roots={};const document={querySelector(s){return roots[s]||(roots[s]=new E(['#event','#condition'].includes(s)?'select':s))},createElement(t){return new E(t)}};"""
        parser=Scripts();parser.feed((output/'index.html').read_text());ctx=quickjs.Context();ctx.eval(dom)
        for script in parser.scripts:ctx.eval(script)
        count=ctx.eval("let n=0;for(const e of data.events){q.value=e.id;for(const name of data.conditions){c.value=name;render();n+=document.querySelector('main').children.length}}n")
        assert count==len(m['events'])*12*4;js={'cards_rendered':count,'full_browser':False}
    result={'interactions':len(m['events']),'shared_parent_full_frame_images_verified':checks,
        'completed_condition_results_verified':len(rows),'unique_model_calls_verified':len(canonical_paths),
        'backend_cache_replays':sum(a['cache_hit'] for a in audits),
        'identical_input_reuse_only':True,'no_gt_in_model_choice_or_decoder':True,
        'decoder_ignores_fake_gt_and_candidate_order':True,'all_scene_candidates_preserved':True,
        'model_vision_and_timestamp_contracts_verified':True,'schema_failures':schema,
        'visual_tokens':sorted(tokens),'effective_composite_frame_sizes':sorted(sizes),
        'source_rgb_field_pixels':[672,672],'no_object_crops_or_anchor_projection_in_input':True,
        'candidate_geometry_scope':'All type-compatible public anchors considered internally for decoding. No all-anchor projection overlay/audit retained.',
        'gt_only_after_frozen_predictions':complete,'viewer_logic':js,'unseen_validation':False}
    save(output/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--complete',action='store_true');p.add_argument('--jscheck-path',type=Path)
    a=p.parse_args();verify(a.output.resolve(),a.complete,a.jscheck_path)
