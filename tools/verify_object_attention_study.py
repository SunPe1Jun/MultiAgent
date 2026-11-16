#!/usr/bin/env python3
"""Audit object/mask attention, real visual inputs, controls and offline tracks."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.object_attention import deduplicate_detections,mask_attention_links,match_masks_one_to_one
from MultiAgent.perspective_attention_study import read,SOURCE
from tools.run_object_attention_study import OUTPUT
from tools.run_object_attention_counterfactual import shifted_links
from tools.verify_attention_region_study import Scripts


def verify(output,jscheck_path=None):
    m=read(output/'manifest.json');assert sha(output/'manifest.json')==read(output/'protocol_lock.json')['manifest_sha256']
    assert read(output/'protocol_lock.json')['before_model_inference'] and not m['inference_gt_loaded'] and not m['anchors_projected']
    assert sha(SOURCE/'manifest.json')==m['source_manifest_sha256']
    assert sha(output/'source/run_object_attention_study.py')==m['code_sha256']
    assert sha(output/'source/object_attention.py')==m['object_code_sha256']
    frames=objects=0
    for e in m['events']:
        for f in e['frames']:
            assert sha(f['source_path'])==f['source_sha256'] and sha(output/f['npz_path'])==f['npz_sha256']
            data=np.load(output/f['npz_path']);maps=data['maps'];masks=data['masks'];h,w=maps.shape[1:]
            expected=deduplicate_detections(f['raw_detections'],w,h)
            assert len(expected)==len(f['objects'])==len(masks)
            for a,b,mask in zip(expected,f['objects'],masks):
                for key in a:assert a[key]==b[key]
                assert b['mask_area_pixels']==int(mask.sum())
            links,display,unknown=mask_attention_links(maps,masks)
            assert links==f['links'];np.testing.assert_allclose(display,data['object_display_density'],atol=1e-12)
            np.testing.assert_allclose(unknown,data['unassigned_maps'],atol=1e-12)
            np.testing.assert_allclose(np.array(links['assigned_seconds'])+links['unassigned_seconds'],links['total_seconds'],atol=1e-8)
            for k,b in enumerate(f['attention_stats']['bins']):
                for o in b['observations']:assert o['gaze_time_seconds']<=f['geometry']['telemetry_time_seconds']+1e-6
            np.testing.assert_allclose(display.sum((1,2))+unknown.sum((1,2)),maps.sum((1,2)),atol=2e-6)
            frames+=1;objects+=len(masks)
    responses={p.stem:read(p) for p in (output/'responses').glob('*.json')};assert len(responses)==len(m['requests'])
    counts=Counter();invalid=[];images={}
    for r in m['requests']:
        assert sha(output/r['path'])==r['sha256'];a=responses[r['id']];raw=read(a['request_path'])
        assert raw['prompt']==r['prompt'] and raw['image_sha256']==[r['sha256']]
        assert raw['max_new_tokens']==160 and raw['do_sample'] is False
        assert raw['processor_audit']['expanded_image_token_count']==1024
        assert raw['visual_tensor_trace']['input_consumption_verified'] and a['vision_verified']
        assert raw['parsed']==a['parsed']
        if a['schema_valid']:
            vals=a['parsed']['selected']+a['parsed']['uncertain'];assert set(vals)<=set(r['available_ids']) and len(set(vals))==len(vals)
        else:invalid.append(r['id'])
        counts[r['condition']]+=1
        key=(r['event_id'],r['frame_id'],r['page'])
        with Image.open(output/r['path']) as im:rgb=np.asarray(im.convert('RGB')).copy()
        # Entire RGB scene/details/text stay identical; only right heat panels vary.
        for k in range(3):rgb[62+k*216:62+k*216+192,756:948]=0
        if key in images:np.testing.assert_array_equal(rgb,images[key])
        else:images[key]=rgb
    assert sha(output/'development_frozen.json')==read(output/'check_started.json')['development_sha256']
    assert read(output/'development_frozen.json')['check_not_started']
    control=output/'counterfactual';cm=read(control/'manifest.json')
    assert sha(control/'manifest.json')==read(control/'protocol_lock.json')['manifest_sha256']
    assert cm['parent_manifest_sha256']==sha(output/'manifest.json')
    ctrl_count=0;main_requests={r['id']:r for r in m['requests']}
    for r in cm['requests']:
        assert sha(control/r['path'])==r['sha256'];parent=main_requests[r['paired_parent_id']]
        assert r['prompt']==parent['prompt'] and r['available_ids']==parent['available_ids']
        a=read(control/'responses'/f"{r['id']}.json");raw=read(a['request_path'])
        assert raw['prompt']==r['prompt'] and raw['image_sha256']==[r['sha256']]
        assert raw['visual_tensor_trace']['input_consumption_verified'] and raw['processor_audit']['expanded_image_token_count']==1024
        with Image.open(control/r['path']) as im:rgb=np.asarray(im.convert('RGB')).copy()
        for k in range(3):rgb[62+k*216:62+k*216+192,756:948]=0
        np.testing.assert_array_equal(rgb,images[(r['event_id'],r['frame_id'],r['page'])]);ctrl_count+=1
    for e in m['events']:
        for f in e['frames']:
            shifted,_=shifted_links(f['objects'],f['links'])
            np.testing.assert_allclose(np.array(shifted['mass_seconds_by_phase_object']).sum(1),np.array(f['links']['mass_seconds_by_phase_object']).sum(1),atol=1e-10)
    tracks=read(output/'tracks/manifest.json');tracked_frames=0;seed_tracks=0
    base={e['event']:e for e in m['events']}
    for event in tracks['results']:
        assert event['parent_manifest_sha256']==sha(output/'manifest.json') and not event['gt_loaded'] and not event['anchors_projected']
        assert sha(output/event['track_masks_path'])==event['track_masks_sha256']
        masks=np.load(output/event['track_masks_path'])['masks'];assert len(masks)==len(event['frames'])
        assert masks.shape[1]==len(event['seed_objects']);seed_tracks+=masks.shape[1]
        for j,f in enumerate(event['frames']):
            assert sha(output/f['native_path'])==f['native_sha256']
            assert f['object_ids']==list(range(1,masks.shape[1]+1))
            assert f['mask_areas']==[int(mask.sum()) for mask in masks[j]]
            tracked_frames+=1
        for o in event['gaze_observations']:
            if o['status']=='PROJECTED':assert abs(o['pts_error_seconds'])<=.08 and o['projection']['status']=='projected_frustum_only'
        np.testing.assert_allclose(np.array(event['object_mass_seconds_by_phase']).sum(1)+event['unassigned_seconds_by_phase'],event['total_seconds_by_phase'],atol=2e-7)
        original=base[event['event']]
        for check,f in zip(event['redetection_compatibility'],original['frames']):
            targets=np.load(output/f['npz_path'])['masks']
            expected=match_masks_one_to_one(masks[check['track_frame_index']],targets,[o['category'] for o in event['seed_objects']],
                [o['category'] for o in f['objects']],minimum_iou=.3)
            assert expected==check['matches']
    centers=read(output/'referent_center_audit.json');assert centers['manifest_sha256']==sha(output/'manifest.json')
    for s in centers['audit_sources']:
        assert sha(s['path'])==s['sha256'] and read(s['path'])['projection_scope']=='event_referents_only'
    js=[]
    if jscheck_path:
        sys.path.insert(0,str(jscheck_path));import quickjs
        dom='''class E{constructor(t='div'){this.tag=t;this.children=[];this.value='';this.src='';this.max=0}append(...xs){for(const x of xs){this.children.push(x);if(this.tag==='select'&&this.value==='')this.value=String(x.value)}}replaceChildren(...xs){this.children=xs;this.value=''}}const roots={};const document={querySelector(s){return roots[s]||(roots[s]=new E(['#event','#condition','#object'].includes(s)?'select':s))},createElement(t){return new E(t)}};'''
        for root,page,kind in [(output,'index.html','objects'),(output/'tracks','index.html','tracks')]:
            parser=Scripts();parser.feed((root/page).read_text());context=quickjs.Context();context.eval(dom)
            if kind=='objects':context.eval("document.querySelector('#condition').value='NO_ADDED_HEAT'")
            for script in parser.scripts:context.eval(script)
            if kind=='objects':
                n=context.eval("let n=0;for(const e of d.m.events){q.value=e.id;for(const v of d.m.conditions){c.value=v;render();n+=document.querySelector('main').children.length}}n")
                assert n==len(m['requests'])
            else:
                n=context.eval("let n=0;for(const e of d){q.value=e.id;changeEvent();for(let i=0;i<e.frames.length;i++){slider.value=i;render();n++}for(const b of e.objects){o.value=b.object_id;render()}}n")
                assert n==tracked_frames
            js.append({'page':str(root/page),'rendered':n,'full_browser':False})
        viewer=read(output/'tracks/viewer_manifest.json')
        for e in viewer['events']:
            for r in e['frames']+e['objects']:assert sha(output/r['path'])==r['sha256']
    result={'object_proposal_frames':frames,'objects':objects,'mass_overlap_unknown_conservation_verified':True,
        'main_model_calls':len(responses),'main_conditions':dict(counts),'main_schema_failures':invalid,
        'counterfactual_model_calls':ctrl_count,'all_1024_visual_tokens_and_consumption_verified':True,
        'same_rgb_ids_crops_prompts_across_heat_controls':True,'development_frozen_before_check':True,
        'offline_track_frames':tracked_frames,'seed_object_tracks':seed_tracks,'nonseed_redetection_matches_reproduced':True,
        'referent_only_post_inference_audit':True,'viewers':js,'target_pixel_accuracy_available':False,'regression_only':True}
    save(output/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--jscheck-path',type=Path)
    a=p.parse_args();verify(a.output.resolve(),a.jscheck_path)
