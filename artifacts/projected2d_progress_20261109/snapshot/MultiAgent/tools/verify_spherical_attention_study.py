#!/usr/bin/env python3
"""Verify frozen visual experiments, spherical mass and viewer source provenance."""
import argparse
from collections import Counter
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from MultiAgent.heatmap_temporal_pilot import sha,save
from MultiAgent.spherical_attention_study import OUTPUT,read
from MultiAgent.spherical_attention import solid_angle_weights


class Scripts(HTMLParser):
    def __init__(self):super().__init__();self.items=[];self.inside=False;self.current=''
    def handle_starttag(self,tag,attrs):
        if tag=='script':self.inside=True;self.current=''
    def handle_endtag(self,tag):
        if tag=='script':self.items.append(self.current);self.inside=False
    def handle_data(self,data):
        if self.inside:self.current+=data


DOM='''class Element{constructor(tag='div'){this.tag=tag;this.children=[];this.textContent='';this.value='';this.src='';}append(...items){for(const item of items){this.children.push(item);if(this.tag==='select'&&this.value==='')this.value=String(item.value)}}replaceChildren(...items){this.children=items}querySelector(tag){return this.children.find(n=>n.tag===tag)||new Element(tag)}getBoundingClientRect(){return {left:0,top:0,width:720,height:360}}}const roots={};const document={querySelector(selector){if(!roots[selector])roots[selector]=new Element(['#event','#mode'].includes(selector)?'select':selector);return roots[selector]},querySelectorAll(tag){return Object.values(roots).flatMap(n=>n.children).filter(n=>n.tag===tag)},createElement(tag){return new Element(tag)}};document.querySelector('#mode').value='overview';document.querySelector('#frame').value='0';'''


def check_js(root,page,quickjs):
    parser=Scripts();parser.feed((root/page).read_text());context=quickjs.Context();context.eval(DOM)
    for script in parser.items:context.eval(script)
    if page=='head_motion_demo.html':
        context.eval("for(let i=0;i<d.events.length;i++){q.value=String(i);for(let j=0;j<d.events[i].frames.length;j++){slider.value=String(j);render();}document.querySelector('#map').onclick.call(document.querySelector('#map'),{clientX:360,clientY:180});}")
    else:
        context.eval("for(let i=0;i<d.events.length;i++){q.value=String(i);for(const value of ['overview','probe','sources']){mode.value=value;render();}}")
    paths=json.loads(context.eval("JSON.stringify([...Object.values(roots).filter(r=>r.src),...Object.values(roots).flatMap(r=>r.children).flatMap(f=>f.children||[]).filter(n=>n.tag==='img')].map(n=>n.src))"))
    for path in paths:assert (root/path).exists(),str(root/path)
    return {'page':str(root/page),'executed_scripts':len(parser.items),'all_selectors_exercised':True,'full_browser_tested':False}


def verify(root,jscheck_path=None):
    m=read(root/'manifest.json')
    assert sha(root/'source'/'spherical_attention.py')==m['sphere_code_sha256']
    assert sha(root/'source'/'spherical_attention_study.py')==m['code_sha256']
    assert len(m['events'])+len(m['fallback_events'])==24
    frames=0
    for e in m['events']:
        for f in e['frames']:
            assert sha(f['source_path'])==f['source_sha256'];frames+=1
            g=f['geometry'];assert g['clock_available']
            assert abs(g['telemetry_time_seconds']+g['selected_clock_offset_seconds']-f['pts'])<1e-6
        v=np.load(root/'events'/e['event']/'volume.npz');maps=v['duration_maps'];coverage=v['camera_coverage_seconds'];owner=v['rgb_source_index']
        for i,b in enumerate(e['volume_stats']['bins']):
            assert abs(float(maps[i].sum())-b['duration_seconds'])<1e-5
            for observation in b['observations']:
                assert observation['gaze_time_seconds']<=e['frames'][-1]['geometry']['telemetry_time_seconds']+1e-6
        assert maps.shape==coverage.shape==(3,180,360)
        assert owner.shape==(360,720) and owner.max()<len(e['frames'])
        assert np.all((owner>=-1)&(owner<len(e['frames'])))
    assert abs(solid_angle_weights(360,180).sum()-4*np.pi)<1e-10
    calls=[];responses=[];cards=0
    for folder in [root,root/'layout_v2']:
        manifest=read(folder/'manifest.json');assert sha(folder/'manifest.json')==read(folder/'protocol_lock.json')['manifest_sha256']
        for e in manifest['events']:
            for c,file in e['conditions'].items():
                assert sha(folder/file['path'])==file['sha256'];cards+=1
                row=read(folder/'responses'/e['id']/f'{c}.json');assert row['status']=='COMPLETE'
                request=read(row['request_path']);assert request['image_sha256']==[file['sha256']]
                assert request['prompt']==manifest['prompt'] and request['do_sample'] is False
                assert request['max_new_tokens']==120
                assert set(row['parsed'])=={'early','late'}
                assert request['processor_audit']['expanded_image_token_count']==1024
                assert request['visual_tensor_trace']['input_consumption_verified']
                assert row['expected']==file['expected_visual_priority']
                assert row['pair_correct']==(row['parsed']==row['expected'])
                responses.append(row)
        audit=next((folder/'model_cache').glob('*/call_audit.jsonl'))
        calls.extend(json.loads(line) for line in audit.read_text().splitlines())
    assert len(calls)==104 and not any(row['cache_hit'] for row in calls)
    # Ensure each clickable atlas pixel's run-length lookup reproduces stored ownership.
    demo=read(root/'head_motion_demo_data.json');rle_pixels=0
    for e in demo['events']:
        owner=np.load(root/'events'/e['event']/'volume.npz')['rgb_source_index']
        decoded=[]
        for row in e['owner_rle']:
            pixels=[];last=0
            for end,index in row:pixels.extend([index]*(end-last));last=end
            decoded.append(pixels)
        np.testing.assert_array_equal(np.array(decoded),owner);rle_pixels+=owner.size
        for f in e['frames']:assert (root/f['map_path']).exists() and sha(root/f['source_path'])==f['source_sha256']
    js=[]
    if jscheck_path:
        sys.path.insert(0,str(jscheck_path));import quickjs
        js=[check_js(folder,page,quickjs) for folder,page in [(root,'index.html'),(root,'head_motion_demo.html'),(root/'layout_v2','index.html')]]
    result={'model_calls':len(calls),'model_responses_complete':dict(Counter(row['status'] for row in responses)),
        'input_sha256_checked':cards,'all_1024_visual_tokens':True,'all_vision_consumption_verified':True,
        'source_frames_checked':frames,'supported_events':len(m['events']),'clock_fallback_events':len(m['fallback_events']),
        'frozen_protocol_and_source_snapshots_verified':True,'spherical_mass_and_time_checks_passed':True,
        'rgb_provenance_rle_pixels_verified':rle_pixels,'viewer_checks':js,
        'candidate_labels_are_object_ids':False,'ground_truth_loaded':False,'final_referent_evaluator_run':False,
        'tests_log':'tests.log','full_repository_suite_rerun':False}
    save(root/'verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--jscheck-path',type=Path)
    a=p.parse_args();verify(a.output.resolve(),a.jscheck_path)
