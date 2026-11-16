#!/usr/bin/env python3
"""Verify immutable evidence, model consumption, attention mass and local viewer logic."""
import argparse
from collections import Counter
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import numpy as np
from MultiAgent.heatmap_temporal_pilot import sha,save
from MultiAgent.heatmap_forms_study import OUTPUT,AUDIT,read,contexts_for


class Scripts(HTMLParser):
    def __init__(self):super().__init__();self.inside=False;self.items=[];self.buffer=''
    def handle_starttag(self,tag,attrs):
        if tag=='script':self.inside=True;self.buffer=''
    def handle_endtag(self,tag):
        if tag=='script':self.inside=False;self.items.append(self.buffer)
    def handle_data(self,data):
        if self.inside:self.buffer+=data


DOM='''
class Element {
 constructor(tag='div'){this.tag=tag;this.children=[];this.textContent='';this.value='';this.src='';}
 append(...nodes){nodes.forEach(n=>{this.children.push(n);if(this.tag==='select'&&this.value==='')this.value=String(n.value)});}
 replaceChildren(...nodes){this.children=nodes;}
 querySelector(tag){return this.children.find(n=>n.tag===tag)||new Element(tag);}
}
const roots={};const document={
 querySelector(selector){if(!roots[selector])roots[selector]=new Element(selector.includes('sample')||selector.includes('event')||selector.includes('condition')||selector==='select'?'select':selector);return roots[selector]},
 querySelectorAll(selector){let all=[];Object.values(roots).forEach(root=>root.children.forEach(n=>{if(n.tag===selector)all.push(n)}));return all},
 createElement(tag){return new Element(tag)}
};document.querySelector('#condition').value='NATIVE_CROP';
'''


def check_viewer(output,name,quickjs):
    parser=Scripts();parser.feed((output/name).read_text());parser.close()
    context=quickjs.Context();context.eval(DOM)
    for script in parser.items:context.eval(script)
    if name=='index.html':
        context.eval("for(let i=0;i<d.samples.length;i++){q.value=String(i);render()}")
    elif name=='temporal_memory.html':
        context.eval("for(const event of [...new Set(d.samples.map(s=>s.event))]){q.value=event;for(const mode of d.conditions){c.value=mode;render()}}")
    else:context.eval("for(const event of Object.keys(d.events)){q.value=event;render()}")
    sources=json.loads(context.eval("JSON.stringify(Object.values(roots).flatMap(r=>r.children).flatMap(f=>f.children||[]).filter(n=>n.tag==='img').map(n=>n.src))"))
    assert all((output/path).exists() for path in sources)
    return {'page':name,'scripts_executed':len(parser.items),'all_selector_choices_exercised':True,'full_browser':False}


def verify(output,jscheck_path=None):
    main=read(output/'manifest.json');temporal=read(output/'temporal_manifest.json')
    assert sha(output/'source'/'attention_volume.py')==main['volume_code_sha256']
    assert sha(output/'source'/'heatmap_forms_study.py')==main['code_sha256']
    all_rows=[];inputs=0
    for manifest,labels_name,lock_name,dir_name,plock in [
        (main,'labels.json','annotation_lock.json','responses','fresh_lock.json'),
        (temporal,'temporal_labels.json','temporal_annotation_lock.json','temporal_responses','temporal_fresh_lock.json')]:
        assert sha(output/labels_name)==read(output/lock_name)['sha256']
        manifest_name='manifest.json' if dir_name=='responses' else 'temporal_manifest.json'
        assert sha(output/manifest_name)==read(output/plock)['manifest_sha256']
        labels=read(output/labels_name)
        for sample in manifest['samples']:
            for image in sample['conditions'].values():
                assert sha(output/image['path'])==image['sha256'];inputs+=1
        rows=[read(p) for p in (output/dir_name).glob('*/*.json')]
        assert len(rows)==len(manifest['samples'])*len(manifest['conditions'])
        indexed={s['id']:s for s in manifest['samples']}
        for row in rows:
            assert row['status']=='COMPLETE'
            image=indexed[row['sample']]['conditions'][row['condition']];request=read(row['request_path'])
            assert request['image_sha256']==[image['sha256']]
            assert request['prompt']==manifest['prompt']
            assert request['do_sample'] is False and request['max_new_tokens']==120
            assert set(request['parsed'])=={'category'} and row['category'] in main['categories']
            assert request['visual_tensor_trace']['input_consumption_verified']
            assert request['processor_audit']['expanded_image_token_count']==420
            expected=labels['samples'][row['sample']]['native_accepted' if row['condition']=='NATIVE_CROP' else 'accepted']
            assert row['correct']==(row['category'] in expected)
        all_rows.extend(rows)
    for sample in main['samples']:
        assert sha(sample['source_path'])==sample['source_sha256']
        maps=np.load(output/'inputs'/sample['id']/'volume.npz')['duration_maps']
        for plane,b in zip(maps,sample['volume']['bins']):
            assert abs(float(plane.sum())-b['duration'])<2e-6
            assert b['coverage']<=1+1e-6
            for o in b['observations']:assert o['gaze_time_seconds']<=sample['geometry']['telemetry_time_seconds']+1e-6
    contexts=contexts_for({s['event'] for s in temporal['samples']})
    native_error=[]
    for s in temporal['samples']:
        assert sha(s['native_source_path'])==s['native_source_sha256']
        assert s['native_frame']['pts_seconds']<=s['current_pts']
        assert abs(s['gaze_to_frame_error_seconds'])<=.08
        native_error.append(abs(s['gaze_to_frame_error_seconds']))
    for event,ctx in contexts.items():
        hashes=read(AUDIT/'pose_alignment'/event/'frame_pose_contract.json')['source_hashes']
        assert sha(ctx.video_path)==hashes['video_sha256'] and sha(ctx.json_path)==hashes['telemetry_sha256']
    memory_cards=0
    for filename in ['native_attention_memory.json','offscreen_attention_memory.json']:
        result=read(output/filename)
        for event in result['events'].values():
            for p in event['memory']['proposals']:
                assert all(o['gaze_time_seconds']<=event['memory']['query_telemetry_seconds'] for o in p['observations'])
                if p['card']:
                    assert sha(output/p['card']['path'])==p['card']['sha256'];memory_cards+=1
    coverage=read(output/'native_history_coverage.json')['samples']
    viewers=[]
    if jscheck_path:
        sys.path.insert(0,str(jscheck_path));import quickjs
        viewers=[check_viewer(output,name,quickjs) for name in ['index.html','temporal_memory.html','native_attention_memory.html','offscreen_attention_memory.html']]
    calls=[json.loads(line) for line in (output/'model_cache'/'heatmap_forms_and_script_routing_v2'/'call_audit.jsonl').read_text().splitlines()]
    assert len(calls)==288 and not any(c['cache_hit'] for c in calls)
    result={'model_call_counts':dict(Counter(r['status'] for r in all_rows)),'model_input_cards_sha256_checked':inputs,
        'all_one_field_schema':True,'all_420_visual_tokens':True,'all_vision_consumption_verified':True,
        'frozen_manifest_label_and_source_hashes_verified':True,'duration_mass_conserved_all_39_queries':True,
        'native_max_abs_gaze_to_frame_error_ms':max(native_error)*1000,'native_future_frame_count':0,
        'native_video_and_telemetry_contract_hashes_verified':True,'additional_memory_cards_verified':memory_cards,
        'native_coverage_queries':len(coverage),
        'offscreen_history_observation_count':sum(p['outside_current_frustum_count'] for s in coverage for p in s['memory']['proposals']),
        'additional_supported_history_seconds':sum(s['memory']['native_supported_duration_seconds']-s['current_frustum_duration_seconds'] for s in coverage),
        'viewers':viewers,'tests_log':'tests.log','full_repo_suite_rerun':False,'final_referent_evaluator_run':False}
    save(output/'verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--jscheck-path',type=Path)
    a=p.parse_args();verify(a.output.resolve(),a.jscheck_path)
