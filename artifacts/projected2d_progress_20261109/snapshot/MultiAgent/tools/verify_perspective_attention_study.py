#!/usr/bin/env python3
"""Audit frozen visual routing inputs, model consumption and historical RGB links."""
import argparse
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from MultiAgent.heatmap_temporal_pilot import sha,save
from MultiAgent.perspective_attention_study import OUTPUT,SOURCE,read
from MultiAgent.perspective_attention import cube_owner,face_mass_audit,sample_face_density


class Scripts(HTMLParser):
    def __init__(self):super().__init__();self.scripts=[];self.inside=False;self.current=''
    def handle_starttag(self,tag,attrs):
        if tag=='script':self.inside=True;self.current=''
    def handle_endtag(self,tag):
        if tag=='script':self.scripts.append(self.current);self.inside=False
    def handle_data(self,data):
        if self.inside:self.current+=data


DOM='''class Element{constructor(tag='div'){this.tag=tag;this.children=[];this.value='';this.src='';this.textContent=''}append(...items){for(const item of items){this.children.push(item);if(this.tag==='select'&&this.value==='')this.value=String(item.value)}}replaceChildren(...items){this.children=items}querySelector(tag){return this.children.find(c=>c.tag===tag)||new Element(tag)}}const roots={};const document={querySelector(selector){if(!roots[selector])roots[selector]=new Element(['#event','#condition','#mode','#method'].includes(selector)?'select':selector);return roots[selector]},createElement(tag){return new Element(tag)}};document.querySelector('#mode').value='joint';document.querySelector('#method').value='phase_grouped';'''


def check_js(root,page,quickjs):
    parser=Scripts();parser.feed((root/page).read_text());context=quickjs.Context();context.eval(DOM)
    for script in parser.scripts:context.eval(script)
    if page=='index.html' and root.name!='multi_region':
        context.eval("for(let i=0;i<m.events.length;i++){q.value=String(i);for(const condition of m.conditions){c.value=condition;for(const value of ['joint','phase','sources']){mode.value=value;render();}}}")
    elif page=='routed_visual_evidence.html':
        context.eval("for(const event of [...new Set(d.rows.map(r=>r.event))]){q.value=event;for(const value of ['joint','phase_grouped','visual_shortlist']){method.value=value;render();}}")
    paths=json.loads(context.eval("JSON.stringify(Object.values(roots).flatMap(r=>r.children).flatMap(r=>r.children).filter(r=>r.tag==='img').map(r=>r.src))"))
    for path in paths:assert (root/path).exists(),str(root/path)
    return {'page':str(root/page),'script_blocks':len(parser.scripts),'selector_logic_exercised':True,'full_browser':False}


def verify(output,jscheck_path=None):
    m=read(output/'manifest.json');group=read(output/'phase_layout_v2/manifest.json');multi=read(output/'multi_region/manifest.json')
    assert sha(output/'source/perspective_attention.py')==m['renderer_sha256']
    assert sha(output/'source/perspective_attention_study.py')==m['code_sha256']
    assert sha(SOURCE/'manifest.json')==m['source_manifest_sha256']
    assert sha(output/'manifest.json')==read(output/'protocol_lock.json')['manifest_sha256']
    assert sha(output/'phase_layout_v2/manifest.json')==read(output/'phase_layout_v2/protocol_lock.json')['manifest_sha256']
    assert read(output/'phase_layout_v2/iteration_protocol.json')['check_has_not_started']
    assert read(output/'development_frozen.json')['check_not_previously_run']
    assert sha(output/'development_frozen.json')==read(output/'check_started.json')['development_sha256']
    assert sha(output/'phase_layout_v2/development_frozen.json')==read(output/'phase_layout_v2/check_started.json')['development_sha256']
    assert sha(output/'multi_region/manifest.json')==read(output/'multi_region/protocol_lock.json')['manifest_sha256']
    cards=0;responses=[];calls=[]
    for root,manifest,kind in [(output,m,'initial'),(output/'phase_layout_v2',group,'grouped'),(output/'multi_region',multi,'shortlist')]:
        indexed={r['id']:r for r in manifest['requests']}
        for r in manifest['requests']:
            path=Path(r['path']) if kind=='shortlist' else root/r['path']
            assert sha(path)==r.get('file_sha256',r['sha256']);cards+=1
        rows=[read(p) for p in (root/'responses').glob('*.json')]
        assert len(rows)==(100 if kind=='initial' else 104)
        for row in rows:
            assert row['status']=='COMPLETE' and row['schema_valid']
            r=indexed[row['id']];request=read(row['request_path'])
            prompt=manifest['prompt'] if kind=='shortlist' else manifest['prompts'][r['prompt_key']]
            assert request['prompt']==prompt
            assert request['image_sha256']==[r.get('file_sha256',r['sha256'])]
            assert request['do_sample'] is False and request['max_new_tokens']==120
            assert request['model']=='/ai/data/Qwen3-VL-8B-Instruct'
            assert request['processor_audit']['expanded_image_token_count']==1024
            assert request['visual_tensor_trace']['input_consumption_verified']
            if kind!='shortlist':assert row['expected']==r['expected'] and row['correct']==(row['parsed']==r['expected'])
            else:
                values=row['parsed']['regions'];assert len(values)<=2 and len(values)==len(set(values))
                assert row['expected_top1']==r['expected_top1']
                assert row['unsupported_proposal_count']==sum(v not in r['half_peak_admissible_labels'] for v in values)
            responses.append(row)
        audit=next((root/'model_cache').glob('*/call_audit.jsonl'))
        calls.extend(json.loads(line) for line in audit.read_text().splitlines())
    assert len(calls)==308 and not any(v['cache_hit'] for v in calls)
    sphere=read(SOURCE/'manifest.json');events={e['id']:e for e in sphere['events']};sources=0
    for e in m['events']:
        data=np.load(SOURCE/'events'/e['event']/'volume.npz');stored=np.load(output/'events'/e['event']/'perspective_density.npz')['density']
        assert len(e['selected_faces'])==6 and set(e['selected_faces'])==set(range(6))
        audit=face_mass_audit(data['duration_maps'],e['selected_faces'])
        np.testing.assert_allclose(audit['retained_duration_fraction'],1.,atol=2e-6)
        for j,face in enumerate(e['selected_faces']):
            np.testing.assert_array_equal(sample_face_density(data['duration_maps'],face,224),stored[:,j])
        for region in e['native_sources_by_region']:
            for s in region:
                assert sha(s['source_path'])==s['source_sha256']
                assert sha(output/s['rendered_path'])==s['rendered_sha256']
                assert s['selection']['source_telemetry_seconds']<=s['selection']['window'][1]+1e-6
                sources+=1
    routes=read(output/'routed_visual_evidence.json');max_error=0.;selected_consistent=0
    for row in routes['rows']:
        path=Path(row['model_response_path']);assert sha(path)==row['model_response_sha256']
        parsed=read(path)['parsed'];base=events[row['event_id']]
        labels=(parsed['regions'] if row['method']=='visual_shortlist' else [parsed['selected'] if row['method']=='phase_grouped' else parsed['early' if row['phase']==0 else 'late']])
        assert row['selected_label'] in labels
        if row['status']!='RGB_EVIDENCE_READY':continue
        obs=row['representative_observation'];ref=Rotation.from_quat(base['canonical_pose']['rotation_xyzw'])
        assert int(cube_owner(ref.inv().apply(obs['world_ray'])))==row['selected_face'];selected_consistent+=1
        assert row['source_frame']['pts_seconds']<=row['window'][1]+row['selected_clock_offset_seconds']+1e-6
        assert abs(row['native_query_telemetry_seconds']+row['selected_clock_offset_seconds']-row['source_frame']['pts_seconds'])<1e-6
        assert sha(output/row['native_source_path'])==row['native_source_sha256']
        assert sha(output/row['evidence_card']['path'])==row['evidence_card']['sha256']
        assert row['object_identity'] is None and row['semantic_model_run'] is False
        assert abs(row['gaze_frame_error_seconds'])<=.08
        max_error=max(max_error,abs(row['gaze_frame_error_seconds']))
    js=[]
    if jscheck_path:
        sys.path.insert(0,str(jscheck_path));import quickjs
        js=[check_js(root,page,quickjs) for root,page in [(output,'index.html'),(output/'phase_layout_v2','index.html'),(output/'multi_region','index.html'),(output,'routed_visual_evidence.html')]]
    result={'model_call_counts':dict(Counter(r['status'] for r in responses)),'model_input_hashes_checked':cards,
        'all_1024_visual_tokens':True,'all_vision_consumption_verified':True,'all_model_schemas_valid':True,
        'fixed_protocols_and_source_hashes_verified':True,'development_before_check_verified':True,
        'all_six_faces_mass_partition_verified':True,'perspective_density_reproduced':True,
        'native_context_sources_verified':sources,'routed_status':routes['status_counts'],
        'model_selected_face_never_replaced_in_routes':selected_consistent,'max_native_frame_error_ms':max_error*1000,
        'viewer_checks':js,'new_semantic_model_calls':0,'final_referent_evaluator_run':False,
        'check_is_independent_unseen_data':False,'tests_log':'tests.log','full_repository_suite_rerun':False}
    save(output/'verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--jscheck-path',type=Path)
    a=p.parse_args();verify(a.output.resolve(),a.jscheck_path)
