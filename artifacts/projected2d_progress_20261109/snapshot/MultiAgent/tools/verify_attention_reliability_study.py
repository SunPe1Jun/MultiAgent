#!/usr/bin/env python3
"""Reproduce frozen paired attention inputs, labels, counterfactuals and statistics."""
import argparse
from collections import Counter
from html.parser import HTMLParser
import io
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from MultiAgent.attention_reliability import injected_history,pair_attention,graphical_pair_heat
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.perspective_attention_study import ROOT,read
from tools.run_attention_reliability_study import OUTPUT,SOURCE,render,control_cases,real_cases,valid_response
from tools.verify_attention_region_study import Scripts


def verify(output,complete=False,jscheck_path=None):
    m=read(output/'manifest.json');assert sha(output/'manifest.json')==read(output/'protocol_lock.json')['manifest_sha256']
    assert read(output/'protocol_lock.json')['before_any_inference']
    assert sha(SOURCE/'manifest.json')==m['source_manifest_sha256']
    assert sha(SOURCE/'referent_center_audit.json')==m['source_center_audit_sha256']
    assert sha(output/'source/run_attention_reliability_study.py')==m['code_sha256']
    assert sha(output/'source/attention_reliability.py')==m['core_sha256']
    parent=read(SOURCE/'manifest.json');controls,c_excluded=control_cases(parent);real,r_excluded=real_cases(parent)
    assert m['excluded_control_events']==c_excluded and m['excluded_real_frame_categories']==r_excluded
    regenerated=controls+real;assert len(regenerated)==len(m['cases'])
    case_by_id={c['id']:c for c in m['cases']};arrays={};rgb_by_id={};mass_cache={}
    for c,base in zip(m['cases'],regenerated):
        for key,value in base.items():assert c[key]==value
        assert sha(c['source_path'])==c['source_sha256'] and sha(c['source_npz_path'])==c['source_npz_sha256']
        assert sha(output/c['pair_npz_path'])==c['pair_npz_sha256']
        d=np.load(output/c['pair_npz_path']);original=np.load(c['source_npz_path'])
        np.testing.assert_array_equal(d['masks'],original['masks'][c['indices']]);np.testing.assert_array_equal(d['raw_maps'],original['maps'])
        arrays[c['id']]={k:d[k] for k in d.files}
        with Image.open(c['source_path']) as im:rgb_by_id[c['id']]=im.convert('RGB')
    responses={p.stem:read(p) for p in (output/'responses').glob('*.json')}
    if complete:assert len(responses)==len(m['requests'])
    assert set(responses)<={r['id'] for r in m['requests']}
    fixed={};cards=0;counts=Counter();invalid=[]
    for r in m['requests']:
        case=case_by_id[r['case_id']];d=arrays[r['case_id']];masks=d['masks']
        key=(r['case_id'],r['condition'])
        if key not in mass_cache:
            if r['family']=='injected_cue':
                mass=np.zeros((3,2)) if r['condition']=='NO_HEAT' else injected_history(masks,reverse=r['condition']=='REVERSED')
                residual=np.zeros_like(d['raw_maps'])
            else:mass,residual,_=pair_attention(d['raw_maps'],masks,r['condition'])
            mass_cache[key]=(mass,residual)
        mass,residual=mass_cache[key]
        labels=r['labels_by_physical_region']
        if r['family']=='injected_cue':
            expected={'early':'none','late':'none'} if r['condition']=='NO_HEAT' else {
                'early':labels[int(mass[0].argmax())],'late':labels[int(mass[2].argmax())]}
        else:expected={'selected':labels[0]}
        assert r['expected']==expected
        image,meta=render(rgb_by_id[r['case_id']],case['objects'],masks,mass,residual,labels,r['layout'])
        buffer=io.BytesIO();image.save(buffer,format='PNG');assert hashlib.sha256(buffer.getvalue()).hexdigest()==r['sha256']
        assert sha(output/r['path'])==r['sha256'] and meta==r['display_support'];cards+=1
        # Heat manipulations cannot change RGB/context/details/printed source IDs.
        pixels=np.array(image);pixels[32:975,650:960]=0
        fixedkey=(r['case_id'],r['permutation'],r['layout'])
        if fixedkey in fixed:np.testing.assert_array_equal(pixels,fixed[fixedkey])
        else:fixed[fixedkey]=pixels
        if r['id'] not in responses:continue
        a=responses[r['id']];raw=read(a['request_path'])
        assert raw['image_sha256']==[r['sha256']] and raw['prompt']==r['prompt']
        assert raw['model']=='/ai/data/Qwen3-VL-8B-Instruct' and raw['do_sample'] is False and raw['max_new_tokens']==100
        assert raw['processor_audit']['expanded_image_token_count']==1024
        assert raw['visual_tensor_trace']['input_consumption_verified'] and a['vision_verified']
        assert raw['parsed']==a['parsed'] and a['schema_valid']==valid_response(a['parsed'],r['family'])
        assert a['matches_label']==(a['schema_valid'] and a['parsed']==r['expected'])
        if not a['schema_valid']:invalid.append(r['id'])
        counts[r['family']]+=1
    audits=[]
    for p in (output/'model_cache').glob('*/call_audit.jsonl'):
        audits.extend(json.loads(s) for s in p.read_text().splitlines())
    if complete:assert len(audits)==len(responses) and not any(r['cache_hit'] for r in audits)
    js=None
    if jscheck_path:
        sys.path.insert(0,str(jscheck_path));import quickjs
        dom="class E{constructor(t='div'){this.tag=t;this.children=[];this.value=''}append(...xs){for(const x of xs){this.children.push(x);if(this.tag==='select'&&!this.value)this.value=String(x.value)}}replaceChildren(){this.children=[]}}const roots={};const document={querySelector(s){return roots[s]||(roots[s]=new E(['#family','#event','#layout'].includes(s)?'select':s))},createElement(t){return new E(t)}};document.querySelector('#family').value='injected_cue';document.querySelector('#layout').value='SEPARATE';"
        parser=Scripts();parser.feed((output/'index.html').read_text());ctx=quickjs.Context();ctx.eval(dom)
        for script in parser.scripts:ctx.eval(script)
        n=ctx.eval("let n=0;for(const f of ['injected_cue','real_center_proxy']){family.value=f;for(const event of [...new Set(d.m.cases.map(c=>c.event))]){q.value=event;for(const l of d.m.layouts){layout.value=l;render();n+=document.querySelector('main').children.length}}}n")
        assert n==len(m['requests'])
        js={'all_condition_cards_rendered':n,'full_browser':False}
    result={'frozen_protocol_and_source_snapshots_verified':True,'injected_control_cases_reproduced':len(controls),
        'conditional_real_proxy_cases_reproduced':len(real),'all_input_images_byte_reproduced':cards,
        'same_rgb_context_details_within_heat_controls':True,'scripted_vs_weak_label_provenance_separated':True,
        'model_calls_verified':len(responses),'calls_by_family':dict(counts),'schema_failures':invalid,
        'all_1024_visual_tokens_and_consumption_verified':True,'viewer_logic':js,
        'independent_object_gt_available':False,'unseen_validation':False,'final_referent_evaluator_run':False}
    save(output/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--complete',action='store_true');p.add_argument('--jscheck-path',type=Path)
    a=p.parse_args();verify(a.output.resolve(),a.complete,a.jscheck_path)
