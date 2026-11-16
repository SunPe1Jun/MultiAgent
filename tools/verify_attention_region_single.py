#!/usr/bin/env python3
"""Verify single-region heat conditions, actual model inputs and development freeze."""
import argparse
from collections import Counter
import io
import hashlib
import json
from pathlib import Path
import sys
from MultiAgent.heatmap_temporal_pilot import sha, save
from MultiAgent.perspective_attention_study import read
from tools.run_attention_region_study import OUTPUT
from tools.run_attention_region_single import single_image
from tools.verify_attention_region_study import Scripts


def verify(parent, complete=False, jscheck_path=None):
    output = parent / 'single_region'; m = read(output / 'manifest.json'); base = read(parent / 'manifest.json')
    assert sha(parent / 'manifest.json') == m['parent_manifest_sha256']
    assert sha(output / 'manifest.json') == read(output / 'protocol_lock.json')['manifest_sha256']
    assert read(output / 'protocol_lock.json')['before_model']
    assert sha(output / 'source.py') == m['code_sha256']
    events = {e['id']: e for e in base['events']}
    rows = {(e['id'], r['id']): r for e in base['events'] for r in e['regions']}
    for event_id, event in events.items():
        ready = sorted([r for r in event['regions'] if r['status'] == 'RGB_EVIDENCE_READY'],
                       key=lambda r: (-r['mass_seconds'], r['component'], r['phase']))
        expected = [next(r for r in ready if r['phase'] == k) for k in (0, 2)]
        expected.append(next(r for r in ready if r['id'] not in {v['id'] for v in expected}))
        chosen = [s['region_id'] for s in m['selected_regions'] if s['event_id'] == event_id]
        assert chosen == [r['id'] for r in expected]
    responses = {p.stem: read(p) for p in (output / 'responses').glob('*.json')}
    if complete: assert len(responses) == len(m['requests'])
    image_count = 0; model_count = 0
    for request in m['requests']:
        path = output / request['image']; assert sha(path) == request['sha256']
        region = rows[(request['event_id'], request['region_id'])]
        image = single_image(parent, region, request['condition']); buffer = io.BytesIO(); image.save(buffer, format='PNG')
        assert hashlib.sha256(buffer.getvalue()).hexdigest() == request['sha256']; image_count += 1
        if request['id'] not in responses: continue
        response = responses[request['id']]; raw = read(response['request_path'])
        assert response['schema_valid'] and response['vision_verified']
        assert raw['prompt'] == request['prompt'] and raw['image_sha256'] == [request['sha256']]
        assert raw['do_sample'] is False and raw['max_new_tokens'] == 120
        assert raw['visual_tensor_trace']['input_consumption_verified']
        assert raw['processor_audit']['expanded_image_token_count'] == 1024
        assert raw['parsed'] == response['parsed']
        assert set(response['parsed']) == {'visible', 'relevance'}
        assert response['parsed']['relevance'] in {'plausible', 'unclear', 'unrelated'}
        model_count += 1
    records = []
    for path in (output / 'model_cache').glob('*/call_audit.jsonl'):
        records.extend(json.loads(line) for line in path.read_text().splitlines())
    assert len(records) == len(responses) and not any(r['cache_hit'] for r in records)
    development = read(output / 'development_frozen.json')
    assert development['manifest_sha256'] == sha(output / 'manifest.json') and development['check_not_started']
    assert development['completed_development_calls'] == len([r for r in m['requests'] if r['split'] == 'development'])
    if complete:
        started = read(output / 'check_started.json')
        assert started['development_frozen_sha256'] == sha(output / 'development_frozen.json')
        assert started['manifest_sha256'] == sha(output / 'manifest.json')
    js = None
    if jscheck_path:
        sys.path.insert(0, str(jscheck_path)); import quickjs
        dom = '''class Element{constructor(tag='div'){this.tag=tag;this.children=[];this.value='';this.src='';this.textContent=''}append(...items){for(const item of items){this.children.push(item);if(this.tag==='select'&&this.value==='')this.value=String(item.value)}}replaceChildren(...items){this.children=items}}const roots={};const document={querySelector(s){if(!roots[s])roots[s]=new Element(['#event','#condition'].includes(s)?'select':s);return roots[s]},createElement(t){return new Element(t)}};document.querySelector('#condition').value='TRUE_HEAT';'''
        parser = Scripts(); parser.feed((output / 'index.html').read_text()); context = quickjs.Context(); context.eval(dom)
        for script in parser.scripts: context.eval(script)
        count = context.eval("let n=0;for(const event of [...new Set(d.m.requests.map(r=>r.event))]){q.value=event;for(const condition of ['TRUE_HEAT','NO_HEAT','REVERSE_TIME']){c.value=condition;render();n+=document.querySelector('main').children.length}}n")
        assert count == len(m['requests'])
        js = {'event_condition_combinations': 39, 'images': count, 'full_browser': False}
    result = {'protocol_parent_snapshot_verified': True, 'selected_regions_reproduced': len(m['selected_regions']),
        'images_byte_reproduced': image_count, 'model_inputs_and_schema_verified': model_count,
        'all_1024_visual_tokens_and_vision_consumption': True, 'development_before_check_verified': complete,
        'viewer_logic': js, 'object_or_relevance_gt_available': False, 'regression_only': True}
    save(output / 'verification.json', result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--parent', type=Path, default=OUTPUT)
    p.add_argument('--complete', action='store_true'); p.add_argument('--jscheck-path', type=Path)
    a = p.parse_args(); verify(a.parent.resolve(), a.complete, a.jscheck_path)
