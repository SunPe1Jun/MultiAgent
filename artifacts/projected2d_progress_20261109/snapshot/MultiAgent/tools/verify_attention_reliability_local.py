#!/usr/bin/env python3
"""Reproduce adaptive local heat inputs and verify their complete model records."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from MultiAgent.attention_reliability import injected_history, pair_attention
from MultiAgent.heatmap_temporal_pilot import save, sha
from MultiAgent.perspective_attention_study import read
from tools.run_attention_reliability_local import CUE_PROMPT, REAL_PROMPT, render_local
from tools.run_attention_reliability_study import OUTPUT, valid_response
from tools.verify_attention_region_study import Scripts


def verify(parent, jscheck_path=None):
    output = parent / 'local_layout_v2'
    manifest = read(output / 'manifest.json')
    original = read(parent / 'manifest.json')
    assert sha(parent / 'manifest.json') == manifest['parent_manifest_sha256']
    lock = read(output / 'protocol_lock.json')
    assert sha(output / 'manifest.json') == lock['manifest_sha256']
    assert lock['before_v2_model'] and manifest['adaptive_regression_only']
    assert sha(output / 'source.py') == manifest['code_sha256']
    cases = {case['id']: case for case in original['cases']}
    original_requests = {(r['case_id'], r['condition'], r['permutation']): r
                         for r in original['requests'] if r['layout'] == 'SEPARATE'}
    responses = {p.stem: read(p) for p in (output / 'responses').glob('*.json')}
    assert set(responses) == {r['id'] for r in manifest['requests']}
    arrays, source_rgb, fixed_rgb, prompt_by_case = {}, {}, {}, {}
    counts = Counter()
    for r in manifest['requests']:
        case_id = r['case_id']
        case = cases[case_id]
        base = original_requests[(case_id, r['condition'], r['permutation'])]
        assert r['labels_by_physical_region'] == base['labels_by_physical_region']
        assert r['expected'] == base['expected']
        assert r['family'] == case['family'] and r['event'] == case['event']
        if case_id not in arrays:
            assert sha(parent / case['pair_npz_path']) == case['pair_npz_sha256']
            with np.load(parent / case['pair_npz_path']) as data:
                arrays[case_id] = {k: data[k] for k in data.files}
            assert sha(case['source_path']) == case['source_sha256']
            with Image.open(case['source_path']) as rgb:
                source_rgb[case_id] = rgb.convert('RGB')
        data = arrays[case_id]
        masks, maps = data['masks'], data['raw_maps']
        if case['family'] == 'injected_cue':
            mass = (np.zeros((3, 2)) if r['condition'] == 'NO_HEAT' else
                    injected_history(masks, reverse=r['condition'] == 'REVERSED'))
            residual = np.zeros_like(maps)
            prompt = CUE_PROMPT
        else:
            mass, residual, _ = pair_attention(maps, masks, r['condition'])
            prompt = REAL_PROMPT.replace('__CATEGORY__', case['category']).replace(
                '__INSTRUCTION__', case['instruction'])
        assert r['prompt'] == prompt
        if case_id in prompt_by_case:
            assert r['prompt'] == prompt_by_case[case_id]
        prompt_by_case[case_id] = r['prompt']
        image, metadata = render_local(source_rgb[case_id], case['objects'], masks,
                                       mass, residual, r['labels_by_physical_region'])
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        assert hashlib.sha256(buffer.getvalue()).hexdigest() == r['sha256']
        assert sha(output / r['path']) == r['sha256']
        assert metadata == r['display_support']
        # Only the six heat panels may differ across the three conditions.
        pixels = np.array(image)
        for x in (370, 688):
            for y in (430, 610, 790):
                pixels[y:y + 152, x:x + 288] = 0
        key = (case_id, r['permutation'])
        if key in fixed_rgb:
            np.testing.assert_array_equal(pixels, fixed_rgb[key])
        else:
            fixed_rgb[key] = pixels
        response = responses[r['id']]
        raw = read(response['request_path'])
        assert raw['prompt'] == r['prompt']
        assert raw['image_sha256'] == [r['sha256']]
        assert raw['model'] == '/ai/data/Qwen3-VL-8B-Instruct'
        assert raw['do_sample'] is False and raw['max_new_tokens'] == 100
        assert raw['processor_audit']['expanded_image_token_count'] == 1024
        assert raw['visual_tensor_trace']['input_consumption_verified']
        assert response['vision_verified'] and response['parsed'] == raw['parsed']
        assert response['schema_valid'] == valid_response(response['parsed'], r['family'])
        assert response['schema_valid']
        assert response['matches_label'] == (response['parsed'] == r['expected'])
        physical = {label: i for i, label in enumerate(r['labels_by_physical_region'])}
        assert response['physical_prediction'] == {
            k: physical.get(value, value) for k, value in response['parsed'].items()}
        counts[r['family']] += 1
    audits = []
    for path in (output / 'model_cache').glob('*/call_audit.jsonl'):
        audits.extend(json.loads(line) for line in path.read_text().splitlines())
    assert len(audits) == len(responses) and not any(a['cache_hit'] for a in audits)
    viewer = None
    if jscheck_path:
        sys.path.insert(0, str(jscheck_path))
        import quickjs
        dom = """class E{constructor(t='div'){this.tag=t;this.children=[];this.value=''}
append(...xs){for(const x of xs){this.children.push(x);if(this.tag==='select'&&!this.value)this.value=String(x.value)}}
replaceChildren(){this.children=[]}}
const roots={};const document={querySelector(s){return roots[s]||(roots[s]=new E(['#family','#event'].includes(s)?'select':s))},createElement(t){return new E(t)}};
document.querySelector('#family').value='injected_cue';"""
        parser = Scripts()
        parser.feed((output / 'index.html').read_text())
        ctx = quickjs.Context()
        ctx.eval(dom)
        for script in parser.scripts:
            ctx.eval(script)
        rendered = ctx.eval("""let n=0;for(const f of ['injected_cue','real_center_proxy']){
family.value=f;for(const event of [...new Set(d.m.requests.map(r=>r.event))]){
q.value=event;render();n+=document.querySelector('main').children.length}}n""")
        assert rendered == len(manifest['requests'])
        viewer = {'all_condition_cards_rendered': rendered, 'full_browser': False}
    result = {
        'frozen_v2_manifest_and_source_verified': True,
        'adaptive_regression_only': True,
        'all_input_images_byte_reproduced': len(manifest['requests']),
        'only_six_heat_panels_change_within_controls': True,
        'identical_prompts_within_conditions_and_id_permutations': True,
        'model_calls_verified': len(responses),
        'calls_by_family': dict(counts),
        'all_schema_valid': True,
        'all_1024_visual_tokens_and_consumption_verified': True,
        'viewer_logic': viewer,
        'independent_object_gt_available': False,
        'unseen_validation': False,
        'final_referent_evaluator_run': False,
    }
    save(output / 'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=OUTPUT)
    parser.add_argument('--jscheck-path', type=Path)
    args = parser.parse_args()
    verify(args.parent.resolve(), args.jscheck_path)
