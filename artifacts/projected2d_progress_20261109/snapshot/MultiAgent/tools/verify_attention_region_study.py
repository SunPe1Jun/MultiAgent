#!/usr/bin/env python3
"""Check connected attention geometry, frozen visuals, native PTS and screening provenance."""
import argparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from MultiAgent.attention_regions import extract_regions, region_observations, representative_observation
from MultiAgent.context import load_contexts
from MultiAgent.local_video_visual import probe_frames
from MultiAgent.gaze_evidence import gaze_records, project_gaze
from MultiAgent.camera_geometry import interpolate_pose
from MultiAgent.spherical_attention_study import AUDIT
from MultiAgent.heatmap_temporal_pilot import sha, save
from MultiAgent.perspective_attention_study import SOURCE, read
from tools.run_attention_region_study import (OUTPUT, render_page, native_angular_lookup,
    heat_tile, sphere_indices, solid_angle_weights, seam_audit)


def png_hash(image):
    data = io.BytesIO(); image.save(data, format='PNG')
    return hashlib.sha256(data.getvalue()).hexdigest()


class Scripts(HTMLParser):
    def __init__(self):
        super().__init__(); self.inside = False; self.scripts = []
    def handle_starttag(self, tag, attrs):
        if tag == 'script': self.inside = True
    def handle_endtag(self, tag):
        if tag == 'script': self.inside = False
    def handle_data(self, data):
        if self.inside: self.scripts.append(data)


def verify(output, jscheck_path=None, require_complete=False):
    m = read(output / 'manifest.json'); lock = read(output / 'protocol_lock.json')
    assert sha(output / 'manifest.json') == lock['manifest_sha256'] and lock['frozen_before_model']
    assert m['regression_only'] and not m['ground_truth_loaded'] and not m['anchors_projected']
    assert sha(m['source_manifest_path']) == m['source_manifest_sha256']
    for path, digest in m['code_snapshots'].items():
        assert sha(output / 'source' / path) == digest
    original = {e['id']: e for e in read(SOURCE / 'manifest.json')['events']}
    selection = read(AUDIT / 'selection.json')
    keys = {e['event'] for e in m['events']}
    contexts = load_contexts({'samples': [r for r in selection['samples'] if f"{r['scene_key']}/{r['row_index']}" in keys]}, load_ground_truth=False)
    contexts = {f'{c.scene_key}/{c.row_index}': c for c in contexts}
    ready = {}; regions = 0; native = 0; max_error = 0.; status = Counter()
    for event in m['events']:
        ctx = contexts[event['event']]
        assert sha(ctx.video_path) == event['source_video_sha256']
        assert sha(ctx.json_path) == event['source_telemetry_sha256']
        decoded, _ = probe_frames(ctx.video_path)
        by_index = {f['source_frame_index']: f for f in decoded}
        records = gaze_records(ctx); record_by_index = {r['source_index']: r for r in records}
        assert sha(event['source_volume_path']) == event['source_volume_sha256']
        assert sha(event['source_clock_contract_path']) == event['source_clock_contract_sha256']
        contract = read(event['source_clock_contract_path'])
        assert event['source_video_sha256'] == contract['source_hashes']['video_sha256']
        assert event['source_telemetry_sha256'] == contract['source_hashes']['telemetry_sha256']
        assert sha(output / event['regions_npz']) == event['regions_npz_sha256']
        data = np.load(output / event['regions_npz']); maps = data['duration_maps']; labels = data['labels']
        np.testing.assert_array_equal(maps, np.load(event['source_volume_path'])['duration_maps'])
        expected_labels, expected_regions, audits = extract_regions(maps, m['retained_mass_fraction'])
        np.testing.assert_array_equal(labels, expected_labels)
        np.testing.assert_array_equal(data['residual_maps'], np.where(labels == 0, maps, 0))
        assert audits == event['mass_audits']
        assert len(event['regions']) == len(expected_regions)
        area = solid_angle_weights(maps.shape[2], maps.shape[1]); density = maps / area[None]
        for row, expected in zip(event['regions'], expected_regions):
            for key, value in expected.items(): assert row[key] == value
            b = original[event['id']]['volume_stats']['bins'][row['phase']]
            assigned = region_observations(b['observations'], event['canonical_pose'], labels[row['phase']], row['component'])
            assert row['observations'] == assigned
            assert row['representative_observation'] == representative_observation(assigned, event['canonical_pose'], row)
            regions += 1; status[row['status']] += 1
            if row['status'] != 'RGB_EVIDENCE_READY': continue
            ready[(event['id'], row['id'])] = row
            path = output / row['native_source_path']
            assert sha(path) == row['native_source_sha256']
            source_time = row['source_frame']['pts_seconds'] - event['clock_offset_seconds']
            assert source_time <= row['window'][1] + 1e-6
            assert abs(source_time - row['representative_observation']['gaze_time_seconds']) <= .08
            assert abs(row['gaze_frame_error_seconds'] - (source_time - row['representative_observation']['gaze_time_seconds'])) < 1e-6
            assert abs(source_time - row['source_pose']['telemetry_time_seconds']) < 1e-6
            assert row['source_frame'] == by_index[row['source_frame']['source_frame_index']]
            eligible = [f for f in decoded if f['pts_seconds'] <= row['window'][1] + event['clock_offset_seconds'] + 1e-6]
            target = row['representative_observation']['gaze_time_seconds'] + event['clock_offset_seconds']
            assert row['source_frame'] == min(eligible, key=lambda f: abs(f['pts_seconds'] - target))
            assert row['source_pose'] == interpolate_pose(records, source_time)
            assert row['projection']['status'] == 'projected_frustum_only'
            max_error = max(max_error, abs(row['gaze_frame_error_seconds'])); native += 1
            with Image.open(path) as image: rgb = image.convert('RGB')
            projected = project_gaze(record_by_index[row['representative_observation']['source_index']], row['source_pose'], *rgb.size)
            assert projected == row['projection']
            box = row['crop_box_xyxy']
            assert png_hash(rgb.crop(box)) == row['crop_image']['sha256']
            assert sha(output / row['context_image']['path']) == row['context_image']['sha256']
            assert sha(output / row['crop_image']['path']) == row['crop_image']['sha256']
            rays = native_angular_lookup(row['source_pose'], event['canonical_pose'], rgb.size)
            x, y = sphere_indices(rays, maps.shape[2], maps.shape[1])
            for phase, heat in enumerate(row['heat_images']):
                assert sha(output / heat['path']) == heat['sha256']
                assert png_hash(heat_tile(density[phase, y, x], float(density.max())).crop(box)) == heat['sha256']
        for audit in event['mass_audits']:
            assert audit['retained_fraction'] is None or audit['retained_fraction'] >= .9 - 1e-9
            assert abs(audit['retained_seconds'] + audit['residual_seconds'] - audit['total_seconds']) < 1e-8
    responses = {p.stem: read(p) for p in (output / 'responses').glob('*.json')}
    requests = {r['id']: r for r in m['requests']}
    if require_complete: assert len(responses) == len(requests)
    assert set(responses) <= set(requests)
    cards = 0
    schema_failures = 0
    for request in m['requests']:
        assert sha(output / request['image']) == request['sha256']
        candidates = [ready[(request['event_id'], rid)] for rid in request['region_ids']]
        assert png_hash(render_page(output, candidates, request['condition'])) == request['sha256']
        cards += 1
        if request['id'] not in responses: continue
        result = responses[request['id']]; raw = read(result['request_path'])
        assert raw['prompt'] == request['prompt']
        assert raw['image_sha256'] == [request['sha256']]
        assert raw['model'] == '/ai/data/Qwen3-VL-8B-Instruct' and not raw['do_sample']
        assert raw['max_new_tokens'] == 512
        assert raw['processor_audit']['expanded_image_token_count'] == 1024
        assert raw['visual_tensor_trace']['input_consumption_verified']
        assert result['vision_verified']
        assert raw['parsed'] == result['parsed']
        if result['schema_valid']:
            assert sorted(d['id'] for d in result['parsed']['decisions']) == sorted(request['region_ids'])
        else:
            schema_failures += 1
    # Every ready region appears exactly once per condition; no proposal lost at page cuts.
    for condition in m['conditions']:
        seen = Counter((r['event_id'], rid) for r in m['requests'] if r['condition'] == condition for rid in r['region_ids'])
        assert set(seen) == set(ready) and set(seen.values()) == {1}
    assert read(output / 'seam_audit.json') == seam_audit()
    model_calls = []
    for path in (output / 'model_cache').glob('*/call_audit.jsonl'):
        model_calls.extend(json.loads(line) for line in path.read_text().splitlines())
    if require_complete:
        assert len(model_calls) == len(requests) and not any(r['cache_hit'] for r in model_calls)
    js = None
    if jscheck_path:
        sys.path.insert(0, str(jscheck_path)); import quickjs
        dom = '''class Element{constructor(tag='div'){this.tag=tag;this.children=[];this.value='';this.src='';this.textContent=''}append(...items){for(const item of items){this.children.push(item);if(this.tag==='select'&&this.value==='')this.value=String(item.value)}}replaceChildren(...items){this.children=items}}const roots={};const document={querySelector(s){if(!roots[s])roots[s]=new Element(['#event','#condition'].includes(s)?'select':s);return roots[s]},createElement(t){return new Element(t)}};document.querySelector('#condition').value='TRUE_HEAT';'''
        parser = Scripts(); parser.feed((output / 'index.html').read_text())
        context = quickjs.Context(); context.eval(dom)
        for script in parser.scripts: context.eval(script)
        count = context.eval("let n=0;for(const event of d.events){q.value=event.id;for(const condition of ['TRUE_HEAT','NO_HEAT','REVERSE_TIME']){c.value=condition;render();n+=document.querySelector('main').children.length}}n")
        assert count == len(requests)
        memory = Scripts(); memory.feed((output / 'region_memory.html').read_text())
        mc = quickjs.Context(); mc.eval(dom)
        for script in memory.scripts: mc.eval(script)
        memory_count = mc.eval("let n=0;for(const event of d.events){q.value=event.id;render();n+=document.querySelector('main').children.length}n")
        assert memory_count == regions
        for preview in read(output / 'memory_previews.json').values():
            assert sha(output / preview['path']) == preview['sha256']
        annotations = read(output / 'annotation_template.json')
        assert annotations['status'] == 'UNANNOTATED' and len(annotations['rows']) == regions
        for row in annotations['rows']:
            assert row['visible_object_labels'] is None and row['instruction_relevance'] is None and row['annotator'] is None
        js = {'event_condition_combinations': len(m['events']) * 3, 'page_images_rendered': count,
              'memory_region_cards_rendered': memory_count, 'full_browser': False}
    result = {'protocol_hash_verified': True, 'code_snapshots_verified': True,
        'sphere_components_reproduced': regions, 'status_counts': dict(status),
        'retained_mass_and_residual_conserved': True, 'native_rgb_routes_verified': native,
        'native_heat_registration_reproduced': native * 3, 'max_pts_error_ms': max_error * 1000,
        'pages_byte_reproduced': cards, 'every_ready_region_retained_in_all_conditions': True,
        'model_calls_verified': len(responses), 'all_1024_tokens_and_vision_verified': True,
        'model_schema_failures_preserved': schema_failures,
        'viewer_logic': js, 'regression_only': True, 'final_referent_evaluator_run': False}
    save(output / 'verification.json', result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--jscheck-path', type=Path)
    p.add_argument('--require-complete', action='store_true')
    a = p.parse_args(); verify(a.output.resolve(), a.jscheck_path, a.require_complete)
