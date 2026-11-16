#!/usr/bin/env python3
"""Isolate occurrence-ID binding on the frozen mention-specific RGB coverage probe."""
import argparse
import hashlib
import json
from pathlib import Path

from MultiAgent.projected2d_attention_study import save_json, sha_file
from MultiAgent.projected2d_main_image import normalize
from MultiAgent.projected2d_routed_rgb import MODEL, ROOT
from tools.probe_referent_temporal_coverage import prompt_for as previous_prompt, verify, report

SOURCE = ROOT / '.local-sync/referent_temporal_coverage_probe_20261109'
OUTPUT = ROOT / '.local-sync/referent_temporal_coverage_v2_20261109'


def prompt_for(event):
    mid = event['task']['mention']['mention_id']
    prompt = previous_prompt(event)
    old = '"mention_id":"..."'
    if prompt.count(old) != 1:
        raise ValueError('Expected a single output ID placeholder')
    binding = ('The output mention_id must be exactly ' + json.dumps(mid) +
               '. It identifies this expression occurrence. A coreference_id is a different '
               'field describing identity relationships and must not replace mention_id.\n')
    return binding + prompt.replace(old, '"mention_id":' + json.dumps(mid))


def prepare(out):
    if (out / 'input_manifest.json').exists():
        raise ValueError('Refuse to overwrite frozen probe')
    verify(SOURCE)
    m = json.loads((SOURCE / 'input_manifest.json').read_text())
    m.update(version='referent_temporal_coverage_v2_occurrence_binding',
             change='Same ten tasks and source RGB images as v1. Bind the exact occurrence ID explicitly in prose and the JSON example.',
             source_manifest_sha256=sha_file(SOURCE / 'input_manifest.json'))
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'input_manifest.json', m)
    paths = [Path(__file__).resolve(), ROOT / 'tools/probe_referent_temporal_coverage.py',
             SOURCE / 'input_manifest.json', SOURCE / 'protocol_lock.json', out / 'input_manifest.json',
             ROOT / 'MultiAgent/qwen_backend.py', ROOT / 'MultiAgent/projected2d_main_image.py',
             ROOT / 'tools/run_full_frame_grounding_v4.py', MODEL / 'config.json']
    save_json(out / 'protocol_lock.json', {'files': {str(p): sha_file(p) for p in paths}})
    print('FROZEN_COVERAGE_V2', len(m['events']), flush=True)


def infer(out):
    from MultiAgent.qwen_backend import QwenBackend
    from tools.run_full_frame_grounding_v4 import clean_visual_trace
    verify(out)
    m = json.loads((out / 'input_manifest.json').read_text())
    backend = QwenBackend(out / 'model_cache', str(MODEL), local_only=True,
                          namespace='referent_temporal_coverage_v2', trace_visual_inputs=True)
    for i, event in enumerate(m['events'], 1):
        task = event['task']; mid = task['mention']['mention_id']
        path = out / 'responses' / f"{event['id']}_{mid}.json"
        images = task['frames']
        captions = [f"Frame {f['frame_id']} | video_time={f['time_seconds']:.3f} seconds" for f in images]
        prompt = prompt_for(event)
        fingerprint = hashlib.sha256(json.dumps([prompt, captions, [f['source_sha256'] for f in images]]).encode()).hexdigest()
        if path.exists():
            if json.loads(path.read_text())['fingerprint'] != fingerprint:
                raise ValueError('Response fingerprint mismatch')
            continue
        for f in images:
            if sha_file(Path(f['source_path'])) != f['source_sha256']:
                raise ValueError('Input RGB changed')
        req = None
        try:
            raw = backend.ask(event['id'] + '/' + mid, 'MENTION_TIME_RGB_V2', prompt,
                              [f['source_path'] for f in images], max_new_tokens=1024, image_captions=captions)
            req = Path(backend.requests[-1]); request = json.loads(req.read_text())
            parsed, errors, coordinates = normalize(raw, {'mentions': [task['mention']],
                'panels': [{'frame_id': f['frame_id']} for f in images]})
            if raw.get('_parse_warning'):
                errors.append(raw['_parse_warning'])
            response = {'status': 'COMPLETE' if not errors else 'PARSE_WITH_WARNINGS',
                'parsed': parsed, 'warnings': errors, 'coordinates': coordinates,
                'raw_response': request['response'], 'request_path': str(req),
                'vision_verified': bool((request.get('visual_tensor_trace') or {}).get('input_consumption_verified')),
                'processor_audit': request.get('processor_audit')}
            if not response['vision_verified']:
                raise ValueError('Vision input was not verified')
        except Exception as exc:
            response = {'status': 'FAILED', 'parsed': {'mentions': []}, 'error': f'{type(exc).__name__}: {exc}'}
        finally:
            if req:
                clean_visual_trace(str(req))
        response.update(event=event['id'], mention_id=mid, fingerprint=fingerprint, prompt=prompt,
                        frames=images, image_captions=captions)
        save_json(path, response)
        print('COVERAGE_V2', i, '/', len(m['events']), event['id'], mid, response['status'], flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'infer', 'report'])
    p.add_argument('--output', type=Path, default=OUTPUT); a = p.parse_args()
    {'prepare': prepare, 'infer': infer, 'report': report}[a.stage](a.output.resolve())


if __name__ == '__main__':
    main()
