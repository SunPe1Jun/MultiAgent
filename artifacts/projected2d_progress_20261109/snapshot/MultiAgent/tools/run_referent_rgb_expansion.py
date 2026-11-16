#!/usr/bin/env python3
"""Larger RGB routing study reusing mention semantics and soft temporal search."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import random

from MultiAgent.heatmap_temporal_pilot import save, sha
from tools.run_gaze_rgb_routing import (
    ROOT, MODEL, VIDEO_MAX_PIXELS, CONDITIONS, SEED, historical_exposure,
    instruction_lookup, prepare_event, read, now,
)

OUTPUT = ROOT / '.local-sync/referent_rgb_expansion32_20261109_repaired'
STUDY_SEED = 202611091


def select(output, per_scene=4):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new empty output directory')
    lookup, sources = instruction_lookup()
    seen, exposure = historical_exposure()
    rng, selected = random.Random(STUDY_SEED), []
    for scene in sorted({scene for scene, _ in lookup}):
        pool = sorted(index for (s, index), text in lookup.items()
                      if s == scene and text and (s, index) not in seen)
        if len(pool) < per_scene:
            raise ValueError(f'Not enough new addresses for {scene}')
        for index in sorted(rng.sample(pool, per_scene)):
            selected.append({'scene_key': scene, 'row_index': index,
                'event_id': f"scene{scene[5:].split('_')[0]}_{index + 1}",
                'instruction': lookup[(scene, index)]})
    save(output / 'selection.json', {'seed': STUDY_SEED, 'samples': selected,
        'per_subscene': per_scene, 'selected_utc': now(),
        'project_previously_addressed_events': len(seen), 'prior_artifacts_scanned': exposure,
        'instruction_sources': sources, 'selection_uses_gt_heat_or_responses': False,
        'scope': 'New relative to available local artifact addresses; absent original-server history/pretraining not verifiable.',
        'planned_conditions': list(CONDITIONS), 'frames_per_event': 12,
        'model': MODEL, 'video_max_pixels': VIDEO_MAX_PIXELS,
        'planned_reuse': ['complete ReferentialAgent fields', 'plan_frames soft action phases',
            'focus_spec primary/context evidence', 'exact source frame and RGB-panel transforms',
            'cardinality, explicit/unique lexical identities and constrained component coreference'],
        'planned_recovery': 'At most one additional whole-event recovery for structural omissions or unresolved/rejected ambiguous concrete mentions; shared policy, no GT triggers.',
        'primary_metrics': ['Exact-set', 'micro-F1'],
        'preparation_source_sha256': {name: sha(ROOT / name) for name in
            ['tools/run_gaze_rgb_routing.py', 'MultiAgent/gaze_rgb_routing.py',
             'MultiAgent/local_video_visual.py', 'MultiAgent/frame_pose_alignment.py']}})
    print('FROZEN_SELECTION', len(selected), [(s['scene_key'], s['row_index']) for s in selected], flush=True)


def prepare_cpu(output, workers=3):
    selection = read(output / 'selection.json')
    events = []
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(prepare_event, item, str(output)): item for item in selection['samples']}
        for future in as_completed(jobs):
            item = jobs[future]
            try:
                events.append(future.result())
            except Exception as exc:
                errors.append({'sample': item, 'error': f'{type(exc).__name__}: {exc}'})
                print('CPU_FAILURE', item['scene_key'], item['row_index'], str(exc), flush=True)
            save(output / 'cpu_progress.json', {'completed': len(events), 'errors': errors,
                'planned': len(selection['samples']), 'gt_loaded': False})
    if errors:
        raise ValueError('CPU preparation failures retained; fix preparation before inference, never replace selected samples')
    save(output / 'cpu_events.json', sorted(events, key=lambda e: (e['scene_key'], e['row_index'])))
    print('CPU_COMPLETE', len(events), 'events', sum(e['clock_available_frames'] > 0 for e in events), 'with supported frames', flush=True)


# Source locks are frozen only after language parsing and visual protocol are
# complete. The earlier sample selection remains immutable.
def study_sources():
    from tools.run_gaze_rgb_routing import SOURCES as BASE_SOURCES
    names = sorted(set(BASE_SOURCES + ['MultiAgent/referent_rgb_selection.py',
        'MultiAgent/preprocess.py', 'MultiAgent/temporal_focus.py',
        'tools/run_referent_rgb_expansion.py']))
    return {name: sha(ROOT / name) for name in names}


def check_study(output):
    m = read(output / 'manifest.json')
    if sha(output / 'manifest.json') != read(output / 'protocol_lock.json')['manifest_sha256']:
        raise ValueError('Frozen study manifest changed')
    if study_sources() != m['source_sha256']:
        raise ValueError('Runtime source differs from frozen study')
    return m



def save_labelled_input(source_path, target, frame):
    """Atomically write a complete PNG; never accept interrupted cache files."""
    import numpy as np
    from PIL import Image
    from MultiAgent.referent_rgb_selection import labelled_rgb
    with Image.open(source_path) as source:
        expected = labelled_rgb(source, frame)
    if target.exists():
        try:
            with Image.open(target) as cached:
                if np.array_equal(np.asarray(cached), np.asarray(expected)):
                    return
        except (OSError, ValueError):
            pass
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.pending.png')
    expected.save(temporary)
    with Image.open(temporary) as written:
        written.verify()
    temporary.replace(target)


def require_verified_inputs(output, manifest):
    path = output / 'input_verification.json'
    if not path.exists():
        raise ValueError('Complete input verification is required before inference')
    verification = read(path)
    expected = sum(len(event['frames']) for event in manifest['events']) * len(manifest['conditions'])
    if (verification.get('manifest_sha256') != sha(output / 'manifest.json')
            or verification.get('all_composite_pixels_rebuilt_from_rgb') != expected
            or verification.get('full_rgb_exact_pixel_matches') != expected
            or verification.get('model_heatmap_pixels') != 0):
        raise ValueError('Input verification does not match the frozen study')


def prepare(output):
    import shutil
    from PIL import Image
    from MultiAgent.agents import ReferentialAgent
    from MultiAgent.context import vrtriref_sample_to_agent_context
    from MultiAgent.qwen_backend import QwenBackend
    from MultiAgent.referent_rgb_selection import (
        VERSION, recover_component_schema, temporal_search, labelled_rgb, grounding_prompt)
    if (output / 'manifest.json').exists():
        raise ValueError('Never overwrite a frozen study')
    events = read(output / 'cpu_events.json')
    selection = read(output / 'selection.json')
    expected = {(s['scene_key'], s['row_index']) for s in selection['samples']}
    if {(e['scene_key'], e['row_index']) for e in events} != expected:
        raise ValueError('CPU sample addresses differ from frozen selection')
    backend = QwenBackend(output / 'language_cache', model=MODEL, local_only=True, namespace=VERSION + '_language')
    parser = ReferentialAgent(backend)
    for event in events:
        print('LANGUAGE', event['event'], flush=True)
        cache = output / 'language' / f"{event['id']}.json"
        if cache.exists():
            parsed = read(cache)
        else:
            row = read(output / 'inputs' / event['scene_key'] / f"{event['row_index']}_row.json")['row']
            context = vrtriref_sample_to_agent_context(row, load_ground_truth=False)
            try:
                parsed = parser.run(context)
            except ValueError as exc:
                raw = read(backend.requests[-1])['parsed']
                try:
                    parsed = recover_component_schema(raw, context)
                    parsed['prior_parser_error'] = str(exc)
                except ValueError as recovery_error:
                    parsed = {'mentions': [], 'parser_failure': str(exc),
                              'recovery_error': str(recovery_error), 'raw': raw}
            parsed['request_paths'] = list(backend.requests)
            save(cache, parsed)
        event['referential'] = parsed
        event['mention_hints'] = parsed['mentions']
        event['parser_valid'] = not bool(parsed.get('parser_failure'))
        event['language_sha256'] = sha(cache)
        event['temporal_search'] = temporal_search(event)
        for frame in event['frames']:
            frame['study_images'] = {}
            for condition in CONDITIONS:
                target = output / 'labelled_inputs' / event['id'] / frame['frame_id'] / f'{condition}.png'
                save_labelled_input(output / frame['images'][condition]['path'], target, frame)
                frame['study_images'][condition] = {'path': str(target.relative_to(output)), 'sha256': sha(target)}
        event['prompts'] = {c: grounding_prompt(event, c) for c in CONDITIONS}
    import hashlib
    requests = []
    for event in events:
        fingerprints = {}
        for condition in CONDITIONS:
            fingerprint = hashlib.sha256(json.dumps([event['prompts'][condition],
                [f['study_images'][condition]['sha256'] for f in event['frames']]], sort_keys=True).encode()).hexdigest()
            rid = f"{event['id']}_{condition}"
            requests.append({'id': rid, 'event_id': event['id'], 'condition': condition,
                'input_fingerprint': fingerprint,
                'canonical_request_id': fingerprints.setdefault(fingerprint, rid)})
    manifest = {'version': VERSION, 'events': events, 'conditions': list(CONDITIONS),
        'requests': requests, 'model': MODEL, 'model_config_sha256': sha(Path(MODEL) / 'config.json'),
        'video_max_pixels': VIDEO_MAX_PIXELS, 'max_new_tokens': 2048,
        'selection_sha256': sha(output / 'selection.json'), 'source_sha256': study_sources(),
        'planned_unique_initial_calls': len({r['canonical_request_id'] for r in requests}),
        'maximum_recovery_calls_per_unique_initial': 1, 'recovery_max_frames': 8,
        'recovery_triggers': ['concrete mention missing/unresolved', 'explicit count incomplete',
            'cardinality conflict', 'visual identity among multiple language-compatible candidates'],
        'stages': ['visual_only', 'language_assembled', 'recovered'],
        'gt_loaded': False, 'default_pipeline_changed': False, 'heatmap_model_input': False,
        'evaluation_policy': 'Freeze all initial/final outputs and response hashes before original-evaluator scoring; report every stage and all selected samples.',
        'decoder_policy': 'Existing 0.15 distance / 0.02 margin; exact selected source frame and language-compatible candidates; no threshold tuning.',
        'known_limits': ['Soft clause time is not measured speech alignment.',
            'Unique language identities can be included without visible support; recorded separately.',
            'Anchor-center geometry, visibility and dynamic correspondence remain uncalibrated.',
            'Raw RGB contains original recorder marks; no-added-input is not gaze-free.',
            'One recovery has additional visual budget; report before/after stages, do not attribute review gains to gaze.',
            'Project-new relative to available artifacts, not a guaranteed pretraining/server holdout.']}
    save(output / 'manifest.json', manifest)
    save(output / 'protocol_lock.json', {'manifest_sha256': sha(output / 'manifest.json'),
        'frozen_utc': now(), 'before_visual_inference': True, 'gt_loaded': False})
    for name in manifest['source_sha256']:
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    print('FROZEN', len(events), 'events', len(requests), 'conditions',
          manifest['planned_unique_initial_calls'], 'unique initial calls', flush=True)


def inference_call(backend, output, event, condition, frames, prompt, key, stage, pixels, max_tokens):
    import time
    import traceback
    from MultiAgent.referent_rgb_selection import validate_choices
    from tools.run_full_frame_grounding_v4 import clean_visual_trace
    paths = [output / f['study_images'][condition]['path'] for f in frames]
    for path, frame in zip(paths, frames):
        if sha(path) != frame['study_images'][condition]['sha256']:
            raise ValueError('Frozen image changed')
    metadata = [{key: f[key] for key in ('frame_id', 'source_frame_id', 'source_frame_index', 'time_seconds')} for f in frames]
    started = time.monotonic()
    parsed, audit, error = None, {}, None
    try:
        parsed = backend.ask_video(key, stage, prompt, paths, metadata,
            video_path=event['video_path'], video_time_range=[frames[0]['time_seconds'], frames[-1]['time_seconds']],
            video_max_pixels=pixels, max_new_tokens=max_tokens)
        audit = read(backend.requests[-1])['processor_audit']
    except Exception as exc:
        error = {'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()}
    checked = validate_choices(parsed, event, condition, {f['frame_id'] for f in frames})
    record = {'parsed': parsed, 'validation': checked, 'processor_audit': audit, 'failure': error,
        'vision_verified': bool(audit.get('visual_tensor_trace', {}).get('input_consumption_verified')),
        'request_path': backend.requests[-1] if backend.requests else None,
        'seconds': time.monotonic() - started, 'supplied_frame_ids': [f['frame_id'] for f in frames]}
    save(output / 'calls' / f'{key}_{stage}.json', record)
    if record['request_path']:
        clean_visual_trace(record['request_path'])
    return record


def infer(output, limit=None):
    from MultiAgent.qwen_backend import QwenBackend
    from MultiAgent.referent_rgb_selection import (
        VERSION, assemble, recovery_mentions, recovery_frames, grounding_prompt, merge_recovery, validate_choices)
    m = check_study(output)
    require_verified_inputs(output, m)
    backend = QwenBackend(output / 'model_cache', model=m['model'], local_only=True,
        namespace=VERSION, trace_visual_inputs=True, video_max_pixels=m['video_max_pixels'])
    events = {e['id']: e for e in m['events']}
    schedule = [r for r in m['requests'] if r['id'] == r['canonical_request_id']]
    random.Random(STUDY_SEED).shuffle(schedule)
    done = 0
    for request in schedule:
        aliases = [r for r in m['requests'] if r['canonical_request_id'] == request['id']]
        if all((output / 'responses' / f"{r['id']}.json").exists() for r in aliases):
            continue
        event, condition = events[request['event_id']], request['condition']
        print('INITIAL', request['id'], flush=True)
        if event['parser_valid']:
            initial = inference_call(backend, output, event, condition, event['frames'], event['prompts'][condition],
                request['id'], 'initial', m['video_max_pixels'], m['max_new_tokens'])
        else:
            initial = {'parsed': None, 'validation': validate_choices(None, event, condition),
                'vision_verified': False, 'failure': {'parser_failure': True}, 'processor_audit': {}}
        checked = initial['validation']
        visual = assemble(event, checked, lexical=False)
        lexical = assemble(event, checked, lexical=True)
        pending = recovery_mentions(event, lexical) if event['parser_valid'] else []
        recovery = None
        merged = checked
        if pending:
            frames = recovery_frames(event, pending, m['recovery_max_frames'])
            prompt = grounding_prompt(event, condition, pending=pending, frames=frames)
            print('RECOVERY', request['id'], 'mentions', pending, 'frames', len(frames), flush=True)
            recovery = inference_call(backend, output, event, condition, frames, prompt, request['id'], 'recovery',
                round(m['video_max_pixels'] * len(frames) / len(event['frames'])), m['max_new_tokens'])
            merged = merge_recovery(checked, recovery['validation'], pending)
        final = assemble(event, merged, lexical=True)
        for alias in aliases:
            save(output / 'responses' / f"{alias['id']}.json", {**alias, 'event': event['event'],
                'scene_key': event['scene_key'], 'row_index': event['row_index'],
                'initial': initial, 'recovery': recovery, 'recovery_mention_ids': pending,
                'final_validation': merged,
                'stages': {'visual_only': visual, 'language_assembled': lexical, 'recovered': final},
                'selected_object_names': final['selected_object_names'],
                'exact_input_reused': alias['id'] != request['id']})
        done += 1
        completed = len(list((output / 'responses').glob('*.json')))
        save(output / 'progress.json', {'completed_conditions': completed, 'planned_conditions': len(m['requests']),
            'last': request['id'], 'gt_loaded': False})
        print('DONE', request['id'], 'selected', final['selected_object_names'],
            'coverage', initial['validation']['coverage_complete'], 'conditions', completed, flush=True)
        if limit and done >= limit:
            break


def evaluate(output):
    from collections import Counter
    from MultiAgent.baseline import module, scene_paths
    from MultiAgent.full_interaction_referring import paired_set_metrics
    m = check_study(output)
    paths = {r['id']: output / 'responses' / f"{r['id']}.json" for r in m['requests']}
    missing = [key for key, path in paths.items() if not path.exists()]
    if missing:
        raise ValueError(f'Cannot evaluate before all selected conditions complete: {len(missing)} missing')
    frozen_path = output / 'predictions_frozen.json'
    hashes = {key: sha(path) for key, path in paths.items()}
    if frozen_path.exists() and read(frozen_path)['response_sha256'] != hashes:
        raise ValueError('Responses changed after freeze')
    if not frozen_path.exists():
        save(frozen_path, {'manifest_sha256': sha(output / 'manifest.json'),
            'response_sha256': hashes, 'frozen_utc': now(), 'before_gt_evaluation': True})
    responses = {key: read(path) for key, path in paths.items()}
    evaluator = module('eval')
    gt_by_scene = {}
    for scene in sorted({e['scene_key'] for e in m['events']}):
        aliases, _ = evaluator.load_anchor_alias_map(scene_paths(scene)['anchor'])
        gt_by_scene[scene] = (aliases, evaluator.load_gt_rows(scene_paths(scene)['gt'], aliases))
    analysis = {'stages': {}, 'events': [], 'gt_loaded_after_prediction_freeze': True,
                'conditions': len(responses), 'unique_initial_inputs': len({r['canonical_request_id'] for r in m['requests']})}
    for stage in m['stages']:
        analysis['stages'][stage] = {}
        for condition in m['conditions']:
            rows, ev_rows, gt_rows_selected = [], [], []
            for event in m['events']:
                r = responses[f"{event['id']}_{condition}"]
                selected = r['stages'][stage]['selected_object_names']
                aliases, gt_rows = gt_by_scene[event['scene_key']]
                gt = gt_rows[event['row_index']]
                target = output / 'predictions' / stage / condition / event['scene_key'] / f"row_{event['row_index']}.json"
                save(target, {'row_index': event['row_index'], 'event_id': event['event_id'],
                    'parsed_response': {'selected_object_names': selected},
                    'response_status': 'ok' if event['parser_valid'] else 'parser_failure', 'used_storyboard_fallback': False})
                rows.append({'gt': gt['mapped_referents'], 'prediction': selected})
                ev_rows.extend(evaluator.build_eval_rows([target], gt_rows, aliases))
                gt_rows_selected.append(gt)
            summary = evaluator.summarize(ev_rows, gt_rows_selected)
            save(output / 'evaluation' / stage / condition / 'original_summary.json', summary)
            evaluator.write_eval_csv(output / 'evaluation' / stage / condition / 'evaluation.csv', ev_rows)
            metrics = paired_set_metrics(rows)
            if abs(metrics['micro_f1'] - summary['micro_f1']) > 1e-12:
                raise ValueError('Independent set metrics disagree with original evaluator')
            analysis['stages'][stage][condition] = {'set_metrics': metrics, 'original_evaluator': summary}
    for event in m['events']:
        _, gt_rows = gt_by_scene[event['scene_key']]
        gt = gt_rows[event['row_index']]['mapped_referents']
        predictions = {stage: {c: responses[f"{event['id']}_{c}"]['stages'][stage]['selected_object_names'] for c in m['conditions']}
                       for stage in m['stages']}
        analysis['events'].append({'id': event['id'], 'event': event['event'], 'instruction': event['instruction'],
            'gt': gt, 'predictions': predictions, 'supported_frames': event['clock_available_frames']})
    unique = {r['canonical_request_id']: responses[r['id']] for r in m['requests'] if r['id'] == r['canonical_request_id']}
    analysis['audit'] = {'parser_failures': sum(not e['parser_valid'] for e in m['events']),
        'initial_schema_valid_conditions': sum(r['initial']['validation']['schema_valid'] for r in responses.values()),
        'initial_coverage_complete_conditions': sum(r['initial']['validation']['coverage_complete'] for r in responses.values()),
        'initial_vision_verified_conditions': sum(r['initial']['vision_verified'] for r in responses.values()),
        'unique_recovery_calls': sum(r['recovery'] is not None for r in unique.values()),
        'recovery_vision_verified': sum(bool(r['recovery'] and r['recovery']['vision_verified']) for r in unique.values()),
        'final_effective_coverage_complete_conditions': sum(r['final_validation'].get('effective_coverage_complete', r['final_validation']['coverage_complete']) for r in responses.values()),
        'initial_visual_token_counts': sorted({r['initial']['processor_audit'].get('expanded_video_token_count') for r in unique.values() if r['initial']['processor_audit']}),
        'recovery_visual_token_counts': sorted({r['recovery']['processor_audit'].get('expanded_video_token_count') for r in unique.values() if r['recovery'] and r['recovery']['processor_audit']}),
        'final_selection_sources': dict(Counter(d['selection_source'] for r in responses.values()
            for d in r['stages']['recovered']['mention_decisions'].values()))}
    save(output / 'analysis.json', analysis)
    print(json.dumps({'metrics': {s: {c: r['set_metrics'] for c, r in conditions.items()} for s, conditions in analysis['stages'].items()},
                      'audit': analysis['audit']}, indent=2), flush=True)


def verify(output):
    import numpy as np
    from PIL import Image
    from MultiAgent.gaze_rgb_routing import render_rgb_evidence, route_regions
    from MultiAgent.referent_rgb_selection import labelled_rgb, grounding_prompt, temporal_search
    m = check_study(output)
    count = 0
    for event in m['events']:
        if temporal_search(event) != event['temporal_search']:
            raise ValueError('Temporal plan no longer reproducible')
        for frame in event['frames']:
            if sha(Path(frame['source_path'])) != frame['source_sha256']:
                raise ValueError('Source image changed')
            with Image.open(frame['source_path']) as source:
                rgb = source.convert('RGB')
            with np.load(output / frame['maps_path']) as archive:
                maps = archive['maps']
            for condition in m['conditions']:
                route = route_regions(maps, rgb.size, clock_supported=frame['geometry']['clock_available'], condition=condition)
                rebuilt = labelled_rgb(render_rgb_evidence(rgb, route), frame)
                path = output / frame['study_images'][condition]['path']
                if sha(path) != frame['study_images'][condition]['sha256']:
                    raise ValueError('Labelled study input hash changed')
                with Image.open(path) as image:
                    pixels = np.asarray(image)
                np.testing.assert_array_equal(pixels, rebuilt)
                np.testing.assert_array_equal(pixels[28:28+rgb.height, :rgb.width], rgb)
                count += 1
        for condition in m['conditions']:
            if grounding_prompt(event, condition) != event['prompts'][condition]:
                raise ValueError('Frozen prompt changed')
    save(output / 'input_verification.json', {'manifest_sha256': sha(output / 'manifest.json'), 'gt_read': False, 'source_and_composite_hashes_match': True,
        'all_composite_pixels_rebuilt_from_rgb': count, 'full_rgb_exact_pixel_matches': count,
        'temporal_plans_reproduced': len(m['events']), 'model_heatmap_pixels': 0})
    print('VERIFIED', count, 'composites', len(m['events']), 'temporal plans', flush=True)


def report(output):
    import html
    m = check_study(output)
    a = read(output / 'analysis.json') if (output / 'analysis.json').exists() else None
    lines = ['<!doctype html><meta charset="utf-8"><title>Referent-aware RGB study</title>',
        '<style>body{font:16px sans-serif;background:#14202b;color:#eee;margin:24px}pre{white-space:pre-wrap}img{width:100%}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}details{margin:16px 0}a{color:#8de}</style>',
        '<h1>Referent-aware RGB + temporal search</h1><p>Complete interaction first, soft mention phases, RGB-only detail windows, exact source-frame points, lexical and coreference assembly. GT only after prediction freeze.</p>']
    if a:
        lines += ['<pre>' + html.escape(json.dumps({s: {c: r['set_metrics'] for c, r in cs.items()} for s, cs in a['stages'].items()}, indent=2)) + '</pre>']
    for event in m['events']:
        lines += ['<h2>' + html.escape(event['event']) + '</h2><p>' + html.escape(event['instruction']) + '</p>',
            '<details><summary>Language mentions and temporal plan</summary><pre>' + html.escape(json.dumps(
                {'mentions': event['mention_hints'], 'search': event['temporal_search']}, ensure_ascii=False, indent=2)) + '</pre></details>']
        for frame in event['frames']:
            lines += ['<details><summary>' + html.escape(frame['frame_id']) + f" · {frame['time_seconds']:.3f}s</summary><div class=grid>"]
            for c in m['conditions']:
                p = html.escape(frame['study_images'][c]['path'])
                lines += ['<div>' + c + '<a href="' + p + '"><img loading=lazy src="' + p + '"></a></div>']
            lines += ['</div></details>']
        if a:
            row = next(e for e in a['events'] if e['id'] == event['id'])
            lines += ['<pre>' + html.escape(json.dumps(row, ensure_ascii=False, indent=2)) + '</pre>']
    (output / 'index.html').write_text('\n'.join(lines))
    print('VIEWER', output / 'index.html', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['select', 'prepare-cpu', 'prepare', 'infer', 'evaluate', 'verify', 'report'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--per-scene', type=int, default=4)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    {'select': lambda p: select(p, args.per_scene), 'prepare-cpu': lambda p: prepare_cpu(p, args.workers),
     'prepare': prepare, 'verify': verify, 'infer': lambda p: infer(p, args.limit), 'evaluate': evaluate, 'report': report}[args.action](args.output.resolve())
