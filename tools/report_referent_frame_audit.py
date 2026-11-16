#!/usr/bin/env python3
"""Finalize frozen frame-choice probes and retain the limits of each conclusion."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

from MultiAgent.projected2d_attention_study import CSV_COLUMNS, evaluator_adapter, read_csv, save_json, sha_file, write_csv
from MultiAgent.projected2d_exact_set import exact_counts
from MultiAgent.projected2d_main_image import verify_lock
from MultiAgent.projected2d_routed_rgb import ROOT

BASE = ROOT / '.local-sync'
A = BASE / 'referent_frame_choice_audit_20261109'
C = BASE / 'referent_frame_checks_v2_20261109'
B = BASE / 'referent_temporal_coverage_probe_20261109'
D = BASE / 'referent_temporal_coverage_v2_20261109'
MAIN = BASE / 'projected2d_main_rgb32_20261109'
PRIOR = BASE / 'referent_rgb_expansion32_20261109_repaired/manifest.json'


def read(path):
    return json.loads(path.read_text())


def frozen_responses(out):
    verify_lock(out)
    frozen = read(out / 'predictions_frozen.json')['responses']
    for name, digest in frozen.items():
        if sha_file(Path(name)) != digest:
            raise ValueError('Frozen prediction changed: ' + name)
    return len(frozen)


def score_comparison():
    rows = []
    for event in read(C / 'input_manifest.json')['events']:
        r = read(C / 'responses' / f"{event['id']}.json")
        pred = evaluator_adapter(r['parsed'])
        rows.append({'scene': event['scene_key'], 'row_index': event['row_index'], 'event_id': event['event_id'],
            'instruction': event['instruction'], 'model_input_image': ';'.join(x['path'] for x in r['input_images']),
            'model_raw_output': r.get('raw_response', ''), 'parsed_json': json.dumps(pred, ensure_ascii=False),
            'parse_ok': str(r['status'] in ('COMPLETE', 'PARSE_WITH_WARNINGS')),
            'prediction_count': len(pred['referents']), 'error_message': r.get('error', '')})
    write_csv(C / 'predictions.csv', rows, CSV_COLUMNS)
    subprocess.run([sys.executable, str(ROOT.parent / 'TriModal-Referring/exam2/evaluate_2d_point_grounding.py'),
        '--manifest', str(A / 'rebuilt_manifest_all.csv'), '--pred_csv', str(C / 'predictions.csv'),
        '--output_dir', str(C / 'evaluation'), '--coordinate_mode', 'panel'], check=True, capture_output=True, text=True)
    return {'metrics': read(C / 'evaluation/2d_eval_summary.json'),
            'exact': exact_counts(read_csv(C / 'evaluation/2d_eval_detail.csv'))}


def coverage_audit():
    counts = Counter(); events = []
    prior = {e['id']: e for e in read(PRIOR)['events']}
    for e in read(MAIN / 'input_manifest.json')['events']:
        mentions = []; panels = {p['frame_id'] for p in e['panels']}
        for mention in e['mentions']:
            counts['mentions'] += 1; mid = mention['mention_id']; h = e['temporal_hints'][mid]
            present = h['soft_phase_panels']; spec = prior[e['id']]['temporal_search']['mentions'][mid]
            counts['uninformative_primary'] += set(present) == panels
            counts['empty_primary'] += not present
            lo, hi = h['soft_phase_range']
            covered = [f['frame_id'] for f in prior[e['id']]['frames'] if lo <= f['time_seconds'] <= hi]
            counts['recoverable_by_existing_full_event'] += not present and bool(covered)
            mentions.append({'mention_id': mid, 'mention': mention['text'], 'soft_phase_range': [lo, hi],
                'present_original_panels': present, 'existing_other_event_frames': covered,
                'existing_primary_frame_ids': spec['primary_frame_ids'],
                'existing_context_frame_ids': spec['context_frame_ids'], 'action_context': spec['action_context']})
        events.append({'id': e['id'], 'panel_range': [min(p['requested_video_time'] for p in e['panels']),
            max(p['requested_video_time'] for p in e['panels'])], 'mentions': mentions})
    result = {'counts': dict(counts), 'events': events,
              'interpretation': 'Coverage is relative to a soft clause-order prior, not verified utterance timing. Original benchmark panels were retained for a matched evaluator; no claim that missing-phase frames are necessary ground truth.'}
    save_json(A / 'coverage_audit.json', result)
    return result


def temporal_probe_summary(out):
    manifest = read(out / 'input_manifest.json'); rows = []; statuses = Counter()
    for event in manifest['events']:
        task = event['task']; mid = task['mention']['mention_id']
        r = read(out / 'responses' / f"{event['id']}_{mid}.json"); statuses[r['status']] += 1
        times = {f['frame_id']: f['time_seconds'] for f in task['frames']}
        predictions = [p for m in r['parsed']['mentions'] for p in m['predictions']]
        lo, hi = task['soft_phase_range']
        rows.append({'event': event['id'], 'mention_id': mid, 'mention': task['mention']['text'],
            'status': r['status'], 'warnings': r.get('warnings', []),
            'first_supplied_frame': task['frames'][0]['frame_id'], 'soft_phase_range': [lo, hi],
            'predictions': [dict(p, time_seconds=times[p['frame_id']],
                within_soft_phase=lo <= times[p['frame_id']] <= hi) for p in predictions]})
    return {'statuses': dict(statuses), 'rows': rows,
            'localized_mentions': sum(bool(r['predictions']) for r in rows),
            'points_on_first_supplied_frame': sum(p['frame_id'] == r['first_supplied_frame'] for r in rows for p in r['predictions']),
            'no_official_scores': 'Original 2D labels cover P1/P2/P3, not these V frames. No 2D performance claim is made.'}


def main():
    counts = {o.name: frozen_responses(o) for o in (MAIN, A, B, C, D)}
    verifications = {}
    for out in (A, B, C, D):
        v = read(out / 'processor_reconstruction_verification.json')
        if v['calls_verified'] != counts[out.name]:
            raise ValueError('Incomplete independent processor verification')
        verifications[out.name] = v['calls_verified']
    a = read(A / 'summary.json'); c = read(C / 'summary.json')
    result = {'date': '2026-11-09', 'task': 'Self-audit of per-referent frame selection in projected-2D RGB grounding',
        'scope': 'Ten previously seen development events, 29 mentions; two of these events also used in 10 single-mention coverage calls per version.',
        'frozen_records_verified': counts, 'new_physical_model_calls': sum(verifications.values()),
        'processor_calls_reconstructed': verifications, 'coverage': coverage_audit()['counts'],
        'order_control': {k: a[k] for k in ('pair_count', 'changed_frame_count', 'first_position_follow_both_count')},
        'matched_panel_conditions': {k: {'stats': v['stats'], 'metrics': v['metrics'],
            'exact': exact_counts(read_csv(A / 'evaluation' / k / '2d_eval_detail.csv'))} for k, v in a['conditions'].items()},
        'comparison_schema_v1_coverage': a['per_referent_comparison_coverage'],
        'comparison_schema_v2': {'counts': c['counts'], 'frame_choices': c['frame_choices'], **score_comparison()},
        'temporal_coverage_v1': temporal_probe_summary(B), 'temporal_coverage_v2': temporal_probe_summary(D),
        'conclusions': [
            'A first-frame tendency is an observed symptom, not an explanation. The prompt asked for comparisons but did not adequately distinguish reference evidence from generic object visibility.',
            'The 2D adapter discarded highlighted occurrence/action context and primary/context frame metadata from the existing referent search plan, retaining only a soft range and intersection with the original panels.',
            'The original 2D panels were retained for evaluation parity; global clause-order priors were not adapted to this sparse support, and missing coverage was not surfaced.',
            'Original green-gaze weak-cue instructions and gaze enlargement were removed, with no equivalent per-referent attention evidence in the main-image control. Their individual causal contributions are unmeasured.',
            'Reversing presentation changes 6 of 23 comparable single-target frame choices; 4 follow first position in both orders. Order sensitivity is present and does not explain all P1 choices.',
            'Correcting the comparison JSON example yields all-frame entries for 29/29 mentions; 20/23 points still select P1. Field compliance is insufficient evidence of valid comparison.',
            'Coverage restoration yields different actual source times for late mentions, but 7/10 corrected single-mention points still choose their local first image. Correct referent identity is not established.',
            'Occurrence ID and coreference ID were confused in three new coverage-probe responses; binding the literal occurrence ID fixes that probe schema. This is not identified as the cause of the old main-image P1 result.',
            'Do not force distinct frames. Keep occurrence-specific evidence, physical identity and final set assembly separate; do not merge occurrences before their temporal grounding.'
        ],
        'unresolved': ['A validated attention cue per occurrence on the same full RGB plane is still needed.',
                       'No new matched 32-event efficacy run has been completed; original low scores remain unchanged.',
                       'Frame-check observations can hallucinate instruction execution; explicit visual verification remains necessary.']}
    save_json(A / 'completion_audit.json', result)
    print(json.dumps({'new_calls': result['new_physical_model_calls'], 'coverage': result['coverage'],
        'order_control': result['order_control'], 'comparison_v2': c['counts'],
        'coverage_v2': {k:v for k,v in result['temporal_coverage_v2'].items() if k!='rows'}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
