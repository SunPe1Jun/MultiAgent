#!/usr/bin/env python3
"""Compare released Experiment 2 predictions with a changed RGB study, without inference."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event_key(row):
    return row['scene'], int(row['row_index'])


def aggregate(rows):
    result = {'events': len(rows)}
    for metric in ('time', 'point', 'joint'):
        thresholds = (None,) if metric == 'time' else (50, 100, 150, 200)
        for threshold in thresholds:
            suffix = '' if threshold is None else f'_{threshold}'
            counts = {part: sum(int(r[f'{metric}_{part}{suffix}']) for r in rows)
                      for part in ('tp', 'fp', 'fn')}
            denominator = 2 * counts['tp'] + counts['fp'] + counts['fn']
            result[metric + suffix] = {**counts, 'f1': 2 * counts['tp'] / denominator if denominator else 0.}
            if metric != 'time':
                eligible = [r for r in rows if int(r[f'{metric}_tp{suffix}']) + int(r[f'{metric}_fn{suffix}']) > 0]
                result[metric + suffix]['evaluable_events'] = len(eligible)
                result[metric + suffix]['exact_events'] = sum(
                    int(r[f'{metric}_fp{suffix}']) == 0 and int(r[f'{metric}_fn{suffix}']) == 0
                    for r in eligible)
    return result


def replay_original(evaluator, manifest, original, args):
    """Released coordinates omit mention text; report any disagreement explicitly."""
    adapted, differences = [], []
    for row in original:
        key = event_key(row)
        source_rows = manifest[key]
        layout = evaluator.build_layout(source_rows, args)
        gt = evaluator.gt_points(source_rows, layout, args)
        payload = {'referents': json.loads(row['predicted_points'])}
        pred = evaluator.parse_predictions({'parsed_json': json.dumps(payload)})
        restored = {'scene': key[0], 'row_index': key[1],
                    'event_id': row['event_id'], 'parsed_json': json.dumps(payload),
                    'parse_ok': row['valid_output']}
        observed = {}
        for part, value in zip(('tp', 'fp', 'fn'), evaluator.greedy_time_match(pred, gt)):
            observed[f'time_{part}'] = value
        for metric, temporal in (('point', False), ('joint', True)):
            for threshold in (50, 100, 150, 200):
                values = evaluator.greedy_point_match(pred, gt, layout, args, threshold, temporal)[:3]
                for part, value in zip(('tp', 'fp', 'fn'), values):
                    observed[f'{metric}_{part}_{threshold}'] = value
        for column, value in observed.items():
            if value != int(row[column]):
                differences.append({'scene': key[0], 'row_index': key[1], 'field': column,
                                    'published': int(row[column]), 'replayed': value})
        adapted.append(restored)
    return adapted, differences


def run(args):
    study = args.study.resolve()
    output = args.output.resolve()
    evidence = args.trimodal.resolve() / 'paper_experiment_evidence'
    original_path = evidence / 'predictions/exp2/qwen3_vl_8b.csv'
    evaluator_path = args.trimodal.resolve() / 'exam2/evaluate_2d_point_grounding.py'
    evaluator_copy = evidence / 'evaluators/evaluate_2d_point_grounding.py'
    manifest_path = study / 'rebuilt_manifest_all.csv'
    frozen_path = study / 'predictions_frozen.json'
    frozen = json.loads(frozen_path.read_text())
    for path, digest in frozen['responses'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Frozen response changed: ' + path)
    protected_paths = [original_path, manifest_path, frozen_path, evaluator_path,
                       study / 'input_manifest.json', study / 'summary.json',
                       args.trimodal.resolve() / 'exam2/run_qwen3vl_2d_point_grounding.py',
                       evidence / 'prompts_and_configs/exp2_qwen8_final.sh',
                       ROOT / 'MultiAgent/projected2d_routed_rgb.py',
                       ROOT / 'MultiAgent/projected2d_routed_composite.py']
    hashes_before = {str(p): sha(p) for p in protected_paths}
    spec = importlib.util.spec_from_file_location('exp2_original_evaluator', evaluator_path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    eval_args = argparse.Namespace(panel_width=512, panel_height=384, columns=3,
                                   gutter=12, label_height=34, coordinate_mode='panel')
    manifest_rows = read_csv(manifest_path)
    manifest = evaluator.manifest_groups(manifest_rows)
    keys = set(manifest)
    all_original = read_csv(original_path)
    original = [r for r in all_original if event_key(r) in keys]
    if {event_key(r) for r in original} != keys or len(original) != len(keys):
        raise ValueError('Original predictions do not cover this exact study subset once')
    adapted, differences = replay_original(evaluator, manifest, original, eval_args)
    summary = json.loads((study / 'summary.json').read_text())['metrics']['FULL_RGB']
    changed = aggregate(read_csv(study / 'evaluation/FULL_RGB/2d_eval_detail.csv'))
    if abs(changed['joint_100']['f1'] - summary['joint_100_f1']) > 1e-12:
        raise ValueError('Current detail and summary disagree')
    model_manifest = json.loads((study / 'input_manifest.json').read_text())
    original_frames = Counter()
    changed_frames = Counter()
    task_differences = []
    mentions = annotated_targets = 0
    for row in original:
        original_frames.update(p['panel_id'] for p in json.loads(row['predicted_points']))
    for event in model_manifest['events']:
        key = event['scene_key'], event['row_index']
        target_count = len({r['referent_name'] for r in manifest[key] if r['referent_name']})
        mentions += len(event['mentions'])
        annotated_targets += target_count
        if len(event['mentions']) != target_count:
            task_differences.append({'scene': key[0], 'row_index': key[1],
                'original_expected_count': target_count, 'changed_mention_count': len(event['mentions']),
                'parser_valid': event['parser_valid'], 'instruction': event['instruction']})
        response = json.loads((study / 'responses' / f"{event['id']}_FULL_RGB.json").read_text())
        for mention in response.get('parsed', {}).get('mentions', []):
            changed_frames.update(p['frame_id'] for p in mention.get('predictions', []))
    all_scores, subset_scores = aggregate(all_original), aggregate(original)
    published = next(r for r in read_csv(evidence / 'model_results.csv')
                     if r['experiment'] == 'experiment2' and r['model'] == 'Qwen3-VL-8B')
    for metric in ('point', 'joint'):
        if abs(all_scores[f'{metric}_100']['f1'] - float(published[f'{metric}100_f1'])) > 1e-12:
            raise ValueError('Released event counts disagree with released full summary')
    result = {
        'date': '2026-11-09', 'audit_kind': 'released_predictions_replay_and_protocol_comparison',
        'new_model_inference_calls': 0,
        'original_qwen8_all4000': all_scores, 'original_qwen8_same_subset': subset_scores,
        'changed_protocol_FULL_RGB_same_subset': changed,
        'same_subset_replay': {'events': len(original), 'metric_counts_per_event': 27,
            'compared_metric_counts': len(original) * 27, 'mismatch_count': len(differences),
            'mismatches': differences,
            'released_prediction_limitation': 'Original mention labels are omitted; replay uses released source-normalized panel points.'},
        'evaluation': {'coordinate_mode': 'panel', 'threshold_units': 'original source-frame pixels',
            'source_sizes': sorted({(int(r['image_width']), int(r['image_height'])) for r in manifest_rows}),
            'evaluator_sha256': sha(evaluator_path),
            'identical_to_released_evaluator_copy': sha(evaluator_path) == sha(evaluator_copy)},
        'protocol_changes': {
            'original': {'input': '3 chronological full RGB + current-gaze crop paired images',
                'full_view_size': [768, 768], 'crop_source_size': [672, 672],
                'crop_output_size': [768, 768], 'paired_image_size': [1554, 768],
                'prompt_mode': 'expected_count', 'target_unit': 'one selected panel point per physical referent',
                'coordinates': 'full_panel or gaze_crop normalized 0..1; original paired_canvas_map',
                'part_target': 'best visible point on indicated part/area',
                'precision': 'BF16, no quantization', 'max_new_tokens': 768},
            'changed': {'input': '3 chronological four-area composites; FULL_RGB repeats full frame four times',
                'source_downsample_before_crop': [512, 512], 'composite_size': [1024, 1080],
                'crop_source_size_at_downsampled_scale': [205, 205],
                'prompt_mode': 'frozen parsed mentions + soft instruction-phase hints; no original expected_count',
                'target_unit': 'one output row per mention; optional one or more target points',
                'coordinates': 'entire composite xy on 0..1000 scale; new script inversion',
                'part_target': 'locate indicated whole parent',
                'precision': 'BF16, no quantization', 'max_new_tokens': 1536},
            'annotated_physical_referents': annotated_targets, 'valid_evaluation_referents': summary['gt_referents'],
            'parsed_mention_rows': mentions, 'events_with_different_task_count': len(task_differences),
            'task_count_differences': task_differences,
            'parser_failed_events': sum(not e['parser_valid'] for e in model_manifest['events'])},
        'panel_prediction_counts': {'original': dict(sorted(original_frames.items())),
                                    'changed_FULL_RGB': dict(sorted(changed_frames.items()))},
        'interpretation': {
            'FULL_RGB_is_original_baseline': False,
            'subset_change_explains_main_drop': False,
            'evaluator_change_explains_main_drop': False,
            'individual_causal_contributions_identified': False,
            'claim_supported': 'The original baseline was not preserved; multiple protocol changes were confounded.',
            'heatmap_effect_on_original_baseline_identified': False},
        'source_sha256': hashes_before,
    }
    for metric in ('point_100', 'joint_100'):
        result.setdefault('score_differences_pp', {})[metric] = {
            'subset_minus_full_original': 100 * (subset_scores[metric]['f1'] - all_scores[metric]['f1']),
            'changed_minus_original_same_subset': 100 * (changed[metric]['f1'] - subset_scores[metric]['f1'])}
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'original_qwen8_same32_replay_predictions.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(adapted[0])); writer.writeheader(); writer.writerows(adapted)
    hashes_after = {str(p): sha(p) for p in protected_paths}
    if hashes_before != hashes_after:
        raise ValueError('A protected source changed during the audit')
    result['protected_sources_unchanged'] = True
    (output / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('same_subset_replay', 'score_differences_pp', 'panel_prediction_counts',
                                             'protected_sources_unchanged')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trimodal', type=Path, default=ROOT.parent / 'TriModal-Referring')
    parser.add_argument('--study', type=Path, default=ROOT / '.local-sync/projected2d_routed_composite_v2_20261109')
    parser.add_argument('--output', type=Path, default=ROOT / '.local-sync/projected2d_baseline_audit_20261109')
    run(parser.parse_args())
