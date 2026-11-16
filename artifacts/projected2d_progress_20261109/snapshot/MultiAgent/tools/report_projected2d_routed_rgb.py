#!/usr/bin/env python3
"""Report paired 2D evaluator counts and fixed duplicate-accounting diagnostics."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
import sys

from MultiAgent.projected2d_attention_study import read_csv, save_json, sha_file
from MultiAgent.projected2d_routed_rgb import GROUPS


def f1(tp, fp, fn):
    denominator = 2*tp + fp + fn
    return 2*tp/denominator if denominator else 0.


def paired_delta(a, b, *, seed=20261109, draws=5000):
    rng = random.Random(seed)
    pairs = list(zip(a, b))
    if not pairs:
        return None
    def delta(rows):
        counts = [sum(v[i] for v in rows) for i in range(6)]
        return f1(*counts[:3]) - f1(*counts[3:])
    combined = [tuple(x+y) for x, y in pairs]
    values = sorted(delta([rng.choice(combined) for _ in combined]) for _ in range(draws))
    return {'difference': delta(combined), 'bootstrap_95_percentile': [values[int(.025*draws)], values[int(.975*draws)]],
            'draws': draws, 'unit': 'interaction', 'seed': seed}


def counts(row, prefix):
    return [int(row[f'{prefix}_{part}_100']) for part in ('tp', 'fp', 'fn')]


def run(output):
    manifest = json.loads((output / 'input_manifest.json').read_text())
    frozen = json.loads((output / 'predictions_frozen.json').read_text())
    for path, digest in frozen['responses'].items():
        if sha_file(Path(path)) != digest:
            raise ValueError('Prediction changed after freeze: ' + path)
    policy = json.loads((output / 'exact_set_policy.json').read_text())
    if sha_file(Path(policy['source'])) != policy['source_sha256']:
        raise ValueError('Exact-set policy changed')
    summary = json.loads((output / 'summary.json').read_text())
    exact = json.loads((output / 'exact_set_analysis.json').read_text())
    rows = {g: {(r['scene'], r['row_index']): r for r in
                read_csv(output / 'evaluation' / g / '2d_eval_detail.csv')} for g in GROUPS}
    keys = sorted(rows[GROUPS[0]])
    if any(sorted(rows[g]) != keys for g in GROUPS):
        raise ValueError('Conditions have different event denominators')
    response_stats = {}
    for group in GROUPS:
        values = [json.loads((output / 'responses' / f"{e['id']}_{group}.json").read_text())
                  for e in manifest['events']]
        response_stats[group] = {'status': dict(Counter(r['status'] for r in values)),
            'vision_verified': sum(bool(r.get('vision_verified')) for r in values),
            'identical_input_reuse': sum('reused_identical_from' in r for r in values),
            'image_tokens': dict(Counter(str((r.get('processor_audit') or {}).get('expanded_image_token_count')) for r in values))}
    deltas = {}
    for control in ('FULL_RGB', 'UNIFORM_RGB', 'MIRRORED_GAZE_RGB'):
        for metric in ('point', 'joint'):
            deltas[f'GAZE_RGB_minus_{control}_{metric}100'] = paired_delta(
                [counts(rows['GAZE_RGB'][k], metric) for k in keys],
                [counts(rows[control][k], metric) for k in keys])
    all_panels = [p for e in manifest['events'] for p in e['panels']]
    audit = {'events': len(manifest['events']), 'panels': len(all_panels),
        'panels_clock_supported': sum(p['clock_supported'] for p in all_panels),
        'events_clock_supported': sum(any(p['clock_supported'] for p in e['panels']) for e in manifest['events']),
        'source_pts_status': dict(Counter(p['pts_verification'] for p in all_panels)),
        'parser_valid_events': sum(e['parser_valid'] for e in manifest['events']),
        'response_stats': response_stats, 'paired_differences': deltas,
        'matching_limit': 'Original evaluator falls back to proximity for nonmatching mention labels; not semantic identity accuracy.',
        'time_limit': 'Acceptable temporal panels are derived by original gaze-to-projected-anchor ranking; not independent human event-time annotations.',
        'geometry_limit': 'Projected anchor point and RGB alignment remain uncalibrated.'}
    save_json(output / 'report.json', audit)
    lines = ['# Projected-2D: RGB routing experiment', '',
        '**Exploratory; projected anchors and reconstructed temporal labels are not independently calibrated.**', '',
        f"Events: {audit['events']}; panels: {audit['panels']}; supported gaze clock: {audit['panels_clock_supported']} panels in {audit['events_clock_supported']} events.",
        'Sample scope: ' + manifest['sample_status'], '',
        '| Condition | Temporal F1 | Point@100 F1 | Joint@100 F1 | Joint exact@100 | Evaluable events |',
        '|---|---:|---:|---:|---:|---:|']
    for group in GROUPS:
        metric = summary['metrics'][group]
        e = exact['conditions'][group]['raw_exact']['joint_exact_100']
        lines.append(f"| {group} | {metric['time_f1']:.4f} | {metric['point_100_f1']:.4f} | {metric['joint_100_f1']:.4f} | {e['exact']} | {e['evaluable_events']}/{e['all_events']} |")
    lines += ['', 'Joint exact means no false positive and no false negative among evaluable referents under the original evaluator; its proximity fallback is not semantic object identity verification.', '',
        'The collapsed diagnostic merges only explicit parent-linked singular coreferences with close same-frame points, without GT; raw output remains primary.', '',
        'The original temporal acceptable-panel ranking itself uses projected anchor proximity to gaze. The released summary has no independent human time or pixel labels.']
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    run(Path(sys.argv[1]))
