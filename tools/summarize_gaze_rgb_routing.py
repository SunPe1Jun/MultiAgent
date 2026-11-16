#!/usr/bin/env python3
"""Summarize frozen predictions and original-evaluator results; no model calls."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from MultiAgent.full_interaction_referring import paired_set_metrics
from MultiAgent.heatmap_temporal_pilot import save, sha
from tools.run_gaze_rgb_routing import OUTPUT, check_lock, read


def micro_contrast(events, control, replicates=5000):
    if not events:
        return None
    def score(rows, condition):
        return paired_set_metrics([{'gt': r['gt'], 'prediction': r['predictions'][condition]} for r in rows])['micro_f1']
    rng = np.random.default_rng(20261109)
    samples = []
    for indices in rng.integers(0, len(events), size=(replicates, len(events))):
        rows = [events[index] for index in indices]
        samples.append(score(rows, 'GAZE_RGB') - score(rows, control))
    return {'difference': score(events, 'GAZE_RGB') - score(events, control),
            'interval_95': np.quantile(samples, [.025, .975]).tolist(),
            'replicates': replicates, 'resampling_unit': 'interaction, all conditions paired',
            'scope': 'Exploratory eight-interaction pilot; empirical interval does not establish generalization or equivalence.'}


def summarize(output):
    manifest = check_lock(output)
    analysis = read(output / 'analysis.json')
    frozen = read(output / 'predictions_frozen.json')
    for request in manifest['requests']:
        if sha(output / 'responses' / f"{request['id']}.json") != frozen['response_sha256'][request['id']]:
            raise ValueError('Response changed after evaluation')
    counts = {}
    for condition in manifest['conditions']:
        responses = [read(output / 'responses' / f"{e['id']}_{condition}.json") for e in manifest['events']]
        mention_rows = [r.get('mention_validation', {}) for r in responses]
        counts[condition] = {'schema_valid': sum(r['validation']['schema_valid'] for r in responses),
            'coverage_complete': sum(r['validation']['coverage_complete'] for r in responses),
            'expected_mentions': sum(len(e['mention_hints']) for e in manifest['events']),
            'grounded_rows': sum(len(r.get('grounded_referents', [])) for r in mention_rows),
            'unresolved_rows': sum(len(r.get('unresolved_mentions', [])) for r in mention_rows),
            'missing_ids': sum(len(r.get('missing_mention_ids', [])) for r in mention_rows),
            'decoder_reasons': dict(Counter(d['reason'] for r in responses for d in r['decoder']['decisions']))}
        if abs(analysis['conditions'][condition]['set_metrics']['micro_f1'] -
               analysis['conditions'][condition]['original_evaluator']['micro_f1']) > 1e-12:
            raise ValueError('Independent set metrics differ from original evaluator')
    strata = {}
    for label, events in [('all', analysis['events']),
                          ('supported', [e for e in analysis['events'] if e['supported_frames'] > 0]),
                          ('fallback_only', [e for e in analysis['events'] if e['supported_frames'] == 0])]:
        strata[label] = {condition: paired_set_metrics([{'gt': e['gt'], 'prediction': e['predictions'][condition]} for e in events])
                        for condition in manifest['conditions']}
    result = {'manifest_sha256': sha(output / 'manifest.json'), 'counts': counts, 'strata': strata,
        'paired_micro_f1': {control: micro_contrast(analysis['events'], control)
                            for control in ('FULL_RGB', 'UNIFORM_RGB', 'MIRRORED_GAZE_RGB')},
        'original_evaluator_agreement': True,
        'raw_green_recorder_marks_retained': True,
        'direct_heatmap_matched_control_run': False,
        'interpretation': 'Measures script-routed RGB details plus provenance. No claim that heatmaps in general are useful/useless; no default-pipeline replacement.'}
    save(output / 'summary.json', result)
    lines = ['# Script-routed RGB pilot — 2026-11-09', '',
        'Heat/duration maps stay inside the script. The model receives full-field RGB and supplementary RGB details. The complete instruction and scene candidate space are retained.', '',
        f"{len(manifest['events'])} interactions; {len(manifest['requests'])} condition results; {analysis['unique_visual_requests']} unique visual requests; "
        f"schema {analysis['schema_valid']}/{len(manifest['requests'])}, mention coverage {analysis['coverage_complete']}/{len(manifest['requests'])}, "
        f"vision consumption verified {analysis['vision_verified']}/{len(manifest['requests'])}. Visual tokens: {analysis['expanded_video_tokens']}.", '',
        '| Condition | TP/FP/FN | Micro-F1 | Exact-set | Any-hit |',
        '|---|---|---:|---:|---:|']
    for condition, report in analysis['conditions'].items():
        m = report['set_metrics']
        lines.append(f"| {condition} | {m['tp']}/{m['fp']}/{m['fn']} | {m['micro_f1']:.4f} | {m['exact_set_count']}/{m['events']} | {m['any_hit_count']}/{m['events']} |")
    lines += ['', 'The same original evaluator scores every condition after all predictions are frozen. Missing/schema-invalid mentions score according to the frozen failure policy. No post-hoc answer repair.', '',
        'Paired Micro-F1 contrasts (GAZE_RGB minus control; interaction bootstrap):', '']
    for control, contrast in result['paired_micro_f1'].items():
        lo, hi = contrast['interval_95']
        lines.append(f"- {control}: {100*contrast['difference']:+.2f} percentage points; empirical 95% interval [{100*lo:.2f}, {100*hi:.2f}].")
    lines += ['', 'These are Micro-F1 differences. The separate analysis.json paired_event_f1_contrasts are differences of mean per-interaction F1 and must not be interchanged.', '',
        'Limitations: only eight interactions, five with supported attention and three full RGB fallbacks; instructions/actors parsed by a model; original recorder marks remain; source RGB is already resized; internal history projection and anchor-center decoding are approximate. Project-new means absent from locally available historical artifact addresses, not absent from missing-server history or model pretraining. No simultaneous direct-heatmap comparison was run.', '',
        'Artifacts: [viewer](index.html), [comparison](comparison.jpg), [analysis](analysis.json), [summary](summary.json), [input verification](verification.json), [routing intervention audit](routing_intervention_audit.json).', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({'conditions': strata['all'], 'paired_micro_f1': result['paired_micro_f1']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    summarize(parser.parse_args().output.resolve())
