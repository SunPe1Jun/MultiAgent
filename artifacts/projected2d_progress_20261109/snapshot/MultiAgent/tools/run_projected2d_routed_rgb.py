#!/usr/bin/env python3
"""Paired RGB routing on reconstructed Experiment 2 panels."""
from __future__ import annotations

import argparse
from argparse import Namespace
from collections import Counter, defaultdict
import json
from pathlib import Path

from MultiAgent.baseline import ROOT as TRI
from MultiAgent.projected2d_attention_study import load_module, read_csv, save_json, sha_file, write_csv

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / '.local-sync/referent_rgb_expansion32_20261109_repaired'
ORIGINAL = ROOT / '.local-sync/referent_rgb_expansion32_20261109'
PAPER_GT = TRI / 'paper_experiment_evidence/manifests/experiment2_projected_gt_manifest.csv'
OUTPUT = ROOT / '.local-sync/projected2d_routed_rgb_20261109'


def build_panels(output: Path):
    """Rebuild original panel selection and compare against the released point manifest."""
    selection = json.loads((PRIOR / 'selection.json').read_text())
    samples = selection['samples']
    gt_rows = read_csv(PAPER_GT)
    gt = defaultdict(list)
    for row in gt_rows:
        gt[(row['scene'], int(row['row_index']))].append(row)
    builder = load_module('projected2d_routed_builder', TRI / 'exam2/build_2d_eval_manifest.py')
    args = Namespace(ffmpeg_path='ffmpeg', ffprobe_path='ffprobe', panels=3,
        video_time_offset_seconds=0.0, auto_video_time_offset=True,
        auto_video_time_offset_source='hybrid', hybrid_offset_threshold_seconds=1.0,
        hybrid_video_time_bias_seconds=0.5, no_extract_frames=False, overwrite_frames=False,
        min_panel_time_gap_seconds=0.25, keep_duplicate_frames=False, acceptable_top_k=2,
        sample_inner_margin_ratio=0.15, panel_selection_strategy='evidence',
        candidate_step_seconds=0.5, evidence_segment_duration_seconds=0.5,
        gaze_stability_window_seconds=0.3, image_width=None, image_height=None,
        start_index=0, limit=None)
    adapter = output / 'reconstructed_inputs'
    all_rows, summaries = [], []
    for scene in sorted({s['scene_key'] for s in samples}):
        scene_samples = [s for s in samples if s['scene_key'] == scene]
        max_index = max(s['row_index'] for s in scene_samples)
        api = [{} for _ in range(max_index + 1)]
        match = []
        for sample in scene_samples:
            index = sample['row_index']
            cached = json.loads((PRIOR / 'inputs' / scene / f'{index}_row.json').read_text())['row']
            if cached['event_id'] != sample['event_id']:
                raise ValueError('Event address mismatch')
            api[index] = {k: v for k, v in cached.items() if not k.startswith('_')}
            names = sorted({r['referent_name'] for r in gt[(scene, index)] if r['referent_name']})
            match.append({'row_index': index, 'gt_referents_mapped': ', '.join(names)})
        api_fields = list(dict.fromkeys(k for row in api for k in row))
        api_path = adapter / f'{scene}_api_input.csv'
        eval_path = adapter / f'{scene}_match_eval.csv'
        write_csv(api_path, api, api_fields)
        write_csv(eval_path, match, ['row_index', 'gt_referents_mapped'])
        paths = builder.ScenePaths(scene, api_path, eval_path,
            TRI / 'paper_experiment_evidence/anchor_tables' / f'{scene}_anchor_table.tsv')
        indices = {s['row_index'] for s in scene_samples}
        choose = builder.choose_rows
        try:
            builder.choose_rows = lambda rows, start, limit: [(i, rows[i]) for i in sorted(indices)]
            rows, summary = builder.process_scene(paths, output / 'manifest_build', args, [])
        finally:
            builder.choose_rows = choose
        all_rows.extend(rows)
        summaries.append(summary)
        print('PANELS', scene, 'events', len(indices), 'rows', len(rows), flush=True)
    builder.write_manifest(output / 'rebuilt_manifest_all.csv', all_rows)
    key = lambda r: (r['scene'], int(r['row_index']), r['referent_name'], r['panel_id'])
    expected = {key(r): r for s in samples for r in gt[(s['scene_key'], s['row_index'])]}
    actual = {key(r): r for r in all_rows}
    columns = ('gt_x', 'gt_y', 'projection_valid', 'status')
    mismatches = []
    for k in sorted(set(expected) | set(actual)):
        a, b = expected.get(k), actual.get(k)
        if a is None or b is None or any(str(a.get(c, '')) != str(b.get(c, '')) for c in columns):
            mismatches.append({'key': k,
                'released': None if a is None else {c: a.get(c) for c in columns},
                'rebuilt': None if b is None else {c: b.get(c) for c in columns}})
    by_event = Counter((r['scene'], int(r['row_index'])) for r in all_rows if r['panel_id'])
    audit = {'sample_count': len(samples), 'released_rows': len(expected), 'rebuilt_rows': len(actual),
        'matching_rows': len(set(expected) | set(actual)) - len(mismatches),
        'mismatch_count': len(mismatches), 'mismatches': mismatches,
        'event_rows': {f'{a}/{b}': n for (a, b), n in by_event.items()},
        'source_selection_sha256': sha_file(PRIOR / 'selection.json'),
        'source_manifest_sha256': sha_file(PAPER_GT), 'builder_summaries': summaries,
        'reconstructed_event_windows': True, 'original_api_inputs_recovered': False,
        'matching_projection_is_not_independent_pixel_calibration': True}
    save_json(output / 'panel_parity_audit.json', audit)
    print('PARITY', audit['matching_rows'], '/', len(set(expected) | set(actual)),
          'mismatches', len(mismatches), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['build-panels', 'prepare', 'infer', 'evaluate', 'exact'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.stage == 'build-panels':
        build_panels(args.output)
    elif args.stage == 'prepare':
        from MultiAgent.projected2d_routed_rgb import prepare_inputs
        prepare_inputs(args.output)
    elif args.stage == 'infer':
        from MultiAgent.projected2d_routed_rgb import infer
        infer(args.output)
    elif args.stage == 'evaluate':
        from MultiAgent.projected2d_routed_rgb import evaluate
        evaluate(args.output)
    elif args.stage == 'exact':
        from MultiAgent.projected2d_exact_set import run
        run(args.output)


if __name__ == '__main__':
    main()
