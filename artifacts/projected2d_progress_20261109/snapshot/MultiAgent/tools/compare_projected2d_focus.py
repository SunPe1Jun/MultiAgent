#!/usr/bin/env python3
"""Evaluate the prior joint-prompt predictions on the fixed focused subset."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

from MultiAgent.baseline import ROOT as TRI
from MultiAgent.projected2d_attention_study import read_csv, save_json
from MultiAgent.projected2d_routed_rgb import GROUPS
from tools.report_projected2d_routed_rgb import counts, paired_delta


def run(source, focused):
    if not (source/'predictions_frozen.json').is_file() or not (focused/'predictions_frozen.json').is_file():
        raise ValueError('Both studies must have frozen predictions')
    manifest = json.loads((focused/'input_manifest.json').read_text())
    focused_summary = json.loads((focused/'summary.json').read_text())['metrics']
    report = {'sample_status': manifest['sample_status'], 'selection_rule': manifest['selection_rule'],
              'interpretation': 'Exploratory paired subset; mention isolation, image ordering and added model calls all changed together.',
              'groups': {}}
    for group in GROUPS:
        target = focused/'evaluation_source_joint_prompt'/group
        subprocess.run([sys.executable, str(TRI/'exam2/evaluate_2d_point_grounding.py'),
            '--manifest', str(focused/'rebuilt_manifest_all.csv'), '--pred_csv', str(source/'predictions'/f'{group}.csv'),
            '--output_dir', str(target), '--coordinate_mode', 'panel'],
            cwd=TRI, check=True, capture_output=True, text=True)
        prior = json.loads((target/'2d_eval_summary.json').read_text())
        prior_rows = {(r['scene'], r['row_index']): r for r in read_csv(target/'2d_eval_detail.csv')}
        new_rows = {(r['scene'], r['row_index']): r for r in read_csv(focused/'evaluation'/group/'2d_eval_detail.csv')}
        keys = sorted(prior_rows)
        if sorted(new_rows) != keys:
            raise ValueError('Focused and source event sets differ')
        frame_counts = {}
        for label, directory in [('joint_prompt', source), ('mention_focus', focused)]:
            counts_by_frame = Counter()
            for event in manifest['events']:
                response = json.loads((directory/'responses'/f"{event['id']}_{group}.json").read_text())
                for mention in response.get('parsed', {}).get('mentions', []):
                    for point in mention.get('predictions', []):
                        counts_by_frame[point['frame_id']] += 1
            frame_counts[label] = dict(counts_by_frame)
        calls = []
        for event in manifest['events']:
            for mention in event['mentions']:
                calls.append(json.loads((focused/'calls'/f"{event['id']}_{mention['mention_id']}_{group}.json").read_text()))
        report['groups'][group] = {'joint_prompt_metrics': prior,
            'mention_focus_metrics': focused_summary[group], 'frame_counts': frame_counts,
            'joint_prompt_logical_calls': len(manifest['events']),
            'mention_focus_logical_calls': len(calls),
            'mention_focus_unique_calls': sum('reused_identical_from' not in c for c in calls),
            'mention_focus_vision_verified': sum(bool(c.get('vision_verified')) for c in calls),
            'mention_focus_status': dict(Counter(c['status'] for c in calls)),
            'delta_joint100_f1': paired_delta([counts(new_rows[k], 'joint') for k in keys],
                                               [counts(prior_rows[k], 'joint') for k in keys]),
            'delta_point100_f1': paired_delta([counts(new_rows[k], 'point') for k in keys],
                                               [counts(prior_rows[k], 'point') for k in keys])}
    save_json(focused/'focus_vs_joint_prompt.json', report)
    for group, row in report['groups'].items():
        print(group, 'joint100', f"{row['joint_prompt_metrics']['joint_100_f1']:.4f}", '->',
              f"{row['mention_focus_metrics']['joint_100_f1']:.4f}",
              'P1', row['frame_counts'], 'logical_calls', row['mention_focus_logical_calls'], flush=True)


if __name__ == '__main__':
    run(Path(sys.argv[1]), Path(sys.argv[2]))
