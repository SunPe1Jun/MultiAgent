"""Post-prediction 2D duplicate accounting and exact-match diagnostics."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import subprocess
import sys

from .baseline import ROOT as TRI
from .projected2d_attention_study import CSV_COLUMNS, evaluator_adapter, read_csv, save_json, write_csv
from .projected2d_routed_rgb import GROUPS

MAX_SAME_FRAME_DISTANCE_PX = 50.0


def collapse_explicit_coreference(parsed, mentions, panels):
    """Collapse only explicit parent-linked singular points agreeing in one source frame."""
    result = copy.deepcopy(parsed)
    definitions = {m['mention_id']: m for m in mentions}
    by_id = {m['mention_id']: m for m in result.get('mentions', [])}
    source_sizes = {p['frame_id']: p['source_size'] for p in panels}
    merged = []
    for mid, definition in definitions.items():
        parent = definition.get('parent_mention_id')
        if not parent or parent not in definitions or parent not in by_id or mid not in by_id:
            continue
        a, b = definitions[parent], definition
        if not a.get('coreference_id') or a.get('coreference_id') != b.get('coreference_id'):
            continue
        if a.get('count') not in (None, 1) or b.get('count') not in (None, 1):
            continue
        primary, other = by_id[parent]['predictions'], by_id[mid]['predictions']
        if len(primary) != 1 or len(other) != 1 or primary[0]['frame_id'] != other[0]['frame_id']:
            continue
        frame_id = primary[0]['frame_id']
        if frame_id not in source_sizes:
            continue
        width, height = source_sizes[frame_id]
        x1, y1 = primary[0]['point_uv']
        x2, y2 = other[0]['point_uv']
        distance = math.hypot((x1-x2)*width, (y1-y2)*height)
        if distance > MAX_SAME_FRAME_DISTANCE_PX:
            continue
        by_id[mid]['predictions'] = []
        merged.append({'parent_mention_id': parent, 'removed_duplicate_mention_id': mid,
                       'frame_id': frame_id, 'distance_px': distance})
    return result, merged


def exact_counts(detail):
    result = {}
    for kind in ('point', 'joint'):
        for threshold in (50, 100, 150, 200):
            eligible_key = 'gt_referent_count' if kind == 'point' else 'time_evaluable_referent_count'
            eligible = [r for r in detail if int(r[eligible_key]) > 0]
            exact = sum(int(r[f'{kind}_fp_{threshold}']) == 0 and int(r[f'{kind}_fn_{threshold}']) == 0
                        for r in eligible)
            result[f'{kind}_exact_{threshold}'] = {'exact': exact, 'evaluable_events': len(eligible),
                                                    'all_events': len(detail)}
    return result


def run(output: Path):
    manifest = json.loads((output / 'input_manifest.json').read_text())
    if not (output / 'predictions_frozen.json').is_file():
        raise ValueError('Freeze all conditions before exact-set analysis')
    report = {'policy': {'explicit_parent_and_shared_coreference_required': True,
        'singular_and_same_frame_required': True, 'distance_px_max': MAX_SAME_FRAME_DISTANCE_PX,
        'uses_gt_for_deduplication': False, 'cross_frame_deduplication': False,
        'official_evaluator_fallback_matching': 'May match other referent labels by proximity; exact is 2D proxy exact, not semantic identity exact.'},
        'conditions': {}}
    for group in GROUPS:
        predictions, merges = [], []
        for event in manifest['events']:
            p = output / 'responses' / f"{event['id']}_{group}.json"
            response = json.loads(p.read_text())
            parsed, merged = collapse_explicit_coreference(response.get('parsed', {}), event['mentions'], event['panels'])
            adapted = evaluator_adapter(parsed)
            predictions.append({'scene': event['scene_key'], 'row_index': event['row_index'],
                'event_id': event['event_id'], 'instruction': event['instruction'],
                'model_input_image': ';'.join(x['path'] for x in response.get('input_images', [])),
                'model_raw_output': response.get('raw_response') or '',
                'parsed_json': json.dumps(adapted, ensure_ascii=False),
                'parse_ok': str(response['status'] in ('COMPLETE', 'PARSE_WITH_WARNINGS')),
                'prediction_count': len(adapted['referents']), 'error_message': response.get('error', '')})
            merges.extend({'event_id': event['id'], **m} for m in merged)
        pred = output / 'predictions' / f'{group}_coref_collapsed.csv'
        write_csv(pred, predictions, CSV_COLUMNS)
        target = output / 'evaluation_coref_collapsed' / group
        subprocess.run([sys.executable, str(TRI / 'exam2/evaluate_2d_point_grounding.py'),
            '--manifest', str(output / 'rebuilt_manifest_all.csv'), '--pred_csv', str(pred),
            '--output_dir', str(target), '--coordinate_mode', 'panel'],
            cwd=TRI, check=True, capture_output=True, text=True)
        raw_detail = read_csv(output / 'evaluation' / group / '2d_eval_detail.csv')
        fixed_detail = read_csv(target / '2d_eval_detail.csv')
        report['conditions'][group] = {'merge_count': len(merges), 'merges': merges,
            'raw_exact': exact_counts(raw_detail), 'coref_collapsed_exact': exact_counts(fixed_detail),
            'coref_collapsed_metrics': json.loads((target / '2d_eval_summary.json').read_text())}
    save_json(output / 'exact_set_analysis.json', report)


if __name__ == '__main__':
    run(Path(sys.argv[1]))
