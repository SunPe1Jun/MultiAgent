"""Instruction-only coverage checks and saved Visual-contract audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from .context import load_contexts
from .mention_mapping import actor_coverage_checks
from .projected2d_attention_study import build_visual_prompt, run_model, save_json, sha_file, write_csv


def coverage_checks(text, mentions):
    """High-confidence actor coverage only; not a complete language evaluator."""
    missing_actors = actor_coverage_checks(text, mentions)
    pronouns = [{'text': m.group(), 'span': [m.start(), m.end()]} for m in re.finditer(r'\b(?:it|them|its|their|he|she|him|her)\b', text, re.I)]
    return {'missing_explicit_actors': missing_actors, 'action_pronouns_for_review': pronouns,
            'language_coreference_notes_present': any(m.get('coreference_note') for m in mentions),
            'scope': 'language-only structural checks, not mention GT accuracy or complete noun-phrase recall'}


def audit(root: Path):
    selected = json.loads((root / 'fixed_samples.json').read_text())
    out = root / 'diagnostics' / 'referential'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ctx in load_contexts(selected, load_ground_truth=False):
        path = root / 'pilot20/samples' / ctx.scene_key / str(ctx.row_index) / 'referential.json'
        parsed = json.loads(path.read_text())
        mentions = parsed.get('mentions') or []
        audit_path = root / 'inputs' / ctx.scene_key / str(ctx.row_index) / 'input_audit.json'
        input_audit = json.loads(audit_path.read_text())
        coverage = coverage_checks(ctx.text, mentions)
        row = {'scene': ctx.scene_key, 'row_index': ctx.row_index, 'instruction': ctx.text,
               'parse_failure': parsed.get('parse_failure'), **coverage,
               'mentions': [{k: m.get(k) for k in ('mention_id', 'text', 'object_type', 'coreference_id',
                                                  'parent_text', 'anchor_text', 'coreference_note',
                                                  'mapping_status', 'candidate_ids')} for m in mentions],
               'old_visual_dropped_fields': ['object_type', 'component', 'parent_text', 'anchor_text',
                                             'coreference_id', 'parent_mention_id', 'coreference_note'],
               'new_prompt_model_status': 'NOT_RUN_ON_OLD_OUTPUTS'}
        prompt_path = out / f'{ctx.scene_key}_{ctx.row_index}_visual_contract_v2.txt'
        prompt_path.write_text(build_visual_prompt(ctx, mentions, input_audit['panels']))
        row['new_prompt_path_using_frozen_old_mentions'] = str(prompt_path)
        rows.append(row)
    summary = {'events': len(rows), 'parse_failures': sum(bool(r['parse_failure']) for r in rows),
               'events_with_missing_actor': sum(bool(r['missing_explicit_actors']) for r in rows),
               'events_with_action_pronouns': sum(bool(r['action_pronouns_for_review']) for r in rows),
               'ground_truth_used': False, 'events_audit': rows,
               'interpretation': 'Existing v1 scores cannot establish coreference/localization reliability when the actor was omitted or antecedent fields were not delivered.'}
    save_json(out / 'contract_audit.json', summary)
    write_csv(out / 'language_coverage.csv', rows, list(rows[0]))
    print(json.dumps({k: v for k, v in summary.items() if k != 'events_audit'}, ensure_ascii=False))
    return summary


def probe(root: Path, events):
    manifest = json.loads((root / 'fixed_samples.json').read_text())
    wanted = set(events)
    manifest['samples'] = [x for x in manifest['samples'] if f"{x['scene_key']}:{x['row_index']}" in wanted]
    contexts = load_contexts(manifest, load_ground_truth=False)
    frame_inputs, ctx_by_key = {}, {}
    for ctx in contexts:
        key = ctx.scene_key, ctx.row_index
        ctx_by_key[key] = ctx
        saved = json.loads((root / 'inputs' / ctx.scene_key / str(ctx.row_index) / 'input_audit.json').read_text())
        rgb = saved['conditions']['RGB']
        if not all(sha_file(Path(r['path'])) == r['sha256'] for r in rgb):
            raise ValueError('Frozen source RGB hashes differ from completed pilot')
        frame_inputs[key] = {'panel_meta': saved['panels'], 'condition_paths': {'RGB': [Path(r['path']) for r in rgb]}}
    output = root / 'diagnostics' / 'referential' / 'local_qwen8b_contract_v3_probe'
    samples = [{**s, 'instruction': ctx_by_key[(s['scene_key'], s['row_index'])].text} for s in manifest['samples']]
    save_json(output / 'probe_config.json', {'model': 'Qwen3-VL-8B-Instruct', 'quantization': 'int8',
        'groups': ['RGB'], 'samples': samples, 'image_selection_changed': False,
        'geometry_changed': False, 'goal': 'new Referential + explicit Visual contract smoke only; not heatmap efficacy evaluation',
        'ground_truth_loaded': False, 'evaluator_status': 'NOT_RUN_geometry_not_calibrated'})
    results, calls = run_model(('RGB',), samples, ctx_by_key, frame_inputs, output, len(samples), quantization='int8')
    summary = {'results': results, 'calls': calls, 'status': 'COMPLETE',
               'evaluator_status': 'NOT_RUN_geometry_not_calibrated'}
    save_json(output / 'probe_summary.json', summary)
    print(f'Completed explicit-task probe: {output}', flush=True)
    return summary


def render_probe(root: Path):
    """Predictions on unchanged RGB; no projected GT mixed into this check."""
    from PIL import Image, ImageDraw, ImageFont
    output = root / 'diagnostics/referential/local_qwen8b_contract_v3_probe'
    summary = json.loads((output / 'probe_summary.json').read_text())
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    audits = []
    for result in summary['results']['RGB']:
        scene, index = result['scene'], result['row_index']
        input_audit = json.loads((root / 'inputs' / scene / str(index) / 'input_audit.json').read_text())
        mention_rows = result.get('parsed', {}).get('mentions', [])
        width, height = 600, 600
        footer = 70 + len(mention_rows) * 24
        sheet = Image.new('RGB', (width * 3, height + footer), '#f8f8f8')
        draw = ImageDraw.Draw(sheet)
        for col, panel in enumerate(input_audit['panels']):
            x0 = col * width
            with Image.open(panel['source_path']) as image:
                sheet.paste(image.convert('RGB').resize((width, height)), (x0, 0))
            draw.text((x0 + 5, height + 5), f"{scene}/{index} {panel['frame_id']} video={panel['video_time_seconds']:.3f}s", fill='black', font=font)
            draw.text((x0 + 5, height + 27), 'Cyan=model v3; no GT shown; unchanged source RGB', fill='black', font=font)
            for n, mention in enumerate(mention_rows):
                draw.text((x0 + 5, height + 53 + n * 24), f"{mention['mention_id']} {mention['mention'][:30]}: {mention['status']}", fill='black', font=font)
                for pred in mention.get('predictions', []):
                    if pred['frame_id'] != panel['frame_id']:
                        continue
                    x, y = x0 + pred['point_uv'][0] * width, pred['point_uv'][1] * height
                    draw.ellipse((x-8, y-8, x+8, y+8), outline='cyan', width=3)
                    draw.text((x+10, y-13+n*2), mention['mention_id'], fill='cyan', font=font)
        path = output / f'{scene}_{index}_predictions_no_gt.jpg'
        sheet.save(path, quality=94)
        audits.append({'scene': scene, 'row_index': index, 'image': str(path),
                       'status': result['status'], 'parse_error': result.get('parse_error'),
                       'model_consumption_verified': (result.get('visual_tensor_trace') or {}).get('input_consumption_verified'),
                       'predictions': result.get('parsed'), 'metrics_status': 'NOT_RUN_geometry_not_calibrated'})
    save_json(output / 'probe_review.json', {'results': audits,
              'caution': 'Protocol completeness is not object identity or coordinate accuracy. Two samples, RGB only; not an attention-method comparison.'})
    return audits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--probe-events', nargs='+')
    parser.add_argument('--render-probe', action='store_true')
    args = parser.parse_args()
    if args.render_probe:
        render_probe(args.root)
    elif args.probe_events:
        probe(args.root, args.probe_events)
    else:
        audit(args.root)


if __name__ == '__main__':
    main()
