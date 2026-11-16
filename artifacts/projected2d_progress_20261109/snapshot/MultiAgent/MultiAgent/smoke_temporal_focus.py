"""Paired temporal-focus experiment reusing frozen parsing, frames and hand cues."""
import argparse
import json
from pathlib import Path

from .context import load_contexts
from .evaluate import evaluate_results
from .pipeline import EXPERIMENTS, run_experiment
from .qwen_backend import QwenBackend
from .temporal_focus import VERSION, focused_visual
from .vision_reasoning import visual_evidence, monolithic_coordinator


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--events', nargs='+', required=True)
    args = p.parse_args()
    baseline_summary = json.loads((args.baseline/'summary.json').read_text())
    manifest = json.loads(Path(baseline_summary['manifest']).read_text())
    manifest['samples'] = [s for s in manifest['samples'] if s['event_id'] in args.events]
    contexts = load_contexts(manifest)
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    backend = QwenBackend(out/'cache')
    rows, baseline_rows = [], []
    for index, ctx in enumerate(contexts, 1):
        key = f'{ctx.scene_key}/{ctx.event_id}'
        source = args.baseline/'traces'/ctx.scene_key/ctx.event_id
        trace = json.loads((source/'trace.json').read_text())
        frames = trace['visual']['frames']
        paths = [str(Path(f['image_rgb']).with_name('view.jpg')) for f in frames]
        isolated = {f['frame_id']: str(source/'visual_isolated'/f"{f['frame_id']}.jpg") for f in frames}
        parse, plan = trace['referential'], trace['temporal_plan']
        print(f'[{index}/{len(contexts)}] {key} focused Visual', flush=True)
        visual = focused_visual(visual_evidence, backend, ctx, parse, paths, frames, plan,
                                trace['hand'], True, False, isolated)
        print(f'[{index}/{len(contexts)}] {key} coordinator', flush=True)
        coordinator = monolithic_coordinator(backend, ctx, parse, paths, frames, visual,
                            use_spatial_cues=not plan['alignment'].get('audio_timing_rejected', False))
        target = out/'traces'/ctx.scene_key/ctx.event_id; target.mkdir(parents=True, exist_ok=True)
        (target/'trace.json').write_text(json.dumps({**trace, 'focus_version': VERSION,
            'frozen_baseline': str(source), 'visual': visual, 'global_coordinator': coordinator},
            ensure_ascii=False, indent=2))
        rows.extend(run_experiment([ctx], EXPERIMENTS['V3'], visual_predictions={key: visual},
            qwen_mentions={key: parse}, monolithic_predictions={key: coordinator}, output_dir=out/'V3'))
        baseline_rows.append(json.loads((args.baseline/'V3'/ctx.scene_key/f'row_{ctx.row_index}.json').read_text()))
        summary = {'version': VERSION, 'completed': len(rows), 'total': len(contexts),
                   'frozen_inputs': ['referential', 'temporal_plan', 'RGB', 'projection', 'hand'],
                   'baseline_metrics': evaluate_results(baseline_rows), 'focus_metrics': evaluate_results(rows)}
        (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
