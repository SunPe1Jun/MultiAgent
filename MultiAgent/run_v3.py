"""ASR-timed visual pipeline: mention clip -> gaze view -> regions -> hand cue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .asr import OfflineASR, align_mentions, clock_alignment, guard_clock
from .baseline import WORKSPACE_ROOT, module
from .context import load_contexts
from .evaluate import evaluate_by_scene, evaluate_results
from .pipeline import EXPERIMENTS, run_experiment
from .preprocess import plan_frames, prepare_views, prepare_isolated_views, VERSION
from .qwen_backend import QwenBackend
from .regions import SceneRegionSegmenter
from .run import evaluate_original
from .vision_reasoning import hand_cues, visual_evidence, monolithic_coordinator
from .temporal_focus import focused_visual
from .agents import ReferentialAgent, sample_key


def main():
    import sys
    if '--align-frame-poses' in sys.argv[1:]:
        from .frame_pose_alignment import main as align_frame_poses
        return align_frame_poses([arg for arg in sys.argv[1:] if arg != '--align-frame-poses'])
    if '--projected2d-study' in sys.argv[1:]:
        from .projected2d_attention_study import main as run_projected2d_study
        return run_projected2d_study([arg for arg in sys.argv[1:] if arg != '--projected2d-study'])
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--experiment', choices=[*EXPERIMENTS, 'all'], default='V3')
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--asr-model', default=str(WORKSPACE_ROOT / 'ASR/faster-whisper-small.en'))
    p.add_argument('--asr-device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--asr-compute-type', default='float16')
    p.add_argument('--timing-mode', choices=['hybrid', 'direct'], default='hybrid',
                   help='Hybrid uses ASR only as supplemental timing hints; direct uses instruction/event timing only.')
    p.add_argument('--qwen-model', default=str(WORKSPACE_ROOT / 'Qwen3-VL-30B-A3B-Instruct'))
    p.add_argument('--base-url', help='Remote OpenAI-compatible service; no local GPU used.')
    p.add_argument('--api-key-env', default='VRTRIREF_API_KEY')
    p.add_argument('--visual-study', choices=['temporal', 'organization'],
                   help='Paired frozen-upstream Visual input study, using this same entrypoint.')
    p.add_argument('--local-dense-video-visual', action='store_true',
                   help='Run local Qwen3-VL Referential + Visual over dense event-video frames; stop before fusion.')
    p.add_argument('--visual-fps', type=float, default=4.0,
                   help='Target Qwen video temporal groups/sec; approximately twice as many source RGB frames/sec.')
    p.add_argument('--visual-min-frames', type=int, default=32,
                   help='Minimum event frames supplied to each local Visual call.')
    p.add_argument('--visual-max-frames', type=int, default=100,
                   help='Maximum event frames supplied to each local Visual call.')
    p.add_argument('--visual-max-pixels', type=int, default=8388608,
                   help='Qwen video processor total pixel budget; controls visual tokens while preserving source-frame times.')
    p.add_argument('--backend-namespace', default='local_dense_video_visual_8b_20261022_v6',
                   help='Isolated local inference cache namespace.')
    p.add_argument('--replay-from', help='Reuse and verify a previous local Visual run\'s Referential and source frames.')
    p.add_argument('--visual-call-granularity', choices=['mention', 'event'], default='mention')
    p.add_argument('--visual-window-groups', type=int, default=0,
                   help='Split all source frames into continuous two-frame groups; 0 uses the entire sequence.')
    p.add_argument('--study-config')
    p.add_argument('--cpu-only', action='store_true')
    p.add_argument('--finalize-study', action='store_true',
                   help='Rebuild audit and paired reports from saved study predictions without inference.')
    p.add_argument('--grounding-model', default=str(WORKSPACE_ROOT / 'grounding-dino-base'))
    p.add_argument('--sam-model', default=str(WORKSPACE_ROOT / 'sam-base'))
    p.add_argument('--regions', action='store_true')
    p.add_argument('--no-gaze', action='store_true')
    p.add_argument('--mention-isolation', action='store_true',
                   help='Run one Visual call per mention with only its candidate 3D/2D context.')
    p.add_argument('--mention-focus', action='store_true',
                   help='Use local-first per-mention frames, expanding to the event only without local evidence.')
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    if args.finalize_study:
        from .visual_study import finalize_saved_study
        return finalize_saved_study(args.output_dir)
    if args.local_dense_video_visual:
        from .local_video_visual import run as run_local_video_visual
        return run_local_video_visual(args)
    if args.visual_study:
        from .visual_study import run_study
        return run_study(args)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    contexts = load_contexts(json.loads(Path(args.manifest).read_text()))
    if args.limit: contexts = contexts[:args.limit]
    names = list(EXPERIMENTS) if args.experiment == 'all' else [args.experiment]
    # The unified workflow always observes Hand as a Visual conditioning cue.
    # It is not emitted as a separate evidence stream by the coordinator.
    sources = {'visual', 'hand'}
    qwen = QwenBackend(args.cache_dir, args.qwen_model, base_url=args.base_url)
    asr = OfflineASR(args.asr_model, out/'cache'/'asr', args.asr_device, args.asr_compute_type)
    segmenter = SceneRegionSegmenter(args.grounding_model, args.sam_model, args.asr_device) if args.regions else None
    referential = ReferentialAgent(qwen)
    parses, visual, hand, coordinators = {}, {}, {}, {}
    for index, ctx in enumerate(contexts, 1):
        key = sample_key(ctx); print(f'[{index}/{len(contexts)}] {key} Referential', flush=True)
        parse = parses[key] = referential.run(ctx)
        duration = module('local').probe_video_duration(Path(ctx.video_path), None)
        if duration is None: raise RuntimeError(f'No video duration: {ctx.video_path}')
        clock = guard_clock(clock_alignment(ctx), duration)
        if args.timing_mode == 'hybrid' and not clock.get('audio_timing_rejected'):
            alignment = align_mentions(ctx, parse, asr, out/'cache')
            alignment['clock'] = clock
        else:
            alignment = {'status': 'clock_rejected' if clock.get('audio_timing_rejected') else 'instruction_event_only', 'clock': clock,
                         'mentions': {}, 'instruction_coverage': 0.0,
                         'policy': 'disabled_semantic_asr'}
        plan = plan_frames(ctx, parse['mentions'], alignment, duration,
                           timed=args.timing_mode == 'hybrid')
        trace_dir = out/'traces'/ctx.scene_key/ctx.event_id; trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir/'referential.json').write_text(json.dumps(parse, ensure_ascii=False, indent=2))
        (trace_dir/'asr_alignment.json').write_text(json.dumps(alignment, ensure_ascii=False, indent=2))
        (trace_dir/'temporal_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2))
        prepared_visual = prepare_views(ctx, parse, plan, trace_dir/'visual',
                                        use_gaze=not args.no_gaze, segmenter=segmenter)
        isolated_paths = (prepare_isolated_views(prepared_visual[1], trace_dir/'visual_isolated')
                          if args.mention_isolation or args.mention_focus else None)
        observations = None
        if 'hand' in sources:
            print(f'[{index}/{len(contexts)}] {key} Hand RGB cue', flush=True)
            hand[key] = hand_cues(qwen, ctx, parse,
                                 [f['image_rgb'] for f in prepared_visual[1]], prepared_visual[1])
            observations = hand[key]
            (trace_dir/'hand_observations.json').write_text(json.dumps(hand[key], ensure_ascii=False, indent=2))
        if 'visual' in sources:
            print(f'[{index}/{len(contexts)}] {key} Visual', flush=True)
            if args.mention_focus:
                visual[key] = focused_visual(
                    visual_evidence, qwen, ctx, parse, prepared_visual[0], prepared_visual[1], plan,
                    observations, not args.no_gaze, segmenter is not None, isolated_paths)
            else:
                visual[key] = visual_evidence(qwen, ctx, parse, prepared_visual[0], prepared_visual[1], plan,
                                              cues=observations, use_gaze=not args.no_gaze,
                                              use_regions=segmenter is not None,
                                              isolate_mentions=args.mention_isolation,
                                              mention_paths=isolated_paths)
            print(f'[{index}/{len(contexts)}] {key} Global coordinator', flush=True)
            coordinators[key] = monolithic_coordinator(qwen, ctx, parse, prepared_visual[0],
                                                       prepared_visual[1], visual[key],
                                                       use_spatial_cues=not plan['alignment'].get('audio_timing_rejected', False))
        trace = {'pipeline_version': VERSION, 'scene_key':ctx.scene_key, 'event_id':ctx.event_id,
                 'referential':parse, 'asr_alignment':alignment, 'temporal_plan':plan,
                 'visual':visual.get(key), 'hand':hand.get(key), 'global_coordinator':coordinators.get(key),
                 'context_provenance':ctx.provenance}
        (trace_dir/'trace.json').write_text(json.dumps(trace, ensure_ascii=False, indent=2))
        # Persist each completed event so an interrupted smoke remains inspectable.
        run_experiment([ctx], EXPERIMENTS[names[0]], visual_predictions=visual,
                       qwen_mentions=parses, monolithic_predictions=coordinators,
                       output_dir=out/names[0])
        (out/'progress.json').write_text(json.dumps({'completed':index, 'total':len(contexts),
                                                    'last_event':key, 'version':VERSION}, indent=2))
    summary = {'manifest':str(Path(args.manifest).resolve()), 'n_contexts':len(contexts),
               'version':VERSION, 'asr_model':args.asr_model, 'timing_mode':args.timing_mode,
               'timing_policy':'instruction_event_plus_asr_hint' if args.timing_mode == 'hybrid' else 'instruction_event_only',
               'regions':args.regions, 'experiments':{}}
    summary['mention_focus'] = args.mention_focus
    for name in names:
        results = run_experiment(contexts, EXPERIMENTS[name], visual_predictions=visual,
                                 qwen_mentions=parses, hand_visual_cues=hand,
                                 monolithic_predictions=coordinators,
                                 output_dir=out/name)
        summary['experiments'][name] = {'metrics':evaluate_results(results),
                                        'by_scene':evaluate_by_scene(results),
                                        'original_evaluator':evaluate_original(results,out/name)}
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k:v['metrics'] for k,v in summary['experiments'].items()}, indent=2), flush=True)


if __name__ == '__main__': main()
