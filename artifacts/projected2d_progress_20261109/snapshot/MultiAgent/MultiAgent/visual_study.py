"""Paired Visual-only experiments with frozen language, timing, Hand and coordinator images."""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

from .agents import ReferentialAgent, sample_key
from .asr import clock_alignment, guard_clock
from .baseline import DATA_ROOT, module
from .context import load_contexts
from .evaluate import evaluate_results, evaluate_by_scene
from .gaze_evidence import (DEFAULTS, gaze_records, build_heatmap, render_presentation,
                           project_gaze, interpolate_camera_pose)
from .pipeline import EXPERIMENTS, run_experiment
from .preprocess import plan_frames, prepare_views
from .qwen_backend import QwenBackend
from .run import evaluate_original
from .vision_reasoning import hand_cues, visual_evidence, monolithic_coordinator


GROUPS = ['R', 'H_inst', 'H_local', 'F_inst', 'F_local']
VERSION = 'temporal_gaze_visual_only_20261021_v2_multiscale'


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def probe_video(path):
    process = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,avg_frame_rate:format=duration',
        '-of', 'json', str(path)], capture_output=True, text=True, check=True)
    return json.loads(process.stdout)


def _fixed_frames(frames, count=3):
    if len(frames) <= count:
        return list(frames)
    indices = [round(i*(len(frames)-1)/(count-1)) for i in range(count)]
    return [frames[i] for i in indices]


def fixed_video_anchor_frames(plan, count=3):
    """Pick an even fixed coverage of the same unweighted event anchors."""
    rows=[r for r in plan['records'] if 'event_context' in r['reasons']]
    return _fixed_frames(rows,count)


def audit_data(ctx, records, clock, video_probe):
    api = module('api')
    errors = []
    norm_errors = []
    for r in records:
        q, v, local_recorded = r['camera_rotation_xyzw'], r['gaze_vector_world'], r['camera_gaze_direction']
        if q is None or v is None or local_recorded is None:
            continue
        local = api.rotate_vector_by_quaternion(v, api.quaternion_conjugate(q))
        a, b = np.asarray(local), np.asarray(local_recorded)
        if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0:
            cos = float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
            errors.append(math.degrees(math.acos(max(-1., min(1., cos)))))
        norm_errors.append(abs(np.linalg.norm(v)-1.))
    positions = [np.asarray(r['camera_position']) for r in records if r['camera_position'] is not None]
    max_displacement = max((float(np.linalg.norm(p-positions[0])) for p in positions), default=None)
    diffs = np.diff([r['telemetry_time_seconds'] for r in records])
    valid_diffs = diffs[diffs > 0]
    return {'scene_key': ctx.scene_key, 'event_id': ctx.event_id,
            'video': video_probe, 'telemetry_samples': len(records),
            'telemetry_median_interval_seconds': float(np.median(valid_diffs)) if len(valid_diffs) else None,
            'telemetry_hz_measured': float(1/np.median(valid_diffs)) if len(valid_diffs) else None,
            'gaze_valid_fraction': sum(r['validity']=='valid' for r in records)/max(1, len(records)),
            'zero_world_intersections': sum(r['gaze_point_world'] == (0., 0., 0.) for r in records),
            'camera_max_displacement_world_units': max_displacement,
            'gaze_world_to_camera_direction_error_degrees_median': float(np.median(errors)) if errors else None,
            'gaze_world_to_camera_direction_error_degrees_p95': float(np.percentile(errors,95)) if errors else None,
            'gaze_vector_norm_error_p95': float(np.percentile(norm_errors,95)) if norm_errors else None,
            'camera_pose_semantics': 'recorded cameraPosition/Rotation; physical center-eye vs capture extrinsics undocumented',
            'conventions_verified': 'xyzw quaternion, inverse world-to-camera rotation, +z forward, +y up, image origin top-left',
            'conventions_assumed_not_calibrated': 'Unity left-handed world; vertical FOV, square pixels; no recording-camera extrinsic or distortion calibration',
            'source_camera': 'recorded pose; exact head/eye/recording-camera offset unavailable',
            'units': 'Unity world units; metric scaling not independently verified',
            'intrinsics': 'scalar cameraFOV only; no per-frame fx/fy/cx/cy or distortion coefficients',
            'depth_or_mesh_available': False, 'dynamic_object_transforms_available': False,
            'dynamic_world_reprojection_status': 'unsupported_dynamic_identity; frustum-only exploratory heatmap',
            'clock': clock, 'gt_accessed_for_input_construction': False,
            'candidate_projection_note': 'Existing evaluator/candidate UV uses square-FOV approximation; unchanged across conditions. Gaze uses explicitly aspect-aware vertical FOV.'}


def prepare_one(ctx, directory, config, backend, cpu_only=False):
    directory.mkdir(parents=True, exist_ok=True)
    records = gaze_records(ctx)
    probe = probe_video(ctx.video_path)
    duration = float(probe['format']['duration'])
    clock_ratio=float(config.get('clock_max_offset_ratio',0.30))
    clock = guard_clock(clock_alignment(ctx), duration, clock_ratio)
    if cpu_only:
        # CPU exports need no fabricated model output. Frame floor is the same
        # existing event policy; the main study later freezes real 8B mentions.
        parse = {'mentions': [], 'source': 'cpu_export_without_language_model'}
    else:
        parse = ReferentialAgent(backend).run(ctx)
    alignment = {'status': 'instruction_event_only', 'clock': clock,
                 'mentions': {}, 'instruction_coverage': 0.,
                 'policy': 'direct timing; no ASR inferred or run in this study'}
    plan = plan_frames(ctx, parse['mentions'], alignment, duration, timed=False,
                       max_clock_offset_ratio=clock_ratio)
    raw_offset=(clock.get('telemetry_to_video') or 0.)
    requested_window=[float(ctx.api_row['t_start'])+raw_offset,
                      float(ctx.api_row['t_end'])+raw_offset]
    clipped_window=[min(max(0.,duration-.1),max(0.,requested_window[0])),
                    min(max(0.,duration-.1),requested_window[1])]
    plan['alignment'].update(requested_event_video_range=requested_window,
                             clipped_event_video_range=clipped_window,
                             event_window_clipped=requested_window!=clipped_window)
    # Event-context anchors are selected without gaze, language clauses, ASR or GT.
    selected_records=fixed_video_anchor_frames(plan,config.get('reference_frame_count',3))
    input_plan={**plan,'records':selected_records}
    original_paths, original_frames = prepare_views(ctx, parse, input_plan, directory/'original_views',
                                                   use_gaze=False, segmenter=None)
    frames = copy.deepcopy(original_frames)
    gaze_clock_available=not bool(plan['alignment'].get('audio_timing_rejected'))
    gaze_offset=(plan['alignment'].get('telemetry_to_video') or 0.) if gaze_clock_available else None
    for frame in frames:
        frame['view_id'] = frame['frame_id']
        frame['source_frame_ids'] = [frame['frame_id']]
        frame['source_time_range'] = [frame['time'], frame['time']]
        frame['telemetry_time_for_gaze']=(frame['time']-gaze_offset) if gaze_clock_available else None
        frame['gaze_clock_status']='metadata_origin_estimate_not_pixel_calibrated' if gaze_clock_available else 'rejected_full_rgb_gaze_disabled'
    raw_paths = [frame['image_rgb'] for frame in frames]
    observations = {'cues': [], 'evidence': [], 'status': 'NOT_RUN_CPU_EXPORT'}
    if not cpu_only:
        observations = hand_cues(backend, ctx, parse, raw_paths, frames)
    audit = audit_data(ctx, records, plan['alignment'], probe)
    save(directory/'gaze_records.json', records)
    save(directory/'data_geometry_audit.json', audit)
    freeze = {'version': VERSION, 'offline': True, 'recording': ctx.video_path,
              'recording_sha256': file_sha(ctx.video_path), 'telemetry': ctx.json_path,
              'telemetry_sha256': file_sha(ctx.json_path), 'instruction': ctx.text,
              'referential': parse, 'plan': plan, 'hand_observations': observations,
              'frames': frames, 'raw_paths': raw_paths,
              'current_architecture_paths': [original_paths[original_frames.index(next(
                  f for f in original_frames if f['frame_id']==frame['frame_id']))] for frame in frames],
              'allowed_observation_range': [min(f['time'] for f in original_frames),
                                            max(f['time'] for f in original_frames)],
              'gaze_clock_status': plan['alignment'].get('calibration_status'),
              'gaze_clock_offset_seconds': gaze_offset,
              'protocol': 'offline_complete_interaction; reference-frame selection is fixed coverage of existing legal plan',
              'candidate_map': [{'mention_id':m['mention_id'], 'candidate_ids':m['candidate_ids']}
                                for m in parse['mentions']],
              'candidate_table': [c.world_row() for c in ctx.candidates],
              'gt_sha256_for_evaluation_only': file_sha(ctx.provenance['gt_file']),
              'upstream_model': backend.model_path if backend else None}
    freeze['frozen_upstream_sha256'] = _hash_json(freeze)
    save(directory/'frozen_upstream.json', freeze)
    variants = {}; cpu_panels = []
    cfg = {**DEFAULTS, **config.get('gaze', {})}
    for frame in frames:
        fid = frame['frame_id']
        rgb = Image.open(frame['image_rgb']).convert('RGB')
        frame_variants = {}
        frame_dir = directory/'inputs'/fid; frame_dir.mkdir(parents=True, exist_ok=True)
        rgb.save(frame_dir/'R.png')
        stats_by_strategy = {}
        for strategy, suffix in [('instant', 'inst'), ('local_history', 'local'), ('utterance_window', 'utterance')]:
            heat, stats, projected = build_heatmap(ctx, records,
                frame_time_video=frame['time'], frame_time_telemetry=frame['telemetry_time_for_gaze'],
                event_window=[float(ctx.api_row['t_start']), float(ctx.api_row['t_end'])],
                clock_available=gaze_clock_available,
                image_size=rgb.size, strategy=strategy, config=cfg)
            # Conservative dynamic safeguard: without motion correspondence,
            # reprojected older intersections are exploratory and explicitly unverified.
            np.savez_compressed(frame_dir/f'H_{suffix}.npz', H_unnormalized=heat)
            save(frame_dir/f'{suffix}_projection.json', {'stats': stats, 'projected_samples': projected})
            stats_by_strategy[strategy] = stats
            for prefix in ['H', 'F']:
                name = f'{prefix}_{suffix}'
                rendered, render_stats = render_presentation(rgb, heat, name, cfg)
                path = frame_dir/f'{name}.png'; rendered.save(path)
                frame_variants[name] = {'path': str(path), 'sha256': file_sha(path),
                                        'size': list(rendered.size), 'render': render_stats,
                                        'heatmap_source': str(frame_dir/f'H_{suffix}.npz')}
        frame_variants['R'] = {'path': str(frame_dir/'R.png'), 'sha256':file_sha(frame_dir/'R.png'),
                               'size':list(rgb.size), 'render': {'status':'raw_rgb_control'}}
        variants[fid] = {'frame':frame, 'variants':frame_variants, 'gaze_evidence':stats_by_strategy}
        panel = Image.new('RGB', (rgb.width*5, rgb.height+28), '#ffffff')
        draw = ImageDraw.Draw(panel)
        for i, group in enumerate(GROUPS):
            panel.paste(Image.open(frame_variants[group]['path']), (i*rgb.width,28))
            draw.text((i*rgb.width+4, 7), f'{fid} {frame["time"]:.3f}s {group}', fill='black')
        panel.thumbnail((1600, 500)); cpu_panels.append(panel)
    contact = Image.new('RGB', (max(p.width for p in cpu_panels), sum(p.height for p in cpu_panels)), '#ffffff')
    y = 0
    for panel in cpu_panels:
        contact.paste(panel, (0,y)); y += panel.height
    contact.save(directory/'cpu_comparison.jpg', quality=92)
    save(directory/'prepared_inputs.json', {'version':VERSION, 'variants':variants,
         'config':cfg, 'offline':True, 'model_input_has_debug_labels':False,
         'heatmap_generation_gt_invariant':'no GT passed to gaze numerical/rendering layer',
         'raw_frame_resolution':probe['streams'][0], 'resize_policy':'aspect-preserving thumbnail max672; no crop/letterbox/distortion correction',
         'server_resize':'remote processor may resize again; exact grid requires server/processor confirmation'})
    return freeze, variants, audit


def _failed_result(ctx, group, exc):
    return {'experiment':group, 'scene_key':ctx.scene_key, 'event_id':ctx.event_id,
            'sample_id':ctx.sample_id, 'row_index':ctx.row_index, 'scene_id':ctx.scene_id,
            'instruction':ctx.text, 'selected_object_names':[], 'selected_object_name':'',
            'response_status':'failed', 'failure':str(exc),
            'gt_referents_mapped':ctx.gt_referents_mapped,
            'adjusted_response':{'selected_object_names':[],'selected_object_name':''}}


def summarize_groups(results, out, frozen, calls):
    summary = {'version':VERSION, 'protocol':'offline paired pilot; no official development split',
               'scope':'entire anchor set only; no mention accuracy, temporal or point metrics',
               'groups':{}, 'frozen_upstream_samples':len(frozen),
               'failed_calls':sum(bool(c.get('error')) for c in calls),
               'real_http_calls':sum(not c.get('cache_hit') for c in calls),
               'cached_replays':sum(bool(c.get('cache_hit')) for c in calls),
               'not_a_gaze_ablation':'Coordinator still receives unchanged original numeric gaze cues; this measures Visual encoding only.'}
    for group, rows in results.items():
        path=out/'results'/group
        for row in rows:
            save(path/row['scene_key']/f'row_{row["row_index"]}.json',row)
        original = evaluate_original(rows,path)
        summary['groups'][group]={'metrics':evaluate_results(rows),'by_scene':evaluate_by_scene(rows),
            'failed_events':sum(r.get('response_status')!='ok' for r in rows),
            'original_evaluator':original,
            'pre_coordinator_fusion_metrics':evaluate_results([{**r,'selected_object_names':r.get('visual_selected_object_names',[]),
                                                   'selected_object_name':''} for r in rows])}
    save(out/'summary.json',summary)
    save(out/'call_audit.json',calls)
    return summary


def _is_correct(row):
    gt=set(row.get('gt_referents_mapped') or [])
    pred=set(row.get('selected_object_names') or [])
    return bool(gt & pred), gt == pred


def paired_comparison(results, out):
    by_group={g:{(r['scene_key'],r['event_id']):r for r in rows}
              for g,rows in results.items()}
    reference=by_group['R']
    comparisons={}
    sample_deltas=[]
    for group in results:
        if group=='R':
            continue
        rows=[]
        transition={'any_hit_improved':0,'any_hit_worsened':0,'any_hit_unchanged':0,
                    'exact_set_improved':0,'exact_set_worsened':0,'exact_set_unchanged':0,
                    'coordinator_changed_visual_set':0,'coordinator_corrected_visual':0,
                    'coordinator_hurt_visual':0}
        for key, row in by_group[group].items():
            base=reference.get(key, {})
            base_hit,base_exact=_is_correct(base)
            hit,exact=_is_correct(row)
            visual=set(row.get('visual_selected_object_names') or [])
            final=set(row.get('selected_object_names') or [])
            visual_hit=bool(set(row.get('gt_referents_mapped') or []) & visual)
            final_hit,_=_is_correct(row)
            transition['any_hit_'+('improved' if hit and not base_hit else 'worsened' if base_hit and not hit else 'unchanged')]+=1
            transition['exact_set_'+('improved' if exact and not base_exact else 'worsened' if base_exact and not exact else 'unchanged')]+=1
            transition['coordinator_changed_visual_set'] += visual != final
            transition['coordinator_corrected_visual'] += final_hit and not visual_hit
            transition['coordinator_hurt_visual'] += visual_hit and not final_hit
            row_delta={'group':group,'scene_key':key[0],'event_id':key[1],
                'R_prediction':sorted(set(base.get('selected_object_names') or [])),
                'group_prediction':sorted(final),'visual_pre_coordinator':sorted(visual),
                'ground_truth_evaluator_only':sorted(set(row.get('gt_referents_mapped') or [])),
                'R_any_hit':base_hit,'group_any_hit':hit,'R_exact_set':base_exact,
                'group_exact_set':exact,'visual_any_hit':visual_hit,'final_any_hit':final_hit,
                'source_frame_ids':row.get('source_frame_ids',[]),
                'source_rgb_frame_sha256':row.get('source_rgb_frame_sha256',[])}
            rows.append(row_delta);sample_deltas.append(row_delta)
        comparisons[group]=transition
    source_checks=[]
    changed_visual_inputs={g:0 for g in results if g!='R'}
    for key in reference:
        event_rows=[by_group[g].get(key,{}) for g in results]
        checks=[tuple(r.get('source_rgb_frame_sha256') or []) for r in event_rows]
        frame_ids=[tuple(r.get('source_frame_ids') or []) for r in event_rows]
        dims=[tuple(tuple(size) for size in r.get('image_sizes') or []) for r in event_rows]
        coordinator_images=[tuple(r.get('coordinator_image_sha256') or []) for r in event_rows]
        baseline_visual=tuple(by_group['R'].get(key,{}).get('input_image_sha256') or [])
        visual_hashes={g:tuple(by_group[g].get(key,{}).get('input_image_sha256') or []) for g in results}
        differs={g:(visual_hashes[g] != baseline_visual) for g in results if g!='R'}
        for group,different in differs.items():
            changed_visual_inputs[group]+=int(different)
        has_source=bool(checks and checks[0] and all(checks))
        source_checks.append({'scene_key':key[0],'event_id':key[1],
            'has_source_frame_audit':has_source,
            'same_source_rgb_pixels_across_groups':has_source and len(set(checks))==1,
            'same_frame_ids_across_groups':has_source and len(set(frame_ids))==1,
            'same_visual_dimensions_across_groups':has_source and len(set(dims))==1,
            'same_coordinator_rgb_across_groups':has_source and len(set(coordinator_images))==1,
            'visual_variant_differs_from_R':differs,
            'source_frame_count':len(checks[0]) if checks else 0})
    verified_source_count=sum(r['has_source_frame_audit'] for r in source_checks)
    payload={'reference_group':'R','paired_sample_count':len(reference),
        'comparison_scope':'same 3 source RGB frames, same dimensions, Visual-only encoding changes',
        'fully_audited_source_sample_count':verified_source_count,
        'incomplete_source_audit_events':[{'scene_key':r['scene_key'],'event_id':r['event_id']}
                                           for r in source_checks if not r['has_source_frame_audit']],
        'same_source_frames_for_all_events_and_groups':verified_source_count==len(source_checks) and all(
            r['same_source_rgb_pixels_across_groups'] and r['same_frame_ids_across_groups']
            and r['same_visual_dimensions_across_groups'] and r['same_coordinator_rgb_across_groups']
            for r in source_checks),
        'events_with_changed_visual_pixels_vs_R':changed_visual_inputs,
        'per_group_transitions_vs_R':comparisons,
        'per_event_deltas':sample_deltas,'source_audit':source_checks}
    save(out/'paired_comparison.json',payload)
    return payload


def _stage_budget(calls):
    stages={}
    for stage in sorted({c.get('stage','unknown') for c in calls}):
        entries=[c for c in calls if c.get('stage','unknown')==stage]
        real=[c for c in entries if not c.get('cache_hit')]
        images=[image for call in entries for image in call.get('images',[])]
        real_images=[image for call in real for image in call.get('images',[])]
        stages[stage]={'logical_requests':len(entries),'real_http_calls':len(real),
            'cache_hits':len(entries)-len(real),'image_count_logical':len(images),
            'image_count_real_http':len(real_images),
            'dimensions':sorted({f"{i.get('width')}x{i.get('height')}" for i in images}),
            'estimated_visual_tokens_logical':sum(i.get('visual_tokens_estimate') or 0 for i in images),
            'estimated_visual_tokens_real_http':sum(i.get('visual_tokens_estimate') or 0 for i in real_images),
            'model_wall_seconds_real_http':sum(c.get('elapsed_seconds',0) for c in real),
            'prompt_tokens_from_server':sum((c.get('usage') or {}).get('prompt_tokens',0) for c in real),
            'completion_tokens_from_server':sum((c.get('usage') or {}).get('completion_tokens',0) for c in real)}
    return stages


def _merge_call_history(current_calls, previous_calls):
    """Keep one real request per active cache key across an incremental rerun."""
    prior_real={}
    for record in previous_calls:
        if not record.get('cache_hit'):
            prior_real.setdefault(record.get('request_path'),record)
    reused=set()
    merged=[]
    for record in current_calls:
        path=record.get('request_path')
        if record.get('cache_hit') and path in prior_real and path not in reused:
            merged.append(prior_real[path])
            reused.add(path)
        else:
            merged.append(record)
    return merged


def write_audit_report(out, results, geometry, calls, paired):
    valid_rates=[r.get('gaze_valid_fraction',0) for r in geometry]
    hz=[r.get('telemetry_hz_measured') for r in geometry if r.get('telemetry_hz_measured')]
    pose_errors=[r.get('gaze_world_to_camera_direction_error_degrees_p95')
                 for r in geometry if r.get('gaze_world_to_camera_direction_error_degrees_p95') is not None]
    movements=[r.get('camera_max_displacement_world_units') for r in geometry
               if r.get('camera_max_displacement_world_units') is not None]
    all_sample_geometry=[]
    for path in (out/'samples').rglob('data_geometry_audit.json'):
        all_sample_geometry.append(json.loads(path.read_text()))
    if all_sample_geometry:
        geometry=all_sample_geometry
    rejected=sum(bool((r.get('clock') or {}).get('audio_timing_rejected')) for r in geometry)
    clipped=sum(bool((r.get('clock') or {}).get('event_window_clipped')) for r in geometry)
    rendered_audits=[]
    for path in (out/'samples').rglob('prepared_inputs.json'):
        payload=json.loads(path.read_text())
        frame_rows=list(payload.get('variants',{}).values())
        rendered_audits.append({'instant':any(
            row.get('gaze_evidence',{}).get('instant',{}).get('valid_projected_count',0)>0
            for row in frame_rows),'local_history':any(
            row.get('gaze_evidence',{}).get('local_history',{}).get('valid_projected_count',0)>0
            for row in frame_rows)})
    instant_events=sum(r['instant'] for r in rendered_audits)
    history_events=sum(r['local_history'] for r in rendered_audits)
    stage_budget=_stage_budget(calls)
    summary=json.loads((out/'summary.json').read_text())
    budget=json.loads((out/'budget.json').read_text())
    manifest=json.loads((out/'manifest.json').read_text())
    run_config=json.loads((out/'config.json').read_text())
    n=len(results['R'])
    successful_samples=sum(row.get('response_status')!='failed' for row in results['R'])
    visual_stage=stage_budget.get('v3_visual',{})
    visual_completed=visual_stage.get('logical_requests',0)//max(1,len(results))
    config_path=run_config.get('study_config_path','configs/temporal_gaze_8b_20261021.json')
    configured_ratio=run_config.get('clock_max_offset_ratio',.30)
    reference_f1=summary['groups']['R']['metrics']['micro_f1']
    local_f1=summary['groups'].get('H_local',{}).get('metrics',{}).get('micro_f1',0.)
    local_hit=summary['groups'].get('H_local',{}).get('metrics',{}).get('any_hit_accuracy',0.)
    raw_hit=summary['groups']['R']['metrics']['any_hit_accuracy']
    fov_f1=summary['groups'].get('F_local',{}).get('metrics',{}).get('micro_f1',0.)
    with (out/'budget.json').open('w',encoding='utf-8') as handle:
        json.dump({**budget,'stage_budget':stage_budget,
            'logical_requests':len(calls),'planned_samples':n,
            'successful_model_samples':successful_samples,
            'visual_calls_per_group':visual_completed,
            'estimated_visual_tokens_per_group':visual_completed*3*441,
            'estimated_visual_tokens_per_group_method':f'{visual_completed} completed Visual calls x 3 images x 21x21 32px merged-grid estimate; remote resize may differ'},
            handle,ensure_ascii=False,indent=2)
        handle.write('\n')
    lines=['# 数据与方法核查','',
        '## 范围与复现','',
        f"本结果按固定实施说明执行五组同帧呈现对照 R/H_inst/H_local/F_inst/F_local。样本 manifest seed={manifest.get('seed')}；选择依据：{manifest.get('selection')}。该 pilot 位于现有 evaluation pool，没有官方开发划分。全部为离线完整交互诊断；每组每样本输入相同的 3 个事件覆盖帧。R 是 8B 同流程下的 raw-RGB 参照条件，不是历史 30B 基线复现。未实现/未声称全景、空间图集或 B2 深度重投影。", '',
        '入口仍为 `python -m MultiAgent.run_v3`，示例：','',
        '```bash',
        'cd /media/kunyu/16TB/spj/Multiagent',
        'source /home/kunyu/miniconda3/etc/profile.d/conda.sh',
        'conda activate trimodal',
        'source /media/kunyu/16TB/spj/setup/use_prepared_inputs.sh',
        f'python -m MultiAgent.run_v3 --manifest {out}/manifest.json --output-dir {out}_rerun --cache-dir runs/visual_evidence_8b_20261021/cache --experiment V3 --visual-study temporal --study-config {config_path} --qwen-model Qwen3-VL-8B-Instruct --base-url http://100.124.154.87:8000/v1 --api-key-env VRTRIREF_API_KEY',
        '```','',
        'CPU 准备命令：','',
        '```bash',
        f'python -m MultiAgent.run_v3 --manifest {out}/manifest.json --output-dir {out}_cpu --cache-dir runs/visual_evidence_8b_20261021/cache --experiment V3 --visual-study temporal --cpu-only --study-config {config_path}',
        '```','',
        '在线调用需设置 `VRTRIREF_API_KEY=EMPTY`；日志不包含密钥。此目录保留固定样本 `manifest.json`、配置、逐样本预测、输入图、原始请求/响应缓存和本报告。', '',
        '仅从已经保存的预测重建配对与指标报告（不发模型请求）：', '',
        '```bash',
        f'python -m MultiAgent.run_v3 --manifest {out}/manifest.json --output-dir {out} --cache-dir runs/visual_evidence_8b_20261021/cache --finalize-study',
        '```','',
        '## 数据、时钟与几何','',
        f'- 本轮时钟比例阈值 {configured_ratio:.2f}（项目默认仍为 0.30）。{n} 条计划样本；时钟/退化窗口保护拒绝 gaze/video 对齐 {rejected} 例，event window 经视频范围裁剪 {clipped} 例。未标定起点误差不会因放宽阈值而消失。拒绝样本继续使用原始 RGB 并进入评价分母。',
        f'- 至少一帧存在可投影热图证据的样本：instant {instant_events}/{len(rendered_audits)}，local_history {history_events}/{len(rendered_audits)}；对齐有效但没有投影证据的帧同样回退原图。',
        f'- gaze telemetry 中有效字段比例均值：{(sum(r.get("gaze_valid_fraction",0) for r in geometry)/len(geometry) if geometry else 0):.3f}。按真实 ISO 时间戳计算的采样率中位数：{(sorted([r["telemetry_hz_measured"] for r in geometry if r.get("telemetry_hz_measured")])[len([r for r in geometry if r.get("telemetry_hz_measured")])//2] if any(r.get("telemetry_hz_measured") for r in geometry) else 0):.2f} Hz。',
        f'- 世界 gazePoint 是有限世界坐标求交点；gazeVector 是世界方向向量，作为无限远方向投影。所有样本均未记录碰撞 anchor ID。',
        f'- 旋转一致性检查：world gazeVector 经过记录 xyzw 四元数逆旋转，与 cameraGazeDirection 比较，样本 p95 的最大值 {max(pose_errors,default=0):.6g}°。这验证内部旋转约定一致，不是相机外参标定。',
        f'- 记录相机位置在每个样本内的最大位移范围 {min(movements,default=0):.3f}–{max(movements,default=0):.3f} Unity world units；物理单位、录制相机与中心眼偏移未知。无同步深度、可重放 mesh 或动态物体变换。遮挡保持 `visibility_unknown`；历史点不跟踪移动对象。',
        '- 可核实代码约定：四元数 (x,y,z,w)，逆旋转 world→camera，相机 +z 前向、+y 上方、左上像素原点；标量 `cameraFOV` 按垂直 FOV、方形像素投影。Unity 左/右手性、逐帧内参和畸变未独立标定。',
        '- 视频实际由现有 ffmpeg 抽帧器按秒 seek，保留抽取时间；短边约束为最大 672 像素，不裁切/letterbox。视觉服务 processor 会否再次缩放以及最终视觉网格无法从兼容 API 验证。',
        '- 热图核以角宽度配置后换算输入分辨率像素；无有效 gaze 时 heatmap 为零、H/F 保留 RGB。无遮挡判断，不归一化弱证据；foveation 使用同一 heatmap 驱动的 2x/4x/8x 降采样/上采样金字塔并保留清晰 RGB 下限，不是 Scene-aware Foveated Rendering 论文复现，也未声称降低视觉 token 或推理耗时。', '',
        '## 冻结路径与推理','',
        '- 同一 8B 服务先完成 Referential、候选映射、直接事件时间计划及原 RGB Hand 观察；五组共用冻结哈希、候选映射、Hand 图像/观察和每 mention 的 Visual 调用粒度。Coordinator 使用原 prompt、规则与原始 RGB 图像，仅 Visual evidence 变化。GT 只用于独立 evaluator；输入构造仅写入 GT 文件哈希，不把标签值送入模型。',
        f'- 每组计划 {n} 条，成功完成上游与模型链路 {successful_samples} 条；另 {n-successful_samples} 条在 Visual/Coordinator 前失败，仍计入原评估分母。每组完成 {visual_completed} 次 Visual 和 Coordinator 逻辑调用；相同 Coordinator 输入由确定性缓存复用。调用审计记录实际 HTTP、缓存命中、stage、耗时、图片哈希/尺寸、原始响应和 usage。远端 processor resize 未知，因此视觉 token 是 32px merged-grid 估计，服务端 prompt_tokens 同时包含文本/图像，不当作视觉 token。',
        f'- 服务端模型 ID：`Qwen3-VL-8B-Instruct`；vLLM OpenAI-compatible `/v1` 服务。实际模型 HTTP {budget.get("real_calls",0)} 次，缓存复用 {summary.get("cached_replays",0)} 次，模型累计 HTTP 等待 {budget.get("model_wall_seconds_sum",0):.1f}s；累计服务端 prompt_tokens {budget.get("usage",{}).get("prompt_tokens",0)}（非纯视觉）。',
        '- 全部阶段复用已部署 Qwen3-VL-8B-Instruct API；本轮未启动或停止推理服务。', '',
        '## 结果读取与解释','',
        f'- 本轮 N={n}：H_local Any-hit {local_hit:.3f} vs R {raw_hit:.3f}，Micro-F1 {local_f1:.3f} vs {reference_f1:.3f}；F_local Micro-F1 {fov_f1:.3f}。各组 Exact-set 以 RESULTS.md 表格为准；本轮探索观察不视作显著性结论。',
        '- 各 gaze 组相对 R 实际改变 Visual 图像像素的样本数：'+', '.join(f'{g}={count}/{n}' for g,count in paired['events_with_changed_visual_pixels_vs_R'].items())+'；其余输入为全 RGB 回退或上游失败。',
        f'- `paired_comparison.json` 对 {paired["fully_audited_source_sample_count"]}/{n} 条完成样本核对同源帧哈希；剩余 {n-paired["fully_audited_source_sample_count"]} 条为上游失败，没有模型输入图。',
        '- `RESULTS.md` 给出原 evaluator 汇总；`paired_comparison.json` 给出相对 R 的逐样本变化；`failure_analysis.json` 对照 Visual 融合前与 Coordinator 后集合，包含漏选、额外对象、同类实例候选混淆。无 mention 级 GT，不报告 mention accuracy。',
        f'- Referential 校验失败 {n-successful_samples} 条；API 失败调用 {summary.get("failed_calls",0)}，已返回但语义抽取为空的响应记录在 failures.json。时钟拒绝样本属于明确 RGB 回退，不能解释成热图成功。',
        f'- 已验证：API `/models` 返回配置模型；关键几何合成测试通过；已完成 {successful_samples}/{n} 条真实模型链路，逐组原 evaluator 分母均保留 {n} 条。',
        '- 仅实现/待验证：gaze 热图/历史窗口和 blur blend 是否改善对象选择；基于文件时间的 gaze/video 起点估计；动态对象历史 gaze 的身份对应。当前缺少深度/mesh，故未做 B2。']
    (out/'AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def finalize_saved_study(output_dir):
    """Rebuild result/audit artifacts from saved predictions without model calls."""
    out=Path(output_dir)
    manifest=json.loads((out/'manifest.json').read_text())
    contexts=load_contexts(manifest)
    config=json.loads((out/'config.json').read_text())
    groups=config.get('groups',GROUPS)
    results={g:[] for g in groups};frozen={}
    for ctx in contexts:
        key=sample_key(ctx)
        for group in groups:
            path=out/'samples'/ctx.scene_key/ctx.event_id/'predictions'/f'{group}.json'
            if not path.exists():
                results[group].append(_failed_result(ctx,group,RuntimeError('saved prediction missing')))
                continue
            row=json.loads(path.read_text())
            freeze_path=out/'samples'/ctx.scene_key/ctx.event_id/'frozen_upstream.json'
            freeze=json.loads(freeze_path.read_text()) if freeze_path.exists() else {}
            source_by_frame={f['frame_id']:{'view_id':f.get('view_id',f['frame_id']),
                'source_frame_ids':f.get('source_frame_ids',[f['frame_id']]),
                'source_time_range':f.get('source_time_range',[f.get('time'),f.get('time')]),
                'source_time_seconds':f.get('time')} for f in freeze.get('frames',[])}
            visual=row.get('visual_evidence') or {}
            for evidence in visual.get('evidence',[]):
                source=source_by_frame.get(evidence.get('frame_id'))
                if source:
                    evidence.update(source)
            row['visual_evidence']=visual
            coordinator=row.get('coordinator') or {}
            visual_names=row.get('visual_selected_object_names') or []
            coord_names=coordinator.get('selected_object_names') or []
            raw_names=(coordinator.get('raw') or {}).get('selected_object_names') or []
            row['pre_coordinator_selected_object_names']=visual_names
            row['coordinator_raw_selected_object_names']=raw_names
            row['coordinator_selected_object_names']=coord_names
            row['coordinator_abstained_and_pipeline_kept_pre']=not bool(
                coordinator.get('selected_object_names') or coordinator.get('temporal_focus'))
            row['source_rgb_frame_sha256']=row.get('coordinator_image_sha256',[])
            save(path,row)
            results[group].append(row)
            frozen[key]=True
    geometry=json.loads((out/'geometry_audit.json').read_text())
    all_geometry=[]
    for ctx in contexts:
        path=out/'samples'/ctx.scene_key/ctx.event_id/'data_geometry_audit.json'
        if path.exists():
            all_geometry.append(json.loads(path.read_text()))
    if all_geometry:
        geometry=all_geometry
        save(out/'geometry_audit.json',geometry)
    calls=json.loads((out/'call_audit.json').read_text())
    summary=summarize_groups(results,out,frozen,calls)
    analyze_results(results,geometry,out,calls)
    paired=paired_comparison(results,out)
    write_audit_report(out,results,geometry,calls,paired)
    return summary


def run_study(args):
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    previous_calls=(json.loads((out/'call_audit.json').read_text())
                    if (out/'call_audit.json').exists() else [])
    config=json.loads(Path(args.study_config).read_text()) if args.study_config else {}
    if not previous_calls and not args.cpu_only:
        namespace=config.get('namespace','temporal_gaze_remote8b_20261021')
        call_log=Path(args.cache_dir)/namespace/'call_audit.jsonl'
        if call_log.exists():
            previous_calls=[json.loads(line) for line in call_log.read_text().splitlines() if line.strip()]
    if args.visual_study=='organization':
        from .spatial_evidence import run_organization_study
        return run_organization_study(args,config)
    if args.mention_focus:
        raise ValueError('mention-focus changes time policy; disable it for the fixed-frame presentation study')
    if args.regions:
        raise ValueError('region generation is frozen off in this controlled pilot')
    manifest=json.loads(Path(args.manifest).read_text())
    if args.limit:manifest['samples']=manifest['samples'][:args.limit]
    save(out/'manifest.json',manifest)
    save(out/'config.json',{'version':VERSION,'config':config,'model':args.qwen_model,
        'base_url':args.base_url,'data_root':str(DATA_ROOT),'cpu_only':args.cpu_only,
        'reference_frames':3,'groups':config.get('groups',GROUPS),'timing_mode':'direct',
        'mention_isolation':args.mention_isolation,'mention_focus':False,'regions':False,
        'selection_precommitted_before_new_model_predictions':True,
        'clock_max_offset_ratio':float(config.get('clock_max_offset_ratio',0.30)),
        'study_config_path':str(Path(args.study_config).resolve()) if args.study_config else None,
        'clock_guard_default_unchanged':0.30})
    backend=QwenBackend(args.cache_dir,args.qwen_model,base_url=args.base_url,
                        api_key=os.environ.get(args.api_key_env, 'EMPTY'),
                        namespace=config.get('namespace','temporal_gaze_remote8b_20261021'))
    if not args.cpu_only:
        if not args.base_url:raise ValueError('This study requires the fixed remote model service')
        save(out/'remote_models.json',backend.check_models())
    contexts=load_contexts(manifest);groups=config.get('groups',GROUPS)
    results={g:[] for g in groups};frozen={};geometry=[];errors=[]
    for index,ctx in enumerate(contexts,1):
        key=sample_key(ctx);directory=out/'samples'/ctx.scene_key/ctx.event_id
        print(f'[{index}/{len(contexts)}] {key} freeze + CPU inputs',flush=True)
        try:
            freeze,variants,audit=prepare_one(ctx,directory,config,backend,args.cpu_only)
        except Exception as exc:
            errors.append({'sample':key,'stage':'frozen_upstream_or_cpu','error':f'{type(exc).__name__}: {exc}'})
            save(directory/'failure.json',errors[-1])
            for group in groups:results[group].append(_failed_result(ctx,group,exc))
            continue
        frozen[key]=freeze;geometry.append(audit)
        if args.cpu_only:continue
        parse,frames,plan=freeze['referential'],freeze['frames'],freeze['plan']
        for group in groups:
            started=time.monotonic();print(f'[{index}/{len(contexts)}] {key} {group}',flush=True)
            paths=[variants[f['frame_id']]['variants'][group]['path'] for f in frames]
            mention_paths={f['frame_id']:p for f,p in zip(frames,paths)}
            try:
                visual=visual_evidence(backend,ctx,parse,paths,frames,plan,
                    cues=freeze['hand_observations'],use_gaze=True,use_regions=False,
                    isolate_mentions=args.mention_isolation,mention_paths=mention_paths,
                    visual_protocol='frozen_rgb_gaze_variants')
                frame_sources={f['frame_id']:{'view_id':f['view_id'],
                    'source_frame_ids':f['source_frame_ids'],
                    'source_time_range':f['source_time_range'],
                    'source_time_seconds':f['time']} for f in frames}
                for evidence in visual.get('evidence',[]):
                    source=frame_sources.get(evidence.get('frame_id'))
                    if source:
                        evidence.update(source)
                # Existing coordinator prompt, rules, metadata and raw RGB stay identical.
                coordinator=monolithic_coordinator(backend,ctx,parse,freeze['raw_paths'],frames,visual,
                    use_spatial_cues=not plan['alignment'].get('audio_timing_rejected',False))
                interim=run_experiment([ctx],EXPERIMENTS['V3'],visual_predictions={key:visual},
                    qwen_mentions={key:parse})[0]
                row=run_experiment([ctx],EXPERIMENTS['V3'],visual_predictions={key:visual},
                    qwen_mentions={key:parse},monolithic_predictions={key:coordinator})[0]
                row.update(visual_selected_object_names=interim['selected_object_names'],
                    pre_coordinator_selected_object_names=interim['selected_object_names'],
                    coordinator_raw_selected_object_names=(coordinator.get('raw') or {}).get('selected_object_names',[]),
                    coordinator_selected_object_names=coordinator['selected_object_names'],
                    coordinator_abstained_and_pipeline_kept_pre=not bool(
                        coordinator.get('selected_object_names') or coordinator.get('temporal_focus')),
                    group=group,offline=True,input_image_paths=paths,
                    input_image_sha256=[file_sha(p) for p in paths],
                    source_frame_ids=[f['frame_id'] for f in frames],
                    source_rgb_frame_sha256=[file_sha(p) for p in freeze['raw_paths']],
                    source_frame_count=len(frames),image_count=len(paths),
                    image_sizes=[variants[f['frame_id']]['variants'][group]['size'] for f in frames],
                    frozen_upstream_sha256=freeze['frozen_upstream_sha256'],
                    coordinator_image_sha256=[file_sha(p) for p in freeze['raw_paths']],
                    hand_image_sha256=[file_sha(p) for p in freeze['raw_paths']],
                    wall_seconds=time.monotonic()-started)
                row['visual_evidence']=visual
                row['coordinator']=coordinator
                save(directory/'predictions'/f'{group}.json',row)
            except Exception as exc:
                row=_failed_result(ctx,group,exc)
                errors.append({'sample':key,'group':group,'stage':'visual_or_coordinator','error':str(exc)})
            results[group].append(row)
            save(out/'progress.json',{'sample':index,'total':len(contexts),'last_sample':key,
                                     'last_group':group,'errors':errors})
    save(out/'geometry_audit.json',geometry);save(out/'failures.json',errors)
    if args.cpu_only:
        summary={'status':'CPU_COMPLETE_MODEL_NOT_RUN','samples':len(frozen),
                 'failed_samples':len(errors),'groups':groups,'geometry':geometry}
        save(out/'summary.json',summary);return summary
    calls=_merge_call_history(backend.call_records,previous_calls)
    summary=summarize_groups(results,out,frozen,calls)
    analyze_results(results,geometry,out,calls)
    paired=paired_comparison(results,out)
    write_audit_report(out,results,geometry,calls,paired)
    return summary


def analyze_results(results,geometry,out,calls):
    rows=[]
    audits={(a['scene_key'],a['event_id']):a for a in geometry}
    for group,events in results.items():
        for event in events:
            gt=set(event.get('gt_referents_mapped',[]));pred=set(event.get('selected_object_names',[]))
            extra=sorted(pred-gt);missing=sorted(gt-pred)
            from .mention_mapping import category
            identity=[{'extra':e,'missed':m} for e in extra for m in missing if category(e)==category(m)]
            a=audits.get((event['scene_key'],event['event_id']),{})
            rows.append({'group':group,'scene_key':event['scene_key'],'event_id':event['event_id'],
                'predicted':sorted(pred),'gt_evaluator_only':sorted(gt),
                'pre_coordinator_fusion':event.get('pre_coordinator_selected_object_names',
                                                   event.get('visual_selected_object_names',[])),
                'coordinator_raw_selection':event.get('coordinator_raw_selected_object_names',
                    (event.get('coordinator') or {}).get('raw',{}).get('selected_object_names',[])),
                'coordinator_selected_after_policy':event.get('coordinator_selected_object_names',
                    (event.get('coordinator') or {}).get('selected_object_names',[])),
                'coordinator_abstained_pipeline_fallback':event.get('coordinator_abstained_and_pipeline_kept_pre',False),
                'missing':missing,'extra':extra,'same_class_instance_confusions':identity,
                'coordinator_added':sorted(pred-set(event.get('pre_coordinator_selected_object_names',
                                                              event.get('visual_selected_object_names',[])))),
                'coordinator_removed':sorted(set(event.get('pre_coordinator_selected_object_names',
                    event.get('visual_selected_object_names',[])))-pred),
                'response_status':event.get('response_status'),
                'clock_rejected':a.get('clock',{}).get('audio_timing_rejected'),
                'head_translation_world_units':a.get('camera_max_displacement_world_units'),
                'gaze_valid_fraction':a.get('gaze_valid_fraction')})
    save(out/'failure_analysis.json',rows)
    remote_calls=[c for c in calls if not c.get('cache_hit')]
    elapsed=sum(c.get('elapsed_seconds',0) for c in remote_calls)
    budget={'real_calls':len(remote_calls),'model_wall_seconds_sum':elapsed,
            'visual_token_estimate_method':'merged 32px image grids, actual remote resize not observable; not total prompt tokens',
            'visual_token_estimate':sum(sum(i.get('visual_tokens_estimate') or 0 for i in c.get('images',[]))
                                        for c in remote_calls),
            'stage_counts':dict(Counter(c['stage'] for c in remote_calls)),
            'usage':{'prompt_tokens':sum((c.get('usage') or {}).get('prompt_tokens',0) for c in remote_calls),
                     'completion_tokens':sum((c.get('usage') or {}).get('completion_tokens',0) for c in remote_calls)},
            'same_visual_source_frames_across_main_groups':True,
            'same_visual_image_dimensions_across_main_groups':True,
            'blur_claims_speedup':False}
    save(out/'budget.json',budget)
    summary=json.loads((out/'summary.json').read_text())
    lines=['# VR-TriRef 8B Visual-only paired pilot','',
           'Offline diagnostic pilot, not a complete-dataset result. No training or official development split.',
           '', '|Group|N|Any-hit|Exact-set|Micro-P|Micro-R|Micro-F1|Failed|',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for g,v in summary['groups'].items():
        m=v['metrics'];lines.append(f"|{g}|{m['n']}|{m['any_hit_accuracy']:.3f}|{m['exact_set_match']:.3f}|{m['micro_precision']:.3f}|{m['micro_recall']:.3f}|{m['micro_f1']:.3f}|{v['failed_events']}|")
    lines += ['', 'All main groups replay the same real 8B Referential, scene candidate mapping, direct event plan and raw-RGB Hand observations. Visual calls use the same granularity. Coordinator prompt/rules/images remain fixed; it recomputes against each group\'s new evidence.',
        '', 'Gaze points are measured world intersections, not target IDs. Historical reprojection is frustum-only and has no occlusion/dynamic-object correspondence guarantee. Camera or clock failures retain RGB and stay in the denominator.',
        '', 'The existing coordinator still receives original gaze numeric/text cues. These comparisons measure additional Visual encoding, not total gaze contribution. Instant and local share their respective numerical H across heatmap/foveation; all images keep geometry and resolution.',
        '', 'Metrics above are generated from this run only. Clock coverage, explicit RGB fallbacks and per-event changes are reported in AUDIT.md and paired_comparison.json; pilot outcomes are not complete-dataset conclusions.',
        '', 'See geometry_audit.json, failures.json, failure_analysis.json, budget.json, call_audit.json and per-sample source/heatmap/projection records.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
