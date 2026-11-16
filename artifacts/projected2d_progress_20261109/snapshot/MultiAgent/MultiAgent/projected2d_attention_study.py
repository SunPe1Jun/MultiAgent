"""Paired Projected-2D visual-input study using the original TriRef evaluator."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

from .agents import ReferentialAgent
from .baseline import module as baseline_module
from .camera_geometry import ATTENTION_CLOCK_POLICY, VERSION as GEOMETRY_VERSION, aligned_frame_geometry
from .context import load_contexts
from .gaze_evidence import DEFAULTS, build_heatmap, gaze_records, render_presentation
from .qwen_backend import QwenBackend


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
TRIREF = WORKSPACE / "Trimodal-Referring"
V3D = WORKSPACE / "V3dMD"
QWEN8 = WORKSPACE / "Qwen3-VL-8B-Instruct"
QWEN30 = WORKSPACE / "Qwen3-VL-30B-A3B-Instruct"
PAPER_GT = TRIREF / "paper_experiment_evidence/manifests/experiment2_projected_gt_manifest.csv"
GROUPS = ("RGB", "CURRENT_HEATMAP", "HISTORY_HEATMAP", "HISTORY_FOVEATION")
CSV_COLUMNS = ("scene", "row_index", "event_id", "instruction", "model_input_image",
               "model_raw_output", "parsed_json", "parse_ok", "prediction_count", "error_message")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list))
                                   else value) for key, value in row.items()})


def audit_annotation_parity(original_rows, rebuilt_rows):
    key = lambda row: tuple(str(row[column]) for column in
                            ("scene", "row_index", "referent_name", "panel_id"))
    original = {key(row): row for row in original_rows}
    columns = ("gt_x", "gt_y", "projection_valid", "status")
    mismatches = []
    for row in rebuilt_rows:
        prior = original.get(key(row))
        if prior is None or any(str(prior.get(c, "")) != str(row.get(c, "")) for c in columns):
            mismatches.append({"key": key(row), "old": prior,
                               "new": {c: row.get(c) for c in columns}})
    return {"compared_rows": len(rebuilt_rows),
            "all_original_projected_gt_fields_identical": not mismatches,
            "mismatches": mismatches, "compared_columns": columns,
            "original_source": str(PAPER_GT)}


def select_fixed_samples(gt_rows, seed=20261022, count=20):
    events = {}
    for row in gt_rows:
        scene, row_index = row["scene"], int(row["row_index"])
        entry = events.setdefault((scene, row_index), {"event_id": row["event_id"], "referents": set(),
                                                        "valid_projection_count": 0})
        if row.get("referent_name"):
            entry["referents"].add(row["referent_name"])
        entry["valid_projection_count"] += row.get("projection_valid") == "True"
    scene_events = {}
    api_by_scene = {}
    for key, row in events.items():
        scene, index = key
        if not row["referents"] or not row["valid_projection_count"]:
            continue
        if scene not in api_by_scene:
            api_by_scene[scene] = read_csv(V3D / "prepared/data" / f"{scene}_api_input.csv")
        api_rows = api_by_scene[scene]
        if index >= len(api_rows):
            continue
        if not Path(api_rows[index]["video_path"]).is_file() or not Path(api_rows[index]["json_path"]).is_file():
            continue
        scene_events.setdefault(scene, []).append({"scene_key": scene, "row_index": index,
            "event_id": row["event_id"], "referent_count": len(row["referents"]),
            "valid_projection_count": row["valid_projection_count"]})
    rng = random.Random(seed)
    for rows in scene_events.values():
        rows.sort(key=lambda row: row["row_index"])
        rng.shuffle(rows)
    selected = []
    for scene in sorted(scene_events):
        if scene_events[scene] and len(selected) < count:
            selected.append(scene_events[scene].pop())
    while len(selected) < count:
        progressed = False
        for scene in sorted(scene_events):
            if scene_events[scene] and len(selected) < count:
                selected.append(scene_events[scene].pop())
                progressed = True
        if not progressed:
            raise RuntimeError(f"Only selected {len(selected)} of {count} eligible source events")
    return selected


def setup_original_manifest(selected, gt_rows, out: Path, *, overwrite=False):
    """Use the original Projected-2D panel selector/projector with saved GT labels."""
    builder = load_module("triref_exam2_manifest", TRIREF / "exam2/build_2d_eval_manifest.py")
    gt_by_key = {}
    for row in gt_rows:
        key = (row["scene"], int(row["row_index"]))
        if row.get("referent_name"):
            gt_by_key.setdefault(key, set()).add(row["referent_name"])
    eval_dir = out / "source_annotation_adapter"
    eval_dir.mkdir(parents=True, exist_ok=True)
    for scene in sorted({row["scene_key"] for row in selected}):
        rows = [{"row_index": item["row_index"],
                 "gt_referents_mapped": ", ".join(sorted(gt_by_key[(scene, item["row_index"])]))}
                for item in selected if item["scene_key"] == scene]
        write_csv(eval_dir / f"{scene}_match_eval.csv", rows, ["row_index", "gt_referents_mapped"])

    args = Namespace(
        ffmpeg_path="ffmpeg", ffprobe_path="ffprobe", panels=3, video_time_offset_seconds=0.0,
        auto_video_time_offset=True, auto_video_time_offset_source="hybrid",
        hybrid_offset_threshold_seconds=1.0, hybrid_video_time_bias_seconds=0.5,
        no_extract_frames=False, overwrite_frames=overwrite, min_panel_time_gap_seconds=0.25,
        keep_duplicate_frames=False, acceptable_top_k=2, sample_inner_margin_ratio=0.15,
        panel_selection_strategy="evidence", candidate_step_seconds=0.5,
        evidence_segment_duration_seconds=0.5, gaze_stability_window_seconds=0.3,
        image_width=None, image_height=None, start_index=0, limit=None)
    all_rows, sample_summaries = [], []
    selected_by_scene = {}
    for item in selected:
        selected_by_scene.setdefault(item["scene_key"], []).append(item["row_index"])
    original_choose = builder.choose_rows
    try:
        for scene, indices in selected_by_scene.items():
            paths = builder.scene_paths(TRIREF, eval_dir, scene)
            for index in indices:
                builder.choose_rows = lambda api_rows, _start, _limit, wanted=index: (
                    [(wanted, api_rows[wanted])] if wanted < len(api_rows) else [])
                rows, summary = builder.process_scene(paths, out / "manifest_build", args, [])
                all_rows.extend(rows)
                sample_summaries.append(summary)
    finally:
        builder.choose_rows = original_choose
    if not all_rows:
        raise RuntimeError("Original Projected-2D manifest builder produced no rows")
    builder.write_manifest(out / "manifest_all.csv", all_rows)
    save_json(out / "original_annotation_parity.json", audit_annotation_parity(gt_rows, all_rows))
    save_json(out / "manifest_build_summary.json", {"samples": sample_summaries,
        "rows": len(all_rows), "builder_source": str(TRIREF / "exam2/build_2d_eval_manifest.py"),
        "annotation_source": str(PAPER_GT), "note": "GT labels/points used for manifest/evaluator only; never included in model input."})
    return all_rows


def make_condition_images(manifest_rows, context_by_key, out: Path, config):
    original_visual = load_module("triref_exam2_visual", TRIREF / "exam2/run_qwen3vl_2d_point_grounding.py")
    records_by_event = {}
    frame_rows_by_event = {}
    for row in manifest_rows:
        key = (row["scene"], int(row["row_index"]))
        frame_rows_by_event.setdefault(key, {}).setdefault(row["panel_id"], row)
    for key, panel_map in frame_rows_by_event.items():
        ctx = context_by_key[key]
        records = gaze_records(ctx)
        records_by_event[key] = records
        event_dir = out / "inputs" / key[0] / str(key[1])
        pose_contract_path = out / "pose_alignment" / key[0] / str(key[1]) / "frame_pose_contract.json"
        pose_contract = json.loads(pose_contract_path.read_text()) if pose_contract_path.is_file() else {}
        pose_by_frame = {row['frame_id']: row for row in pose_contract.get('frames', [])}
        condition_paths = {group: [] for group in GROUPS}
        panel_meta = []
        projections = []
        for panel_id in sorted(panel_map, key=lambda value: int(value[1:])):
            row = panel_map[panel_id]
            source = Path(row["frame_path"])
            if not source.is_file():
                projections.append({"frame_id": panel_id, "status": "missing_source_frame", "source_path": str(source)})
                continue
            with Image.open(source) as opened:
                rgb = opened.convert("RGB")
            requested_video_time = float(row["video_frame_time"])
            pose_frame = pose_by_frame.get(panel_id, {})
            frame_time = pose_frame.get('frame_time', {})
            reference_camera = pose_frame.get('reference_camera', {})
            source_sha256 = sha_file(source)
            geometry = aligned_frame_geometry(pose_frame, source_sha256, requested_video_time,
                                              float(row['video_time_offset_seconds'] or 0.0))
            video_time = geometry['video_time_seconds']
            telemetry_time = geometry['telemetry_time_seconds']
            clock_available = geometry['clock_available']
            common = {**DEFAULTS, **config["gaze"]}
            pose_input = geometry['reference_pose']
            heat_current, stat_current, proj_current = build_heatmap(ctx, records,
                frame_time_video=video_time, frame_time_telemetry=telemetry_time,
                event_window=[float(ctx.api_row["t_start"]), float(ctx.api_row["t_end"])],
                clock_available=clock_available, image_size=rgb.size, strategy="instant",
                config=common, reference_pose=pose_input)
            heat_history, stat_history, proj_history = build_heatmap(ctx, records,
                frame_time_video=video_time, frame_time_telemetry=telemetry_time,
                event_window=[float(ctx.api_row["t_start"]), float(ctx.api_row["t_end"])],
                clock_available=clock_available, image_size=rgb.size, strategy="local_history",
                config=common, reference_pose=pose_input)
            rendered_current, current_render = render_presentation(rgb, heat_current, "H_inst", common)
            rendered_history, history_render = render_presentation(rgb, heat_history, "H_local", common)
            rendered_foveation, foveation_render = render_presentation(rgb, heat_history, "F_local", common)
            frame_dir = event_dir / panel_id
            frame_dir.mkdir(parents=True, exist_ok=True)
            paths = {}
            # Match intermediate and final codecs across the four conditions.
            # Retain the old RGB composites at their original paths for audit.
            for group, image in [("RGB", rgb), ("CURRENT_HEATMAP", rendered_current),
                                 ("HISTORY_HEATMAP", rendered_history),
                                 ("HISTORY_FOVEATION", rendered_foveation)]:
                filename = "RGB_CODEC_ALIGNED.jpg" if group == "RGB" else f"{group}.jpg"
                path = frame_dir / filename
                image.save(path, quality=94)
                paths[group] = path
            full_paths = dict(paths)
            paired_meta = None
            for group in GROUPS:
                metadata = {"panel_id": panel_id, "frame_path": str(paths[group]),
                    "video_frame_time": str(video_time), "json_sample_time": str(telemetry_time),
                    "gaze_u_norm": (next((p.get('normalized_uv', [None, None])[0] for p in proj_current
                                           if p.get('status') == 'projected_frustum_only'), '')),
                    "gaze_v_norm": (next((p.get('normalized_uv', [None, None])[1] for p in proj_current
                                           if p.get('status') == 'projected_frustum_only'), '')),
                    "gaze_projection_valid": str(any(p.get('status') == 'projected_frustum_only'
                                                     for p in proj_current))}
                expanded, expanded_meta = original_visual.expand_panel_context_images(
                    key, [paths[group]], [metadata], out / "model_inputs" /
                    ("RGB_CODEC_ALIGNED" if group == "RGB" else group),
                    Namespace(panel_context_mode="paired_crop", gaze_crop_ratio=0.35,
                              crop_output_size=768, overwrite=True))
                paths[group] = expanded[0]
                condition_paths[group].append(paths[group])
                if paired_meta is None:
                    paired_meta = expanded_meta[0]
                else:
                    assert paired_meta["crop_bbox_norm"] == expanded_meta[0]["crop_bbox_norm"]
            panel_meta.append({"frame_id": panel_id, "source_frame_id": source.name,
                "video_time_seconds": video_time, "requested_video_time_seconds": requested_video_time,
                "actual_container_pts_seconds": frame_time.get('actual_container_pts_seconds'),
                "telemetry_time_seconds": telemetry_time,
                "selected_clock_offset_seconds": geometry['selected_clock_offset_seconds'],
                "selected_pose_clock_source": geometry['selected_pose_clock_source'],
                "camera_source_field": geometry['camera_source_field'],
                "clock_policy": geometry['clock_policy'],
                "original_clock_offset_seconds": reference_camera.get('clock_offset_seconds'),
                "original_pose_clock_source": reference_camera.get('clock_status'),
                "candidate_clock_status": pose_contract.get('clock', {}).get('status'),
                "pose_status": geometry['pose_status'], "pose_reasons": geometry['pose_reasons'],
                "video_time_offset_seconds": row.get('video_time_offset_seconds'),
                "width": rgb.width, "height": rgb.height, "source_path": str(source),
                "source_sha256": source_sha256,
                "frame_geometry_status": geometry["status"],
                "attention_clock_supported": clock_available,
                "input_representation": "original_paired_crop",
                "intermediate_encoding": "JPEG_quality94_all_four_conditions",
                "crop_bbox_norm": paired_meta["crop_bbox_norm"],
                "crop_source": paired_meta["crop_source"],
                "full_frame_conditions": {group: str(path) for group, path in full_paths.items()},
                "gaze_projection_valid": row.get("gaze_projection_valid"),
                "current_gaze_count": stat_current["valid_projected_count"],
                "history_gaze_count": stat_history["valid_projected_count"],
                "current_gaze_duration": stat_current["valid_evidence_duration_seconds"],
                "history_gaze_duration": stat_history["valid_evidence_duration_seconds"],
                "current_kernel_sigma_pixels": stat_current["kernel_sigma_pixels_at_model_input"],
                "history_kernel_sigma_pixels": stat_history["kernel_sigma_pixels_at_model_input"],
                "current_render": current_render, "history_render": history_render,
                "foveation_render": foveation_render,
                "processed_sha256": {group: sha_file(paths[group]) for group in GROUPS}})
            np.savez_compressed(frame_dir / "attention_audit.npz",
                current_H_unnormalized=heat_current, history_H_unnormalized=heat_history)
            projections.append({"frame_id": panel_id, "current": proj_current, "history": proj_history,
                                "manifest_gaze_uv_for_visual_audit_only": [row.get("gaze_u_norm"), row.get("gaze_v_norm")]})
            rgb.close()
        if len(panel_meta) != len(panel_map):
            # Keep the event in its fixed denominator; absent input is a recorded failure.
            for group in GROUPS:
                condition_paths[group] = [p for p in condition_paths[group] if p.exists()]
        save_json(event_dir / "input_audit.json", {"scene": key[0], "row_index": key[1],
            "event_id": ctx.event_id, "panels": panel_meta, "projection_records": projections,
            "source_panel_count": len(panel_map), "built_panel_count": len(panel_meta),
            "conditions": {group: [{"path": str(path), "sha256": sha_file(path)} for path in paths]
                           for group, paths in condition_paths.items()}})
        frame_rows_by_event[key] = {"panel_meta": panel_meta, "condition_paths": condition_paths,
                                    "projection_records": projections}
    return frame_rows_by_event


def prepare_pose_source_audits(manifest_rows, context_by_key, out: Path):
    """Expose only frozen RGB/frame metadata to the GT-free PTS/pose fitter."""
    by_event = {}
    for row in manifest_rows:
        key = (row['scene'], int(row['row_index']))
        by_event.setdefault(key, {}).setdefault(row['panel_id'], row)
    for (scene, row_index), panels in by_event.items():
        target = out / 'inputs' / scene / str(row_index) / 'input_audit.json'
        if target.exists() and json.loads(target.read_text()).get('panels'):
            continue
        ctx = context_by_key[(scene, row_index)]
        values = []
        for panel_id in sorted(panels, key=lambda item: int(item[1:])):
            row = panels[panel_id]
            source = Path(row['frame_path'])
            values.append({'frame_id': panel_id, 'source_path': str(source.resolve()),
                'source_sha256': sha_file(source), 'width': int(row['image_width']),
                'height': int(row['image_height']), 'video_time_seconds': float(row['video_frame_time']),
                'video_time_offset_seconds': row.get('video_time_offset_seconds', ''),
                'telemetry_time_seconds': float(row['json_sample_time']),
                'video_path': str(Path(ctx.video_path).resolve())})
        save_json(target, {'scene': scene, 'row_index': row_index,
            'event_id': ctx.event_id, 'source_panel_count': len(values), 'panels': values,
            'ground_truth_loaded': False, 'source_media_only': True})


def reproject_gt_manifest(rows, out: Path):
    """Create a parallel evaluator manifest using world anchors and PTS poses."""
    from .projection_diagnostics import project_world_anchor
    from collections import Counter

    output, audit, contracts = [], [], {}
    for original in rows:
        row = dict(original)
        key = (row['scene'], int(row['row_index']))
        if key not in contracts:
            path = out / 'pose_alignment' / key[0] / str(key[1]) / 'frame_pose_contract.json'
            try:
                contracts[key] = json.loads(path.read_text())
            except OSError:
                contracts[key] = {'frames': [], 'clock': {'status': 'POSE_CONTRACT_MISSING'}}
        contract = contracts[key]
        frame = next((item for item in contract.get('frames', [])
                      if item.get('frame_id') == row['panel_id']), None)
        source = Path(row['frame_path'])
        geometry = aligned_frame_geometry((frame or {}) if source.is_file() else {},
            sha_file(source) if source.is_file() else '', float(row['video_frame_time']),
            float(row.get('video_time_offset_seconds') or 0.))
        camera = geometry['reference_pose'] or {}
        if camera.get('status') == 'POSE_AVAILABLE':
            sample = {'cameraPosition': dict(zip('xyz', camera['position_world'])),
                      'cameraRotation': dict(zip('xyzw', camera['rotation_xyzw'])),
                      'cameraFOV': camera['fov_vertical_degrees']}
            projected = project_world_anchor(
                [float(row['anchor_x']), float(row['anchor_y']), float(row['anchor_z'])], sample,
                int(row['image_width']), int(row['image_height']))
        else:
            projected = {'status': camera.get('status', 'POSE_UNAVAILABLE'),
                         'in_frustum': False, 'normalized_uv_unclipped': None,
                         'pixel_xy_unclipped': None}
        old = {'gt_x': row.get('gt_x', ''), 'gt_y': row.get('gt_y', ''),
               'gt_u_norm': row.get('gt_u_norm', ''), 'gt_v_norm': row.get('gt_v_norm', ''),
               'projection_valid': row.get('projection_valid', '')}
        normalized, pixel = projected.get('normalized_uv_unclipped'), projected.get('pixel_xy_unclipped')
        valid = bool(projected.get('in_frustum') and normalized and pixel)
        row.update(original_gt_x=old['gt_x'], original_gt_y=old['gt_y'],
                   original_gt_u_norm=old['gt_u_norm'], original_gt_v_norm=old['gt_v_norm'],
                   original_projection_valid=old['projection_valid'],
                   pose_alignment_status=camera.get('clock_status', camera.get('status', 'POSE_UNAVAILABLE')),
                   pose_candidate_clock_status=contract.get('clock', {}).get('status'),
                   source_frame_pts_seconds=(frame or {}).get('frame_time', {}).get('actual_container_pts_seconds'),
                   telemetry_pose_time_seconds=geometry['telemetry_time_seconds'],
                   selected_clock_offset_seconds=geometry['selected_clock_offset_seconds'],
                   selected_pose_clock_source=geometry['selected_pose_clock_source'],
                   frame_geometry_status=geometry['status'], clock_policy=geometry['clock_policy'],
                   gt_projection_status=projected.get('status'), projection_valid=str(valid))
        if valid:
            row.update(gt_x=f'{pixel[0]:.2f}', gt_y=f'{pixel[1]:.2f}',
                       gt_u_norm=f'{normalized[0]:.6f}', gt_v_norm=f'{normalized[1]:.6f}',
                       status='point_evaluable', status_detail='')
        else:
            row.update(gt_x='', gt_y='', gt_u_norm='', gt_v_norm='',
                       status='projection_invalid',
                       status_detail=f"PTS pose reprojection unavailable/outside: {projected.get('status')}")
        output.append(row)
        audit.append({'scene': key[0], 'row_index': key[1], 'frame_id': row['panel_id'],
            'referent_name': row.get('referent_name'), 'pose_status': camera.get('status', 'POSE_UNAVAILABLE'),
            'pose_clock_source': geometry['selected_pose_clock_source'],
            'selected_clock_offset_seconds': geometry['selected_clock_offset_seconds'],
            'frame_geometry_status': geometry['status'], 'clock_policy': geometry['clock_policy'],
            'candidate_clock_status': contract.get('clock', {}).get('status'),
            'pts_seconds': row.get('source_frame_pts_seconds'),
            'telemetry_pose_time_seconds': row.get('telemetry_pose_time_seconds'),
            'old_gt_pixel_xy': [old['gt_x'], old['gt_y']], 'new_gt_pixel_xy_unclipped': pixel,
            'new_gt_uv_unclipped': normalized, 'new_projection_status': projected.get('status'),
            'old_projection_valid': old['projection_valid'], 'new_projection_valid': str(valid)})
    write_csv(out / 'gt_pose_reprojection_audit.csv', audit, list(audit[0]) if audit else [])
    save_json(out / 'gt_pose_reprojection_summary.json', {
        'rows': len(audit), 'clock_policy': ATTENTION_CLOCK_POLICY,
        'projection_scope': 'event_referents_from_manifest_only',
        'projection_status_counts': dict(Counter(r['new_projection_status'] for r in audit)),
        'pose_status_counts': dict(Counter(r['pose_status'] for r in audit)),
        'gt_used_for_pose_fit_or_model_input': False,
        'evaluator_matching_rules_unchanged': True,
        'original_manifest_and_gt_columns_preserved': True})
    return output


VISUAL_PROMPT_VERSION = 'explicit_referent_contract_v3'


def build_visual_prompt_legacy(ctx, mentions, panel_meta):
    safe_mentions = [{key: mention.get(key) for key in ["mention_id", "text", "span", "kind", "role", "relation", "count"]}
                     for mention in mentions]
    panel_lines = [f"- {row['frame_id']}: source_frame_id={row['source_frame_id']}, video_time={row['video_time_seconds']:.3f}s, crop_bbox_in_source=[{row.get('crop_bbox_norm', '')}]"
                   for row in panel_meta]
    return f'''You are the Visual agent in the Projected-2D referring-point task.
Use the original instruction and each supplied mention occurrence to locate every intended referent in the chronological egocentric panels.
Each image follows the original paired layout: LEFT is the full source frame and RIGHT is its gaze crop. The crop bounding box is shared across all conditions.
Use the crop to inspect detail, but return point_uv in the LEFT full source frame coordinate system; map crop points through crop_bbox_in_source.
The panels may show original RGB, a subtle current-gaze heatmap, an accumulated recent-gaze heatmap, or gaze-guided foveation.
Warm red tint and brighter grayscale areas are weak attention evidence, not target labels. Relatively sharper areas in foveation are also weak cues; blur does not imply object absence.
Always ground the point on visible object pixels using RGB appearance and action. Do not assume the gaze cue is correct. Do not use world gaze coordinates, camera pose, anchor IDs, target names, GT coordinates, or hidden annotation mappings.
Original instruction:
{ctx.text}
Panel identifiers and times:
{chr(10).join(panel_lines)}
Mention occurrences:
{json.dumps(safe_mentions, ensure_ascii=False)}
For each mention, return status localized, ambiguous, or not_visible. Return one or more predictions when the expression refers to multiple targets; each prediction has frame_id from the panel list and point_uv normalized to [0,1] within that original source frame, origin at top-left. Do not return canvas coordinates. Give a short visible evidence phrase. If localization is uncertain, use ambiguous/not_visible rather than inventing coordinates.
Return only JSON with schema: {{"mentions":[{{"mention_id":"m0","status":"localized","predictions":[{{"frame_id":"P1","point_uv":[0.5,0.5]}}],"evidence":"..."}}]}}'''


def build_visual_prompt(ctx, mentions, panel_meta):
    fields = ('mention_id', 'text', 'span', 'object_type', 'kind', 'role', 'count',
              'component', 'parent_text', 'anchor_text', 'coreference_id',
              'parent_mention_id', 'coreference_note', 'relation')
    tasks = []
    for mention in mentions:
        task = {k: mention.get(k) for k in fields}
        task['localization_scope'] = ('whole parent object indicated by the component phrase'
                                      if mention.get('component') or mention.get('parent_text')
                                      else 'spatial destination/region' if mention.get('kind') == 'location'
                                      else 'visible instance of the named object')
        tasks.append(task)
    frames = [{k: p[k] for k in ('frame_id', 'source_frame_id', 'video_time_seconds', 'crop_bbox_norm')}
              for p in panel_meta]
    skeleton = {'mentions': [{'mention_id': m['mention_id'],
                              'status': 'localized|ambiguous|not_visible',
                              'predictions': [], 'evidence': ''} for m in mentions]}
    return f'''You are the Visual agent. Solve given-expression 2D temporal localization.
Your task is to find the visible entity or destination denoted by EACH exact expression in localization_tasks. Do not merely name a category or describe the scene.
Read the entire instruction to identify who acts, what is moved, and the destination. The expression and object_type tell you WHAT to find; role, relations and the instruction tell you WHICH instance. Use color, shape, position, other visible objects and action to distinguish same-class instances.
coreference_id, parent_mention_id and coreference_note are language hypotheses from the Referential agent. Repeated mentions with a supported common antecedent denote the same entity, but its image coordinates may change between frames. Repeated words alone do not establish coreference. If the supplied language hypothesis conflicts with the instruction or image, report ambiguity instead of inventing certainty.
For a component expression, follow localization_scope and locate its whole parent object: this is the original object-level task, not component recognition. For a destination such as "there", locate the indicated visible spatial region only when context/cues support it. Do not substitute an arbitrary nearest object.
Each image is a paired layout: LEFT is the full source frame, RIGHT is a crop of that SAME frame. Image k corresponds to source_frames[k]. crop_bbox_norm is x0,y0,x1,y1 in normalized source coordinates. Convert a crop point (a,b) to source (x0+a*(x1-x0), y0+b*(y1-y0)). Never return composite-canvas coordinates or a default centre. Check that the final point actually overlays the object on the LEFT source image.
All conditions receive this identical prompt. Images may be RGB, current/recent gaze overlays, or gaze-guided foveation. A warm tint or relatively sharp region is only weak observed attention evidence. Confirm the referent in RGB; gaze can be on another object, and peripheral blur does not mean absence. No numeric gaze or world coordinates, hidden answer identifiers, GT positions or annotation-derived object counts are supplied.
For every task, return exactly one mention entry with the exact mention_id. status is localized, ambiguous, or not_visible. For localized predictions, identify the intended visible entity on the actual source frame. Use point_uv in [0,1], origin at top-left. Multi-target expressions may have multiple predictions. Choose an evidence frame that really shows each target; return no fabricated point for a frame where the target is outside the image or not identifiable. Do not copy one location across all mentions. Give short visual evidence identifying the referent and any unresolved confusion.
Instruction:
{ctx.text}
source_frames:
{json.dumps(frames, ensure_ascii=False)}
localization_tasks:
{json.dumps(tasks, ensure_ascii=False)}
Return only JSON. Complete ALL {len(mentions)} entries in this exact mention order and replace each status with one allowed value. For each localized entry, populate predictions with objects containing frame_id and point_uv=[u,v] measured from the image; there are deliberately no example/default coordinates to copy. Keep predictions empty only when evidence is insufficient. Required output structure:
{json.dumps(skeleton, ensure_ascii=False)}'''


def normalize_model_mentions(parsed, referential):
    raw = parsed.get("mentions") if isinstance(parsed, dict) else None
    if not isinstance(raw, list):
        return None, "response_missing_mentions_array"
    allowed = {row["mention_id"]: row for row in referential.get("mentions", [])}
    output = []
    errors = []
    for row in raw:
        if not isinstance(row, dict) or row.get("mention_id") not in allowed:
            errors.append("unknown_or_malformed_mention_id")
            continue
        status = row.get("status")
        if status not in {"localized", "ambiguous", "not_visible"}:
            errors.append(f"invalid_status:{row.get('mention_id')}")
            status = "ambiguous"
        predictions = row.get("predictions", [])
        valid_predictions = []
        if isinstance(predictions, list):
            for item in predictions:
                if not isinstance(item, dict):
                    continue
                frame_id, point = item.get("frame_id"), item.get("point_uv")
                try:
                    u, v = map(float, point)
                except (TypeError, ValueError):
                    continue
                if frame_id not in {"P1", "P2", "P3"} or not (np.isfinite(u) and np.isfinite(v) and 0 <= u <= 1 and 0 <= v <= 1):
                    continue
                valid_predictions.append({"frame_id": frame_id, "point_uv": [u, v]})
        else:
            errors.append(f"invalid_predictions:{row['mention_id']}")
        output.append({"mention_id": row["mention_id"], "mention": allowed[row["mention_id"]]["text"],
                       "status": status, "predictions": valid_predictions,
                       "evidence": str(row.get("evidence") or "")})
    missing = set(allowed) - {row["mention_id"] for row in output}
    for mid in sorted(missing):
        output.append({"mention_id": mid, "mention": allowed[mid]["text"], "status": "ambiguous",
                       "predictions": [], "evidence": "Missing or invalid model row; retained as empty."})
        errors.append(f"missing_mention:{mid}")
    return {"mentions": output}, ";".join(errors)


def evaluator_adapter(parsed):
    referents = []
    for mention in (parsed or {}).get("mentions", []):
        for prediction in mention.get("predictions", []):
            point = prediction["point_uv"]
            referents.append({"mention": mention["mention"], "panel_id": prediction["frame_id"],
                              "coordinate_space": "full_panel", "x_norm": point[0], "y_norm": point[1]})
    return {"referents": referents}


def run_model(groups, sample_rows, context_by_key, frame_inputs, out: Path, limit: int,
              *, model_path, quantization="none"):
    backend = QwenBackend(out.parent / "cache", str(model_path), local_only=True,
        namespace=f"projected2d_visual_guidance_{Path(model_path).name}_{VISUAL_PROMPT_VERSION}_{quantization}",
        trace_visual_inputs=True)
    original_load = backend.load

    def load_with_reserved_vram():
        if backend.runner is not None:
            return
        local = baseline_module("local")
        original_loader = local.load_model_with_fallbacks

        def bounded_loader(model_name, model_kwargs, runtime, use_flash_attn):
            bounded_kwargs = dict(model_kwargs)
            if quantization == "int8":
                from transformers import BitsAndBytesConfig
                bounded_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                bounded_kwargs["device_map"] = {"": 0}
            elif Path(model_path).name == QWEN8.name:
                bounded_kwargs["max_memory"] = {0: "10GiB", "cpu": "96GiB"}
                bounded_kwargs["offload_folder"] = str(out / "model_offload")
            return original_loader(model_name, bounded_kwargs, runtime, use_flash_attn)

        local.load_model_with_fallbacks = bounded_loader
        try:
            original_load()
        finally:
            local.load_model_with_fallbacks = original_loader
        device_map = getattr(backend.runner.model, "hf_device_map", {})
        save_json(out / "model_device_map.json", {
            "model": str(model_path), "quantization": quantization,
            "max_memory": ({"cuda:0": "10GiB", "cpu": "96GiB"}
                           if quantization == "none" and Path(model_path).name == QWEN8.name else None),
            "device_map": device_map,
        })

    backend.load = load_with_reserved_vram
    results = {group: [] for group in groups}
    sample_by_key = {(row["scene_key"], row["row_index"]): row for row in sample_rows}
    for index, item in enumerate(sample_rows[:limit], 1):
        key = (item["scene_key"], item["row_index"])
        ctx = context_by_key[key]
        sample_dir = out / "samples" / key[0] / str(key[1])
        sample_dir.mkdir(parents=True, exist_ok=True)
        parse_path = sample_dir / "referential.json"
        prompt_path = sample_dir / "visual_prompt.txt"
        smoke_dir = out.parent / "smoke4" / "samples" / key[0] / str(key[1])
        if out.name == "pilot20" and not parse_path.exists() and (smoke_dir / "referential.json").exists():
            save_json(parse_path, json.loads((smoke_dir / "referential.json").read_text()))
        if parse_path.exists():
            referential = json.loads(parse_path.read_text())
        else:
            try:
                referential = ReferentialAgent(backend).run(ctx)
            except Exception as exc:
                referential = {"mentions": [], "parse_failure": f"{type(exc).__name__}: {exc}"}
            save_json(parse_path, referential)
        mention_rows = referential.get("mentions", []) if isinstance(referential, dict) else []
        panel_meta = frame_inputs[key]["panel_meta"]
        prompt = build_visual_prompt(ctx, mention_rows, panel_meta)
        prompt_path.write_text(prompt, encoding="utf-8")
        for group in groups:
            images = frame_inputs[key]["condition_paths"][group]
            input_image_audit = [{"path": str(path), "sha256": sha_file(path)} for path in images]
            result_path = sample_dir / f"{group}.json"
            if out.name == "pilot20" and not result_path.exists() and (smoke_dir / f"{group}.json").exists():
                prior = json.loads((smoke_dir / f"{group}.json").read_text())
                if prior.get("prompt") == prompt and prior.get("input_images") == input_image_audit:
                    save_json(result_path, {**prior, "replayed_from": str(smoke_dir / f"{group}.json")})
            if result_path.exists():
                saved = json.loads(result_path.read_text())
                if (saved.get("status") in ("COMPLETE", "PARSE_WITH_WARNINGS")
                        and saved.get("prompt") == prompt
                        and saved.get("input_images") == input_image_audit):
                    results[group].append(saved)
                    continue
                attempts = sample_dir / "attempts"
                attempts.mkdir(exist_ok=True)
                attempt_index = 1
                while (attempts / f"{group}_{attempt_index:02d}.json").exists():
                    attempt_index += 1
                save_json(attempts / f"{group}_{attempt_index:02d}.json", saved)
            if len(images) != 3 or not mention_rows:
                result = {"scene": key[0], "row_index": key[1], "event_id": ctx.event_id,
                          "condition": group, "status": "NOT_RUN", "error": referential.get("parse_failure") or
                          f"expected 3 panels, found {len(images)} or no mentions",
                          "parsed": {"mentions": [{"mention_id": m["mention_id"], "mention": m["text"],
                              "status": "ambiguous", "predictions": [], "evidence": "Upstream failure; empty prediction retained."}
                              for m in mention_rows]}, "raw_response": None, "prompt": prompt}
                save_json(result_path, result)
                results[group].append(result)
                continue
            started = time.monotonic()
            try:
                parsed_raw = backend.ask(f"{key[0]}/row_{key[1]}", f"projected2d_{group}", prompt,
                                         [str(path) for path in images], max_new_tokens=768)
                request = Path(backend.requests[-1])
                raw_saved = json.loads(request.read_text()) if request.exists() else {}
                normalized, parse_error = normalize_model_mentions(parsed_raw, referential)
                result = {"scene": key[0], "row_index": key[1], "event_id": ctx.event_id,
                    "condition": group, "status": "COMPLETE" if not parse_error else "PARSE_WITH_WARNINGS",
                    "raw_response": raw_saved.get("response"), "parsed_raw": parsed_raw,
                    "parsed": normalized or {"mentions": []}, "parse_error": parse_error,
                    "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "request_path": str(request), "elapsed_seconds": time.monotonic()-started,
                    "input_images": input_image_audit,
                    "adapter_output": evaluator_adapter(normalized),
                    "visual_tensor_trace": raw_saved.get("visual_tensor_trace"),
                    "processor_audit": raw_saved.get("processor_audit")}
            except Exception as exc:
                result = {"scene": key[0], "row_index": key[1], "event_id": ctx.event_id,
                    "condition": group, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                    "parsed": {"mentions": [{"mention_id": m["mention_id"], "mention": m["text"],
                        "status": "ambiguous", "predictions": [], "evidence": "Call failure; empty prediction retained."}
                        for m in mention_rows]}, "raw_response": None, "prompt": prompt,
                    "elapsed_seconds": time.monotonic()-started, "input_images": input_image_audit,
                    "request_path": backend.requests[-1] if backend.requests else None}
            save_json(result_path, result)
            results[group].append(result)
            print(f"[{index}/{limit}] {key[0]}/{ctx.event_id} {group}: {result['status']}", flush=True)
    return results, backend.call_records


def write_evaluator_predictions(results, samples, out: Path, split: str,
                                manifest_variants: dict[str, Path]):
    sample_by_key = {(row["scene_key"], row["row_index"]): row for row in samples}
    pred_dir = out / split / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    evaluator = load_module("triref_exam2_evaluator", TRIREF / "exam2/evaluate_2d_point_grounding.py")
    for group in GROUPS:
        pred_rows = []
        row_map = {(row["scene"], row["row_index"]): row for row in results[group]}
        for key, sample in sample_by_key.items():
            result = row_map.get(key, {})
            adapted = evaluator_adapter(result.get("parsed", {"mentions": []}))
            pred_rows.append({"scene": key[0], "row_index": key[1], "event_id": sample["event_id"],
                "instruction": sample.get("instruction", ""), "model_input_image": ";".join(
                    item["path"] for item in result.get("input_images", [])),
                "model_raw_output": result.get("raw_response") or "",
                "parsed_json": json.dumps(adapted, ensure_ascii=False),
                "parse_ok": str(result.get("status") in ("COMPLETE", "PARSE_WITH_WARNINGS")),
                "prediction_count": len(adapted["referents"]),
                "error_message": result.get("error", result.get("parse_error", ""))})
        pred_csv = pred_dir / f"{group}.csv"
        write_csv(pred_csv, pred_rows, CSV_COLUMNS)
        for variant, manifest_path in manifest_variants.items():
            eval_dir = out / split / "evaluation" / variant / group
            evaluator_args = [sys.executable, str(TRIREF / "exam2/evaluate_2d_point_grounding.py"),
                "--manifest", str(manifest_path), "--pred_csv", str(pred_csv),
                "--output_dir", str(eval_dir), "--coordinate_mode", "panel", "--panel_width", "512",
                "--panel_height", "384", "--columns", "3"]
            completed = subprocess.run(evaluator_args, cwd=TRIREF, capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(f"Original evaluator failed for {variant}/{group}: {completed.stderr[-3000:]}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/projected2d_attention_qwen8b_20261022")
    parser.add_argument("--split", choices=("smoke4", "pilot20", "pilot100"), default="smoke4")
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--model-path", default=str(QWEN8))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite-frames", action="store_true")
    parser.add_argument("--quantization", choices=("none", "int8"), default="none")
    parser.add_argument("--max-events", type=int)
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    # New language/visual contract must not overwrite the completed v1 pilot.
    if not out.name.endswith('_contract_v3'):
        out = out.with_name(out.name + '_contract_v3')
    if args.quantization != "none":
        out = out.with_name(out.name + "_" + args.quantization)
    out.mkdir(parents=True, exist_ok=True)
    gt_rows = read_csv(PAPER_GT)
    sample_path = out / "fixed_samples.json"
    if sample_path.exists():
        selected = json.loads(sample_path.read_text())["samples"]
    else:
        count = 100 if args.split == 'pilot100' else args.sample_count
        selected = select_fixed_samples(gt_rows, count=count)
        save_json(sample_path, {"seed": 20261022,
            "selection": "scene-stratified random sample from the original Projected-2D manifest; media/projection availability only; no model score filtering",
            "sample_count": len(selected),
            "source": str(PAPER_GT), "samples": selected})
    if args.split == 'pilot100' and len(selected) != 100:
        raise ValueError(f"pilot100 requires exactly 100 fixed samples, found {len(selected)}")
    prepare_all = setup_original_manifest(selected, gt_rows, out, overwrite=args.overwrite_frames)
    selected_by_key = {(row["scene_key"], row["row_index"]) for row in selected}
    fixed_rows = [row for row in prepare_all if (row["scene"], int(row["row_index"])) in selected_by_key]
    # Keep manifest rows limited to the fixed sample list; preserve the original columns/protocol.
    manifest_path = out / "manifest_all.csv"
    write_csv(manifest_path, fixed_rows, list(fixed_rows[0]))
    chosen = selected[:4] if args.split == "smoke4" else selected
    if args.max_events is not None:
        chosen = chosen[:args.max_events]
    chosen_keys = {(row["scene_key"], row["row_index"]) for row in chosen}
    chosen_manifest_rows = [row for row in fixed_rows if (row["scene"], int(row["row_index"])) in chosen_keys]
    contexts_manifest = {"samples": chosen}
    contexts = load_contexts(contexts_manifest, load_ground_truth=False)
    context_by_key = {(ctx.scene_key, ctx.row_index): ctx for ctx in contexts}
    chosen = [{**item, "instruction": context_by_key[(item["scene_key"], item["row_index"])].text}
              for item in chosen]
    model_path = Path(args.model_path).resolve()
    if not (model_path / 'config.json').is_file():
        raise FileNotFoundError(f"Local model config not found: {model_path / 'config.json'}")
    config = {"geometry_version": GEOMETRY_VERSION,
        "attention_clock_policy": ATTENTION_CLOCK_POLICY,
        "kernel_fov_source": "same_reference_camera_as_gaze_projection",
        "model": str(model_path), "model_config_sha256": sha_file(model_path / 'config.json'),
        "groups": GROUPS, "panels": 3,
        "frame_selection": "original TriRef evidence panel selector; fixed per-event panel IDs/times across conditions",
        "panel_selection_strategy": "evidence", "sample_inner_margin_ratio": 0.15,
        "panel_context_mode": "paired_crop", "gaze_crop_ratio": 0.35, "crop_output_size": 768,
        "prompt_mode": "given_expression_no_annotation_derived_expected_count",
        "visual_prompt_version": VISUAL_PROMPT_VERSION,
        "visual_input_encoding": "all conditions: decoded source -> identity/heatmap/foveation -> JPEG quality94 -> original gaze crop -> original paired composite JPEG quality92",
        "acceptable_top_k": 2, "dtype": "bfloat16", "quantization": args.quantization,
        "clock_validation_status": "PTS_RGB_MOTION_FIT_WITH_ORIGINAL_CLOCK_FALLBACK; evaluator GT labels remain the original Projected-2D annotations",
        "pose_alignment": "per-event exact source JPEG/container PTS and linear-position/SLERP camera pose; validated candidate else original offset fallback; original evaluator labels unchanged",
        "score_interpretation": "ORIGINAL_PROJECTED2D_EVALUATOR_LABELS_UNCHANGED; PTS pose contract affects gaze presentation only",
        "do_sample": False, "max_new_tokens": 768,
        "candidate_mapping": "frozen ReferentialAgent; candidate IDs omitted from Visual prompt",
        "history_seconds": config_gaze()["local_history_seconds"],
        "gaze": {**DEFAULTS, "local_history_seconds": 2.0, "kernel_sigma_degrees": 2.0,
                 "overlay_max_alpha": 0.28, "foveation_min_clear_fraction": 0.55,
                 "foveation_downsample_factors": [2, 4, 8]},
        "ground_truth_loaded_for_context": False, "gt_in_model_prompt": False,
        "evaluator": str(TRIREF / "exam2/evaluate_2d_point_grounding.py")}
    save_json(out / "config.json", config)
    prepare_pose_source_audits(chosen_manifest_rows, context_by_key, out)
    from .frame_pose_alignment import run as run_frame_pose_alignment
    pose_root = out / 'pose_alignment'
    alignment_events = [f"{item['scene_key']}:{item['row_index']}" for item in chosen]
    alignment = run_frame_pose_alignment(out, pose_root, {}, 'ffmpeg', 'ffprobe', reuse=True,
                                         events=alignment_events)
    config['pose_alignment_summary'] = str(pose_root / 'summary.json')
    config['pose_alignment_counts'] = alignment.get('counts', {})
    save_json(out / "config.json", config)
    pts_manifest_path = out / args.split / 'manifest_pts_pose_reprojected.csv'
    pts_rows = reproject_gt_manifest(chosen_manifest_rows, out)
    write_csv(pts_manifest_path, pts_rows, list(pts_rows[0]))
    frame_inputs = make_condition_images(chosen_manifest_rows, context_by_key, out, config)
    write_csv(out / args.split / "manifest_all.csv", chosen_manifest_rows, list(chosen_manifest_rows[0]))
    if args.prepare_only:
        print(f"Prepared {len(chosen)} events; no model calls.")
        return
    results, calls = run_model(GROUPS, chosen, context_by_key, frame_inputs,
                              out / args.split, len(chosen), model_path=model_path,
                              quantization=args.quantization)
    # Evaluator manifest is the exact fixed subset, including all projection and temporal fields.
    write_csv(out / args.split / "manifest_all.csv", chosen_manifest_rows, list(chosen_manifest_rows[0]))
    write_evaluator_predictions(results, chosen, out, args.split, {
        'original_gt': out / args.split / 'manifest_all.csv',
        'pts_pose_gt': pts_manifest_path})
    save_json(out / args.split / "summary.json", {"status": "COMPLETE", "split": args.split,
        "sample_count": len(chosen), "groups": list(GROUPS), "calls": calls,
        "metrics": {variant: {group: json.loads((out / args.split / "evaluation" / variant / group /
                           "2d_eval_summary.json").read_text()) for group in GROUPS}
                    for variant in ('original_gt', 'pts_pose_gt')},
        "failures": [{"scene": row["scene"], "row_index": row["row_index"], "condition": row["condition"],
                      "status": row["status"], "error": row.get("error", row.get("parse_error"))}
                     for group_rows in results.values() for row in group_rows if row["status"] not in ("COMPLETE", "PARSE_WITH_WARNINGS")]})
    print(f"Completed {args.split}: {out / args.split / 'summary.json'}")


def config_gaze():
    return {**DEFAULTS, "local_history_seconds": 2.0, "kernel_sigma_degrees": 2.0}


if __name__ == "__main__":
    main()
