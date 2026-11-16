"""Traceable local probe for gaze-image encoding and Visual-agent cue reading."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .context import load_contexts
from .gaze_evidence import (DEFAULTS, build_heatmap, gaze_records,
                            interpolate_camera_pose, render_presentation)
from .local_video_visual import qwen_temporal_groups, visual_prompt
from .qwen_backend import QwenBackend


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
MODEL_DEFAULT = WORKSPACE / "Qwen3-VL-8B-Instruct"
LEGACY_ATTENTION_NOTE = """Some frames may subtly tint pixels near measured gaze history or gently blur peripheral RGB. This is uncertain attention evidence, never an object label or target probability. Do not assume the hottest point is the target. Inspect the chronological frames and report whether a cue is visible, its approximate image center, or null when uncertain. Then use RGB, instruction, and the unchanged closed-set candidate metadata to assess the mention. The cue is optional corroboration, not a replacement for visible appearance, action, or relations."""
STEPWISE_ATTENTION_NOTE = """Use this gaze-guided visual protocol on the chronological video:
1. Inspect each temporal view in order and first identify whether it contains raw RGB, a red/orange gaze-history overlay, or gaze-guided foveation.
2. For every view, report the cue type, estimate its center in normalized image coordinates [u, v] with the top-left origin, and say which visible region it falls on. Use null if the cue or its center is not visually recoverable.
3. Track how that visual cue changes across time. A red overlay indicates projected gaze history; stronger tint means more accumulated gaze evidence. In foveation, relatively sharper regions encode greater gaze emphasis and blur is a rendering choice, not object absence or occlusion.
4. Only after that trace, compare the cue with the current language mention, visible candidates, and action. Gaze is weak, potentially misaligned corroboration; it is not target identity or a target probability. Prefer RGB appearance, action, and relations when evidence conflicts, and leave identity unresolved when needed.

Do not use or infer numeric gaze coordinates, world points, camera pose, or target labels. The only gaze cue available is what is visibly encoded in the supplied pixels. Do not assume the screen center is attended."""


def read_json(path):
    return json.loads(Path(path).read_text())


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path) -> str:
    return sha_bytes(Path(path).read_bytes())


def uniform_indices(length, count):
    if length < count:
        raise ValueError(f"Need {count} source frames, found {length}")
    indices = [round(i * (length - 1) / (count - 1)) for i in range(count)]
    if len(set(indices)) != count:
        raise ValueError("Uniform frame selection produced duplicate source frames")
    return indices


def select_probe_frames(frames, records, offset, count, config):
    """Optional GT-free quality/pose selection within the unchanged event window."""
    if not config.get('keyframe_quality_and_pose_selection'):
        return [frames[i] for i in uniform_indices(len(frames), count)], {
            'policy': 'uniform_time_strata_over_existing_offline_event_window_no_GT',
            'candidate_frame_count': len(frames), 'quality_filter_enabled': False,
        }
    rows, eligible = [], []
    for frame in frames:
        pose = interpolate_camera_pose(records, float(frame['time_seconds'])-float(offset)
                                       if offset is not None else None,
                                       config.get('gaze', {}).get('max_gap_seconds', .25))
        source = Path(frame['path'])
        if not source.is_absolute():
            source = (ROOT / source).resolve()
        with Image.open(source) as image:
            gray = np.asarray(image.convert('L').resize((128, 128)), dtype=np.float32)
        visible_fraction = float((gray > 12).mean())
        reasons = []
        if pose is None:
            reasons.append('no_corresponding_camera_pose_at_existing_clock_offset')
        if visible_fraction < .1 or float(gray.std()) < 3:
            reasons.append('low_content_or_mostly_black_frame')
        rows.append({'frame_id': frame['frame_id'], 'source_frame_id': frame.get('source_frame_id'),
                     'time_seconds': frame['time_seconds'], 'eligible': not reasons,
                     'nonblack_fraction': visible_fraction, 'grayscale_std': float(gray.std()),
                     'rejection_reasons': reasons})
        if not reasons:
            eligible.append(frame)
    fallback = len(eligible) < count
    pool = frames if fallback else eligible
    selected = [pool[i] for i in uniform_indices(len(pool), count)]
    return selected, {
        'policy': 'quality_and_pose_available_then_uniform_time_strata_no_GT',
        'candidate_frame_count': len(frames), 'eligible_frame_count': len(eligible),
        'quality_filter_enabled': True, 'fallback_to_unfiltered_event': fallback,
        'fallback_reason': 'fewer_than_requested_frames_with_corresponding_pose_and_content' if fallback else None,
        'observation_window_changed': False, 'clock_offset_changed': False,
        'frame_audit': rows,
    }


def rgb_pixel_sha(image):
    return sha_bytes(np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes())


def cue_prompt(base, arm, temporal_groups):
    contract = "Return compact JSON with these keys: evidence, visual_observations,\nand identity_status."
    replacement = ("Return compact JSON with these keys: attention_trace, evidence, "
                   "visual_observations, and identity_status. Each attention_trace row "
                   "must contain view_id, cue_type (none, heatmap, foveation, unclear), "
                   "cue_center_uv ([u,v] or null), cue_confidence, and cue_observation.")
    if contract not in base:
        raise ValueError("Unexpected local Visual prompt contract")
    prompt = base.replace(contract, replacement, 1)
    note = LEGACY_ATTENTION_NOTE if arm == "legacy_soft" else STEPWISE_ATTENTION_NOTE
    view_ids = [group["view_id"] for group in temporal_groups]
    trace_rule = (
        "\n\nRequired attention_trace audit output: return exactly one row, in this order, "
        f"for each view ID {json.dumps(view_ids)}. Do not return an empty attention_trace. "
        "If no cue is visible, set cue_type=none, cue_center_uv=null, cue_confidence=0, "
        "and explain that no cue is detectable. If it is ambiguous, use cue_type=unclear "
        "and a null center. Never fabricate a center. This trace requirement overrides "
        "the general instruction to use empty arrays for absent object evidence; keep "
        "unsupported candidate evidence empty. For this singular mention, report at most "
        "one positive same-class candidate.\n")
    marker = "\n\nInstruction: "
    if marker not in prompt:
        raise ValueError("Unexpected local Visual prompt input boundary")
    return prompt.replace(marker, "\n\nVisual attention protocol:\n" + note + trace_rule + marker, 1)


def model_center(row):
    point = row.get("cue_center_uv")
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    try:
        u, v = float(point[0]), float(point[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(u) and math.isfinite(v) and 0 <= u <= 1 and 0 <= v <= 1):
        return None
    return [u, v]


def weighted_center(heat):
    total = float(heat.sum())
    if total <= 0:
        return None
    height, width = heat.shape
    ys, xs = np.indices(heat.shape)
    return [float((heat * (xs + 0.5)).sum() / total / width),
            float((heat * (ys + 0.5)).sum() / total / height)]


def prediction_summary(parsed, candidate_ids):
    evidence = parsed.get("evidence", []) if isinstance(parsed, dict) else []
    positive = []
    invalid = []
    for row in evidence:
        if not isinstance(row, dict):
            continue
        name = row.get("candidate_id")
        try:
            score = float(row.get("support_score", 0) or 0)
            confidence = float(row.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            continue
        if score > 0 and confidence > 0:
            if name in candidate_ids:
                positive.append(name)
            else:
                invalid.append(name)
    return {"positive_candidate_ids": sorted(set(positive)),
            "out_of_scope_positive_ids": sorted(set(invalid)),
            "identity_status": parsed.get("identity_status") if isinstance(parsed, dict) else None,
            "attention_trace": parsed.get("attention_trace", []) if isinstance(parsed, dict) else []}


def make_contact_sheet(directory, frame_ids, paths_by_condition):
    thumb = 168
    label_h = 24
    rows = ["R", "H_local", "F_local"]
    canvas = Image.new("RGB", ((thumb + 2) * len(frame_ids), (thumb + label_h + 2) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for r, condition in enumerate(rows):
        y = r * (thumb + label_h + 2)
        draw.text((3, y + 5), condition, fill="black")
        for col, frame_id in enumerate(frame_ids):
            x = col * (thumb + 2)
            draw.text((x + 3, y + 5), frame_id, fill="black")
            image = Image.open(paths_by_condition[condition][frame_id]).convert("RGB")
            image.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
            canvas.paste(image, (x, y + label_h))
    path = directory / "input_contact_sheet.jpg"
    canvas.save(path, quality=92)
    return path


def prepare_sample(ctx, sample_dir, args, config, old_root, local_root):
    gaze_dir = old_root / "samples" / ctx.scene_key / ctx.event_id
    local_dir = local_root / "samples" / ctx.scene_key / ctx.event_id
    frozen = read_json(gaze_dir / "frozen_upstream.json")
    gaze_input = read_json(gaze_dir / "gaze_records.json")
    old_inputs = read_json(gaze_dir / "prepared_inputs.json")
    frame_manifest = read_json(local_dir / "frame_manifest.json")
    referential = read_json(local_dir / "referential.json")
    mention_id = config["mention_id"]
    mentions = {row["mention_id"]: row for row in referential.get("mentions", [])}
    mention = mentions.get(mention_id)
    if not mention:
        raise ValueError(f"Frozen local Referential has no {mention_id}")
    candidate_ids = set(mention.get("candidate_ids") or [])
    if len(candidate_ids) < 2:
        raise ValueError(f"{ctx.scene_key}/{ctx.event_id}/{mention_id} lacks a multi-candidate closed set")
    if ctx.provenance.get("ground_truth_loaded") is not False:
        raise ValueError("Probe context must be loaded with load_ground_truth=False")

    frames_all = frame_manifest["frames"]
    metadata = []
    images = {condition: {} for condition in config["conditions"]}
    expected_centers = {}
    stats_by_frame = {}
    raw_records = gaze_input
    alignment = (frozen.get("plan", {}).get("alignment") or {})
    clock_available = not bool(alignment.get("audio_timing_rejected"))
    offset = frozen.get("gaze_clock_offset_seconds") if clock_available else None
    selected, selection_audit = select_probe_frames(
        frames_all, gaze_input, offset, config['source_frame_count'], config)
    gaze_cfg = {**DEFAULTS, **config.get("gaze", {})}
    output_sample = sample_dir / ctx.scene_key / ctx.event_id
    input_dir = output_sample / "inputs"

    for index, row in enumerate(selected):
        frame_id = row["frame_id"]
        source = Path(row["path"])
        if not source.is_absolute():
            source = (ROOT / source).resolve()
        rgb = Image.open(source).convert("RGB")
        video_time = float(row["time_seconds"])
        telemetry_time = video_time - float(offset) if offset is not None else None
        heat, stats, projected = build_heatmap(
            ctx, raw_records,
            frame_time_video=video_time,
            frame_time_telemetry=telemetry_time,
            event_window=[float(ctx.api_row["t_start"]), float(ctx.api_row["t_end"])],
            clock_available=clock_available,
            image_size=rgb.size,
            strategy=config["gaze_strategy"],
            config=gaze_cfg)
        frame_dir = input_dir / frame_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(frame_dir / "attention_map.npz", H_unnormalized=heat)
        save(frame_dir / "projection.json", {"stats": stats, "projected_samples": projected})
        rendered = {"R": (rgb.copy(), {"status": "raw_rgb_control"})}
        for condition in ("H_local", "F_local"):
            rendered[condition] = render_presentation(rgb, heat, condition, gaze_cfg)
        per_frame_centers = []
        for condition, (image, render_stats) in rendered.items():
            path = frame_dir / f"{condition}.png"
            image.save(path)
            pixel_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            raw_array = np.asarray(rgb, dtype=np.uint8)
            diff = np.any(pixel_array != raw_array, axis=2)
            images[condition][frame_id] = str(path)
            if condition != "R":
                center = weighted_center(heat)
                if center is not None:
                    per_frame_centers.append(center)
            stats_by_frame.setdefault(frame_id, {})[condition] = {
                "path": str(path), "file_sha256": sha_file(path),
                "decoded_rgb_sha256": rgb_pixel_sha(image),
                "render": render_stats,
                "changed_pixel_fraction_vs_R": float(diff.mean()),
                "mean_absolute_rgb_delta_vs_R": float(np.abs(pixel_array.astype(np.int16) - raw_array.astype(np.int16)).mean()),
            }
        metadata.append({"frame_id": frame_id, "view_id": frame_id,
                         "source_frame_id": row.get("source_frame_id"),
                         "source_frame_index": row.get("source_frame_index"),
                         "time_seconds": video_time,
                         "timestamp_source": row.get("timestamp_source"),
                         "width": rgb.width, "height": rgb.height,
                         "raw_file_sha256": sha_file(source),
                         "raw_decoded_rgb_sha256": rgb_pixel_sha(rgb),
                         "selection_reason": selection_audit['policy']})
        stats_by_frame[frame_id]["gaze"] = {"projection": stats,
                                               "expected_attention_center_uv": weighted_center(heat)}
        rgb.close()

    temporal_groups = qwen_temporal_groups(metadata)
    for group in temporal_groups:
        member_centers = [stats_by_frame[fid]["gaze"]["expected_attention_center_uv"]
                          for fid in group["frame_ids"]]
        member_centers = [point for point in member_centers if point]
        expected_centers[group["view_id"]] = (
            [float(np.mean([p[0] for p in member_centers])),
             float(np.mean([p[1] for p in member_centers]))]
            if member_centers else None)
    sheet = make_contact_sheet(output_sample, [f["frame_id"] for f in metadata],
                               {condition: images[condition] for condition in ('R', 'H_local', 'F_local')})
    freeze = {
        "sample": {"scene_key": ctx.scene_key, "event_id": ctx.event_id,
                   "instruction": ctx.text, "mention_id": mention_id,
                   "mention": mention, "allowed_candidate_ids": sorted(candidate_ids)},
        "referential_source": str(local_dir / "referential.json"),
        "referential_sha256": sha_file(local_dir / "referential.json"),
        "source_video": frame_manifest["video_path"],
        "source_video_sha256": frame_manifest["video_sha256"],
        "source_frame_count_in_full_local_run": len(frames_all),
        "source_frame_count_sent_to_probe": len(metadata),
        "keyframe_selection_audit": selection_audit,
        "frames": metadata, "temporal_groups": temporal_groups,
        "expected_attention_center_by_group_uv": expected_centers,
        "gaze_records_source": str(gaze_dir / "gaze_records.json"),
        "gaze_records_sha256": sha_file(gaze_dir / "gaze_records.json"),
        "gaze_record_count": len(gaze_input),
        "clock_available": clock_available,
        "clock_rejection": alignment.get("clock", {}).get("audio_timing_rejected"),
        "telemetry_to_video_offset_seconds": offset,
        "gaze_strategy": config["gaze_strategy"],
        "coordinate_contract": "existing aspect-aware vertical-FOV projection; frustum-only, visibility unknown",
        "attention_maps": stats_by_frame,
        "input_contact_sheet": str(sheet),
        "ground_truth_loaded": False,
        "ground_truth_used_for_selection_or_prediction": False,
        "upstream_replayed": "local v4 scene-isolated Referential and source video frames; no new upstream model call",
        "offline_full_interaction": True,
    }
    save(output_sample / "frozen_input_audit.json", freeze)
    return referential, mention, metadata, temporal_groups, images, expected_centers, freeze


def run(args):
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    config = read_json(config_path)
    old_root = (ROOT / config["gaze_source_run"]).resolve()
    local_root = (ROOT / config["local_visual_source_run"]).resolve()
    source_manifest = read_json(local_root / "manifest.json")
    requested = {tuple(key) for key in config["sample_keys"]}
    items = [row for row in source_manifest["samples"]
             if (row["scene_key"], row["event_id"]) in requested]
    found = {(row["scene_key"], row["event_id"]) for row in items}
    if found != requested:
        raise ValueError(f"Sample keys missing from frozen source manifest: {sorted(requested-found)}")
    items.sort(key=lambda row: config["sample_keys"].index([row["scene_key"], row["event_id"]]))
    if args.limit:
        items = items[:args.limit]
    manifest = {"seed": 20261022, "selection": "pre-fixed sample keys in config; no GT or prediction filtering",
                "samples": items}
    contexts = load_contexts(manifest, load_ground_truth=False)
    context_by_key = {(ctx.scene_key, ctx.event_id): ctx for ctx in contexts}

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    save(out / "config.json", {**config, "config_path": str(config_path),
         "model": str(Path(args.model).resolve()), "backend_namespace": args.namespace,
         "cache_dir": str(Path(args.cache_dir).resolve()), "local_backend": True,
         "frame_selection": "8 uniformly spaced existing actual PTS frames, same IDs in every condition",
         "visual_processor_temporal_patch_size": 2,
         "prompt_comparison": "legacy soft cue wording vs paper-inspired stepwise visual cue trace; same output schema",
         "visual_tokens_are_processor_audited": True})
    save(out / "manifest.json", manifest)
    backend = QwenBackend(args.cache_dir, args.model, namespace=args.namespace,
                          local_only=True, video_max_pixels=args.video_max_pixels,
                          trace_visual_inputs=args.trace_visual_inputs)
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(20261022)

    results, failures = [], []
    for index, item in enumerate(items, 1):
        key = (item["scene_key"], item["event_id"])
        ctx = context_by_key[key]
        print(f"[{index}/{len(items)}] {key[0]}/{key[1]} prepare 8-frame R/H/F", flush=True)
        sample_dir = out / "samples"
        try:
            (referential, mention, metadata, groups, image_paths,
             expected_centers, freeze) = prepare_sample(ctx, sample_dir, args, config, old_root, local_root)
            prompt_manifest = {"qwen_temporal_groups": groups}
            base = visual_prompt(ctx, referential, mention, prompt_manifest)
            candidates = set(mention.get("candidate_ids") or [])
            predictions = {}
            for arm in config["prompt_arms"]:
                prompt = cue_prompt(base, arm, groups)
                for condition in config["conditions"]:
                    stage = f"attention_probe_{arm}_{condition}"
                    cache_key = f"{ctx.scene_key}/{ctx.event_id}/{config['mention_id']}"
                    started = time.monotonic()
                    parsed = backend.ask_video(
                        cache_key, stage, prompt,
                        [image_paths[condition][row["frame_id"]] for row in metadata],
                        metadata, max_new_tokens=2048,
                        video_path=ctx.video_path,
                        video_time_range=[metadata[0]["time_seconds"], metadata[-1]["time_seconds"]],
                        video_max_pixels=args.video_max_pixels)
                    request_path = Path(backend.requests[-1])
                    request = read_json(request_path.with_suffix(".request.json")) if request_path.with_suffix(".request.json").exists() else read_json(request_path)
                    response = read_json(request_path)
                    pred = prediction_summary(parsed, candidates)
                    pred.update({"condition": condition, "prompt_arm": arm,
                                 "raw_response": response.get("response"),
                                 "parsed_response": parsed,
                                 "request_path": str(request_path),
                                 "request_manifest_path": str(request_path.with_suffix(".request.json")),
                                 "processor_audit": response.get("processor_audit"),
                                 "actual_input_frames": response.get("images"),
                                 "local_call_elapsed_seconds": response.get("elapsed_seconds"),
                                 "wall_seconds_this_invocation": time.monotonic()-started,
                                 "requested_video_time_range": request.get("video_time_range")})
                    trace_by_id = {row.get("view_id"): row for row in pred["attention_trace"]
                                   if isinstance(row, dict)}
                    center_errors = []
                    for group in groups:
                        trace = trace_by_id.get(group["view_id"], {})
                        expected = expected_centers.get(group["view_id"])
                        observed = model_center(trace)
                        error = (float(math.dist(expected, observed))
                                 if expected is not None and observed is not None else None)
                        if error is not None:
                            center_errors.append(error)
                        trace["expected_gaze_history_center_uv_for_audit_only"] = expected
                        trace["center_error_uv_distance"] = error
                    pred["attention_trace_center_error_mean"] = (
                        float(np.mean(center_errors)) if center_errors else None)
                    predictions[f"{arm}:{condition}"] = pred
                    save(out / "samples" / ctx.scene_key / ctx.event_id / "predictions" /
                         f"{arm}_{condition}.json", pred)
                    print(f"  {arm}/{condition}: positive={pred['positive_candidate_ids']} "
                          f"trace_rows={len(pred['attention_trace'])}", flush=True)
            for arm in config["prompt_arms"]:
                raw = set(predictions[f"{arm}:R"]["positive_candidate_ids"])
                for condition in ("H_local", "F_local"):
                    pred = predictions[f"{arm}:{condition}"]
                    pred["candidate_set_changed_vs_R_same_prompt"] = (
                        set(pred["positive_candidate_ids"]) != raw)
                    save(out / "samples" / ctx.scene_key / ctx.event_id / "predictions" /
                         f"{arm}_{condition}.json", pred)
            failure_path = out / "samples" / ctx.scene_key / ctx.event_id / "failure.json"
            if failure_path.exists():
                recovery_path = out / "samples" / ctx.scene_key / ctx.event_id / "recovered_attempt_failures.json"
                prior = read_json(recovery_path) if recovery_path.exists() else []
                prior.append({**read_json(failure_path), "recovered_by_successful_rerun": True})
                save(recovery_path, prior)
                failure_path.unlink()
            results.append({"scene_key": ctx.scene_key, "event_id": ctx.event_id,
                            "mention_id": config["mention_id"],
                            "source_frames": len(metadata), "temporal_groups": len(groups),
                            "clock_available": freeze["clock_available"],
                            "conditions_completed": len(predictions),
                            "predictions": predictions})
        except Exception as exc:
            failure = {"scene_key": ctx.scene_key, "event_id": ctx.event_id,
                       "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            save(out / "samples" / ctx.scene_key / ctx.event_id / "failure.json", failure)
            print(f"  FAILED: {failure['error']}", flush=True)
    save(out / "summary.json", {
        "status": "COMPLETE" if not failures else "COMPLETE_WITH_FAILURES",
        "planned_events": len(items), "completed_events": len(results),
        "failed_events": len(failures), "conditions_per_event": len(config["conditions"])*len(config["prompt_arms"]),
        "expected_model_calls": len(items)*len(config["conditions"])*len(config["prompt_arms"]),
        "source_frame_count_per_call": config["source_frame_count"],
        "temporal_groups_per_call": config["source_frame_count"]//2,
        "failures": failures, "events": results,
        "call_audit": backend.call_records,
        "ground_truth_loaded": False, "coordinator_run": False,
        "evaluation_scope": "cue perception/local Visual evidence only; no mention accuracy or final-set metrics",
    })
    print(json.dumps({"completed": len(results), "failed": len(failures),
                      "model_calls": len(backend.call_records)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_attention_guidance_probe_20261022.json")
    parser.add_argument("--output-dir", default="runs/local_attention_guidance_probe_20261022")
    parser.add_argument("--cache-dir", default="runs/local_attention_guidance_probe_20261022_cache")
    parser.add_argument("--model", default=str(MODEL_DEFAULT))
    parser.add_argument("--namespace", default="local_attention_guidance_probe_qwen8b_20261022")
    parser.add_argument("--video-max-pixels", type=int, default=8_388_608)
    parser.add_argument("--trace-visual-inputs", action="store_true",
                        help="Hash processor tensors and verify the exact tensor reaches Qwen vision encoder.")
    parser.add_argument("--limit", type=int, help="Run the first N fixed events for a smoke check")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
