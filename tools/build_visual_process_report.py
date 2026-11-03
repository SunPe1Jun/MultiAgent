"""Build a read-only audit bundle from existing visual experiment artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import statistics

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
ORG = RUNS / "local_visual_organization_guidance_probe_20261022_pilot2"
SENS = RUNS / "heatmap_pixel_sensitivity_20261022_scene1_163_V051"
REMOTE = RUNS / "visual_evidence_8b_20261021"
ORDER = ("C0", "A-H", "B-H", "A-F", "B-F")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def media_check(rows, trace):
    restored = trace.get("restored_frame_paths", [])
    checks = []
    for index, item in enumerate(rows):
        path = resolve(item["path"])
        record = {"index": index + 1, "path": str(path), "sha256": sha(path),
                  "recorded_sha256": item.get("sha256"),
                  "file_hash_matches": sha(path) == item.get("sha256")}
        if index < len(restored):
            target = resolve(restored[index])
            a = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
            b = np.asarray(Image.open(target).convert("RGB"), dtype=np.int16)
            record.update(restored_path=str(target), restored_shape=list(b.shape),
                          same_shape=a.shape == b.shape)
            if a.shape == b.shape:
                delta = np.abs(a - b)
                record.update(restored_max_rgb_delta=int(delta.max()),
                              restored_mean_rgb_delta=float(delta.mean()),
                              restored_pixels_exact=bool(np.array_equal(a, b)))
        checks.append(record)
    return checks


def tensor_check(trace, enabled):
    if not enabled:
        return {"status": "not_rehashed", "recorded_verified": trace.get("input_consumption_verified")}
    import torch
    from MultiAgent.visual_input_trace import tensor_record
    tensors = torch.load(resolve(trace["processor_inputs_path"]), map_location="cpu", weights_only=True)
    records = {key: tensor_record(value) for key, value in tensors.items()}
    matches = {key: records[key] == value for key, value in trace["processor_tensors"].items()}
    key = "pixel_values_videos" if trace["kind"] == "video" else "pixel_values"
    dtype_name = trace["expected_vision_tensor"]["dtype"].split(".")[-1]
    cast = tensor_record(tensors[key].to(dtype=getattr(torch, dtype_name)))
    calls = trace.get("vision_forward_calls", [])
    return {"status": "rehashed_from_saved_processor_tensors", "processor_records_match": matches,
            "cast_pixel_hash_matches_expected": cast == trace["expected_vision_tensor"],
            "cast_pixel_hash_matches_logged_vision_calls": bool(calls) and all(
                row["input"] == cast for row in calls), "vision_call_count": len(calls),
            "recorded_verified": trace.get("input_consumption_verified")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/visual_process_report_20261022")
    parser.add_argument("--verify-tensors", action="store_true")
    args = parser.parse_args()
    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompts").mkdir(exist_ok=True)
    (out / "raw_responses").mkdir(exist_ok=True)
    bundle = {"date": "2026-10-22", "protocol": "read-only retrospective audit; no model calls",
              "new_inference_calls": 0, "samples": [], "calls": [], "historical_metrics": {}}
    inputs, frame_rows, gallery = [], [], []
    sources = set()
    organization = read(ORG / "summary.json")
    for sample in organization["samples"]:
        scene, event = sample["scene_key"], sample["event_id"]
        folder = ORG / "samples" / scene / event
        frozen_path = folder / "frozen_input_audit.json"
        frozen = read(frozen_path)
        spatial = folder / "inputs" / "B1_directional_panorama"
        geometry = read(spatial / "geometry.json")
        raw_gaze = read(resolve(frozen["gaze_records_source"]))
        frame_manifest = read(RUNS / "local_dense_video_visual_8b_20261022_pilot100_v4" /
                              "samples" / scene / event / "frame_manifest.json")
        old = read(REMOTE / "pilot100_relaxed_clock" / "samples" / scene / event / "frozen_upstream.json")
        maps = np.load(spatial / "attention_and_provenance.npz")
        coverage = maps["coverage_count"] > 0
        latitude = (.5 - (np.arange(coverage.shape[0]) + .5) / coverage.shape[0]) * np.pi
        spherical_coverage = float((coverage * np.cos(latitude)[:, None]).sum() /
                                   (coverage.shape[1] * np.cos(latitude).sum()))
        gaze_times = [row["telemetry_time_seconds"] for row in raw_gaze]
        intervals = np.diff(gaze_times)
        offset = frozen["telemetry_to_video_offset_seconds"]
        selected_start, selected_end = frozen["frames"][0]["time_seconds"], frozen["frames"][-1]["time_seconds"]
        source_checks = {}
        for field, expected in [("source_video", "source_video_sha256"),
                                ("referential_source", "referential_sha256"),
                                ("gaze_records_source", "gaze_records_sha256")]:
            p = resolve(frozen[field])
            source_checks[field] = {"path": str(p), "matches": sha(p) == frozen[expected]}
            sources.add(p)
        a_indices = set()
        for frame in frozen["frames"]:
            fid = frame["frame_id"]
            projection = read(folder / "inputs" / fid / "projection.json")
            a_indices.update(row["source_index"] for row in projection["projected_samples"]
                             if row.get("status") == "projected_frustum_only")
            stats = projection["stats"]
            attention = frozen["attention_maps"][fid]
            frame_rows.append({"scene_key": scene, "event_id": event, **frame,
                               "telemetry_time_seconds": frame["time_seconds"] - offset,
                               "valid_gaze_count": stats["valid_projected_count"],
                               "valid_gaze_duration_seconds": stats["valid_evidence_duration_seconds"],
                               "heat_max": stats["max_unnormalized_heat"],
                               "heat_sum": stats["unnormalized_heat_sum"],
                               "heat_kernel_sigma_pixels": stats["kernel_sigma_pixels_at_model_input"],
                               "overlay_mean_absolute_rgb_delta": attention["H_local"]["mean_absolute_rgb_delta_vs_R"],
                               "foveation_mean_absolute_rgb_delta": attention["F_local"]["mean_absolute_rgb_delta_vs_R"]})
        b_gaze = read(spatial / "gaze_time_index.json")
        b_indices = {row["source_index"] for row in b_gaze}
        sample_audit = {"scene_key": scene, "event_id": event, "instruction": frozen["sample"]["instruction"],
                        "mention_id": sample["mention_id"], "candidate_ids": sample["candidate_ids"],
                        "source_checks": source_checks,
                        "selection": {k: v for k, v in frozen["keyframe_selection_audit"].items() if k != "frame_audit"},
                        "video": {"duration_seconds": frame_manifest["duration_seconds"],
                                  "source_stream": frame_manifest["source_stream"],
                                  "selected_time_range_seconds": [selected_start, selected_end],
                                  "telemetry_equivalent_time_range_seconds": [selected_start - offset, selected_end - offset]},
                        "clock": old["plan"]["alignment"],
                        "gaze": {"total_records": len(raw_gaze), "median_interval_seconds": statistics.median(intervals),
                                 "estimated_hz": 1 / statistics.median(intervals),
                                 "last_telemetry_seconds": gaze_times[-1],
                                 "records_after_selected_video_end": sum(t > selected_end - offset for t in gaze_times),
                                 "a_unique_projectable_source_indices": sorted(a_indices),
                                 "b1_unique_source_indices": sorted(b_indices),
                                 "a_b1_source_sets_equal": a_indices == b_indices,
                                 "b1_rgb_gaze_max_association_delta_seconds": max(row["rgb_gaze_delta_seconds"] for row in b_gaze)},
                        "B1": {k: geometry[k] for k in ["status", "failure_reasons", "coverage_fraction",
                            "camera_max_displacement_world_units", "gaze_unique_source_count",
                            "gaze_valid_evidence_duration_seconds", "temporal_layer_range_seconds"]},
                        "B1_solid_angle_coverage_fraction": spherical_coverage,
                        "B1_registration": [{k: r.get(k) for k in ["source_view_ids", "status", "accepted_matches", "median_px", "p95_px"]}
                                            for r in geometry["registration_residuals"]],
                        "model_B_projection_type": sample["projection_type"]}
        bundle["samples"].append(sample_audit)
        gallery_sample = {"key": f"{scene}/{event}", "instruction": frozen["sample"]["instruction"], "conditions": {}}
        for condition in ORDER:
            prediction_path = folder / "predictions" / (condition + ".json")
            prediction = read(prediction_path)
            request_path = resolve(prediction["request_manifest_path"])
            request = read(request_path)
            response_path = resolve(prediction["request_path"])
            response = read(response_path)
            audit = prediction["processor_audit"]
            trace = prediction["visual_tensor_trace"]
            trace_path = resolve(trace["trace_path"])
            checks = media_check(request["images"], trace)
            prompt = request["prompt"]
            name = f"{scene}_{event}_{condition}"
            (out / "prompts" / (name + ".txt")).write_text(prompt, encoding="utf-8")
            (out / "raw_responses" / (name + ".txt")).write_text(response["response"], encoding="utf-8")
            call = {"scene_key": scene, "event_id": event, "mention_id": sample["mention_id"],
                    "condition": condition, "backend": prediction["backend"],
                    "source_frames": [f["source_frame_id"] for f in sample["source_frames"]],
                    "image_count": len(request["images"]),
                    "vision_tokens": audit.get("processor_image_tokens_estimate", audit.get("processor_video_tokens_estimate")),
                    "input_token_count": audit["input_token_count"],
                    "grid_thw": audit.get("image_grid_thw", audit.get("video_grid_thw")),
                    "elapsed_seconds": prediction["elapsed_seconds"], "inference_seconds": prediction.get("inference_seconds"),
                    "cue_reported": prediction["trace_views_with_reported_cue"], "trace_rows": prediction["trace_rows_returned"],
                    "positive_candidate_ids": prediction["positive_candidate_ids"], "identity_status": prediction["identity_status"],
                    "prompt_sha256_matches": hashlib.sha256(prompt.encode()).hexdigest() == prediction["prompt_sha256"],
                    "raw_response_matches_prediction": response["response"] == prediction["raw_response"],
                    "request_path": str(request_path), "response_path": str(response_path), "trace_path": str(trace_path),
                    "tensor_check": tensor_check(trace, args.verify_tensors), "media_checks": checks}
            bundle["calls"].append(call)
            sources.update([prediction_path, request_path, response_path, trace_path, frozen_path])
            for image, check in zip(request["images"], checks):
                inputs.append({"scene_key": scene, "event_id": event, "condition": condition, **image,
                               "file_hash_matches": check["file_hash_matches"],
                               "restored_pixels_exact": check.get("restored_pixels_exact"),
                               "restored_max_rgb_delta": check.get("restored_max_rgb_delta"),
                               "restored_path": check.get("restored_path")})
            gallery_sample["conditions"][condition] = {
                "budget": {k: call[k] for k in ["source_frames", "image_count", "vision_tokens", "elapsed_seconds", "cue_reported", "trace_rows", "positive_candidate_ids"]},
                "images": [{"src": os.path.relpath(resolve(image["path"]), out), "sha256": image["sha256"],
                            "path": image["path"], "width": image["width"], "height": image["height"]} for image in request["images"]],
                "prompt": prompt, "raw_response": response["response"]}
        gallery_sample["conditions"]["B1_CPU_NOT_SENT"] = {
            "budget": {"status": geometry["status"], "reasons": geometry["failure_reasons"], "sent_to_model": False},
            "images": [{"src": os.path.relpath(spatial / name, out), "sha256": sha(spatial / name), "path": str(spatial / name)}
                       for name in ["RGB.png", "coverage.png", "heatmap.png", "temporal_heatmap.png", "heatmap_overlay.png", "foveation.png"]],
            "prompt": "未作为本轮模型输入。两条样本都显式回退 B3。", "raw_response": "NOT_RUN"}
        gallery.append(gallery_sample)
    for pilot in ["pilot20_corrected", "pilot100_relaxed_clock"]:
        summary_path = REMOTE / pilot / "summary.json"
        summary = read(summary_path)
        rows = []
        for group, result in summary["groups"].items():
            rows.append({"group": group, **result["metrics"]})
        bundle["historical_metrics"][pilot] = rows
        write_csv(out / (pilot + "_metrics.csv"), rows)
        sources.add(summary_path)
    historical_failures = read(REMOTE / "pilot100_relaxed_clock" / "failure_analysis.json")
    bundle["historical_100_error_counts"] = {
        group: {key: sum(bool(row.get(key)) for row in historical_failures if row["group"] == group)
                for key in ["missing", "extra", "same_class_instance_confusions", "coordinator_abstained_pipeline_fallback"]}
        for group in ["R", "H_inst", "H_local", "F_inst", "F_local"]}
    sensitivity = read(SENS / "summary.json")
    bundle["sensitivity"] = {}
    for kind, result in sensitivity["results"].items():
        trace = result["visual_tensor_trace"]
        audit = read(SENS / "input_audit.json")["groups"][kind]
        response_path = resolve(result["request_path"])
        response = read(response_path)
        images = response["images"]
        checks = media_check(images, trace)
        bundle["sensitivity"][kind] = {"parse_status": result["parse_status"],
            "raw_response": result["raw_response"], "tensor_check": tensor_check(trace, args.verify_tensors),
            "media_checks": checks, "vision_tokens": result["processor_audit"]["processor_image_tokens_estimate"],
            "elapsed_seconds": result["elapsed_seconds"],
            "coordinate_range_explicit_in_prompt": "[0,1]" in audit["prompt"] or "normalized" in audit["prompt"],
            "nonzero_maps_detected": sum(row["model_response"].get("detected") is True for row in result["checks"] if row["expected"]["variant"] != "zero"),
            "nonzero_map_count": sum(row["expected"]["variant"] != "zero" for row in result["checks"]),
            "quadrant_correct_including_zero": sum(row["quadrant_correct"] for row in result["checks"])}
        (out / "prompts" / ("sensitivity_" + kind + ".txt")).write_text(audit["prompt"], encoding="utf-8")
        (out / "raw_responses" / ("sensitivity_" + kind + ".txt")).write_text(result["raw_response"], encoding="utf-8")
        sources.update([SENS / "summary.json", SENS / "input_audit.json", response_path, resolve(trace["trace_path"])])
    bundle["checks"] = {"organization_calls": len(bundle["calls"]),
        "image_file_checks": len(inputs), "all_organization_file_hashes_match": all(row["file_hash_matches"] for row in inputs),
        "all_organization_restored_pixels_exact": all(row["restored_pixels_exact"] for row in inputs),
        "all_organization_prompt_hashes_match": all(row["prompt_sha256_matches"] for row in bundle["calls"]),
        "all_organization_raw_responses_match": all(row["raw_response_matches_prediction"] for row in bundle["calls"])}
    save(out / "audit.json", bundle)
    write_csv(out / "input_inventory.csv", inputs)
    write_csv(out / "frames.csv", frame_rows)
    write_csv(out / "call_inventory.csv", [{k: v for k, v in row.items() if k not in ["media_checks", "tensor_check"]} for row in bundle["calls"]])
    for name in ["visual_organization_audit.py", "local_attention_probe.py", "spatial_evidence.py", "gaze_evidence.py",
                 "local_video_visual.py", "visual_input_trace.py", "qwen_backend.py", "vision_reasoning.py", "run_v3.py", "context.py", "heatmap_sensitivity.py"]:
        sources.add(ROOT / "MultiAgent" / name)
    sources.update([ROOT / "configs/visual_organization_guidance_probe_20261022.json", ORG / "config.json", ORG / "manifest.json", Path(__file__)])
    save(out / "source_hashes.json", [{"path": str(path.resolve()), "sha256": sha(path)} for path in sorted(sources)])
    data = json.dumps(gallery, ensure_ascii=False).replace("</", "<\\/")
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>视觉全过程报告：实际输入图册</title><style>
body{font:15px/1.65 system-ui,sans-serif;margin:24px;color:#202622;background:#fff}main{max-width:1500px;margin:auto}
h1{font-size:24px}label{display:inline-block;margin:0 18px 12px 0}select,button{font:inherit;padding:5px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f3;padding:14px;max-height:500px;overflow:auto}
#media{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}figure{margin:0;border-bottom:1px solid #cfd5d0;padding-bottom:12px}img{width:100%;height:auto;display:block}figcaption{overflow-wrap:anywhere;font-size:12px}summary{cursor:pointer;font-weight:600}a{color:#176b45}
</style><main><h1>视觉全过程报告：实际模型输入与原始响应</h1>
<p>来源：2026-10-22 已保存的本地 8B 运行。此图册读取历史文件，没有新增推理。C0/A-F 为视频帧；A-H 是交替热图/RGB 静态图；B-H/B-F 是 B3 回退图集。B1_CPU_NOT_SENT 是未交给模型的全景尝试。</p>
<label>样本 <select id="sample"></select></label><label>条件 <select id="condition"></select></label><p id="instruction"></p><pre id="budget"></pre><section id="media"></section>
<details><summary>实际提示词</summary><pre id="prompt"></pre></details><details><summary>模型原始响应</summary><pre id="response"></pre></details>
<p><a href="../../docs/VISUAL_END_TO_END_REPORT_20261022_zh.md">完整报告</a> · <a href="audit.json">结构化核查</a> · <a href="input_inventory.csv">实际图片清单</a> · <a href="frames.csv">源帧与 gaze 统计</a></p></main>
<script>const samples=__DATA__;const sample=document.getElementById('sample'),condition=document.getElementById('condition');
samples.forEach((s,i)=>sample.add(new Option(s.key,i)));['C0','A-H','B-H','A-F','B-F','B1_CPU_NOT_SENT'].forEach(c=>condition.add(new Option(c,c)));
function render(){const s=samples[Number(sample.value)],c=s.conditions[condition.value];document.getElementById('instruction').textContent=s.instruction;document.getElementById('budget').textContent=JSON.stringify(c.budget,null,2);document.getElementById('prompt').textContent=c.prompt;document.getElementById('response').textContent=c.raw_response;const media=document.getElementById('media');media.replaceChildren();c.images.forEach((item,i)=>{const f=document.createElement('figure'),a=document.createElement('a'),im=document.createElement('img'),cap=document.createElement('figcaption');a.href=item.src;a.target='_blank';im.src=item.src;im.loading='lazy';im.alt=condition.value+' image '+(i+1);a.append(im);cap.textContent=(i+1)+'. '+item.path+' | sha256: '+item.sha256;f.append(a,cap);media.append(f);});}sample.onchange=render;condition.onchange=render;render();</script></html>'''.replace("__DATA__", data)
    (out / "visual_gallery.html").write_text(page, encoding="utf-8")
    print(json.dumps({"output": str(out), "checks": bundle["checks"],
                      "tensor_verification": args.verify_tensors,
                      "sensitivity": {key: {k:v for k,v in value.items() if k not in ["raw_response", "media_checks", "tensor_check"]} for key,value in bundle["sensitivity"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
