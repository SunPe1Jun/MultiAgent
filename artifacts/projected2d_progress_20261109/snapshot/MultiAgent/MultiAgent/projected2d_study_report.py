"""Render saved Projected-2D predictions; annotations are read only here."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .projected2d_attention_study import GROUPS, save_json, sha_file, write_csv


def image_delta(reference: str, condition: str):
    """Measure differences in images restored from the actual processor tensors."""
    with Image.open(reference) as image:
        base = np.asarray(image.convert("RGB"), dtype=np.int16)
    with Image.open(condition) as image:
        other = np.asarray(image.convert("RGB"), dtype=np.int16)
    if base.shape != other.shape:
        raise ValueError(f"Paired processor image sizes differ: {base.shape}, {other.shape}")
    difference = np.abs(base - other)
    return {"width": int(base.shape[1]), "height": int(base.shape[0]),
            "mean_absolute_rgb_difference_0_255": float(difference.mean()),
            "fraction_pixels_changed_more_than_8": float((difference.max(axis=2) > 8).mean())}


def export_model_input_provenance(saved, samples, output: Path, model_path: str):
    """Decode captured input IDs, rather than reconstructing an assumed prompt."""
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    rows = []
    for index, sample in enumerate(samples):
        item_dir = output / "actual_model_inputs" / sample["scene_key"] / str(sample["row_index"])
        event = {"scene": sample["scene_key"], "row_index": sample["row_index"],
                 "conditions": {}, "prompt_ids_identical": None, "image_grids_identical": None}
        for group in GROUPS:
            result = saved[group][index]
            trace = result.get("visual_tensor_trace") or {}
            if result["status"] not in ("COMPLETE", "PARSE_WITH_WARNINGS"):
                event["conditions"][group] = {"status": result["status"], "captured": False}
                continue
            item_dir.mkdir(parents=True, exist_ok=True)
            tensors = torch.load(trace["processor_inputs_path"], map_location="cpu", weights_only=True)
            decoded = tokenizer.decode(tensors["input_ids"][0].tolist(), skip_special_tokens=False)
            prompt_file = item_dir / f"{group}_decoded_actual_input.txt"
            prompt_file.write_text(decoded, encoding="utf-8")
            records = trace["processor_tensors"]
            condition = {
                "status": result["status"], "captured": True,
                "decoded_actual_prompt_path": str(prompt_file),
                "input_ids_sha256": records["input_ids"]["sha256"],
                "image_grid_sha256": records["image_grid_thw"]["sha256"],
                "pixel_values_sha256": records["pixel_values"]["sha256"],
                "input_consumption_verified": trace["input_consumption_verified"],
                "input_file_hashes_verified": all(sha_file(Path(row["path"])) == row["sha256"]
                                                  for row in result["input_images"]),
                "processor_restored_images": trace["restored_frame_paths"],
            }
            event["conditions"][group] = condition
            del tensors
        completed = [row for row in event["conditions"].values() if row.get("captured")]
        if len(completed) == len(GROUPS):
            event["prompt_ids_identical"] = len({row["input_ids_sha256"] for row in completed}) == 1
            event["image_grids_identical"] = len({row["image_grid_sha256"] for row in completed}) == 1
            base = event["conditions"]["RGB"]
            for group in GROUPS[1:]:
                condition = event["conditions"][group]
                condition["pixels_differ_from_RGB"] = condition["pixel_values_sha256"] != base["pixel_values_sha256"]
                condition["processor_image_deltas_from_RGB"] = [
                    image_delta(a, b) for a, b in zip(base["processor_restored_images"],
                                                     condition["processor_restored_images"], strict=True)]
        rows.append(event)
    save_json(output / "actual_model_inputs" / "provenance.json", {
        "source": "captured input_ids and pixel_values from the actual generate call",
        "interpretation": "Verifies delivered pixels and explicit cue instructions; does not prove attention use or improved reasoning.",
        "events": rows})
    return rows


def export_attention_strength(root: Path, samples, output: Path, config):
    rows = []
    gaze = config["gaze"]
    for sample in samples:
        event_dir = root / "inputs" / sample["scene_key"] / str(sample["row_index"])
        audit = json.loads((event_dir / "input_audit.json").read_text())
        for panel in audit["panels"]:
            with np.load(event_dir / panel["frame_id"] / "attention_audit.npz") as maps:
                for name, key in (("current", "current_H_unnormalized"), ("history", "history_H_unnormalized")):
                    heat = maps[key]
                    focus = np.clip(1 - np.exp(-np.maximum(heat, 0) * gaze["display_scale"]), 0, 1)
                    alpha = focus * gaze["overlay_max_alpha"]
                    rows.append({"scene": sample["scene_key"], "row_index": sample["row_index"],
                        "frame_id": panel["frame_id"], "map": name,
                        "valid_projected_observations": panel[f"{name}_gaze_count"],
                        "unnormalized_heat_max": float(heat.max()),
                        "unnormalized_heat_sum": float(heat.sum()),
                        "effective_max_overlay_alpha": float(alpha.max()),
                        "fraction_pixels_overlay_alpha_above_0.01": float((alpha > .01).mean()),
                        "fraction_pixels_overlay_alpha_above_0.05": float((alpha > .05).mean()),
                        "fraction_pixels_focus_above_0.5": float((focus > .5).mean())})
    write_csv(output / "attention_strength.csv", rows, list(rows[0]))
    return rows


def report(root: Path, split: str):
    directory = root / split
    summary = json.loads((directory / "summary.json").read_text())
    samples = json.loads((root / "fixed_samples.json").read_text())["samples"]
    samples = samples[:summary["sample_count"]]
    with (directory / "manifest_all.csv").open(encoding="utf-8-sig") as stream:
        manifest = list(csv.DictReader(stream))
    config = json.loads((root / "config.json").read_text())
    parity_path = root / "original_annotation_parity.json"
    parity = json.loads(parity_path.read_text()) if parity_path.exists() else None
    parity_description = (f"The prepared subset has {parity['compared_rows']} rows and "
                          f"{len(parity['mismatches'])} differences." if parity else
                          "The original annotation parity audit has not been generated.")
    groups = []
    saved = {}
    event_metrics = []
    for group in GROUPS:
        items = [json.loads((directory / "samples" / item["scene_key"] /
                            str(item["row_index"]) / f"{group}.json").read_text())
                 for item in samples]
        saved[group] = items
        successes = [x for x in items if x["status"] in ("COMPLETE", "PARSE_WITH_WARNINGS")]
        traces = [x.get("visual_tensor_trace") or {} for x in successes]
        requests = [json.loads(Path(x["request_path"]).read_text()) for x in successes]
        metrics = summary["metrics"][group]
        groups.append({
            "condition": group, "events": len(items),
            "successful_visual_calls": len(successes),
            "parse_warning_events": sum(x["status"] == "PARSE_WITH_WARNINGS" for x in items),
            "json_repaired_events": sum(bool(x.get("parsed_raw", {}).get("_parse_warning")) for x in successes),
            "upstream_or_call_failure_events": len(items) - len(successes),
            "vision_consumption_verified": sum(bool(x.get("input_consumption_verified")) for x in traces),
            "actual_input_images": sum(len(x.get("input_images", [])) for x in successes),
            "image_tokens_per_call": sorted({x["processor_audit"]["expanded_image_token_count"] for x in successes}),
            "elapsed_seconds": sum(x.get("elapsed_seconds", 0.0) for x in items),
            "recorded_inference_seconds": sum(x.get("inference_seconds", 0.0) for x in requests),
            "replayed_smoke_calls": sum(bool(x.get("replayed_from")) for x in items),
            "temporal_f1": metrics["time_f1"],
            **{f"{kind}@{k}_f1": metrics[f"{kind}_{k}_f1"]
               for kind in ("point", "joint") for k in (50, 100, 150, 200)},
        })
        with (directory / "evaluation" / group / "2d_eval_detail.csv").open(encoding="utf-8-sig") as stream:
            event_metrics.extend({"condition": group, **row} for row in csv.DictReader(stream))
    paired = []
    for index, item in enumerate(samples):
        predictions = {g: saved[g][index] for g in GROUPS}
        complete = all(x["status"] in ("COMPLETE", "PARSE_WITH_WARNINGS") for x in predictions.values())
        if complete:
            prompts = {hashlib.sha256(x["prompt"].encode()).hexdigest() for x in predictions.values()}
            token_budgets = {x["processor_audit"]["expanded_image_token_count"] for x in predictions.values()}
            assert len(prompts) == len(token_budgets) == 1
            assert all(len(x["input_images"]) == 3 for x in predictions.values())
        paired.append({"scene": item["scene_key"], "row_index": item["row_index"],
                       "paired_calls_completed": complete,
                       "status": {g: x["status"] for g, x in predictions.items()}})
    output = directory / "report"
    output.mkdir(exist_ok=True)
    source_files = [Path(config["evaluator"]),
                    Path(config["evaluator"]).parent / "build_2d_eval_manifest.py",
                    Path(config["evaluator"]).parent / "run_qwen3vl_2d_point_grounding.py",
                    Path(__file__).with_name("projected2d_attention_study.py"),
                    Path(__file__).with_name("gaze_evidence.py"),
                    Path(__file__).with_name("qwen_backend.py")]
    environment = {"python": sys.version, "executable": sys.executable,
        "source_hash_scope": "current report generation; not an immutable snapshot of code at original inference time",
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "accelerate", "bitsandbytes", "numpy", "Pillow")},
        "source_sha256": {str(path): sha_file(path) for path in source_files}}
    save_json(output / "environment_and_sources.json", environment)
    write_csv(output / "per_event_original_metrics.csv", event_metrics, list(event_metrics[0]))
    provenance = export_model_input_provenance(saved, samples, output, config["model"])
    attention_strength = export_attention_strength(root, samples, output, config)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    examples = []
    metric_map = {(row["condition"], row["scene"], int(row["row_index"])): row for row in event_metrics}
    paired_changes = []
    for index, item in enumerate(samples):
        scene, row_index = item["scene_key"], item["row_index"]
        baseline = saved["RGB"][index]
        for group in GROUPS[1:]:
            current = saved[group][index]
            signature = lambda result: [(m["mention_id"], m["status"], m["predictions"])
                                        for m in result.get("parsed", {}).get("mentions", [])]
            before = metric_map[("RGB", scene, row_index)]
            after = metric_map[(group, scene, row_index)]
            paired_changes.append({"scene": scene, "row_index": row_index, "condition": group,
                "complete_pair": paired[index]["paired_calls_completed"],
                "localization_output_changed": signature(current) != signature(baseline),
                "delta_prediction_count": int(after["prediction_count"])-int(before["prediction_count"]),
                "delta_time_tp": int(after["time_tp"])-int(before["time_tp"]),
                "delta_point_tp_100": int(after["point_tp_100"])-int(before["point_tp_100"]),
                "delta_joint_tp_100": int(after["joint_tp_100"])-int(before["joint_tp_100"])})
    write_csv(output / "paired_changes.csv", paired_changes, list(paired_changes[0]))
    eligible = [index for index, row in enumerate(paired) if row["paired_calls_completed"]]
    example_indices = []
    def change(index, group):
        sample = samples[index]
        key = (sample["scene_key"], sample["row_index"])
        return (int(metric_map[(group, *key)]["joint_tp_100"])
                - int(metric_map[("RGB", *key)]["joint_tp_100"]))
    for group, sign in (("HISTORY_HEATMAP", 1), ("HISTORY_HEATMAP", -1), ("HISTORY_FOVEATION", -1)):
        candidates = [i for i in eligible if i not in example_indices]
        if candidates:
            example_indices.append(max(candidates, key=lambda i: sign*change(i, group)))
    example_records = []
    for index, item in enumerate(samples):
        if index not in example_indices:
            continue
        scene, row_index = item["scene_key"], item["row_index"]
        audit = json.loads((root / "inputs" / scene / str(row_index) / "input_audit.json").read_text())
        width, height, label = 340, 340, 52
        event_manifest = [r for r in manifest if r['scene'] == scene and int(r['row_index']) == row_index]
        footer = 36 + 17 * max(sum(r['panel_id'] == p['frame_id'] for r in event_manifest)
                               for p in audit['panels'])
        row_height = height + label + footer
        sheet = Image.new("RGB", (width * 4, row_height * 3), "white")
        draw = ImageDraw.Draw(sheet)
        for row, panel in enumerate(audit["panels"]):
            fid = panel["frame_id"]
            for col, group in enumerate(GROUPS):
                source = panel["full_frame_conditions"][group]
                with Image.open(source) as image:
                    resized = image.convert("RGB").resize((width, height))
                x0, y0 = col * width, row * row_height + label
                sheet.paste(resized, (x0, y0))
                draw.text((x0+4, y0-label+3), f"{group} {fid}", fill="black", font=font)
                draw.text((x0+4, y0-label+21), "Cyan=PRED / Magenta=projected GT", fill="black", font=font)
                all_gt_rows = [r for r in event_manifest if r['panel_id'] == fid]
                draw.text((x0 + 4, y0 + height + 4), 'GT below: original frustum only; RGB sync UNKNOWN', fill='black', font=font)
                for k, target in enumerate(all_gt_rows):
                    valid = target['projection_valid'] == 'True'
                    draw.text((x0 + 4, y0 + height + 23 + k * 17),
                              f"{target['referent_name']}: {'in frustum / occlusion unknown' if valid else 'INVALID: outside/behind/missing pose'}",
                              fill='#8f008f' if valid else '#a02010', font=font)
                gt_rows = [r for r in all_gt_rows if r['projection_valid'] == 'True']
                for target in gt_rows:
                    x, y = x0 + float(target["gt_u_norm"])*width, y0 + float(target["gt_v_norm"])*height
                    draw.line((x-7,y,x+7,y),fill="magenta",width=2)
                    draw.line((x,y-7,x,y+7),fill="magenta",width=2)
                    draw.text((x+4,y+5),target["referent_name"],fill="magenta",font=font)
                for mention in saved[group][index]["parsed"]["mentions"]:
                    for point in mention["predictions"]:
                        if point["frame_id"] != fid:
                            continue
                        x,y=x0+point["point_uv"][0]*width,y0+point["point_uv"][1]*height
                        draw.ellipse((x-6,y-6,x+6,y+6),outline="cyan",width=2)
                        draw.text((x+7,y-14),mention["mention_id"],fill="cyan",font=font)
        target = output / f"{scene}_{row_index}_input_prediction_gt.jpg"
        sheet.save(target, quality=94)
        examples.append(str(target))
        example_records.append({"scene": scene, "row_index": row_index,
            "selection": "Largest history-heatmap improvement, largest history-heatmap decline, and largest foveation decline in Joint@100 TP; ties use fixed order; no change to experiment denominator.",
            "comparison_image": str(target),
            "panels": audit["panels"],
            "predictions": {group: saved[group][index]["parsed"] for group in GROUPS},
            "metrics": {group: metric_map[(group, scene, row_index)] for group in GROUPS},
            "projected_gt_for_posthoc_visualization_only": [r for r in manifest
                if r["scene"] == scene and int(r["row_index"]) == row_index]})
        model_sheet = Image.new("RGB", (480 * 4, (236 + label) * 3), "white")
        model_draw = ImageDraw.Draw(model_sheet)
        for col, group in enumerate(GROUPS):
            paths = saved[group][index]["visual_tensor_trace"]["restored_frame_paths"]
            for row, source in enumerate(paths):
                with Image.open(source) as image:
                    resized = image.convert("RGB").resize((480, 236))
                x0, y0 = col * 480, row * (236 + label) + label
                model_sheet.paste(resized, (x0, y0))
                model_draw.text((x0+4, y0-label+5), f"{group} P{row+1}", fill="black", font=font)
                model_draw.text((x0+4, y0-label+23), "Actual processor pixels / full + crop", fill="black", font=font)
        model_sheet.save(output / f"{scene}_{row_index}_actual_processor_inputs.jpg", quality=94)
    save_json(output / "examples.json", example_records)
    save_json(output / "audit.json", {"config": config, "groups": groups, "paired": paired,
                                      "examples": examples,
                                      "actual_input_provenance": str(output / "actual_model_inputs/provenance.json"),
                                      "annotation_scope": "report/evaluation only; fixed sample eligibility uses original annotation availability"})
    baseline = groups[0]
    lines = [f"# Projected-2D {split}", "", f"Model: Qwen3-VL-8B; quantization={config.get('quantization', 'none')}; no comparison to historical model results.", "",
             "**PROVISIONAL_GEOMETRY_NOT_CALIBRATED**: RGB/telemetry alignment is unverified. The 2026-10-23 GT-free motion audit covers the fixed 20 events and supports clock-alignment concerns; candidate residuals are lower in 16 events and higher in 4. These saved scores cannot establish heatmap/foveation efficacy. See diagnostics/REVIEW_20261023_zh.md at the experiment root.", "",
             "|Condition|Successful calls / events|Temporal F1|Point@100 F1|Joint@100 F1|Joint@200 F1|Delta Joint@100|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in groups:
        lines.append(f"|{row['condition']}|{row['successful_visual_calls']}/{row['events']}|{row['temporal_f1']:.4f}|{row['point@100_f1']:.4f}|{row['joint@100_f1']:.4f}|{row['joint@200_f1']:.4f}|{row['joint@100_f1']-baseline['joint@100_f1']:+.4f}|")
    lines += ["", "All fixed events, including upstream failures, remain in the original evaluator denominator.",
              "The original projected-anchor proxy and matching rules are unchanged. No mention-level accuracy is asserted.",
              "Inputs reuse multi_image + paired_crop; prompts use instruction and frozen extracted mentions, without annotation-derived expected_count.",
              "These are given-expression extension results, not an exact reproduction of the historical GT-count-conditioned wrapper.",
              "", "Saved examples draw projected GT only after inference. They are never inference inputs.",
              "", "## Actual inputs and outputs", "",
              "All conditions use the same instruction, frozen Referential mentions, source frames, paired-crop bounding boxes, media form, decoder settings, and expanded image-token budget.",
              "The original protocol selects three evidence panels per event. This diagnostic is not the earlier 34–100-frame video experiment.",
              "Each condition is one multi-image Visual call per event with all frozen mentions; outputs retain per-mention status, frame_id, point_uv and evidence. No Coordinator/fusion call is included in this Visual diagnostic.",
              "", "|Condition|Actual visual input|Output|", "|---|---|---|",
              "|RGB|Original source pixels; original gaze-centred paired crop|Per-mention normalized source-frame points|",
              "|CURRENT_HEATMAP|Current gaze reprojected onto the reference camera, overlay before identical paired crop|Same schema|",
              "|HISTORY_HEATMAP|Past 2 seconds of observed gazePoint reprojected using each reference camera; mild overlay|Same schema|",
              "|HISTORY_FOVEATION|Same numerical history heatmap; smooth peripheral weakening with retained RGB detail|Same schema|",
              "", "RGB already contains the original gaze-centred crop. The comparisons measure adding overlays/foveation to that existing input, not introducing gaze to a gaze-free baseline.",
              "Original evaluation is event/referent level, using projected 3D anchor points and acceptable panels. It is not independent hand-labelled per-mention pixel GT, and no mention accuracy is reported.",
              "The adapter flattens all per-mention point predictions into the original referents schema. It does not use GT to merge repeated mentions or remove false positives; the original one-referent matching may penalize repeated predictions of one physical target.",
              "", "## Full original metrics", "",
              "|Condition|Point@50|Point@100|Point@150|Point@200|Joint@50|Joint@100|Joint@150|Joint@200|",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in groups:
        values = [row[f"{kind}@{k}_f1"] for kind in ("point", "joint") for k in (50, 100, 150, 200)]
        lines.append(f"|{row['condition']}|" + "|".join(f"{value:.4f}" for value in values) + "|")
    lines += ["", "## Visual budget and recorded cost", "",
              "Each completed event uses three source frames and three paired full-frame/crop images. Crops do not add source frames.",
              "Image tokens below are counted from the actual expanded image placeholders, not total prompt tokens.",
              "Inference seconds sum the saved original calls, including exact-match replayed calls; they are not the wall time of the last resume.",
              "", "|Condition|Visual calls|Images|Image tokens / call|Inference seconds|Parse-warning events|",
              "|---|---:|---:|---|---:|---:|"]
    for row in groups:
        lines.append(f"|{row['condition']}|{row['successful_visual_calls']}|{row['actual_input_images']}|{row['image_tokens_per_call']}|{row['recorded_inference_seconds']:.1f}|{row['parse_warning_events']}|")
    lines += ["", "## Effective attention strength", "",
              "Statistics use all fixed source panels, including panels from upstream-failure events.",
              "", "|Map|Panels|Zero-map panels|Median maximum alpha|Median pixel fraction alpha > 0.01|",
              "|---|---:|---:|---:|---:|"]
    for name in ("current", "history"):
        rows = [row for row in attention_strength if row['map'] == name]
        lines.append(f"|{name}|{len(rows)}|{sum(row['unnormalized_heat_max'] == 0 for row in rows)}|"
                     f"{np.median([row['effective_max_overlay_alpha'] for row in rows]):.4f}|"
                     f"{np.median([row['fraction_pixels_overlay_alpha_above_0.01'] for row in rows]):.4f}|")
    lines += ["", "## Paired output changes", "",
              "A changed prediction is evidence of a response to the input, not evidence of a correct response.",
              "", "|Condition|Completed pairs with changed localization|Joint@100 TP increased / decreased / unchanged|",
              "|---|---:|---|"]
    for group in GROUPS[1:]:
        rows = [row for row in paired_changes if row['condition'] == group and row['complete_pair']]
        up = sum(row['delta_joint_tp_100'] > 0 for row in rows)
        down = sum(row['delta_joint_tp_100'] < 0 for row in rows)
        equal = sum(row['delta_joint_tp_100'] == 0 for row in rows)
        lines.append(f"|{group}|{sum(row['localization_output_changed'] for row in rows)}/{len(rows)}|{up} / {down} / {equal}|")
    verified_pairs = [row for row in provenance if row["prompt_ids_identical"] is not None]
    lines += ["", "## Delivery verification", "",
              f"Captured full input IDs and real vision-forward tensor hashes for {sum(row['successful_visual_calls'] for row in groups)} successful calls.",
              f"Fully paired events: {len(verified_pairs)}; identical actual input IDs: {sum(row['prompt_ids_identical'] for row in verified_pairs)}; identical image grids: {sum(row['image_grids_identical'] for row in verified_pairs)}.",
              "Decoded system/user prompts, processor-restored paired images, pixel deltas and file hashes are indexed under actual_model_inputs/provenance.json.",
              "Delivery and prompt instructions are verified. Whether the model benefits from the cue is evaluated by the paired metrics, not inferred from the tensor audit.",
              "The stored alpha=0.28 is a cap, not the actual tint strength. attention_strength.csv records effective alpha after duration weighting and rendering for every source panel.",
              "Source P1/P2/P3 IDs identify exact saved JPEGs by path and SHA-256; video_time is the original extractor's requested seek time. Exact decoded video PTS/frame number is not captured by the original extractor.",
              "Current gaze uses the nearest valid observation within its configured tolerance and can fall slightly after the panel time; history uses observations through the reference time. The overall experiment is offline.",
              "Finite gazePoint intersections are world coordinates; gazeVector fallback is a direction. Historical points are reprojected to the reference camera, but occlusion and dynamic-object correspondence cannot be verified without depth/mesh.",
              "Original source hashes and package versions are saved in environment_and_sources.json. Coordinate consistency over the fixed 20 events is saved at the experiment root in coordinate_audit.json; this does not validate video clock alignment.",
              "", "## Inherited annotation limitation", "",
              "original_annotation_parity.json compares all rebuilt GT pixel coordinates, projection validity and status to the original released manifest. " + parity_description,
              "Posthoc image inspection reveals visible offsets in some projected-anchor labels: for example, scene4_room2/145 P1 places the person proxy near the wall rather than the visible person at the left. This is an inherited proxy/alignment concern, not a new manual pixel annotation or a verified clock calibration.",
              "The cause (clock alignment, anchor placement or camera projection) has not been isolated. The evaluator and labels remain unchanged; the reported pilot metrics must not be interpreted as validated human-labelled pixel localization accuracy.",
              "Equal image tokens and shared GPU load mean the elapsed differences do not establish inference acceleration from foveation.",
              "", "## Failures", ""]
    for row in summary["failures"]:
        lines.append(f"- {row['scene']}/{row['row_index']} {row['condition']}: {row['status']}; {row['error']}")
    if not summary["failures"]:
        lines.append("No upstream or model-call failures.")
    (output / "REPORT.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--split", choices=("smoke4", "pilot20"), required=True)
    args = parser.parse_args()
    report(args.root, args.split)


if __name__ == "__main__":
    main()
