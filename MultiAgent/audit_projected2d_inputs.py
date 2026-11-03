"""Rebuild saved attention maps without GT and export reference-camera provenance."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .context import load_contexts
from .gaze_evidence import build_heatmap, gaze_records, interpolate_camera_pose
from .projected2d_attention_study import read_csv, save_json


def audit(root: Path):
    config = json.loads((root / "config.json").read_text())
    manifest = json.loads((root / "fixed_samples.json").read_text())
    frame_manifest = {(row["scene"], int(row["row_index"]), row["panel_id"]): row
                      for row in read_csv(root / "manifest_all.csv")}
    events = []
    for ctx in load_contexts(manifest, load_ground_truth=False):
        records = gaze_records(ctx)
        event_dir = root / "inputs" / ctx.scene_key / str(ctx.row_index)
        saved_audit = json.loads((event_dir / "input_audit.json").read_text())
        frames = []
        for panel in saved_audit["panels"]:
            meta = frame_manifest[(ctx.scene_key, ctx.row_index, panel["frame_id"])]
            offset_source = meta.get("video_time_offset_source") or ""
            clock_available = bool(offset_source) and not offset_source.startswith("auto_unavailable")
            video_time, telemetry_time = panel["video_time_seconds"], panel["telemetry_time_seconds"]
            pose = interpolate_camera_pose(records, telemetry_time, config["gaze"]["max_gap_seconds"])
            intrinsics = None
            if pose:
                focal = panel["height"]/(2*math.tan(math.radians(pose["fov_vertical_degrees"])/2))
                intrinsics = {"fx": focal, "fy": focal, "cx": panel["width"]/2,
                              "cy": panel["height"]/2, "origin": "top_left",
                              "camera_axes": "Unity +x right, +y up, +z forward"}
            frame = {"frame_id": panel["frame_id"], "source_frame_id": panel["source_frame_id"],
                "video_path": meta["video_path"], "source_path": panel["source_path"],
                "source_sha256": panel["source_sha256"],
                "video_time_seconds": video_time, "telemetry_time_seconds": telemetry_time,
                "video_time_semantics": "Original extractor requested seek time, not measured decoded PTS",
                "video_time_offset_seconds": float(meta["video_time_offset_seconds"]),
                "video_time_offset_source": offset_source,
                "reference_camera_pose": pose, "intrinsics": intrinsics, "maps": {}}
            with np.load(event_dir / panel["frame_id"] / "attention_audit.npz") as expected:
                for label, strategy, key in (("current", "instant", "current_H_unnormalized"),
                                              ("history", "local_history", "history_H_unnormalized")):
                    heat, stats, projections = build_heatmap(ctx, records,
                        frame_time_video=video_time, frame_time_telemetry=telemetry_time,
                        event_window=[float(ctx.api_row["t_start"]), float(ctx.api_row["t_end"])],
                        clock_available=clock_available, image_size=(panel["width"], panel["height"]),
                        strategy=strategy, config=config["gaze"])
                    identical = bool(np.array_equal(heat, expected[key]))
                    frame["maps"][label] = {"recomputed_map_exactly_matches_saved_input_map": identical,
                        "stats": stats, "source_observations_and_reprojections": projections}
                    if not identical:
                        raise ValueError(f"Attention map mismatch: {ctx.scene_key}/{ctx.row_index}/{panel['frame_id']}/{label}")
                    if not np.any(heat):
                        groups = ["CURRENT_HEATMAP"] if label == "current" else ["HISTORY_HEATMAP", "HISTORY_FOVEATION"]
                        fallback_matches = {group: panel["processed_sha256"][group] == panel["processed_sha256"]["RGB"]
                                            for group in groups}
                        frame["maps"][label]["no_attention_fallback_equals_RGB_input"] = fallback_matches
                        if not all(fallback_matches.values()):
                            raise ValueError(f"No-gaze fallback differs from RGB: {ctx.scene_key}/{ctx.row_index}/{panel['frame_id']}")
            frames.append(frame)
        events.append({"scene": ctx.scene_key, "row_index": ctx.row_index,
                       "event_id": ctx.event_id, "raw_gaze_records": records, "frames": frames})
    result = {"protocol": "offline; GT not loaded or used; deterministic CPU input audit",
        "all_saved_numerical_attention_maps_reproduced_exactly": True,
        "coordinate_convention": "gazePoint finite world intersection; gazeVector world direction; camera-to-world quaternion xyzw; world-to-camera uses inverse rotation after subtracting position for points",
        "limitations": ["FOV-derived ideal pinhole intrinsics, not a separately calibrated camera matrix",
                        "Clock offset reused from original protocol, not independently calibrated",
                        "No mesh/depth occlusion or moving-object correspondence verification"],
        "events": events}
    save_json(root / "geometry_provenance.json", result)
    print(json.dumps({"events": len(events), "frames": sum(len(e["frames"]) for e in events),
                      "all_maps_exactly_reproduced": True}, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    audit(args.root)


if __name__ == "__main__":
    main()
