#!/usr/bin/env python3
"""Evaluate exam2 Qwen 2D point predictions against projected anchor proxy GT."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


DETAIL_COLUMNS = (
    "scene",
    "row_index",
    "event_id",
    "gt_referent_count",
    "time_evaluable_referent_count",
    "prediction_count",
    "time_matched",
    "point_matched_50",
    "point_matched_100",
    "point_matched_150",
    "point_matched_200",
    "joint_matched_50",
    "joint_matched_100",
    "joint_matched_150",
    "joint_matched_200",
    "time_tp",
    "time_fp",
    "time_fn",
    "point_tp_50",
    "point_fp_50",
    "point_fn_50",
    "point_tp_100",
    "point_fp_100",
    "point_fn_100",
    "point_tp_150",
    "point_fp_150",
    "point_fn_150",
    "point_tp_200",
    "point_fp_200",
    "point_fn_200",
    "joint_tp_50",
    "joint_fp_50",
    "joint_fn_50",
    "joint_tp_100",
    "joint_fp_100",
    "joint_fn_100",
    "joint_tp_150",
    "joint_fp_150",
    "joint_fn_150",
    "joint_tp_200",
    "joint_fp_200",
    "joint_fn_200",
    "min_point_distance_px",
    "mean_joint_distance_px_100",
    "parse_ok",
    "error_message",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 2D point predictions using anchor-projection proxy GT.")
    parser.add_argument("--manifest", required=True, help="manifest_all.csv from build_2d_eval_manifest.py")
    parser.add_argument("--pred_csv", required=True, help="Prediction CSV from run_qwen3vl_2d_point_grounding.py")
    parser.add_argument("--output_dir", required=True, help="Evaluation output directory.")
    parser.add_argument("--thresholds", default="50,100,150,200", help="Pixel thresholds. Default: 50,100,150,200")
    parser.add_argument("--panel_width", type=int, default=512, help="Model panel width. Default: 512")
    parser.add_argument("--panel_height", type=int, default=384, help="Model panel height. Default: 384")
    parser.add_argument("--columns", type=int, default=3, help="Model sheet columns. Default: 3")
    parser.add_argument("--gutter", type=int, default=12, help="Model sheet gutter. Default: 12")
    parser.add_argument("--label_height", type=int, default=34, help="Panel label height. Default: 34")
    parser.add_argument("--start_index", type=int, default=0, help="Minimum row_index per scene. Default: 0.")
    parser.add_argument("--limit", type=int, help="Maximum row_index count per scene.")
    parser.add_argument("--scenes", nargs="*", help="Optional scenes to evaluate.")
    parser.add_argument(
        "--coordinate_mode",
        choices=("auto", "panel", "sheet"),
        default="auto",
        help="Evaluate prediction coordinates as panel-local, sheet-global, or auto. Default: auto.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_int(value: Any) -> Optional[int]:
    text = normalize_text(value)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def parse_float(value: Any) -> Optional[float]:
    text = normalize_text(value)
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DETAIL_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in DETAIL_COLUMNS})


def manifest_groups(rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, int], List[Mapping[str, str]]]:
    grouped: Dict[Tuple[str, int], List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        scene = normalize_text(row.get("scene"))
        row_index = parse_int(row.get("row_index"))
        if scene and row_index is not None:
            grouped[(scene, row_index)].append(row)
    return grouped


def prediction_map(rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, int], Mapping[str, str]]:
    result: Dict[Tuple[str, int], Mapping[str, str]] = {}
    for row in rows:
        scene = normalize_text(row.get("scene"))
        row_index = parse_int(row.get("row_index"))
        if scene and row_index is not None:
            result[(scene, row_index)] = row
    return result


def selected_key(scene: str, row_index: int, args: argparse.Namespace) -> bool:
    if args.scenes and scene not in set(args.scenes):
        return False
    if row_index < args.start_index:
        return False
    if args.limit is not None and row_index >= args.start_index + args.limit:
        return False
    return True


def parse_predictions(row: Mapping[str, str]) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(normalize_text(row.get("parsed_json")) or "{}")
    except json.JSONDecodeError:
        return []
    raw = payload.get("referents") if isinstance(payload, Mapping) else None
    if not isinstance(raw, list):
        return []
    predictions: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        panel_id = normalize_text(item.get("panel_id")).upper()
        if panel_id and not panel_id.startswith("P"):
            panel_id = "P" + panel_id
        sheet_x_norm = parse_float(item.get("sheet_x_norm"))
        sheet_y_norm = parse_float(item.get("sheet_y_norm"))
        x_norm = parse_float(item.get("x_norm"))
        y_norm = parse_float(item.get("y_norm"))
        if not panel_id:
            continue
        if sheet_x_norm is not None and not (0.0 <= sheet_x_norm <= 1.0):
            sheet_x_norm = None
        if sheet_y_norm is not None and not (0.0 <= sheet_y_norm <= 1.0):
            sheet_y_norm = None
        if x_norm is not None and not (0.0 <= x_norm <= 1.0):
            x_norm = None
        if y_norm is not None and not (0.0 <= y_norm <= 1.0):
            y_norm = None
        predictions.append(
            {
                "mention": normalize_text(item.get("mention")),
                "panel_id": panel_id,
                "sheet_x_norm": sheet_x_norm,
                "sheet_y_norm": sheet_y_norm,
                "x_norm": x_norm,
                "y_norm": y_norm,
            }
        )
    return predictions


def panel_sort_key(panel_id: str) -> int:
    number = parse_int(panel_id.upper().replace("P", ""))
    return number if number is not None else 999


def build_layout(rows: Sequence[Mapping[str, str]], args: argparse.Namespace) -> Dict[str, Any]:
    panel_ids = sorted(
        {
            normalize_text(row.get("panel_id")).upper()
            for row in rows
            if normalize_text(row.get("panel_id")) and normalize_text(row.get("frame_extracted")) != "False"
        },
        key=panel_sort_key,
    )
    if not panel_ids:
        panel_ids = sorted({normalize_text(row.get("panel_id")).upper() for row in rows if normalize_text(row.get("panel_id"))}, key=panel_sort_key)
    columns = max(1, int(args.columns))
    sheet_rows = int(math.ceil(len(panel_ids) / columns)) if panel_ids else 1
    sheet_w = columns * args.panel_width + (columns + 1) * args.gutter
    sheet_h = sheet_rows * (args.panel_height + args.label_height) + (sheet_rows + 1) * args.gutter
    positions: Dict[str, Tuple[int, int]] = {}
    for idx, panel_id in enumerate(panel_ids):
        row_pos = idx // columns
        col_pos = idx % columns
        x0 = args.gutter + col_pos * (args.panel_width + args.gutter)
        y0 = args.gutter + row_pos * (args.panel_height + args.label_height + args.gutter)
        positions[panel_id] = (x0, y0)
    return {"sheet_w": sheet_w, "sheet_h": sheet_h, "positions": positions}


def content_point_to_sheet(
    panel_id: str,
    u_norm: float,
    v_norm: float,
    image_width: float,
    image_height: float,
    layout: Mapping[str, Any],
    args: argparse.Namespace,
) -> Optional[Tuple[float, float]]:
    positions = layout["positions"]
    if panel_id not in positions or image_width <= 0 or image_height <= 0:
        return None
    scale = min(args.panel_width / image_width, args.panel_height / image_height)
    content_w = image_width * scale
    content_h = image_height * scale
    pad_x = (args.panel_width - content_w) / 2.0
    pad_y = (args.panel_height - content_h) / 2.0
    x0, y0 = positions[panel_id]
    return x0 + pad_x + u_norm * content_w, y0 + pad_y + v_norm * content_h


def panel_area_point_to_sheet(
    panel_id: str,
    x_norm: float,
    y_norm: float,
    layout: Mapping[str, Any],
    args: argparse.Namespace,
) -> Optional[Tuple[float, float]]:
    positions = layout["positions"]
    if panel_id not in positions:
        return None
    x0, y0 = positions[panel_id]
    return x0 + x_norm * args.panel_width, y0 + y_norm * args.panel_height


def prediction_sheet_point(prediction: Mapping[str, Any], layout: Mapping[str, Any], args: argparse.Namespace) -> Optional[Tuple[float, float]]:
    sheet_x = prediction.get("sheet_x_norm")
    sheet_y = prediction.get("sheet_y_norm")
    if sheet_x is not None and sheet_y is not None:
        return float(sheet_x) * float(layout["sheet_w"]), float(sheet_y) * float(layout["sheet_h"])
    x_norm = prediction.get("x_norm")
    y_norm = prediction.get("y_norm")
    panel_id = normalize_text(prediction.get("panel_id")).upper()
    if x_norm is None or y_norm is None or not panel_id:
        return None
    return panel_area_point_to_sheet(panel_id, float(x_norm), float(y_norm), layout, args)


def gt_points(rows: Sequence[Mapping[str, str]], layout: Mapping[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        if normalize_text(row.get("projection_valid")) != "True":
            continue
        panel_id = normalize_text(row.get("panel_id")).upper()
        referent = normalize_text(row.get("referent_name"))
        gt_u = parse_float(row.get("gt_u_norm"))
        gt_v = parse_float(row.get("gt_v_norm"))
        width = parse_float(row.get("image_width"))
        height = parse_float(row.get("image_height"))
        if not panel_id or not referent or gt_u is None or gt_v is None or not width or not height:
            continue
        key = (panel_id, referent)
        if key in seen:
            continue
        seen.add(key)
        sheet_point = content_point_to_sheet(panel_id, gt_u, gt_v, width, height, layout, args)
        if sheet_point is None:
            continue
        points.append(
            {
                "referent": referent,
                "panel_id": panel_id,
                "gt_u_norm": gt_u,
                "gt_v_norm": gt_v,
                "image_width": width,
                "image_height": height,
                "sheet_x": sheet_point[0],
                "sheet_y": sheet_point[1],
                "evidence_acceptable": normalize_text(row.get("evidence_acceptable")) == "True",
            }
        )
    return points


def label_key(value: Any) -> str:
    text = normalize_text(value).lower()
    return "".join(ch for ch in text if ch.isalnum())


def point_distance(prediction: Mapping[str, Any], gt: Mapping[str, Any], layout: Mapping[str, Any], args: argparse.Namespace) -> Optional[float]:
    if args.coordinate_mode == "panel":
        panel_id = normalize_text(prediction.get("panel_id")).upper()
        if panel_id != normalize_text(gt.get("panel_id")).upper():
            return None
        x_norm = prediction.get("x_norm")
        y_norm = prediction.get("y_norm")
        if x_norm is None or y_norm is None:
            return None
        pred_x = float(x_norm) * float(gt["image_width"])
        pred_y = float(y_norm) * float(gt["image_height"])
        gt_x = float(gt["gt_u_norm"]) * float(gt["image_width"])
        gt_y = float(gt["gt_v_norm"]) * float(gt["image_height"])
        return math.hypot(pred_x - gt_x, pred_y - gt_y)
    if args.coordinate_mode == "sheet":
        pred_point = prediction_sheet_point(prediction, layout, args)
        if pred_point is None:
            return None
        return math.hypot(pred_point[0] - float(gt["sheet_x"]), pred_point[1] - float(gt["sheet_y"]))
    if prediction.get("sheet_x_norm") is None or prediction.get("sheet_y_norm") is None:
        panel_id = normalize_text(prediction.get("panel_id")).upper()
        if panel_id != normalize_text(gt.get("panel_id")).upper():
            return None
        x_norm = prediction.get("x_norm")
        y_norm = prediction.get("y_norm")
        if x_norm is None or y_norm is None:
            return None
        pred_x = float(x_norm) * float(gt["image_width"])
        pred_y = float(y_norm) * float(gt["image_height"])
        gt_x = float(gt["gt_u_norm"]) * float(gt["image_width"])
        gt_y = float(gt["gt_v_norm"]) * float(gt["image_height"])
        return math.hypot(pred_x - gt_x, pred_y - gt_y)
    pred_point = prediction_sheet_point(prediction, layout, args)
    if pred_point is None:
        return None
    return math.hypot(pred_point[0] - float(gt["sheet_x"]), pred_point[1] - float(gt["sheet_y"]))


def candidate_rows(
    prediction: Mapping[str, Any],
    gts: Sequence[Mapping[str, Any]],
    matched_referents: set[str],
    require_acceptable: bool,
) -> List[Mapping[str, Any]]:
    candidates = [
        gt
        for gt in gts
        if str(gt["referent"]) not in matched_referents
        and (not require_acceptable or bool(gt.get("evidence_acceptable")))
    ]
    mention_key = label_key(prediction.get("mention"))
    label_candidates = [gt for gt in candidates if mention_key and mention_key == label_key(gt.get("referent"))]
    return label_candidates if label_candidates else candidates


def greedy_point_match(
    predictions: Sequence[Mapping[str, Any]],
    gts: Sequence[Mapping[str, Any]],
    layout: Mapping[str, Any],
    args: argparse.Namespace,
    threshold: float,
    require_acceptable: bool,
) -> Tuple[int, int, int, List[float]]:
    matched_referents: set[str] = set()
    distances: List[float] = []
    tp = 0
    fp = 0
    total_gt = len({str(gt["referent"]) for gt in gts if not require_acceptable or bool(gt.get("evidence_acceptable"))})
    for prediction in predictions:
        candidates = candidate_rows(prediction, gts, matched_referents, require_acceptable)
        scored: List[Tuple[float, Mapping[str, Any]]] = []
        for gt in candidates:
            distance = point_distance(prediction, gt, layout, args)
            if distance is not None:
                scored.append((distance, gt))
        if not scored:
            fp += 1
            continue
        best_distance, best = min(scored, key=lambda item: item[0])
        if best_distance <= threshold:
            tp += 1
            matched_referents.add(str(best["referent"]))
            distances.append(best_distance)
        else:
            fp += 1
    fn = max(0, total_gt - tp)
    return tp, fp, fn, distances


def greedy_time_match(predictions: Sequence[Mapping[str, Any]], gts: Sequence[Mapping[str, Any]]) -> Tuple[int, int, int]:
    matched_referents: set[str] = set()
    tp = 0
    fp = 0
    total_gt = len({str(gt["referent"]) for gt in gts if bool(gt.get("evidence_acceptable"))})
    for prediction in predictions:
        panel_id = normalize_text(prediction.get("panel_id")).upper()
        if not panel_id:
            fp += 1
            continue
        candidates = [
            gt
            for gt in candidate_rows(prediction, gts, matched_referents, True)
            if normalize_text(gt.get("panel_id")).upper() == panel_id
        ]
        if not candidates:
            fp += 1
            continue
        matched_referents.add(str(candidates[0]["referent"]))
        tp += 1
    fn = max(0, total_gt - tp)
    return tp, fp, fn


def min_point_distance(predictions: Sequence[Mapping[str, Any]], gts: Sequence[Mapping[str, Any]], layout: Mapping[str, Any], args: argparse.Namespace) -> Optional[float]:
    distances = [
        distance
        for prediction in predictions
        for gt in gts
        for distance in [point_distance(prediction, gt, layout, args)]
        if distance is not None
    ]
    return min(distances) if distances else None


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def add_metric(summary: Dict[str, Any], prefix: str, tp: int, fp: int, fn: int) -> None:
    scores = prf(tp, fp, fn)
    summary[f"{prefix}_precision"] = scores["precision"]
    summary[f"{prefix}_recall"] = scores["recall"]
    summary[f"{prefix}_f1"] = scores["f1"]


def main() -> None:
    args = parse_args()
    manifest = manifest_groups(read_csv_rows(Path(args.manifest)))
    predictions = prediction_map(read_csv_rows(Path(args.pred_csv)))
    output_dir = Path(args.output_dir).resolve()
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    if 50.0 not in thresholds:
        thresholds.append(50.0)
    if 100.0 not in thresholds:
        thresholds.append(100.0)
    if 150.0 not in thresholds:
        thresholds.append(150.0)
    if 200.0 not in thresholds:
        thresholds.append(200.0)

    detail_rows: List[Dict[str, Any]] = []
    totals: Dict[str, Any] = {
        "events": 0,
        "gt_referents": 0,
        "time_evaluable_referents": 0,
        "predictions": 0,
        "time_tp": 0,
        "time_fp": 0,
        "time_fn": 0,
        "point_tp_50": 0,
        "point_fp_50": 0,
        "point_fn_50": 0,
        "point_tp_100": 0,
        "point_fp_100": 0,
        "point_fn_100": 0,
        "point_tp_150": 0,
        "point_fp_150": 0,
        "point_fn_150": 0,
        "point_tp_200": 0,
        "point_fp_200": 0,
        "point_fn_200": 0,
        "joint_tp_50": 0,
        "joint_fp_50": 0,
        "joint_fn_50": 0,
        "joint_tp_100": 0,
        "joint_fp_100": 0,
        "joint_fn_100": 0,
        "joint_tp_150": 0,
        "joint_fp_150": 0,
        "joint_fn_150": 0,
        "joint_tp_200": 0,
        "joint_fp_200": 0,
        "joint_fn_200": 0,
    }
    joint_distances_100: List[float] = []
    point_distances_100: List[float] = []

    for key in sorted(manifest, key=lambda item: (item[0], item[1])):
        scene, row_index = key
        if not selected_key(scene, row_index, args):
            continue
        rows = manifest[key]
        pred_row = predictions.get(key, {})
        pred_items = parse_predictions(pred_row)
        layout = build_layout(rows, args)
        gts = gt_points(rows, layout, args)
        gt_referents = {str(gt["referent"]) for gt in gts}
        time_referents = {str(gt["referent"]) for gt in gts if bool(gt.get("evidence_acceptable"))}
        time_tp, time_fp, time_fn = greedy_time_match(pred_items, gts)
        point_tp50, point_fp50, point_fn50, _point_dist50 = greedy_point_match(pred_items, gts, layout, args, 50.0, False)
        point_tp100, point_fp100, point_fn100, point_dist100 = greedy_point_match(pred_items, gts, layout, args, 100.0, False)
        point_tp150, point_fp150, point_fn150, _point_dist150 = greedy_point_match(pred_items, gts, layout, args, 150.0, False)
        point_tp200, point_fp200, point_fn200, _point_dist200 = greedy_point_match(pred_items, gts, layout, args, 200.0, False)
        joint_tp50, joint_fp50, joint_fn50, _joint_dist50 = greedy_point_match(pred_items, gts, layout, args, 50.0, True)
        joint_tp100, joint_fp100, joint_fn100, joint_dist100 = greedy_point_match(pred_items, gts, layout, args, 100.0, True)
        joint_tp150, joint_fp150, joint_fn150, _joint_dist150 = greedy_point_match(pred_items, gts, layout, args, 150.0, True)
        joint_tp200, joint_fp200, joint_fn200, _joint_dist200 = greedy_point_match(pred_items, gts, layout, args, 200.0, True)
        min_dist = min_point_distance(pred_items, gts, layout, args)

        totals["events"] += 1
        totals["gt_referents"] += len(gt_referents)
        totals["time_evaluable_referents"] += len(time_referents)
        totals["predictions"] += len(pred_items)
        for name, value in (
            ("time_tp", time_tp),
            ("time_fp", time_fp),
            ("time_fn", time_fn),
            ("point_tp_50", point_tp50),
            ("point_fp_50", point_fp50),
            ("point_fn_50", point_fn50),
            ("point_tp_100", point_tp100),
            ("point_fp_100", point_fp100),
            ("point_fn_100", point_fn100),
            ("point_tp_150", point_tp150),
            ("point_fp_150", point_fp150),
            ("point_fn_150", point_fn150),
            ("point_tp_200", point_tp200),
            ("point_fp_200", point_fp200),
            ("point_fn_200", point_fn200),
            ("joint_tp_50", joint_tp50),
            ("joint_fp_50", joint_fp50),
            ("joint_fn_50", joint_fn50),
            ("joint_tp_100", joint_tp100),
            ("joint_fp_100", joint_fp100),
            ("joint_fn_100", joint_fn100),
            ("joint_tp_150", joint_tp150),
            ("joint_fp_150", joint_fp150),
            ("joint_fn_150", joint_fn150),
            ("joint_tp_200", joint_tp200),
            ("joint_fp_200", joint_fp200),
            ("joint_fn_200", joint_fn200),
        ):
            totals[name] += value
        point_distances_100.extend(point_dist100)
        joint_distances_100.extend(joint_dist100)

        detail_rows.append(
            {
                "scene": scene,
                "row_index": row_index,
                "event_id": normalize_text(rows[0].get("event_id")),
                "gt_referent_count": len(gt_referents),
                "time_evaluable_referent_count": len(time_referents),
                "prediction_count": len(pred_items),
                "time_matched": time_tp,
                "point_matched_50": point_tp50,
                "point_matched_100": point_tp100,
                "point_matched_150": point_tp150,
                "point_matched_200": point_tp200,
                "joint_matched_50": joint_tp50,
                "joint_matched_100": joint_tp100,
                "joint_matched_150": joint_tp150,
                "joint_matched_200": joint_tp200,
                "time_tp": time_tp,
                "time_fp": time_fp,
                "time_fn": time_fn,
                "point_tp_50": point_tp50,
                "point_fp_50": point_fp50,
                "point_fn_50": point_fn50,
                "point_tp_100": point_tp100,
                "point_fp_100": point_fp100,
                "point_fn_100": point_fn100,
                "point_tp_150": point_tp150,
                "point_fp_150": point_fp150,
                "point_fn_150": point_fn150,
                "point_tp_200": point_tp200,
                "point_fp_200": point_fp200,
                "point_fn_200": point_fn200,
                "joint_tp_50": joint_tp50,
                "joint_fp_50": joint_fp50,
                "joint_fn_50": joint_fn50,
                "joint_tp_100": joint_tp100,
                "joint_fp_100": joint_fp100,
                "joint_fn_100": joint_fn100,
                "joint_tp_150": joint_tp150,
                "joint_fp_150": joint_fp150,
                "joint_fn_150": joint_fn150,
                "joint_tp_200": joint_tp200,
                "joint_fp_200": joint_fp200,
                "joint_fn_200": joint_fn200,
                "min_point_distance_px": f"{min_dist:.2f}" if min_dist is not None else "",
                "mean_joint_distance_px_100": f"{safe_div(sum(joint_dist100), len(joint_dist100)):.2f}" if joint_dist100 else "",
                "parse_ok": normalize_text(pred_row.get("parse_ok")),
                "error_message": normalize_text(pred_row.get("error_message")),
            }
        )

    summary = dict(totals)
    add_metric(summary, "time", int(totals["time_tp"]), int(totals["time_fp"]), int(totals["time_fn"]))
    add_metric(summary, "point_50", int(totals["point_tp_50"]), int(totals["point_fp_50"]), int(totals["point_fn_50"]))
    add_metric(summary, "point_100", int(totals["point_tp_100"]), int(totals["point_fp_100"]), int(totals["point_fn_100"]))
    add_metric(summary, "point_150", int(totals["point_tp_150"]), int(totals["point_fp_150"]), int(totals["point_fn_150"]))
    add_metric(summary, "point_200", int(totals["point_tp_200"]), int(totals["point_fp_200"]), int(totals["point_fn_200"]))
    add_metric(summary, "joint_50", int(totals["joint_tp_50"]), int(totals["joint_fp_50"]), int(totals["joint_fn_50"]))
    add_metric(summary, "joint_100", int(totals["joint_tp_100"]), int(totals["joint_fp_100"]), int(totals["joint_fn_100"]))
    add_metric(summary, "joint_150", int(totals["joint_tp_150"]), int(totals["joint_fp_150"]), int(totals["joint_fn_150"]))
    add_metric(summary, "joint_200", int(totals["joint_tp_200"]), int(totals["joint_fp_200"]), int(totals["joint_fn_200"]))
    summary["mean_point_distance_px_100"] = safe_div(sum(point_distances_100), len(point_distances_100))
    summary["mean_joint_distance_px_100"] = safe_div(sum(joint_distances_100), len(joint_distances_100))

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "2d_eval_detail.csv"
    summary_path = output_dir / "2d_eval_summary.json"
    write_csv(detail_path, detail_rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote detail: {detail_path}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
