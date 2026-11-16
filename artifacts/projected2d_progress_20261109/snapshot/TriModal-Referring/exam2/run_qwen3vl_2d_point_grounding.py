#!/usr/bin/env python3
"""Run Qwen3-VL for multi-panel 2D referring point grounding.

The model sees several unannotated panels per event in one call. GT anchor
projection points from the manifest are not drawn into the model input; they
are used only by the evaluation script.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


OUTPUT_COLUMNS = (
    "scene",
    "row_index",
    "event_id",
    "instruction",
    "model_input_image",
    "model_raw_output",
    "parsed_json",
    "parse_ok",
    "prediction_count",
    "error_message",
)


SYSTEM_PROMPT = (
    "You are a precise multimodal 2D referring-point grounding model for egocentric VR screenshots. "
    "You receive chronological panels P1, P2, P3, etc. as separate images or as a contact sheet. "
    "The visible green gaze dot in the panels is only an attention cue; it is not the answer and can be noisy. "
    "Your task is to identify the referent or referents intended by the instruction, choose the best evidence panel for each referent, "
    "and output one normalized 2D point per referent inside that selected panel. "
    "Return strict JSON only, with no markdown and no commentary."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Qwen3-VL on exam2 multi-panel 2D point-grounding inputs.")
    parser.add_argument("--repo_root", default=".", help="Repository root. Default: current directory.")
    parser.add_argument("--manifest", required=True, help="manifest_all.csv from build_2d_eval_manifest.py")
    parser.add_argument("--output_csv", required=True, help="Output CSV for model predictions.")
    parser.add_argument("--output_json_dir", required=True, help="Directory for per-event raw JSON outputs.")
    parser.add_argument("--model_input_dir", help="Directory for optional unannotated model contact sheets. Default: <output_json_dir>/model_inputs")
    parser.add_argument("--model_name", required=True, help="Local Qwen3-VL model path.")
    parser.add_argument("--dtype", default="auto", help="Torch dtype: auto, float16, bfloat16, float32.")
    parser.add_argument("--use_flash_attn", action="store_true", help="Try flash_attention_2 first, then fall back.")
    parser.add_argument("--local_files_only", action="store_true", help="Load model and processor from local files only.")
    parser.add_argument("--max_new_tokens", type=int, default=768, help="Maximum new tokens. Default: 768.")
    parser.add_argument("--offload_folder", help="Optional Accelerate disk offload folder.")
    parser.add_argument("--start_index", type=int, default=0, help="First row_index per scene to run. Default: 0.")
    parser.add_argument("--limit", type=int, help="Maximum row_index count per scene.")
    parser.add_argument("--scenes", nargs="*", help="Optional scenes to run.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-event JSON outputs.")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue when one event fails.")
    parser.add_argument("--panel_width", type=int, default=512, help="Panel width in model sheet. Default: 512")
    parser.add_argument("--panel_height", type=int, default=384, help="Panel height in model sheet. Default: 384")
    parser.add_argument("--columns", type=int, default=3, help="Panel columns. Default: 3")
    parser.add_argument(
        "--input_mode",
        choices=("multi_image", "contact_sheet"),
        default="multi_image",
        help="multi_image sends each panel as a separate image in one model call; contact_sheet keeps the old concatenated sheet. Default: multi_image.",
    )
    parser.add_argument(
        "--panel_caption_mode",
        choices=("none", "text"),
        default="none",
        help="How to bind panel ids to images in multi_image mode. none keeps the v4 image-only order; text inserts a text caption before each image. Default: none.",
    )
    parser.add_argument(
        "--panel_context_mode",
        choices=("full", "full_crop", "paired_crop"),
        default="full",
        help=(
            "full uses one full image per panel; full_crop adds one separate gaze-centered crop reference per panel; "
            "paired_crop sends one paired image per panel with LEFT=full panel and RIGHT=gaze crop. Default: full."
        ),
    )
    parser.add_argument("--gaze_crop_ratio", type=float, default=0.35, help="Crop side ratio relative to the shorter image side. Default: 0.35.")
    parser.add_argument("--crop_output_size", type=int, default=768, help="Square crop output size in pixels. Default: 768.")
    parser.add_argument(
        "--paired_crop_coordinate_policy",
        choices=("none", "source_or_canvas_map", "paired_canvas_map"),
        default="paired_canvas_map",
        help=(
            "For paired_crop inputs, map model coordinates back to full-panel coordinates. "
            "paired_canvas_map ignores unreliable coordinate_space labels and maps by paired-image geometry; "
            "source_or_canvas_map uses coordinate_space when present and falls back to paired-canvas geometry. Default: paired_canvas_map."
        ),
    )
    parser.add_argument(
        "--prompt_mode",
        choices=("expected_count", "gt_referents", "instruction_only"),
        default="expected_count",
        help="expected_count gives only the expected output count; gt_referents is oracle label mode; instruction_only asks the model to infer everything. Default: expected_count.",
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
        value_float = float(text)
    except ValueError:
        return None
    return value_float if math.isfinite(value_float) else None


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def group_manifest(rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, int], List[Mapping[str, str]]]:
    grouped: Dict[Tuple[str, int], List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        scene = normalize_text(row.get("scene"))
        row_index = parse_int(row.get("row_index"))
        panel_id = normalize_text(row.get("panel_id"))
        if scene and row_index is not None and panel_id:
            grouped[(scene, row_index)].append(row)
    return grouped


def selected_key(key: Tuple[str, int], args: argparse.Namespace) -> bool:
    scene, row_index = key
    if args.scenes and scene not in set(args.scenes):
        return False
    if row_index < args.start_index:
        return False
    if args.limit is not None and row_index >= args.start_index + args.limit:
        return False
    return True


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def resize_panel(image: Image.Image, panel_size: Tuple[int, int]) -> Image.Image:
    target_w, target_h = panel_size
    src_w, src_h = image.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", panel_size, (18, 18, 18))
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas


def build_model_contact_sheet(
    key: Tuple[str, int],
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    panel_size: Tuple[int, int],
    columns: int,
    overwrite: bool,
) -> Tuple[Path, List[Dict[str, str]]]:
    scene, row_index = key
    panel_to_row: Dict[str, Mapping[str, str]] = {}
    for row in rows:
        panel = normalize_text(row.get("panel_id"))
        if normalize_text(row.get("frame_extracted")) == "False":
            continue
        image_path = Path(normalize_text(row.get("frame_path")))
        if not image_path.exists():
            continue
        if panel and panel not in panel_to_row:
            panel_to_row[panel] = row
    panels = sorted(panel_to_row, key=lambda panel: parse_int(panel.replace("P", "")) or 999)
    if not panels:
        raise RuntimeError(f"No panels for {scene} row {row_index}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{scene}_row_{row_index}.jpg"
    panel_meta: List[Dict[str, str]] = []
    if output_path.exists() and not overwrite:
        for panel in panels:
            row = panel_to_row[panel]
            panel_meta.append({
                "panel_id": panel,
                "video_frame_time": normalize_text(row.get("video_frame_time") or row.get("frame_time")),
                "json_sample_time": normalize_text(row.get("json_sample_time")),
                "frame_path": normalize_text(row.get("frame_path")),
            })
        return output_path, panel_meta

    panel_w, panel_h = panel_size
    columns = max(1, columns)
    panel_count = len(panels)
    sheet_rows = int(math.ceil(panel_count / columns))
    gutter = 12
    label_h = 34
    sheet_w = columns * panel_w + (columns + 1) * gutter
    sheet_h = sheet_rows * (panel_h + label_h) + (sheet_rows + 1) * gutter
    sheet = Image.new("RGB", (sheet_w, sheet_h), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    font = load_font(20)
    small_font = load_font(15)

    for idx, panel in enumerate(panels):
        row = panel_to_row[panel]
        image_path = Path(normalize_text(row.get("frame_path")))
        image = Image.open(image_path).convert("RGB")
        image = resize_panel(image, panel_size)
        row_pos = idx // columns
        col_pos = idx % columns
        x0 = gutter + col_pos * (panel_w + gutter)
        y0 = gutter + row_pos * (panel_h + label_h + gutter)
        sheet.paste(image, (x0, y0))
        draw.rectangle((x0, y0, x0 + panel_w, y0 + panel_h), outline=(210, 210, 210), width=2)
        video_time = normalize_text(row.get("video_frame_time") or row.get("frame_time"))
        draw.text((x0 + 8, y0 + 7), panel, fill=(255, 255, 255), font=font)
        draw.text((x0 + 62, y0 + 10), f"video_t={video_time}s", fill=(235, 235, 235), font=small_font)
        draw.text((x0 + 8, y0 + panel_h + 7), f"{panel}  t={video_time}s", fill=(235, 235, 235), font=small_font)
        panel_meta.append({
            "panel_id": panel,
            "video_frame_time": video_time,
            "json_sample_time": normalize_text(row.get("json_sample_time")),
            "frame_path": str(image_path),
        })
    sheet.save(output_path, quality=92)
    return output_path, panel_meta


def collect_model_panels(
    key: Tuple[str, int],
    rows: Sequence[Mapping[str, str]],
) -> Tuple[List[Path], List[Dict[str, str]]]:
    scene, row_index = key
    panel_to_row: Dict[str, Mapping[str, str]] = {}
    for row in rows:
        panel = normalize_text(row.get("panel_id"))
        if normalize_text(row.get("frame_extracted")) == "False":
            continue
        image_path = Path(normalize_text(row.get("frame_path")))
        if not image_path.exists():
            continue
        if panel and panel not in panel_to_row:
            panel_to_row[panel] = row
    panels = sorted(panel_to_row, key=lambda panel: parse_int(panel.replace("P", "")) or 999)
    if not panels:
        raise RuntimeError(f"No panels for {scene} row {row_index}")
    image_paths: List[Path] = []
    panel_meta: List[Dict[str, str]] = []
    for panel in panels:
        row = panel_to_row[panel]
        image_path = Path(normalize_text(row.get("frame_path")))
        image_paths.append(image_path)
        panel_meta.append(
            {
                "panel_id": panel,
                "image_role": "full",
                "video_frame_time": normalize_text(row.get("video_frame_time") or row.get("frame_time")),
                "json_sample_time": normalize_text(row.get("json_sample_time")),
                "frame_path": str(image_path),
                "gaze_u_norm": normalize_text(row.get("gaze_u_norm")),
                "gaze_v_norm": normalize_text(row.get("gaze_v_norm")),
                "gaze_projection_valid": normalize_text(row.get("gaze_projection_valid")),
            }
        )
    return image_paths, panel_meta


def clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def create_gaze_crop(
    image_path: Path,
    output_path: Path,
    gaze_u_norm: Optional[float],
    gaze_v_norm: Optional[float],
    crop_ratio: float,
    output_size: int,
    overwrite: bool,
) -> Tuple[Path, Tuple[float, float, float, float], str]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    center_u = gaze_u_norm if gaze_u_norm is not None else 0.5
    center_v = gaze_v_norm if gaze_v_norm is not None else 0.5
    center_x = clamp_float(center_u, 0.0, 1.0) * width
    center_y = clamp_float(center_v, 0.0, 1.0) * height
    side = max(32, int(min(width, height) * clamp_float(crop_ratio, 0.1, 0.9)))
    half = side / 2.0
    left = int(round(clamp_float(center_x - half, 0.0, max(0.0, width - side))))
    top = int(round(clamp_float(center_y - half, 0.0, max(0.0, height - side))))
    right = min(width, left + side)
    bottom = min(height, top + side)
    if overwrite or not output_path.exists():
        crop = image.crop((left, top, right, bottom)).resize((output_size, output_size), Image.Resampling.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path, quality=92)
    bbox = (left / width, top / height, right / width, bottom / height)
    source = "gaze" if gaze_u_norm is not None and gaze_v_norm is not None else "center_fallback"
    return output_path, bbox, source


def draw_boxed_label(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int],
    background: Tuple[int, int, int],
) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad_x = 8
    pad_y = 5
    draw.rectangle(
        (bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y),
        fill=background,
    )
    draw.text((x, y), text, fill=fill, font=font)


def create_paired_full_crop_image(
    image_path: Path,
    crop_path: Path,
    output_path: Path,
    panel_id: str,
    crop_bbox_norm: Tuple[float, float, float, float],
    output_size: int,
    overwrite: bool,
) -> Path:
    if output_path.exists() and not overwrite:
        return output_path

    full_image = Image.open(image_path).convert("RGB")
    crop_image = Image.open(crop_path).convert("RGB").resize((output_size, output_size), Image.Resampling.LANCZOS)
    src_w, src_h = full_image.size
    full_w = max(1, int(round(output_size * src_w / max(1, src_h))))
    full_resized = full_image.resize((full_w, output_size), Image.Resampling.LANCZOS)

    gutter = 18
    canvas_w = full_w + gutter + output_size
    canvas_h = output_size
    canvas = Image.new("RGB", (canvas_w, canvas_h), (16, 16, 16))
    canvas.paste(full_resized, (0, 0))
    canvas.paste(crop_image, (full_w + gutter, 0))

    draw = ImageDraw.Draw(canvas)
    font = load_font(24)
    small_font = load_font(18)
    full_color = (78, 180, 255)
    crop_color = (255, 210, 60)

    draw.rectangle((0, 0, full_w - 1, output_size - 1), outline=full_color, width=5)
    crop_x0 = full_w + gutter
    draw.rectangle((crop_x0, 0, crop_x0 + output_size - 1, output_size - 1), outline=crop_color, width=5)
    draw.rectangle((full_w, 0, full_w + gutter - 1, output_size - 1), fill=(16, 16, 16))

    left, top, right, bottom = crop_bbox_norm
    box = (
        int(round(left * full_w)),
        int(round(top * output_size)),
        int(round(right * full_w)),
        int(round(bottom * output_size)),
    )
    draw.rectangle(box, outline=crop_color, width=4)

    draw_boxed_label(draw, (14, 14), f"{panel_id} FULL PANEL", font, (255, 255, 255), (20, 70, 105))
    draw_boxed_label(
        draw,
        (14, 50),
        "coordinate_space=full_panel",
        small_font,
        (255, 255, 255),
        (20, 70, 105),
    )
    draw_boxed_label(
        draw,
        (crop_x0 + 14, 14),
        f"{panel_id} GAZE CROP",
        font,
        (20, 20, 20),
        (255, 210, 60),
    )
    draw_boxed_label(
        draw,
        (crop_x0 + 14, 50),
        "coordinate_space=gaze_crop",
        small_font,
        (20, 20, 20),
        (255, 210, 60),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    return output_path


def expand_panel_context_images(
    key: Tuple[str, int],
    image_paths: Sequence[Path],
    panel_meta: Sequence[Mapping[str, str]],
    model_input_dir: Path,
    args: argparse.Namespace,
) -> Tuple[List[Path], List[Dict[str, str]]]:
    if args.panel_context_mode == "full":
        return list(image_paths), [dict(item) for item in panel_meta]
    scene, row_index = key
    expanded_paths: List[Path] = []
    expanded_meta: List[Dict[str, str]] = []
    crop_dir = model_input_dir / "gaze_crops" / scene / f"row_{row_index}"
    paired_dir = model_input_dir / "paired_full_crop" / scene / f"row_{row_index}"
    for image_path, meta in zip(image_paths, panel_meta):
        panel_id = normalize_text(meta.get("panel_id"))
        gaze_u = parse_float(meta.get("gaze_u_norm")) if normalize_text(meta.get("gaze_projection_valid")) == "True" else None
        gaze_v = parse_float(meta.get("gaze_v_norm")) if normalize_text(meta.get("gaze_projection_valid")) == "True" else None
        crop_path, bbox, source = create_gaze_crop(
            image_path,
            crop_dir / f"{panel_id}_crop.jpg",
            gaze_u,
            gaze_v,
            float(args.gaze_crop_ratio),
            int(args.crop_output_size),
            args.overwrite,
        )
        crop_bbox_text = ",".join(f"{value:.4f}" for value in bbox)
        if args.panel_context_mode == "paired_crop":
            paired_path = create_paired_full_crop_image(
                image_path,
                crop_path,
                paired_dir / f"{panel_id}_paired.jpg",
                panel_id,
                bbox,
                int(args.crop_output_size),
                args.overwrite,
            )
            paired_meta = dict(meta)
            paired_meta.update(
                {
                    "image_role": "paired_full_crop",
                    "frame_path": str(paired_path),
                    "source_frame_path": normalize_text(meta.get("frame_path")),
                    "crop_path": str(crop_path),
                    "crop_source": source,
                    "crop_bbox_norm": crop_bbox_text,
                }
            )
            expanded_paths.append(paired_path)
            expanded_meta.append(paired_meta)
        else:
            full_meta = dict(meta)
            full_meta["image_role"] = "full"
            expanded_paths.append(image_path)
            expanded_meta.append(full_meta)
            expanded_paths.append(crop_path)
            expanded_meta.append(
                {
                    "panel_id": panel_id,
                    "image_role": "gaze_crop",
                    "video_frame_time": normalize_text(meta.get("video_frame_time")),
                    "json_sample_time": normalize_text(meta.get("json_sample_time")),
                    "frame_path": str(crop_path),
                    "source_frame_path": normalize_text(meta.get("frame_path")),
                    "crop_source": source,
                    "crop_bbox_norm": crop_bbox_text,
                }
            )
    return expanded_paths, expanded_meta


def load_qwen_module(repo_root: Path) -> Any:
    module_path = repo_root / "scripts" / "grounding" / "run_qwen3vl_local_keyframe_grounding.py"
    spec = importlib.util.spec_from_file_location("qwen3vl_local_grounding", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Qwen helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gt_referent_names(rows: Sequence[Mapping[str, str]]) -> List[str]:
    names: List[str] = []
    seen = set()
    for row in rows:
        name = normalize_text(row.get("referent_name"))
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def build_prompt(rows: Sequence[Mapping[str, str]], panel_meta: Sequence[Mapping[str, str]], prompt_mode: str) -> str:
    first = rows[0]
    instruction = normalize_text(first.get("instruction"))
    panel_items: List[Mapping[str, str]] = []
    seen_panels = set()
    for item in panel_meta:
        panel_id = normalize_text(item.get("panel_id"))
        if not panel_id or panel_id in seen_panels:
            continue
        seen_panels.add(panel_id)
        panel_items.append(item)
    panel_lines = [
        f"- {item['panel_id']}: video_time={item['video_frame_time']}s, json_sample_time={item['json_sample_time']}s"
        for item in panel_items
    ]
    image_lines: List[str] = []
    for index, item in enumerate(panel_meta, start=1):
        panel_id = normalize_text(item.get("panel_id"))
        role = normalize_text(item.get("image_role")) or "full"
        if role == "gaze_crop":
            bbox = normalize_text(item.get("crop_bbox_norm"))
            image_lines.append(f"- Image {index}: {panel_id} gaze-centered crop reference; crop_bbox_in_full_panel=[{bbox}]")
        elif role == "paired_full_crop":
            bbox = normalize_text(item.get("crop_bbox_norm"))
            image_lines.append(
                f"- Image {index}: {panel_id} paired image; LEFT side is the full panel; "
                f"RIGHT side is a gaze-centered crop; crop_bbox_in_full_panel=[{bbox}]"
            )
        else:
            image_lines.append(f"- Image {index}: {panel_id} full panel")
    referents = gt_referent_names(rows)
    if prompt_mode == "gt_referents" and referents:
        target_block = f"""Target referents to localize:
{chr(10).join(f"- {name}" for name in referents)}

This is an oracle localization run. Output exactly one entry for each target referent listed above. Do not add extra referents that are not listed. The labels are evaluation labels; use the instruction and visual evidence to map each label to the visible object or region."""
        task_lead = "For each listed target referent:"
    elif prompt_mode == "expected_count" and referents:
        target_block = f"""Expected number of output referents: {len(referents)}

Infer the target referents from the instruction. Do not use hidden dataset labels such as car1 or forklift2; describe each output with a short phrase from the instruction."""
        task_lead = "For each intended referent:"
    else:
        target_block = "Target referents to localize:\nInfer them from the instruction."
        task_lead = "Identify every concrete referent in the instruction that can be localized in one of the panels. For each referent:"
    return f"""Instruction:
{instruction}

{target_block}

Panels:
{chr(10).join(panel_lines)}

Image order:
{chr(10).join(image_lines)}

Task:
{task_lead}
- compare all panels before deciding; do not prefer P1 or the earliest image unless it is the best evidence;
- panel_id follows the image order and the Panels list above;
- choose the single best panel_id from the visible panels;
- output x_norm and y_norm as normalized coordinates inside the coordinate_space you choose;
- for a paired image, use coordinate_space="full_panel" if the point is on the LEFT full panel;
- for a paired image, use coordinate_space="gaze_crop" if the point is easier to localize on the RIGHT crop;
- do not use coordinates of the whole paired image canvas; choose either full_panel or gaze_crop;
- use the green gaze dot only as a noisy attention cue, together with the image and language;
- do not output the green dot unless it is actually on the intended referent;
- if a referent is an object part, surface, or placement area, output the best visible point on that part/area rather than the geometric center of the whole image.

Required strict JSON schema:
{{
  "referents": [
    {{
      "mention": "target referent label or short phrase",
      "panel_id": "P3",
      "coordinate_space": "full_panel",
      "x_norm": 0.0,
      "y_norm": 0.0,
      "confidence": 0.0
    }}
  ]
}}

Use null coordinates only when no reliable point can be localized. Return JSON only."""


def build_model_inputs(
    processor: Any,
    images: Sequence[Image.Image],
    prompt_text: str,
    panel_meta: Sequence[Mapping[str, str]],
    panel_caption_mode: str,
) -> Mapping[str, Any]:
    image_content: List[Dict[str, str]] = []
    for index, _image in enumerate(images):
        meta = panel_meta[index] if index < len(panel_meta) else {}
        panel_id = normalize_text(meta.get("panel_id")) or f"P{index + 1}"
        role = normalize_text(meta.get("image_role")) or "full"
        video_time = normalize_text(meta.get("video_frame_time"))
        json_time = normalize_text(meta.get("json_sample_time"))
        caption = f"Panel {panel_id} {role}"
        if video_time:
            caption += f", video_time={video_time}s"
        if json_time:
            caption += f", json_sample_time={json_time}s"
        if panel_caption_mode == "text":
            image_content.append({"type": "text", "text": caption})
        image_content.append({"type": "image"})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [*image_content, {"type": "text", "text": prompt_text}]},
    ]
    if hasattr(processor, "apply_chat_template"):
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return processor(text=[chat_text], images=list(images), return_tensors="pt")
    return processor(text=[SYSTEM_PROMPT + "\n\n" + prompt_text], images=list(images), return_tensors="pt")


def strip_json_line_comments(value: str) -> str:
    """Remove JSONC-style line comments without touching quoted strings."""
    output: List[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(value):
        char = value[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < len(value) and value[index + 1] == "/":
            index += 2
            while index < len(value) and value[index] not in "\r\n":
                index += 1
            continue
        output.append(char)
        index += 1
    return "".join(output)


def extract_json(raw_response: str) -> Optional[Dict[str, Any]]:
    candidates: List[str] = []
    stripped = raw_response.strip()
    if stripped:
        candidates.append(stripped)
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw_response, flags=re.DOTALL))
    candidates.extend(re.findall(r"\{.*\}", raw_response, flags=re.DOTALL))
    for candidate in candidates:
        cleaned = re.sub(r"^```(?:json)?", "", candidate.strip()).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        cleaned = strip_json_line_comments(cleaned)
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return normalize_prediction_payload(payload)
    return None


def normalize_prediction_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    raw_referents = payload.get("referents")
    if not isinstance(raw_referents, list):
        raw_referents = []
    referents: List[Dict[str, Any]] = []
    for item in raw_referents:
        if not isinstance(item, Mapping):
            continue
        panel_id = normalize_text(item.get("panel_id")).upper()
        if panel_id and not panel_id.startswith("P"):
            panel_id = "P" + panel_id
        sheet_x_norm = parse_float(item.get("sheet_x_norm"))
        sheet_y_norm = parse_float(item.get("sheet_y_norm"))
        x_norm = parse_float(item.get("x_norm"))
        y_norm = parse_float(item.get("y_norm"))
        confidence = parse_float(item.get("confidence"))
        coordinate_space = normalize_coordinate_space(
            item.get("coordinate_space")
            or item.get("point_source")
            or item.get("source")
            or item.get("space")
        )
        referents.append({
            "mention": normalize_text(item.get("mention")),
            "panel_id": panel_id,
            "coordinate_space": coordinate_space,
            "sheet_x_norm": sheet_x_norm if sheet_x_norm is not None and 0.0 <= sheet_x_norm <= 1.0 else None,
            "sheet_y_norm": sheet_y_norm if sheet_y_norm is not None and 0.0 <= sheet_y_norm <= 1.0 else None,
            "x_norm": x_norm if x_norm is not None and 0.0 <= x_norm <= 1.0 else None,
            "y_norm": y_norm if y_norm is not None and 0.0 <= y_norm <= 1.0 else None,
            "confidence": max(0.0, min(1.0, confidence)) if confidence is not None else None,
        })
    return {"referents": referents}


def normalize_coordinate_space(value: Any) -> str:
    text = normalize_text(value).lower().replace("-", "_").replace(" ", "_")
    if text in {"crop", "gaze_crop", "right_crop", "zoom", "zoom_crop", "cropped"}:
        return "gaze_crop"
    if text in {"full", "full_panel", "left_full", "left_panel", "panel", "image"}:
        return "full_panel"
    return ""


def parse_bbox_norm(value: Any) -> Optional[Tuple[float, float, float, float]]:
    parts = [parse_float(part) for part in normalize_text(value).split(",")]
    if len(parts) != 4 or any(part is None for part in parts):
        return None
    left, top, right, bottom = (float(part) for part in parts if part is not None)
    if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
        return None
    return left, top, right, bottom


def paired_canvas_regions(image_path: Path) -> Optional[Tuple[float, float]]:
    if not image_path.exists():
        return None
    with Image.open(image_path) as image:
        width, height = image.size
    crop_size = height
    gutter = 18
    full_width = width - crop_size - gutter
    if width <= 0 or full_width <= 0:
        return None
    full_end = full_width / width
    crop_start = (full_width + gutter) / width
    return full_end, crop_start


def map_prediction_coordinates(
    parsed: Optional[Mapping[str, Any]],
    panel_meta: Sequence[Mapping[str, str]],
    policy: str,
) -> Optional[Dict[str, Any]]:
    if parsed is None:
        return None
    if policy == "none":
        return dict(parsed)

    meta_by_panel = {normalize_text(item.get("panel_id")): item for item in panel_meta}
    mapped_referents: List[Dict[str, Any]] = []
    for ref in parsed.get("referents", []):
        if not isinstance(ref, Mapping):
            continue
        item = dict(ref)
        panel_id = normalize_text(item.get("panel_id"))
        meta = meta_by_panel.get(panel_id)
        x_norm = parse_float(item.get("x_norm"))
        y_norm = parse_float(item.get("y_norm"))
        coordinate_space = normalize_coordinate_space(item.get("coordinate_space"))
        item["raw_x_norm"] = x_norm
        item["raw_y_norm"] = y_norm
        item["raw_coordinate_space"] = coordinate_space
        item["coordinate_transform"] = "none"
        item["coordinate_space"] = "full_panel"

        if meta and x_norm is not None and y_norm is not None:
            bbox = parse_bbox_norm(meta.get("crop_bbox_norm"))
            regions = paired_canvas_regions(Path(normalize_text(meta.get("frame_path"))))
            if policy == "paired_canvas_map" and regions is not None:
                full_end, crop_start = regions
                if x_norm <= full_end:
                    item["x_norm"] = clamp_float(x_norm / full_end, 0.0, 1.0)
                    item["y_norm"] = clamp_float(y_norm, 0.0, 1.0)
                    item["coordinate_transform"] = "paired_canvas_left_to_full_panel"
                elif bbox is not None and x_norm >= crop_start:
                    crop_width = max(1e-6, 1.0 - crop_start)
                    crop_x = clamp_float((x_norm - crop_start) / crop_width, 0.0, 1.0)
                    left, top, right, bottom = bbox
                    item["x_norm"] = clamp_float(left + crop_x * (right - left), 0.0, 1.0)
                    item["y_norm"] = clamp_float(top + y_norm * (bottom - top), 0.0, 1.0)
                    item["coordinate_transform"] = "paired_canvas_right_crop_to_full_panel"
            elif coordinate_space == "gaze_crop" and bbox is not None:
                left, top, right, bottom = bbox
                item["x_norm"] = clamp_float(left + x_norm * (right - left), 0.0, 1.0)
                item["y_norm"] = clamp_float(top + y_norm * (bottom - top), 0.0, 1.0)
                item["coordinate_transform"] = "gaze_crop_to_full_panel"
            elif coordinate_space not in {"full_panel", "gaze_crop"}:
                if regions is not None:
                    full_end, crop_start = regions
                    if x_norm <= full_end:
                        item["x_norm"] = clamp_float(x_norm / full_end, 0.0, 1.0)
                        item["y_norm"] = clamp_float(y_norm, 0.0, 1.0)
                        item["coordinate_transform"] = "paired_canvas_left_to_full_panel"
                    elif bbox is not None and x_norm >= crop_start:
                        crop_width = 1.0 - crop_start
                        crop_x = clamp_float((x_norm - crop_start) / crop_width, 0.0, 1.0)
                        left, top, right, bottom = bbox
                        item["x_norm"] = clamp_float(left + crop_x * (right - left), 0.0, 1.0)
                        item["y_norm"] = clamp_float(top + y_norm * (bottom - top), 0.0, 1.0)
                        item["coordinate_transform"] = "paired_canvas_right_crop_to_full_panel"
        mapped_referents.append(item)
    return {"referents": mapped_referents}


def run_one_event(
    runner: Any,
    qwen_module: Any,
    image_paths: Sequence[Path],
    panel_meta: Sequence[Mapping[str, str]],
    prompt_text: str,
    panel_caption_mode: str,
    max_new_tokens: int,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    model_inputs = build_model_inputs(runner.processor, images, prompt_text, panel_meta, panel_caption_mode)
    model_inputs = qwen_module.move_batch_to_device(model_inputs, runner.device)
    with runner.runtime.torch.inference_mode():
        generated = runner.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    generated_ids = qwen_module.trim_generated_ids(model_inputs, generated)
    raw_response = qwen_module.decode_response(runner.processor, generated_ids).strip()
    return raw_response, extract_json(raw_response)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_json_dir = Path(args.output_json_dir).resolve()
    model_input_dir = Path(args.model_input_dir).resolve() if args.model_input_dir else output_json_dir / "model_inputs"
    output_json_dir.mkdir(parents=True, exist_ok=True)
    model_input_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_csv_rows(manifest_path)
    grouped = group_manifest(manifest_rows)
    selected = [(key, grouped[key]) for key in sorted(grouped, key=lambda item: (item[0], item[1])) if selected_key(key, args)]
    qwen_module = load_qwen_module(repo_root)
    runner = qwen_module.LocalQwen3VLRunner(
        model_name=args.model_name,
        dtype_name=args.dtype,
        use_flash_attn=args.use_flash_attn,
        max_new_tokens=args.max_new_tokens,
        local_files_only=args.local_files_only,
        input_mode="image",
        max_video_frames=1,
        ffmpeg_path=None,
        ffprobe_path=None,
        prompt_variant="debug",
        offload_folder=args.offload_folder,
    )
    runner.load()

    output_rows: List[Dict[str, Any]] = []
    for key, rows in selected:
        scene, row_index = key
        first = rows[0]
        event_id = normalize_text(first.get("event_id"))
        per_event_json = output_json_dir / f"{scene}_row_{row_index}.json"
        try:
            if args.input_mode == "contact_sheet":
                image_path, panel_meta = build_model_contact_sheet(
                    key,
                    rows,
                    model_input_dir,
                    (args.panel_width, args.panel_height),
                    args.columns,
                    args.overwrite,
                )
                image_paths = [image_path]
            else:
                image_paths, panel_meta = collect_model_panels(key, rows)
                image_paths, panel_meta = expand_panel_context_images(key, image_paths, panel_meta, model_input_dir, args)
                image_path = image_paths[0]
            prompt_text = build_prompt(rows, panel_meta, args.prompt_mode)
            coordinate_policy = args.paired_crop_coordinate_policy if args.panel_context_mode == "paired_crop" else "none"
            if per_event_json.exists() and not args.overwrite:
                payload = json.loads(per_event_json.read_text(encoding="utf-8"))
                raw_output = normalize_text(payload.get("model_raw_output"))
                raw_parsed = payload.get("raw_parsed_json")
                if not isinstance(raw_parsed, Mapping):
                    raw_parsed = payload.get("parsed_json") if isinstance(payload.get("parsed_json"), Mapping) else None
                parsed = map_prediction_coordinates(raw_parsed, panel_meta, coordinate_policy)
            else:
                raw_output, raw_parsed = run_one_event(
                    runner,
                    qwen_module,
                    image_paths,
                    panel_meta,
                    prompt_text,
                    args.panel_caption_mode,
                    args.max_new_tokens,
                )
                parsed = map_prediction_coordinates(raw_parsed, panel_meta, coordinate_policy)
                payload = {
                    "scene": scene,
                    "row_index": row_index,
                    "event_id": event_id,
                    "instruction": normalize_text(first.get("instruction")),
                    "input_mode": args.input_mode,
                    "panel_context_mode": args.panel_context_mode,
                    "panel_caption_mode": args.panel_caption_mode,
                    "paired_crop_coordinate_policy": coordinate_policy,
                    "model_input_image": ";".join(str(path) for path in image_paths),
                    "model_input_images": [str(path) for path in image_paths],
                    "prompt_text": prompt_text,
                    "panel_meta": list(panel_meta),
                    "model_raw_output": raw_output,
                    "raw_parsed_json": raw_parsed or {},
                    "parsed_json": parsed or {},
                    "parse_ok": raw_parsed is not None,
                }
                per_event_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            prediction_count = len(parsed.get("referents", [])) if isinstance(parsed, Mapping) else 0
            output_rows.append({
                "scene": scene,
                "row_index": row_index,
                "event_id": event_id,
                "instruction": normalize_text(first.get("instruction")),
                "model_input_image": ";".join(str(path) for path in image_paths),
                "model_raw_output": raw_output,
                "parsed_json": json.dumps(parsed or {}, ensure_ascii=False),
                "parse_ok": str(parsed is not None),
                "prediction_count": prediction_count,
                "error_message": "",
            })
            print(f"[ok] {scene} row_{row_index} predictions={prediction_count}")
        except Exception as exc:
            if not args.continue_on_error:
                raise
            output_rows.append({
                "scene": scene,
                "row_index": row_index,
                "event_id": event_id,
                "instruction": normalize_text(first.get("instruction")),
                "model_input_image": "",
                "model_raw_output": "",
                "parsed_json": "",
                "parse_ok": "False",
                "prediction_count": 0,
                "error_message": f"{type(exc).__name__}: {exc}",
            })
            print(f"[error] {scene} row_{row_index}: {exc}", file=sys.stderr)
    write_csv(output_csv, output_rows)
    print(f"Wrote predictions: {output_csv}")


if __name__ == "__main__":
    main()
