from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict

from PIL import Image, ImageDraw, ImageFont

from .context import AgentContext


def _frame(video: str, timestamp: float, out: Path) -> bool:
    if not video or not Path(video).exists(): return False
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-ss", str(max(0, timestamp)), "-i", video, "-frames:v", "1", "-q:v", "2", str(out)]
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0 and out.exists()


def render_debug(ctx: AgentContext, result: Dict[str, Any], output_path: str | Path) -> str:
    output_path = Path(output_path)
    frame_path = output_path.with_suffix(".frame.jpg")
    ts = float((ctx.event.get("time_window") or {}).get("t_peak", 0.0) or 0.0)
    ok = _frame(ctx.video_path, ts, frame_path)
    if ok:
        image = Image.open(frame_path).convert("RGB")
        image.thumbnail((1100, 700))
    else:
        image = Image.new("RGB", (1100, 700), (30, 30, 30))
    canvas = Image.new("RGB", (1100, image.height + 250), (20, 20, 20)); canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas); font = ImageFont.load_default()
    lines = [f"event={ctx.event_id}  t_peak={ts:.3f}s  frame={'yes' if ok else 'unavailable'}",
             f"instruction: {ctx.text}", f"prediction: {result.get('selected_object_name','')}  GT: {','.join(ctx.gt_referents_mapped)}"]
    ranked = result.get("ranked_candidates", [])[:8]
    lines.append("scores: " + " | ".join(f"{x['candidate_id']}={x['score']:.3f}" for x in ranked))
    spatial = ctx.peak_spatial
    lines.append(f"gaze_hit={spatial.get('camera_hit_point')} hand_hit={spatial.get('right_index_ray_hit_point')}")
    for i, line in enumerate(lines): draw.text((12, image.height + 12 + i * 28), line[:180], fill=(240, 240, 240), font=font)
    canvas.save(output_path)
    return str(output_path)
