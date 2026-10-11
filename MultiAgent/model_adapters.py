"""Optional adapters for the local models already used by the repository.

These adapters are deliberately lazy: importing the package does not load a multi-billion
parameter checkpoint. The closed-set pipeline remains runnable with recorded telemetry alone.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


class ExistingQwenPredictionIndex:
    """Scene-qualified keys and the COMPLETE baseline referent set."""
    def __init__(self, data_dir, suffix="_local_3d_outputs_qwen3vl30b_mention_first_v3"):
        from .baseline import module
        ev = module('eval')
        self.predictions = {}
        for directory in sorted(Path(data_dir).glob(f"*{suffix}")):
            scene = directory.name[:-len(suffix)]
            for path in sorted(directory.glob('row_*.json')):
                row = json.loads(path.read_text())
                self.predictions[f"{scene}/{row['event_id']}"] = ev.find_prediction_labels(row)


class QwenCoordinator:
    """Prompt/parse adapter for a future Qwen3-VL structured coordinator call.

    The prompt is deterministic and carries only closed-set names and evidence. A caller may
    connect it to the repository's local Qwen runner; no model is loaded during construction.
    """
    def build_prompt(self, context: Any, evidence: list[Mapping[str, Any]]) -> str:
        names = [c.object_name for c in context.candidates]
        payload = {"event_id": context.event_id, "instruction": context.text, "candidates": names, "evidence": evidence}
        return ("You are a coordinator for closed-set VR referent selection. Return JSON only. "
                "Select one or more names from candidates; never invent a name.\n" + json.dumps(payload, ensure_ascii=False))

    @staticmethod
    def parse(text: str, candidate_names: list[str]) -> Dict[str, Any]:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match: return {"selected_object_names": [], "parse_error": "no_json"}
        try: data = json.loads(match.group(0))
        except json.JSONDecodeError as exc: return {"selected_object_names": [], "parse_error": str(exc)}
        selected = data.get("selected_object_names") or ([data.get("selected_object_name")] if data.get("selected_object_name") else [])
        allowed = {x.lower(): x for x in candidate_names}
        selected = [allowed[x.lower()] for x in selected if isinstance(x, str) and x.lower() in allowed]
        data["selected_object_names"] = selected
        data["selected_object_name"] = selected[0] if selected else ""
        return data


def run_existing_qwen_single_event(*, input_csv: str, row_index: int, output_json: str, anchor_table: str,
                                   model_path: str, offload_folder: str | None = None, prompt_strategy: str = "mention_first") -> Dict[str, Any]:
    """Invoke the repository's proven local runner for one real event (opt-in and expensive)."""
    from .baseline import ROOT
    root = ROOT
    script = root / "scripts/grounding/run_qwen3vl_local_single_event_3d.py"
    cmd = [sys.executable, str(script), "--input_csv", input_csv, "--row_index", str(row_index),
           "--output_json", output_json, "--scene_anchor_csv", anchor_table, "--model_name", model_path,
           "--prompt_strategy", prompt_strategy, "--local_files_only"]
    if offload_folder: cmd += ["--offload_folder", offload_folder]
    subprocess.run(cmd, cwd=root, check=True)
    return json.loads(Path(output_json).read_text(encoding="utf-8"))
