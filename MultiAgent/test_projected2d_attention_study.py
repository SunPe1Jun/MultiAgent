"""Protocol checks independent of model accuracy and CUDA availability."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from .projected2d_attention_study import (
    CSV_COLUMNS, TRIREF, audit_annotation_parity, build_visual_prompt, evaluator_adapter, normalize_model_mentions,
    read_csv, write_csv,
)


class Projected2DProtocolTests(unittest.TestCase):
    def test_inherited_annotation_audit_records_coordinate_changes(self):
        source = {"scene": "scene1", "row_index": "705", "referent_name": "target",
                  "panel_id": "P1", "gt_x": "200.00", "gt_y": "300.00",
                  "projection_valid": "True", "status": "point_evaluable"}
        rebuilt = {**source, "gt_x": "201.00"}
        audit = audit_annotation_parity([source], [rebuilt])
        self.assertFalse(audit["all_original_projected_gt_fields_identical"])
        self.assertEqual(len(audit["mismatches"]), 1)
        self.assertEqual(source["gt_x"], "200.00")

    def test_instruction_and_source_frame_protocol_reach_prompt(self):
        text = "Move those two trucks beside that building."
        mention = {"mention_id": "m0", "text": "those two trucks", "count": 2,
                   "object_type": "truck", "coreference_id": "r7", "anchor_text": "those two trucks",
                   "candidate_ids": ["SECRET_ANCHOR_ID"], "gt_point": [0.12345, 0.98765]}
        panels = [{"frame_id": "P1", "source_frame_id": "P1.jpg",
                   "video_time_seconds": 3.25, "crop_bbox_norm": "0.1,0.2,0.5,0.6"}]
        prompt = build_visual_prompt(SimpleNamespace(text=text), [mention], panels)
        self.assertIn(text, prompt)
        self.assertIn('"video_time_seconds": 3.25', prompt)
        self.assertIn('"object_type": "truck"', prompt)
        self.assertIn('"coreference_id": "r7"', prompt)
        self.assertIn('"anchor_text": "those two trucks"', prompt)
        self.assertIn("0.1,0.2,0.5,0.6", prompt)
        self.assertNotIn("SECRET_ANCHOR_ID", prompt)
        self.assertNotIn("0.12345", prompt)

    def test_plural_predictions_and_missing_mentions_survive_adapter(self):
        referential = {"mentions": [{"mention_id": "m0", "text": "two trucks"},
                                   {"mention_id": "m1", "text": "there"}]}
        response = {"mentions": [{"mention_id": "m0", "status": "localized",
            "predictions": [{"frame_id": "P1", "point_uv": [0.2, 0.3]},
                            {"frame_id": "P3", "point_uv": [0.7, 0.6]}], "evidence": "two visible vehicles"}]}
        parsed, warning = normalize_model_mentions(response, referential)
        self.assertEqual(len(evaluator_adapter(parsed)["referents"]), 2)
        self.assertEqual(parsed["mentions"][1]["predictions"], [])
        self.assertIn("missing_mention:m1", warning)

    def test_original_evaluator_keeps_sparse_high_row_indices(self):
        source = Path(__file__).resolve().parents[1] / "runs/projected2d_attention_qwen8b_20261022/manifest_all.csv"
        if not source.exists():
            self.skipTest("Prepared original manifest unavailable")
        manifest = read_csv(source)
        keys = {(x["scene"], x["row_index"]) for x in manifest}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_csv(tmp / "manifest.csv", manifest, list(manifest[0]))
            predictions = [{"scene": scene, "row_index": index, "event_id": "",
                "instruction": "", "model_input_image": "", "model_raw_output": "",
                "parsed_json": '{"referents":[]}', "parse_ok": "false",
                "prediction_count": 0, "error_message": "test empty predictions"} for scene,index in keys]
            write_csv(tmp / "pred.csv", predictions, CSV_COLUMNS)
            completed = subprocess.run([sys.executable, str(TRIREF / "exam2/evaluate_2d_point_grounding.py"),
                "--manifest", str(tmp / "manifest.csv"), "--pred_csv", str(tmp / "pred.csv"),
                "--output_dir", str(tmp / "eval"), "--coordinate_mode", "panel"], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            metrics = json.loads((tmp / "eval/2d_eval_summary.json").read_text())
            self.assertEqual(metrics["events"], len(keys))
            self.assertGreater(metrics["point_fn_100"], 0)
            self.assertEqual(metrics["predictions"], 0)


if __name__ == "__main__":
    unittest.main()
