"""Visual-only loading must not read evaluation labels or another scene table."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from .context import vrtriref_sample_to_agent_context


class ContextIsolationTests(unittest.TestCase):
    def test_visual_only_context_does_not_load_evaluator_or_labels(self):
        api = SimpleNamespace(
            load_scene_anchor_table=Mock(return_value=[{
                'object_name': 'chair5', 'x_world': 1, 'y_world': 2, 'z_world': 3,
                'aliases': ['seat']}]),
            collect_timed_samples=Mock(return_value=[(.1, {})]))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            event = root / 'event.json'
            event.write_text('{}')
            telemetry = root / 'raw.json'
            telemetry.write_text('[{}]')
            paths = {'anchor': root / 'scene2_anchor_table.tsv',
                     'gt': root / 'labels_must_not_be_read.xlsx', 'input': root / 'input.csv'}
            row = {'_scene_key': 'scene2', '_row_index': 0, 'event_id': 'scene2_test',
                   'sample_id': 'sample', 'scene_id': 'scene2', 'instruction_text': 'Move the seat.',
                   'video_path': str(root / 'clip.mp4'), 'json_path': str(telemetry),
                   'event_json_path': str(event), 't_start': 0, 't_end': 1}
            with patch('MultiAgent.context.scene_paths', return_value=paths), \
                    patch('MultiAgent.context.module', return_value=api) as imported:
                ctx = vrtriref_sample_to_agent_context(row, load_ground_truth=False)
        imported.assert_called_once_with('api')
        api.load_scene_anchor_table.assert_called_once_with(paths['anchor'])
        self.assertEqual([candidate.candidate_id for candidate in ctx.candidates], ['chair5'])
        self.assertFalse(ctx.provenance['ground_truth_loaded'])
        self.assertTrue(ctx.gt['not_loaded_for_visual_only'])


if __name__ == '__main__':
    unittest.main()
