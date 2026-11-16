"""Run boundary checks: GT cannot be opened before prediction freezing."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_gaze_rgb_routing as run


class RoutingRunProtocolTests(unittest.TestCase):
    def manifest(self):
        return {'requests': [{'id': 'scene2_52_FULL_RGB'}, {'id': 'scene2_52_GAZE_RGB'}]}

    def test_missing_condition_blocks_evaluation_before_evaluator_import(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with patch.object(run, 'check_lock', return_value=self.manifest()), patch.object(run, 'module') as evaluator:
                with self.assertRaisesRegex(ValueError, 'every condition'):
                    run.evaluate(output)
                evaluator.assert_not_called()
                self.assertFalse((output / 'predictions_frozen.json').exists())

    def test_changed_frozen_response_blocks_gt_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / 'manifest.json').write_text('{}')
            (output / 'responses').mkdir()
            for request in self.manifest()['requests']:
                (output / 'responses' / (request['id'] + '.json')).write_text('{"prediction":[]}')
            with patch.object(run, 'check_lock', return_value=self.manifest()):
                run.freeze_predictions(output)
            frozen = json.loads((output / 'predictions_frozen.json').read_text())
            self.assertTrue(frozen['before_gt_evaluation'])
            (output / 'responses' / 'scene2_52_GAZE_RGB.json').write_text('{"prediction":["changed"]}')
            with patch.object(run, 'check_lock', return_value=self.manifest()), patch.object(run, 'module') as evaluator:
                with self.assertRaisesRegex(ValueError, 'changed after freeze'):
                    run.evaluate(output)
                evaluator.assert_not_called()

    def test_runtime_source_change_blocks_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / 'manifest.json').write_text(json.dumps({'source_sha256': {'runner': 'old'}}))
            (output / 'protocol_lock.json').write_text(json.dumps({'manifest_sha256': run.sha(output / 'manifest.json')}))
            with patch.object(run, 'source_lock', return_value={'runner': 'new'}):
                with self.assertRaisesRegex(ValueError, 'Runtime sources'):
                    run.check_lock(output)


if __name__ == '__main__':
    unittest.main()
