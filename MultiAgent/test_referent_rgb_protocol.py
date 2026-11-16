from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from MultiAgent.heatmap_temporal_pilot import save, sha
from tools import run_referent_rgb_expansion as run


class ExpandedProtocolTests(unittest.TestCase):
    def test_missing_conditions_block_gt_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            m = {'requests': [{'id': 'unfinished'}]}
            with patch.object(run, 'check_study', return_value=m), patch('MultiAgent.baseline.module') as module:
                with self.assertRaisesRegex(ValueError, 'all selected conditions'):
                    run.evaluate(output)
                module.assert_not_called()
                self.assertFalse((output/'predictions_frozen.json').exists())

    def test_response_freeze_precedes_first_evaluator_call(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save(output/'manifest.json', {})
            save(output/'responses'/'one.json', {})
            m = {'requests': [{'id': 'one'}]}
            def stop(name):
                self.assertTrue((output/'predictions_frozen.json').exists())
                self.assertEqual(run.read(output/'predictions_frozen.json')['response_sha256']['one'],
                                 sha(output/'responses'/'one.json'))
                raise RuntimeError('stopped before GT access')
            with patch.object(run, 'check_study', return_value=m), patch('MultiAgent.baseline.module', side_effect=stop):
                with self.assertRaisesRegex(RuntimeError, 'stopped before GT'):
                    run.evaluate(output)

    def test_interrupted_png_cache_is_rebuilt_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root/'source.png', root/'labelled.png'
            Image.new('RGB', (200, 256), 'green').save(source)
            target.write_bytes(b'')
            run.save_labelled_input(source, target, {'frame_id': 'V001', 'time_seconds': 1.})
            with Image.open(target) as image:
                image.load()
                self.assertEqual(image.size, (200, 256))
                self.assertEqual(image.getpixel((0, 28)), (0, 128, 0))
            self.assertFalse(target.with_name(target.name+'.pending.png').exists())
            previous = sha(target)
            run.save_labelled_input(source, target, {'frame_id': 'V001', 'time_seconds': 1.})
            self.assertEqual(sha(target), previous)

    def test_inference_requires_complete_matching_input_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save(output/'manifest.json', {})
            manifest = {'events': [{'frames': [1]}], 'conditions': ['one', 'two']}
            with self.assertRaisesRegex(ValueError, 'required before inference'):
                run.require_verified_inputs(output, manifest)
            record = {'manifest_sha256': sha(output/'manifest.json'),
                      'all_composite_pixels_rebuilt_from_rgb': 2,
                      'full_rgb_exact_pixel_matches': 2, 'model_heatmap_pixels': 0}
            save(output/'input_verification.json', record)
            run.require_verified_inputs(output, manifest)
            save(output/'input_verification.json', {**record, 'full_rgb_exact_pixel_matches': 1})
            with self.assertRaisesRegex(ValueError, 'does not match'):
                run.require_verified_inputs(output, manifest)

    def test_source_changes_block_use_of_frozen_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            save(output/'manifest.json', {'source_sha256': {'one': 'old'}})
            save(output/'protocol_lock.json', {'manifest_sha256':sha(output/'manifest.json')})
            with patch.object(run, 'study_sources', return_value={'one':'new'}):
                with self.assertRaisesRegex(ValueError, 'Runtime source'):
                    run.check_study(output)


if __name__ == '__main__':
    unittest.main()
