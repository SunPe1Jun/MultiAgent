"""Focused contract tests for 2D RGB routing."""
import unittest

from .projected2d_attention_study import evaluator_adapter, normalize_model_mentions
from .projected2d_routed_rgb import prompt_for


class Routed2DTests(unittest.TestCase):
    def setUp(self):
        self.event = {'instruction': 'Move the chair, then inspect its seat.',
            'mentions': [{'mention_id': 'm0', 'text': 'the chair', 'object_type': 'chair',
                          'coreference_id': 'r0'},
                         {'mention_id': 'm1', 'text': 'its seat', 'component': 'seat',
                          'coreference_id': 'r0'}],
            'temporal_hints': {'m0': {'soft_phase_range': [0, 2], 'soft_phase_panels': ['P1']}},
            'panels': [{'frame_id': 'P1', 'requested_video_time': 1., 'display_size': [512, 512],
                        'boxes': {'GAZE_RGB': [[0, 0, 512, 512]] * 3}}]}

    def test_prompt_is_visual_and_preserves_soft_time_and_coreference(self):
        prompt = prompt_for(self.event, 'GAZE_RGB')
        self.assertIn('full RGB source frame', prompt)
        self.assertIn('all panels', prompt)
        self.assertIn('soft_phase_panels', prompt)
        self.assertIn('coreference_id', prompt)
        self.assertNotIn('candidate_ids', prompt)
        self.assertNotIn('gt_referent', prompt)
        self.assertNotIn('anchor', prompt)

    def test_original_evaluator_adapter_retains_points_and_empty_mentions(self):
        raw = {'mentions': [{'mention_id': 'm0', 'status': 'localized',
            'predictions': [{'frame_id': 'P1', 'point_uv': [.2, .3]}]},
            {'mention_id': 'm1', 'status': 'not_visible', 'predictions': []}]}
        parsed, warning = normalize_model_mentions(raw, {'mentions': self.event['mentions']})
        self.assertFalse(warning)
        self.assertEqual(evaluator_adapter(parsed)['referents'], [{'mention': 'the chair',
            'panel_id': 'P1', 'coordinate_space': 'full_panel', 'x_norm': .2, 'y_norm': .3}])


if __name__ == '__main__':
    unittest.main()
