"""Checks for causal time controls, fallback and missing-prediction denominators."""
import unittest

import numpy as np
from PIL import Image

from .heatmap_temporal_pilot import render_temporal, score_response, temporal_maps


class TemporalPilotTests(unittest.TestCase):
    def test_reverse_time_keeps_alpha_and_spatial_mass(self):
        rows = [{'status': 'projected_frustum_only', 'gaze_time_seconds': t,
                 'pixel_xy': [x, 20.], 'weight_seconds': .1}
                for t, x in [(0.1, 10.), (1., 30.), (1.9, 50.)]]
        maps = temporal_maps(rows, 2., 2., (64, 40))
        rgb = Image.new('RGB', (64, 40), (100, 100, 100))
        normal, alpha = render_temporal(rgb, maps)
        reverse, reverse_alpha = render_temporal(rgb, maps, reverse=True)
        np.testing.assert_array_equal(alpha, reverse_alpha)
        self.assertFalse(np.array_equal(normal, reverse))
        future = [{**rows[0], 'gaze_time_seconds': 2.01}]
        with self.assertRaises(ValueError):
            temporal_maps(future, 2., 2., (64, 40))

    def test_zero_maps_preserve_rgb(self):
        image = Image.new('RGB', (32, 40), (11, 87, 235))
        for reverse in (True, False):
            rendered, alpha = render_temporal(image, np.zeros((3, 40, 32), np.float32), reverse)
            np.testing.assert_array_equal(rendered, image)
            self.assertEqual(float(alpha.sum()), 0.)

    def test_missing_or_duplicate_frame_is_a_failure_not_excluded(self):
        frame = {'frame_id': 'V001', 'motion_bin': 'high_ge45', 'clock_supported': True,
                 'conditions': {'ORDERED_HEATMAP': {'image_size': [448, 480],
                    'expected_focus_xy': [224., 240.], 'expected_older_xy': [50., 240.],
                    'expected_newer_xy': [350., 240.]}}}
        event = {'frames': [frame]}
        scored, valid = score_response({'frames': []}, event, 'ORDERED_HEATMAP')
        self.assertFalse(valid)
        self.assertTrue(scored[0]['focus_xy']['eligible'])
        self.assertFalse(scored[0]['focus_xy']['success_at_10pct'])
        answer = {'frame_id': 'V001', 'focus_xy': [500, 500]}
        scored, valid = score_response({'frames': [answer, answer]}, event, 'ORDERED_HEATMAP')
        self.assertFalse(valid)
        self.assertFalse(scored[0]['focus_xy']['success_at_10pct'])


if __name__ == '__main__':
    unittest.main()
