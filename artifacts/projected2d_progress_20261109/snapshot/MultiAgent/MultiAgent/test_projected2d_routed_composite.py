import unittest

from .projected2d_routed_composite import composite_to_source, normalize, prompt_for


class CompositeCoordinatesTests(unittest.TestCase):
    def setUp(self):
        self.panel = {'frame_id': 'P1', 'display_size': [512, 512],
            'requested_video_time': 3., 'boxes': {'GAZE_RGB': [[100, 120, 300, 320],
                [0, 0, 512, 512], [0, 100, 512, 300]]}}
        self.event = {'instruction': 'Move the chair.', 'mentions': [{'mention_id': 'm0', 'text': 'the chair'}],
                      'temporal_hints': {}, 'panels': [self.panel]}

    def xy(self, x, y):
        return [x/1024*1000, y/1080*1000]

    def test_full_source_center_maps_exactly(self):
        uv, area = composite_to_source(self.panel, 'GAZE_RGB', self.xy(256, 28+256))
        self.assertEqual(area, 0)
        self.assertAlmostEqual(uv[0], .5)
        self.assertAlmostEqual(uv[1], .5)

    def test_detail_center_maps_to_source_crop_center(self):
        uv, area = composite_to_source(self.panel, 'GAZE_RGB', self.xy(512+256, 28+256))
        self.assertEqual(area, 1)
        self.assertAlmostEqual(uv[0], 200/512)
        self.assertAlmostEqual(uv[1], 220/512)

    def test_header_and_detail_letterbox_are_rejected(self):
        with self.assertRaises(ValueError):
            composite_to_source(self.panel, 'GAZE_RGB', self.xy(100, 5))
        with self.assertRaises(ValueError):
            composite_to_source(self.panel, 'GAZE_RGB', self.xy(512+100, 540+28+20))

    def test_normalize_accepts_supplied_frame_and_rejects_unknown(self):
        row = {'mention_id': 'm0', 'status': 'localized', 'predictions': [
            {'frame_id': 'P1', 'composite_xy': self.xy(256, 28+256)},
            {'frame_id': 'P3', 'composite_xy': self.xy(256, 28+256)}]}
        parsed, errors, conversions = normalize({'mentions': [row]}, self.event, 'GAZE_RGB')
        self.assertEqual(len(parsed['mentions'][0]['predictions']), 1)
        self.assertTrue(any('panel was not supplied' in e for e in errors))
        self.assertEqual(len(conversions), 1)

    def test_duplicate_mention_rows_are_not_scored_twice(self):
        row = {'mention_id': 'm0', 'status': 'localized', 'predictions': [
            {'frame_id': 'P1', 'composite_xy': self.xy(256, 28+256)}]}
        parsed, errors, conversions = normalize({'mentions': [row, row]}, self.event, 'GAZE_RGB')
        self.assertEqual(parsed['mentions'][0]['predictions'], [])
        self.assertTrue(any('duplicate_mention' in e for e in errors))
        self.assertEqual(conversions, [])

    def test_uncertain_row_cannot_smuggle_a_point(self):
        row = {'mention_id': 'm0', 'status': 'ambiguous', 'predictions': [
            {'frame_id': 'P1', 'composite_xy': self.xy(256, 28+256)}]}
        parsed, errors, conversions = normalize({'mentions': [row]}, self.event, 'GAZE_RGB')
        self.assertEqual(parsed['mentions'][0]['predictions'], [])
        self.assertTrue(any('non-localized' in e for e in errors))
        self.assertEqual(conversions, [])

    def test_prompt_specifies_composite_and_does_not_include_gt_or_scene_ids(self):
        text = prompt_for(self.event, 'GAZE_RGB')
        self.assertIn('ENTIRE composite', text)
        self.assertIn('script handles', text)
        self.assertNotIn('candidate_ids', text)
        self.assertNotIn('gt_x', text)


if __name__ == '__main__':
    unittest.main()
