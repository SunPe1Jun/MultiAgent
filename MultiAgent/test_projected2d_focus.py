import unittest

from .projected2d_focus import focus_event, prompt_for


class FocusedProtocolTests(unittest.TestCase):
    def setUp(self):
        self.mention = {'mention_id': 'm1', 'text': 'the chair'}
        self.event = {'instruction': 'Move the chair.', 'mentions': [self.mention],
            'temporal_hints': {'m1': {'soft_phase_range': [8, 10]}},
            'panels': [{'frame_id': f'P{i}', 'requested_video_time': time,
                        'display_size': [512, 512],
                        'boxes': {'GAZE_RGB': [[0, 0, 512, 512]] * 3}}
                       for i, time in [(1, 2), (2, 6), (3, 9)]]}

    def test_phase_promotes_nearest_but_preserves_all_panels(self):
        focused, hint = focus_event(self.event, self.mention)
        self.assertEqual([p['frame_id'] for p in focused['panels']], ['P3', 'P2', 'P1'])
        self.assertEqual([p['frame_id'] for p in self.event['panels']], ['P1', 'P2', 'P3'])
        self.assertEqual(hint['soft_phase_range'], [8, 10])

    def test_prompt_identifies_nonchronological_order_and_single_mention(self):
        focused, hint = focus_event(self.event, self.mention)
        prompt = prompt_for(focused, self.mention, hint, 'GAZE_RGB')
        self.assertIn('out of chronological order', prompt)
        self.assertIn('ONE current mention', prompt)
        self.assertIn('ENTIRE composite', prompt)
        self.assertNotIn('candidate_ids', prompt)
        self.assertNotIn('gt_x', prompt)


if __name__ == '__main__':
    unittest.main()
