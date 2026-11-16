import unittest

from .projected2d_exact_set import collapse_explicit_coreference, exact_counts


class ExactSetTests(unittest.TestCase):
    def setUp(self):
        self.mentions = [{'mention_id': 'm0', 'coreference_id': 'r0', 'count': 1},
                         {'mention_id': 'm1', 'coreference_id': 'r0', 'parent_mention_id': 'm0', 'count': 1}]
        self.panels = [{'frame_id': 'P1', 'source_size': [1920, 1920]},
                       {'frame_id': 'P2', 'source_size': [1920, 1920]}]
        self.parsed = {'mentions': [{'mention_id': 'm0', 'predictions': [{'frame_id': 'P1', 'point_uv': [.3, .4]}]},
                                    {'mention_id': 'm1', 'predictions': [{'frame_id': 'P1', 'point_uv': [.31, .4]}]}]}

    def test_same_frame_explicit_parent_collapses_without_mutating_input(self):
        result, merges = collapse_explicit_coreference(self.parsed, self.mentions, self.panels)
        self.assertEqual(len(merges), 1)
        self.assertEqual(result['mentions'][1]['predictions'], [])
        self.assertEqual(len(self.parsed['mentions'][1]['predictions']), 1)

    def test_shared_coreference_alone_or_separate_frame_is_not_enough(self):
        self.mentions[1].pop('parent_mention_id')
        self.assertEqual(collapse_explicit_coreference(self.parsed, self.mentions, self.panels)[1], [])
        self.mentions[1]['parent_mention_id'] = 'm0'
        self.parsed['mentions'][1]['predictions'][0]['frame_id'] = 'P2'
        self.assertEqual(collapse_explicit_coreference(self.parsed, self.mentions, self.panels)[1], [])

    def test_far_or_plural_points_remain_separate(self):
        self.parsed['mentions'][1]['predictions'][0]['point_uv'] = [.5, .4]
        self.assertEqual(collapse_explicit_coreference(self.parsed, self.mentions, self.panels)[1], [])
        self.parsed['mentions'][1]['predictions'][0]['point_uv'] = [.31, .4]
        self.mentions[1]['count'] = 2
        self.assertEqual(collapse_explicit_coreference(self.parsed, self.mentions, self.panels)[1], [])

    def test_exact_excludes_unscorable_events_and_reports_denominator(self):
        row = {'gt_referent_count': '1', 'time_evaluable_referent_count': '1'}
        for kind in ('point', 'joint'):
            for threshold in (50, 100, 150, 200):
                row[f'{kind}_fp_{threshold}'] = '0'
                row[f'{kind}_fn_{threshold}'] = '0'
        empty = {**row, 'gt_referent_count': '0', 'time_evaluable_referent_count': '0'}
        self.assertEqual(exact_counts([row, empty])['joint_exact_100'],
                         {'exact': 1, 'evaluable_events': 1, 'all_events': 2})


if __name__ == '__main__':
    unittest.main()
