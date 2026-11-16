import unittest

from tools.report_projected2d_routed_rgb import paired_delta


class PairedReportTests(unittest.TestCase):
    def test_identical_counts_have_zero_delta_and_interval(self):
        counts = [[1, 0, 1], [0, 1, 1], [0, 0, 0]]
        result = paired_delta(counts, counts, draws=100)
        self.assertEqual(result['difference'], 0.)
        self.assertEqual(result['bootstrap_95_percentile'], [0., 0.])

    def test_perfect_match_beats_empty_prediction(self):
        result = paired_delta([[1, 0, 0]], [[0, 0, 1]], draws=100)
        self.assertEqual(result['difference'], 1.)
        self.assertEqual(result['bootstrap_95_percentile'], [1., 1.])


if __name__ == '__main__':
    unittest.main()
