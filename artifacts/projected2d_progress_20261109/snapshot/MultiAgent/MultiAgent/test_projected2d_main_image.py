import copy
import math
import unittest

from .projected2d_main_image import normalize, prompt_for
from .qwen_backend import _image_content_with_captions


class MainImageContractTests(unittest.TestCase):
    def setUp(self):
        self.event = {'instruction': 'Move that chair near the table.',
            'mentions': [{'mention_id': 'm0', 'text': 'that chair', 'coreference_id': 'r0'}],
            'temporal_hints': {'m0': {'soft_phase_range': [1., 4.]}},
            'panels': [{'frame_id': 'P2', 'requested_video_time': 2.},
                       {'frame_id': 'P3', 'requested_video_time': 4.}]}

    def payload(self, point, frame='P2', status='localized'):
        return {'mentions': [{'mention_id': 'm0', 'status': status,
                              'predictions': [{'frame_id': frame, 'point_xy': point}]}]}

    def test_direct_frame_coordinate_is_resolution_independent(self):
        for size in ([512, 512], [768, 768], [1920, 1080]):
            event = copy.deepcopy(self.event)
            event['panels'][0]['source_size'] = size
            parsed, errors, conversions = normalize(self.payload([500, 400]), event)
            self.assertEqual(errors, [])
            self.assertEqual(parsed['mentions'][0]['predictions'][0]['point_uv'], [.5, .4])
            self.assertEqual(conversions[0]['source_uv'], [.5, .4])

    def test_frame_missing_from_input_is_rejected(self):
        parsed, errors, _ = normalize(self.payload([500, 400], frame='P1'), self.event)
        self.assertEqual(parsed['mentions'][0]['predictions'], [])
        self.assertTrue(any('not supplied' in x for x in errors))

    def test_invalid_coordinates_are_rejected(self):
        for xy in ([True, 400], [math.nan, 400], [400, math.inf], [-1, 400],
                   [1001, 400], ['500', 400], [400], None):
            parsed, errors, _ = normalize(self.payload(xy), self.event)
            self.assertEqual(parsed['mentions'][0]['predictions'], [])
            self.assertTrue(errors)

    def test_uncertainty_cannot_produce_scored_points(self):
        parsed, errors, _ = normalize(self.payload([500, 400], status='ambiguous'), self.event)
        self.assertEqual(parsed['mentions'][0]['predictions'], [])
        self.assertTrue(errors)

    def test_duplicate_mentions_not_counted_twice(self):
        row = self.payload([500, 400])['mentions'][0]
        parsed, errors, _ = normalize({'mentions': [row, row]}, self.event)
        self.assertEqual(len(parsed['mentions']), 1)
        self.assertEqual(parsed['mentions'][0]['predictions'], [])
        self.assertTrue(any('duplicate_mention' in x for x in errors))

    def test_missing_mentions_remain_explicit(self):
        parsed, errors, _ = normalize({}, self.event)
        self.assertEqual(parsed['mentions'][0]['mention_id'], 'm0')
        self.assertEqual(parsed['mentions'][0]['status'], 'ambiguous')
        self.assertTrue(errors)

    def test_prompt_preserves_task_and_hints_without_collage_or_gt(self):
        text = prompt_for(self.event)
        self.assertIn('that chair', text)
        self.assertIn('soft_phase_range', text)
        self.assertIn('0..1000', text)
        self.assertIn('ENTIRE RGB image', text)
        for forbidden in ('composite_xy', 'quadrant', 'expected_count', 'gt_x', 'candidate_id'):
            self.assertNotIn(forbidden, text)


class CaptionedImageContentTests(unittest.TestCase):
    def test_default_content_stays_unchanged(self):
        images = [{'type': 'image'}, {'type': 'image'}]
        self.assertEqual(_image_content_with_captions(images), images)

    def test_caption_is_adjacent_to_its_actual_frame(self):
        images = [{'type': 'image', 'key': 'a'}, {'type': 'image', 'key': 'b'}]
        content = _image_content_with_captions(images, ['Frame P2', 'Frame P3'])
        self.assertEqual([p['type'] for p in content], ['text', 'image', 'text', 'image'])
        self.assertEqual(content[0]['text'], 'Frame P2')
        self.assertEqual(content[1], images[0])
        self.assertEqual(content[2]['text'], 'Frame P3')
        self.assertEqual(content[3], images[1])

    def test_malformed_caption_mapping_fails(self):
        for captions in ([], ['P1', 'P2'], [None], ['']):
            with self.assertRaises(ValueError):
                _image_content_with_captions([{'type': 'image'}], captions)


if __name__ == '__main__':
    unittest.main()
