"""Checks for source integrity and nonsemantic RGB evidence routing."""
import json
import unittest

import numpy as np
from PIL import Image

from .gaze_rgb_routing import (
    CONDITIONS, HEADER, RoutingConfig, grounding_prompt, render_rgb_evidence, route_regions, literal_mention_hints,
)
from .full_frame_grounding_v4 import validate_mention_contract
from . import test_full_frame_grounding_v4 as contract_tests


class GazeRGBRoutingTests(unittest.TestCase):
    def maps(self):
        maps = np.zeros((3, 80, 120), dtype=float)
        maps[0, 25, 15] = .4
        maps[1, 40, 60] = .4
        maps[2, 55, 105] = .4
        return maps

    def test_separated_history_keeps_multiple_regions_in_source_bounds(self):
        route = route_regions(self.maps(), (120, 80), clock_supported=True)
        boxes = [slot['bbox_source_xyxy'] for slot in route['slots']]
        self.assertEqual(len(set(map(tuple, boxes))), 3)
        for slot, point in zip(route['slots'], [(15, 25), (60, 40), (105, 55)]):
            x0, y0, x1, y1 = slot['bbox_source_xyxy']
            self.assertTrue(0 <= x0 < x1 <= 120 and 0 <= y0 < y1 <= 80)
            self.assertTrue(x0 <= point[0] < x1 and y0 <= point[1] < y1)
            self.assertAlmostEqual(slot['captured_support_seconds'], .4)

    def test_unsupported_clock_never_routes_even_large_density(self):
        for condition in CONDITIONS:
            route = route_regions(self.maps(), (120, 80), clock_supported=False, condition=condition)
            self.assertEqual([slot['bbox_source_xyxy'] for slot in route['slots']], [[0, 0, 120, 80]] * 3)

    def test_sparse_phase_does_not_get_brightened_into_evidence(self):
        maps = self.maps()
        maps[1] *= .01
        route = route_regions(maps, (120, 80), clock_supported=True)
        self.assertEqual(route['slots'][1]['fallback_reason'], 'insufficient_phase_support')
        self.assertEqual(route['slots'][1]['bbox_source_xyxy'], [0, 0, 120, 80])
        self.assertIsNone(route['slots'][0]['fallback_reason'])

    def test_wrong_gaze_control_changes_crop_without_reflecting_rgb(self):
        maps = self.maps()
        true = route_regions(maps, (120, 80), clock_supported=True)
        wrong = route_regions(maps, (120, 80), clock_supported=True, condition='MIRRORED_GAZE_RGB')
        self.assertNotEqual(true['slots'][0]['bbox_source_xyxy'], wrong['slots'][0]['bbox_source_xyxy'])
        self.assertEqual(true['slots'][0]['phase_support_seconds'], wrong['slots'][0]['phase_support_seconds'])
        yy, xx = np.indices((80, 120))
        pixels = np.stack([xx, yy, xx + yy], axis=-1).astype('uint8')
        image = render_rgb_evidence(Image.fromarray(pixels), wrong)
        np.testing.assert_array_equal(np.asarray(image)[HEADER:HEADER + 80, :120], pixels)

    def test_all_conditions_preserve_full_rgb_exactly_and_same_dimensions(self):
        rng = np.random.default_rng(17)
        pixels = rng.integers(0, 256, (80, 120, 3), dtype=np.uint8)
        source = Image.fromarray(pixels)
        for condition in CONDITIONS:
            route = route_regions(self.maps(), source.size, clock_supported=True, condition=condition)
            image = render_rgb_evidence(source, route)
            self.assertEqual(image.size, (240, 216))
            np.testing.assert_array_equal(np.asarray(image)[HEADER:HEADER + 80, :120], pixels)
            self.assertFalse(route['candidate_filtering'])
            self.assertFalse(route['heatmap_sent_to_model'])

    def test_fallback_inputs_are_identical_across_conditions(self):
        source = Image.new('RGB', (120, 80), (31, 101, 157))
        results = [np.asarray(render_rgb_evidence(source,
            route_regions(self.maps() * 0, source.size, clock_supported=True, condition=c))) for c in CONDITIONS]
        for result in results[1:]:
            np.testing.assert_array_equal(results[0], result)

    def test_uniform_detail_positions_do_not_depend_on_gaze_location(self):
        first = route_regions(self.maps(), (120, 80), clock_supported=True, condition='UNIFORM_RGB')
        second = route_regions(self.maps()[:, ::-1, ::-1], (120, 80), clock_supported=True, condition='UNIFORM_RGB')
        self.assertEqual([s['bbox_source_xyxy'] for s in first['slots']],
                         [s['bbox_source_xyxy'] for s in second['slots']])

    def test_invalid_density_and_config_are_rejected(self):
        for maps in [np.zeros((2, 80, 120)), np.full((3, 80, 120), -1.), np.full((3, 80, 120), np.nan)]:
            with self.assertRaises(ValueError):
                route_regions(maps, (120, 80), clock_supported=True)
        for kwargs in [{'crop_fraction': 0}, {'minimum_support_seconds': 0}, {'maximum_overlap_iou': 2}]:
            with self.assertRaises(ValueError):
                RoutingConfig(**kwargs)

    def test_model_payload_contains_rgb_rectangles_but_no_attention_scores(self):
        event = contract_tests.MentionAccountingContractTests().event()
        for frame in event['frames']:
            frame['time_seconds'] = 1.
            frame['routing'] = {'GAZE_RGB': route_regions(np.zeros((3, 672, 672)), (672, 672), clock_supported=False)}
        payload = json.loads(grounding_prompt(event, 'GAZE_RGB').split('INPUT: ', 1)[1])
        text = json.dumps(payload)
        self.assertIn('detail_source_rectangles_xyxy', text)
        for forbidden in ['phase_support_seconds', 'captured_fraction', 'gaze_point', 'heatmap', 'candidate_id']:
            self.assertNotIn(forbidden, text)
        self.assertEqual(len(payload['mention_units']), 2)
        # Full RGB remains at the exact V4 panel coordinates; output decoding
        # does not need a new coordinate convention or tuned scene-ID decoder.
        response = {'mentions': [contract_tests.MentionAccountingContractTests().grounded('M001'),
                                 contract_tests.MentionAccountingContractTests().grounded('M002')]}
        checked = validate_mention_contract(response, event)
        self.assertTrue(checked['coverage_complete'])
        self.assertAlmostEqual(checked['grounded_referents'][0]['x'], .75)
        self.assertAlmostEqual(checked['grounded_referents'][0]['y'], .5)


class LiteralMentionBoundaryTests(unittest.TestCase):
    def test_pronoun_component_keeps_literal_span_without_repairing_parent_identity(self):
        text = 'Move a chair and inspect its seat.'
        raw = {'mentions': [{'text': 'a chair'}, {'text': 'its seat', 'parent_text': 'a chair',
                            'anchor_text': 'a chair', 'candidate_ids': ['chair7']}]}
        hints = literal_mention_hints(raw, text)
        self.assertEqual(hints[1]['text'], 'its seat')
        self.assertEqual(text[slice(*hints[1]['span'])], 'its seat')
        self.assertNotIn('parent_text', hints[1])
        self.assertNotIn('candidate_ids', hints[1])
        self.assertEqual(raw['mentions'][1]['parent_text'], 'a chair')

    def test_explicit_actor_omission_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing explicit actor'):
            literal_mention_hints({'mentions': [{'text': 'the cup'}]}, 'Tell that person to pick up the cup.')

    def test_repeated_occurrences_have_distinct_literal_spans(self):
        text = 'Move that car beside that car.'
        hints = literal_mention_hints({'mentions': [{'text': 'that car'}, {'text': 'that car'}]}, text)
        self.assertNotEqual(hints[0]['span'], hints[1]['span'])
        self.assertNotEqual(hints[0]['mention_id'], hints[1]['mention_id'])

    def test_nonliteral_and_excess_occurrences_are_rejected(self):
        for rows in [[{'text': 'chair7'}], [{'text': 'that chair'}, {'text': 'that chair'}]]:
            with self.assertRaises(ValueError):
                literal_mention_hints({'mentions': rows}, 'Move that chair.')

    def test_empty_and_instance_selection_payloads_are_rejected(self):
        for payload in [{'mentions': []}, {'mentions': [{'text': 'that chair'}], 'selected_object_names': ['chair7']}]:
            with self.assertRaises(ValueError):
                literal_mention_hints(payload, 'Move that chair.')


if __name__ == '__main__':
    unittest.main()
