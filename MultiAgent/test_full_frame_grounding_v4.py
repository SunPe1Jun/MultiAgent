import unittest

from .full_frame_grounding_v4 import grounding_prompt, validate_mention_contract
from . import test_full_frame_grounding as grounding_tests


class MentionAccountingContractTests(unittest.TestCase):
    def event(self):
        event = grounding_tests.FullFrameGroundingTests().event()
        for frame in event['frames']:
            frame['source_size'] = [672, 672]
        event['mention_hints'] = [
            {'mention_id': 'M001', 'text': 'that chair', 'role': 'object'},
            {'mention_id': 'M002', 'text': 'that chair', 'role': 'object'},
        ]
        return event

    def grounded(self, mention_id, x=375, y=260):
        return {
            'mention_id': mention_id,
            'mention': 'that chair',
            'status': 'grounded',
            'category': 'chair',
            'view_id': 'G001',
            'x': x,
            'y': y,
            'unresolved_reason': None,
        }

    def test_duplicate_literal_mentions_have_separate_accounting(self):
        event = self.event()
        result = validate_mention_contract({'mentions': [self.grounded('M001'), self.grounded('M002')]}, event)
        self.assertTrue(result['schema_valid'])
        self.assertTrue(result['coverage_complete'])
        self.assertEqual([row['mention_id'] for row in result['grounded_referents']], ['M001', 'M002'])
        self.assertAlmostEqual(result['grounded_referents'][0]['x'], .75)
        self.assertAlmostEqual(result['grounded_referents'][0]['y'], .5)

    def test_missing_row_is_coverage_failure_not_schema_repair(self):
        event = self.event()
        result = validate_mention_contract({'mentions': [self.grounded('M001')]}, event)
        self.assertTrue(result['schema_valid'])
        self.assertFalse(result['coverage_complete'])
        self.assertEqual(result['missing_mention_ids'], ['M002'])

    def test_grounded_row_may_omit_null_unresolved_reason(self):
        event = self.event()
        row = self.grounded('M001')
        row.pop('unresolved_reason')
        result = validate_mention_contract({'mentions': [row, self.grounded('M002')]}, event)
        self.assertTrue(result['schema_valid'])
        self.assertTrue(result['coverage_complete'])
        self.assertIsNone(result['grounded_referents'][0]['unresolved_reason'])

    def test_unresolved_row_uses_one_uniform_shape(self):
        event = self.event()
        unresolved = {
            'mention_id': 'M002', 'mention': 'that chair', 'status': 'unresolved',
            'category': None, 'view_id': None, 'x': None, 'y': None,
            'unresolved_reason': 'uncertain_identity',
        }
        result = validate_mention_contract({'mentions': [self.grounded('M001'), unresolved]}, event)
        self.assertTrue(result['schema_valid'])
        self.assertTrue(result['coverage_complete'])
        self.assertEqual(len(result['unresolved_mentions']), 1)

    def test_full_object_row_in_unresolved_is_rejected(self):
        event = self.event()
        bad = dict(self.grounded('M001'))
        bad['status'] = 'unresolved'
        bad['unresolved_reason'] = 'uncertain_identity'
        result = validate_mention_contract({'mentions': [bad, self.grounded('M002')]}, event)
        self.assertFalse(result['schema_valid'])
        self.assertIn('unresolved_contains_grounding_fields', result['errors'])
        self.assertTrue(result['coverage_complete'] is False)

    def test_points_outside_rgb_panel_never_reach_decoder(self):
        for x, y in [(700, 260), (375, 800), (375, 0), (True, 260), (375, float('nan'))]:
            with self.subTest(x=x, y=y):
                result = validate_mention_contract({'mentions': [self.grounded('M001', x, y), self.grounded('M002')]}, self.event())
                self.assertFalse(result['schema_valid'])
                self.assertFalse(result['coverage_complete'])
                self.assertEqual([r['mention_id'] for r in result['grounded_referents']], ['M002'])

    def test_duplicate_ids_do_not_choose_one_of_conflicting_rows(self):
        rows = [self.grounded('M001', 100), self.grounded('M001', 400), self.grounded('M002')]
        result = validate_mention_contract({'mentions': rows}, self.event())
        self.assertEqual(result['duplicate_mention_ids'], ['M001'])
        self.assertEqual([r['mention_id'] for r in result['grounded_referents']], ['M002'])
        self.assertFalse(result['coverage_complete'])

    def test_malformed_field_types_reject_without_crashing(self):
        for field, value in [('mention_id', []), ('status', {}), ('view_id', []), ('category', {}), ('mention', [])]:
            with self.subTest(field=field):
                row = self.grounded('M001'); row[field] = value
                result = validate_mention_contract({'mentions': [row, self.grounded('M002')]}, self.event())
                self.assertFalse(result['schema_valid'])
                self.assertFalse(result['coverage_complete'])

    def test_input_duplicate_ids_fail_before_inference(self):
        event = self.event(); event['mention_hints'][1]['mention_id'] = 'M001'
        with self.assertRaises(ValueError):
            validate_mention_contract({'mentions': []}, event)

    def test_prompt_exposes_stable_ids_and_not_anchor_ids(self):
        prompt = grounding_prompt(self.event())
        self.assertIn('M001', prompt)
        self.assertIn('M002', prompt)
        self.assertNotIn('chair1', prompt)
        self.assertNotIn('chair2', prompt)


if __name__ == '__main__':
    unittest.main()
