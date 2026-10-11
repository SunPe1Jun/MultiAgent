"""Regression checks for JSON transport; no inference or dataset replacement."""
import json
import unittest

from .qwen_backend import QwenBackend


class ResponseParsingTests(unittest.TestCase):
    def test_bare_hand_array_requires_hand_schema(self):
        cue = {'frame_id': 'F1', 'mention_ids': ['m0'], 'cue_type': 'pointing',
               'confidence': .9, 'explanation': 'visible hand'}
        self.assertEqual(QwenBackend._parse_response(json.dumps([cue])), {'cues': [cue]})
        self.assertIsNone(QwenBackend._parse_response('[{"frame_id":"F1","candidate_id":"box1"}]'))

    def test_indented_truncated_evidence(self):
        response = '{\n "evidence": [\n {"mention_id":"m0","confidence":0}, {"mention_id":"m1'
        parsed = QwenBackend._parse_response(response)
        self.assertEqual(parsed['evidence'], [{'mention_id': 'm0', 'confidence': 0}])
        self.assertEqual(parsed['_parse_warning'], 'truncated_evidence_response')

    def test_grouped_evidence_preserves_all_rows_and_scores(self):
        rows = [{'mention_id': 'm0', 'support_score': .7, 'explanation': 'literal } ]'},
                {'mention_id': 'm1', 'support_score': .2}]
        # The real failure emitted complete arrays but omitted group braces.
        response = '```json\n[{"evidence":' + json.dumps(rows[:1]) + ',{"evidence":' + json.dumps(rows[1:]) + ']\n```'
        self.assertEqual(QwenBackend._parse_response(response), {'evidence': rows})

    def test_valid_grouped_evidence(self):
        response = '[{"evidence":[{"mention_id":"m0"}]},{"evidence":[{"mention_id":"m1"}]}]'
        self.assertEqual(QwenBackend._parse_response(response),
                         {'evidence': [{'mention_id': 'm0'}, {'mention_id': 'm1'}]})

    def test_incomplete_row_is_not_recovered(self):
        self.assertIsNone(QwenBackend._parse_response('[{"evidence":[{"explanation":"unfinished]'))

    def test_unrelated_multiple_objects_are_not_merged(self):
        self.assertIsNone(QwenBackend._parse_response('[{"a":1},{"a":2}]'))

    def test_truncated_evidence_keeps_only_complete_rows(self):
        response = '```json\n{"evidence":[{"mention_id":"m0","support_score":0.8},'
        parsed = QwenBackend._parse_response(response)
        self.assertEqual(parsed['evidence'], [{'mention_id': 'm0', 'support_score': 0.8}])
        self.assertEqual(parsed['_parse_warning'], 'truncated_evidence_response')

    def test_standard_response_and_missing_outer_brace(self):
        for response in ['{"evidence":[]}', '```json\n{"evidence":[]\n```']:
            self.assertEqual(QwenBackend._parse_response(response), {'evidence': []})


if __name__ == '__main__':
    unittest.main()
