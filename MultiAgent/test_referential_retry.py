"""Validation feedback must preserve raw language and enforce the retry bound."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from .agents import ReferentialAgent
from .mention_mapping import map_phrase, resolve_mentions
from .asr import match_mention
from .context import Candidate


class ReferentialRetryTests(unittest.TestCase):
    def setUp(self):
        self.ctx = SimpleNamespace(scene_key='scene1', event_id='scene1_769',
                                   text='Pick up this box from the platform, set it near that warhouse',
                                   candidates=[])
        self.bad = {'mentions': [{'text': 'warehouse', 'object_type': 'warehouse'}]}
        self.good = copy.deepcopy(self.bad)
        self.good['mentions'][0]['text'] = 'warhouse'

    def test_feedback_retries_without_weakening_literal_check(self):
        backend = Mock()
        backend.ask.side_effect = [self.bad, self.good]
        result = ReferentialAgent(backend).run(self.ctx)
        self.assertEqual(result['mentions'][0]['text'], 'warhouse')
        self.assertEqual(result['mentions'][0]['object_type'], 'warehouse')
        self.assertEqual(len(result['validation_retries']), 1)
        self.assertEqual(backend.ask.call_args.args[1], 'referential_validation_retry_1')
        self.assertIn(self.ctx.text, backend.ask.call_args.args[2])
        self.assertEqual(self.bad['mentions'][0]['text'], 'warehouse')

    def test_invalid_retries_still_fail_after_bound(self):
        backend = Mock()
        backend.ask.return_value = self.bad
        with self.assertRaisesRegex(ValueError, 'literal instruction span'):
            ReferentialAgent(backend).run(self.ctx)
        self.assertEqual(backend.ask.call_count, 3)

    def test_valid_response_does_not_retry(self):
        backend = Mock()
        backend.ask.return_value = self.good
        ReferentialAgent(backend).run(self.ctx)
        backend.ask.assert_called_once()

    def test_deictic_location_is_deferred_to_multimodal_context(self):
        ctx = SimpleNamespace(scene_key='scene2', event_id='scene2_123',
                              text='Slide that chair to there.',
                              candidates=[Candidate('chair5','chair5',0,0,0),
                                          Candidate('desk8','desk8',0,0,0)])
        out = resolve_mentions(ctx, {'mentions': [
            {'text': 'there', 'object_type': 'location', 'kind': 'location'}]})
        self.assertEqual(out['mentions'][0]['candidate_ids'], ['chair5', 'desk8'])
        self.assertEqual(out['mentions'][0]['candidate_mapping_reason'],
                         'contextual_location_pool')

    def test_asr_head_recovery_keeps_low_confidence(self):
        words = [{'word': 'that', 'start': 0.0, 'end': .1},
                 {'word': 'fire', 'start': .1, 'end': .3}]
        out = match_mention('that file', words, 'Push that file', 'file')
        self.assertTrue(out['matched'])
        self.assertEqual(out['alignment_quality'], 'head_only')
        self.assertLess(out['confidence'], .65)

    def test_scene_local_alias_overrides_global_alias(self):
        candidates = [
            Candidate('dining_table', 'dining_table', 0, 0, 0,
                      ('dining table', 'table')),
            Candidate('desk', 'desk', 0, 0, 0, ()),
        ]
        self.assertEqual(map_phrase('table', candidates), ['dining_table'])
        self.assertEqual(map_phrase('dining table', candidates), ['dining_table'])

    def test_scene_local_plural_alias_maps_to_local_instances(self):
        candidates = [
            Candidate('curtains1', 'curtains1', 0, 0, 0, ('curtains',)),
            Candidate('curtains2', 'curtains2', 0, 0, 0, ('curtains',)),
        ]
        self.assertEqual(map_phrase('curtains', candidates),
                         ['curtains1', 'curtains2'])

    def test_bare_class_is_not_treated_as_numbered_instance(self):
        ctx = SimpleNamespace(text='Pick up the chair.', candidates=[
            Candidate('chair', 'chair', 0, 0, 0),
            Candidate('chair2', 'chair2', 0, 0, 0),
        ])
        out = resolve_mentions(ctx, {'mentions': [
            {'text': 'the chair', 'object_type': 'chair'}]})
        self.assertEqual(out['mentions'][0]['candidate_ids'], ['chair', 'chair2'])
        self.assertEqual(out['mentions'][0]['explicit_ids'], [])

    def test_component_parent_inherits_scene_local_coreference(self):
        ctx = SimpleNamespace(text="Align it with the dining table and the table's edge.", candidates=[
            Candidate('dining_table', 'dining_table', 0, 0, 0,
                      ('dining table', 'table')),
            Candidate('desk', 'desk', 0, 0, 0),
        ])
        out = resolve_mentions(ctx, {'mentions': [
            {'text': 'the dining table', 'object_type': 'table',
             'coreference_id': 'r1'},
            {'text': "the table's edge", 'object_type': 'edge',
             'component': 'edge', 'parent_text': 'the table',
             'coreference_id': 'r1'},
        ]})
        self.assertEqual(out['mentions'][1]['candidate_ids'], ['dining_table'])
        self.assertEqual(out['mentions'][1]['candidate_mapping_reason'], 'coreference_parent')

    def test_explicit_numbered_id_remains_specific(self):
        ctx = SimpleNamespace(text='Move chair2.', candidates=[
            Candidate('chair', 'chair', 0, 0, 0), Candidate('chair2', 'chair2', 0, 0, 0)])
        m = resolve_mentions(ctx, {'mentions':[{'text':'chair2', 'object_type':'chair'}]})['mentions'][0]
        self.assertEqual(m['candidate_ids'], ['chair2'])

    def test_spatial_side_recovers_anchor_without_locking_bare_id(self):
        ctx = SimpleNamespace(text='Move it toward the bedroom side.', candidates=[
            Candidate('bedroom', 'bedroom', 0, 0, 0)])
        m = resolve_mentions(ctx, {'mentions':[{'text':'the bedroom side', 'object_type':'side'}]})['mentions'][0]
        self.assertEqual(m['candidate_ids'], ['bedroom'])
        self.assertFalse(m['explicit_ids'])

    def test_near_relation_does_not_turn_unmapped_door_into_truck(self):
        ctx = SimpleNamespace(text='Open the door near the truck.', candidates=[
            Candidate('truck1', 'truck1', 0, 0, 0)])
        m = resolve_mentions(ctx, {'mentions':[{'text':'the door near the truck', 'object_type':'door'}]})['mentions'][0]
        self.assertEqual(m['candidate_ids'], [])

    def test_nested_parent_is_deduplicated_but_later_occurrence_survives(self):
        ctx = SimpleNamespace(text='Open the door of that van, then move that van.',
                              candidates=[Candidate('van1', 'van1', 0, 0, 0)])
        out = resolve_mentions(ctx, {'mentions': [
            {'text': 'the door of that van', 'object_type': 'door',
             'component': 'door', 'parent_text': 'that van'},
            {'text': 'that van', 'object_type': 'van'},
            {'text': 'that van', 'object_type': 'van'},
        ]})
        self.assertEqual([m['mention_id'] for m in out['mentions']], ['m0', 'm2'])
        self.assertEqual(out['mentions'][1]['span'], [37, 45])
        self.assertEqual(out['suppressed_nested_mentions'][0]['covered_by_mention_id'], 'm0')

    def test_spatial_anchor_is_not_deduplicated_as_a_parent(self):
        ctx = SimpleNamespace(text='Open the door near that van.', candidates=[
            Candidate('door1', 'door1', 0, 0, 0), Candidate('van1', 'van1', 0, 0, 0)])
        out = resolve_mentions(ctx, {'mentions': [
            {'text': 'the door', 'object_type': 'door'},
            {'text': 'that van', 'object_type': 'van'},
        ]})
        self.assertEqual(len(out['mentions']), 2)
        self.assertFalse(out['suppressed_nested_mentions'])
