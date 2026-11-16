"""Validation feedback must preserve raw language and enforce the retry bound."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from .agents import ReferentialAgent
from .mention_mapping import map_phrase, repair_explicit_component_parent, resolve_mentions
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

    def test_parent_mapping_error_gets_one_targeted_retry_then_fails_closed(self):
        ctx = SimpleNamespace(scene_key='scene5', event_id='scene5_1',
                              text='Move the crate beside the truck.', candidates=[
                                  Candidate('crate1', 'crate1', 0, 0, 0),
                                  Candidate('truck1', 'truck1', 0, 0, 0)])
        response = {'mentions': [{'text': 'the crate', 'object_type': 'crate',
                                  'parent_text': 'the truck'}]}
        backend = Mock()
        backend.ask.return_value = response
        with self.assertRaisesRegex(ValueError, 'parent_text .* must be a literal span'):
            ReferentialAgent(backend).run(ctx)
        self.assertEqual(backend.ask.call_count, 4)
        self.assertIn('relation words such as "beside" do not',
                      backend.ask.call_args.args[2])

    def test_explicit_of_phrase_is_merged_from_literal_instruction_only(self):
        ctx = SimpleNamespace(scene_key='scene1', event_id='scene1_216',
            text='Turn toward the side door of the rear building.', candidates=[
                Candidate('door1', 'door1', 0, 0, 0), Candidate('building2', 'building2', 0, 0, 0)])
        payload = {'mentions': [
            {'text': 'the side door', 'object_type': 'door', 'component': 'side door',
             'parent_text': 'the rear building', 'coreference_id': 'r0'},
            {'text': 'the rear building', 'object_type': 'building', 'coreference_id': 'r1'}]}
        repaired, audit = repair_explicit_component_parent(payload, ctx.text)
        self.assertEqual(repaired['mentions'][0]['text'], 'the side door of the rear building')
        self.assertEqual(len(repaired['mentions']), 1)
        self.assertEqual(len(audit), 1)
        resolved = resolve_mentions(ctx, repaired)
        self.assertEqual(resolved['mentions'][0]['candidate_ids'], ['building2'])
        self.assertNotIn('GT', json.dumps(audit).upper())

    def test_component_repair_does_not_remove_a_later_separate_parent_occurrence(self):
        instruction = 'Open the side door of the rear building, then inspect the rear building.'
        payload = {'mentions': [
            {'text': 'the side door', 'object_type': 'door', 'component': 'side door',
             'parent_text': 'the rear building'},
            {'text': 'the rear building', 'object_type': 'building'},
            {'text': 'the rear building', 'object_type': 'building'}]}
        repaired, audit = repair_explicit_component_parent(payload, instruction)
        self.assertEqual(repaired['mentions'][0]['text'], 'the side door of the rear building')
        self.assertEqual([row['text'] for row in repaired['mentions']],
                         ['the side door of the rear building', 'the rear building'])
        self.assertEqual(audit[0]['removed_duplicate_parent_index'], 1)

    def test_relation_phrase_is_not_repaired_as_a_component_parent(self):
        instruction = 'Move the crate beside the truck.'
        payload = {'mentions': [{'text': 'the crate', 'object_type': 'crate',
                                 'component': '', 'parent_text': 'the truck'}]}
        repaired, audit = repair_explicit_component_parent(payload, instruction)
        self.assertEqual(audit, [])
        self.assertEqual(repaired['mentions'][0]['text'], 'the crate')

    def test_valid_response_does_not_retry(self):
        backend = Mock()
        backend.ask.return_value = self.good
        ReferentialAgent(backend).run(self.ctx)
        backend.ask.assert_called_once()

    def test_whole_instruction_cannot_be_an_instruction_type_mention(self):
        payload = {'mentions': [{'text': self.ctx.text, 'object_type': 'instruction'}]}
        with self.assertRaisesRegex(ValueError, 'instruction-level text'):
            resolve_mentions(self.ctx, payload)

    def test_pronoun_parent_repair_requires_matching_prior_literal_antecedent(self):
        ctx = SimpleNamespace(text='Move the crate and inspect its panel.', candidates=[
            Candidate('crate', 'crate', 0, 0, 0)])
        payload = {'mentions': [
            {'text': 'the crate', 'object_type': 'crate', 'coreference_id': 'r1'},
            {'text': 'its panel', 'object_type': 'panel', 'component': 'panel',
             'parent_text': 'the crate', 'anchor_text': 'the crate', 'coreference_id': 'r9'}]}
        with self.assertRaisesRegex(ValueError, 'parent_text'):
            resolve_mentions(ctx, payload)

    def test_deictic_location_is_deferred_to_multimodal_context(self):
        ctx = SimpleNamespace(scene_key='scene2', event_id='scene2_123',
                              text='Slide that chair to there.',
                              candidates=[Candidate('chair5','chair5',0,0,0),
                                          Candidate('desk8','desk8',0,0,0)])
        out = resolve_mentions(ctx, {'mentions': [
            {'text': 'there', 'object_type': 'location', 'kind': 'location'}]})
        self.assertEqual(out['mentions'][0]['candidate_ids'], [])
        self.assertEqual(out['mentions'][0]['candidate_mapping_reason'],
                         'contextual_location_unresolved')

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

    def test_possessive_component_maps_to_whole_parent_anchor(self):
        ctx = SimpleNamespace(text="Attach that drone's battery.", candidates=[
            Candidate('drone1', 'drone1', 0, 0, 0),
            Candidate('battery1', 'battery1', 0, 0, 0),
        ])
        out = resolve_mentions(ctx, {'mentions': [
            {'text': "that drone's battery", 'object_type': 'battery',
             'component': 'battery', 'kind': 'object'}]})
        mention = out['mentions'][0]
        self.assertEqual(mention['parent_text'], 'that drone')
        self.assertEqual(mention['candidate_ids'], ['drone1'])
        self.assertEqual(mention['candidate_mapping_reason'], 'component_parent')

    def test_pronominal_component_parent_recovers_scene5_empty_retry_failure(self):
        instruction = 'instruct this drone to identify the crate  beside the truck and then focus on its side panel.'
        candidates = [Candidate('drone1', 'drone1', 0, 0, 0),
                      Candidate('crate1', 'crate1', 0, 0, 0),
                      Candidate('truck1', 'truck1', 0, 0, 0)]
        ctx = SimpleNamespace(scene_key='scene5', event_id='scene5_387',
                              text=instruction, candidates=candidates)
        first = {'mentions': [
            {'text': 'this drone', 'object_type': 'drone', 'coreference_id': 'r0'},
            {'text': 'the crate', 'object_type': 'crate', 'coreference_id': 'r1'},
            {'text': 'the truck', 'object_type': 'truck', 'coreference_id': 'r2'},
            {'text': 'its side panel', 'object_type': 'side panel',
             'component': 'side panel', 'parent_text': 'the crate', 'coreference_id': 'r3'},
        ]}
        second = {'mentions': [
            {'text': 'this drone', 'object_type': 'drone', 'coreference_id': 'r0'},
            {'text': 'the crate beside the truck', 'object_type': 'crate',
             'parent_text': 'the crate beside the truck', 'coreference_id': 'r1'},
            {'text': 'its side panel', 'object_type': 'panel', 'component': 'side',
             'parent_text': 'the crate beside the truck', 'coreference_id': 'r2'},
        ]}
        corrected = {'mentions': [
            {'text': 'this drone', 'object_type': 'drone', 'coreference_id': 'r0'},
            {'text': 'the crate', 'object_type': 'crate', 'coreference_id': 'r1'},
            {'text': 'the truck', 'object_type': 'truck', 'coreference_id': 'r2'},
            {'text': 'its side panel', 'object_type': 'panel', 'component': 'side panel',
             'parent_text': 'the crate', 'anchor_text': 'the crate', 'coreference_id': 'r1'},
        ]}
        backend = Mock()
        backend.ask.side_effect = [first, second, corrected]
        result = ReferentialAgent(backend).run(ctx)
        self.assertEqual(backend.ask.call_count, 3)
        self.assertIn('Copy every mention text with its original whitespace',
                      backend.ask.call_args_list[1].args[2])
        self.assertIn('leave parent_text empty',
                      backend.ask.call_args_list[1].args[2])
        component = next(row for row in result['mentions'] if row['text'] == 'its side panel')
        self.assertEqual(component['candidate_ids'], ['crate1'])
        self.assertEqual(component['parent_mention_id'], 'm1')
        self.assertEqual(component['candidate_mapping_reason'], 'coreference_component_parent')
        self.assertEqual(component['repaired_pronoun_parent_text'], 'the crate')

        failing_backend = Mock()
        failing_backend.ask.side_effect = [first, second, {'mentions': []}]
        with self.assertRaisesRegex(ValueError, 'Empty extraction for nonempty instruction'):
            ReferentialAgent(failing_backend).run(ctx)
        self.assertEqual(failing_backend.ask.call_count, 3)
