import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from .context import Candidate
from .temporal_focus import focus_spec, focused_visual
from .vision_reasoning import visual_evidence, monolithic_coordinator


class TemporalFocusTests(unittest.TestCase):
    def setUp(self):
        self.text = 'Move that chair, then inspect that chair.'
        self.mention = {'mention_id': 'm1', 'text': 'that chair', 'span': [30, 40],
                        'candidate_ids': ['chair1', 'chair2'], 'count': 1}
        self.ctx = SimpleNamespace(text=self.text, scene_key='test', event_id='test',
                                   peak_spatial={}, api_row={}, candidates=[
                                       Candidate('chair1', 'chair1', 0, 0, 0),
                                       Candidate('chair2', 'chair2', 1, 0, 0)])
        self.frames = [{'frame_id': f'F{i}', 'time': i, 'image_rgb': f'{i}.jpg'}
                       for i in range(1, 10)]
        self.window = {'video_range': [0, 10], 'phase_hint': [5, 10],
                       'asr_hint': {'video_range': [7, 9]}}
        self.plan = {'mention_windows': {'m1': self.window}, 'alignment': {}}
        self.parse = {'mentions': [self.mention]}
        self.paths = [f['image_rgb'] for f in self.frames]

    def test_repeated_occurrence_highlight_and_rejected_clock(self):
        spec = focus_spec(self.text, self.mention, self.window, self.frames)
        self.assertIn('inspect [CURRENT MENTION: that chair]', spec['highlighted_instruction'])
        self.assertEqual(spec['primary_frame_ids'], ['F7', 'F8', 'F9'])
        self.assertNotIn('Move', spec['action_context'])
        rejected = focus_spec(self.text, self.mention, self.window, self.frames, True)
        self.assertEqual(rejected['primary_range'], [5, 10])
        self.assertEqual(rejected['time_source'], 'soft_instruction_phase')

    def test_local_context_evidence_rejected_then_expanded(self):
        def reply(key, stage, prompt, paths):
            return {'evidence': [{'mention_id': 'm1', 'candidate_id': 'chair2',
                'frame_id': 'F6', 'support_score': .9, 'confidence': .9,
                'temporal_justification': 'Visible continuous manipulation into the second action.'}]}
        backend = Mock(); backend.ask.side_effect = reply
        result = focused_visual(visual_evidence, backend, self.ctx, self.parse,
            self.paths, self.frames, self.plan, None, False, False, None)
        self.assertEqual(backend.ask.call_count, 2)
        self.assertEqual(result['rejected_evidence'][0]['reason'], 'context_only_local_evidence')
        self.assertEqual(result['focus_audit']['m1']['status'], 'expanded_supported')
        self.assertEqual(result['evidence'][0]['candidate_id'], 'chair2')
        self.assertLess(len(backend.ask.call_args_list[0].args[3]), len(self.frames))
        self.assertEqual(len(backend.ask.call_args_list[1].args[3]), len(self.frames))

    def test_valid_primary_evidence_does_not_expand(self):
        backend = Mock(); backend.ask.return_value = {'evidence': [
            {'mention_id': 'm1', 'candidate_id': 'chair2', 'frame_id': 'F8',
             'support_score': .9, 'confidence': .9}]}
        result = focused_visual(visual_evidence, backend, self.ctx, self.parse,
            self.paths, self.frames, self.plan, None, False, False, None)
        self.assertEqual(backend.ask.call_count, 1)
        self.assertEqual(result['focus_audit']['m1']['status'], 'local_supported')

    def test_unsupplied_frame_cannot_be_cited(self):
        backend = Mock(); backend.ask.return_value = {'evidence': [
            {'mention_id': 'm1', 'candidate_id': 'chair2', 'frame_id': 'F1',
             'support_score': .9, 'confidence': .9}]}
        spec = focus_spec(self.text, self.mention, self.window, self.frames)
        result = visual_evidence(backend, self.ctx, self.parse, self.paths[6:],
            self.frames[6:], self.plan, isolate_mentions=True,
            focus_context={**spec, 'stage': 'local'})
        self.assertFalse(result['evidence'])
        self.assertEqual(result['rejected_evidence'][0]['reason'], 'positive_evidence_missing_frame')

    def test_global_coordinator_cannot_transfer_other_occurrence_evidence(self):
        backend = Mock(); backend.ask.return_value = {
            'selected_object_names': ['chair1'],
            'decisions': [{'mention_id': 'm1', 'candidate_id': 'chair1'}]}
        result = monolithic_coordinator(backend, self.ctx, self.parse, self.paths,
            self.frames, {'temporal_focus': True, 'evidence': [
                {'mention_id': 'm0', 'candidate_id': 'chair1', 'support_score': 1, 'confidence': 1}]})
        self.assertEqual(result['selected_object_names'], [])
        self.assertEqual(result['decisions'], [])
        self.assertEqual(len(result['rejected_decisions']), 1)

    def test_singleton_survives_coordinator_abstention(self):
        backend = Mock(); backend.ask.return_value = {'selected_object_names': [], 'decisions': []}
        mention = {**self.mention, 'candidate_ids': ['chair2']}
        result = monolithic_coordinator(backend, self.ctx, {'mentions': [mention]}, self.paths,
            self.frames, {'temporal_focus': True, 'evidence': []})
        self.assertEqual(result['selected_object_names'], ['chair2'])

    def test_single_clause_preserves_five_frame_coverage(self):
        spec = focus_spec(self.text, self.mention, {'video_range': [0,10], 'phase_hint': [0,10]},
                          self.frames[:5], True)
        self.assertEqual(spec['primary_frame_ids'], ['F1', 'F2', 'F3', 'F4', 'F5'])
