import copy
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from .context import Candidate
from .gaze_rgb_routing import route_regions, render_rgb_evidence
from .referent_rgb_selection import (
    assemble, grounding_prompt, labelled_rgb, merge_recovery, panel_to_source, composite_to_source,
    recover_component_schema, recovery_frames, recovery_mentions, temporal_search,
    validate_choices,
)


class ReferentRGBTests(unittest.TestCase):
    def event(self):
        pose = {'position_world': [0, 0, 0], 'rotation_xyzw': [0, 0, 0, 1], 'fov_vertical_degrees': 90}
        frames = []
        for i in range(12):
            route = route_regions(np.zeros((3, 100, 100)), (100, 100), clock_supported=False)
            frames.append({'frame_id': f'V{i+1:03d}', 'source_frame_id': f'frame_{i}',
                'source_frame_index': i, 'time_seconds': float(i), 'source_size': [100, 100],
                'geometry': {'clock_available': True, 'reference_pose': pose},
                'routing': {'GAZE_RGB': route}})
        event = {'instruction': 'Move that chair, then inspect that chair and the cup.',
            'frames': frames,
            'scene_candidate_table': [
                {'candidate_id': 'chair1', 'aliases': [], 'position_world': [0, 0, 2]},
                {'candidate_id': 'chair2', 'aliases': [], 'position_world': [1, 0, 2]},
                {'candidate_id': 'cup', 'aliases': [], 'position_world': [0, 1, 2]}],
            'mention_hints': [
                {'mention_id': 'm0', 'text': 'that chair', 'span': [5, 15], 'count': 1,
                 'candidate_ids': ['chair1', 'chair2'], 'candidate_mapping_reason': 'type_or_alias'},
                {'mention_id': 'm1', 'text': 'that chair', 'span': [30, 40], 'count': 1,
                 'candidate_ids': ['chair1', 'chair2'], 'candidate_mapping_reason': 'type_or_alias'},
                {'mention_id': 'm2', 'text': 'the cup', 'span': [45, 52], 'count': 1,
                 'candidate_ids': ['cup'], 'candidate_mapping_reason': 'type_or_alias'}]}
        event['temporal_search'] = temporal_search(event)
        return event

    def row(self, mid='m0', frame='V001', x=500, panel=0, category='chair'):
        return {'mention_id': mid, 'status': 'grounded', 'instances': [
            {'frame_id': frame, 'x': (panel % 2 + x/1000) * 500,
             'y': ((panel // 2) * 128 + 28 + 50) / 256 * 1000, 'category': category}], 'unresolved_reason': None}

    def checked(self, event, rows):
        return validate_choices({'mentions': rows}, event, 'GAZE_RGB')

    def test_detail_coordinate_maps_to_exact_source_crop(self):
        frame = self.event()['frames'][0]
        frame['routing']['GAZE_RGB']['slots'][0]['bbox_source_xyxy'] = [60, 20, 100, 60]
        self.assertEqual(panel_to_source(frame, 'GAZE_RGB', 1, 500, 500), (.8, .4))
        self.assertEqual(panel_to_source(frame, 'GAZE_RGB', 1, 0, 0), (.6, .2))
        self.assertEqual(panel_to_source(frame, 'GAZE_RGB', 0, 750, 250), (.75, .25))

    def test_whole_composite_coordinates_preserve_v4_geometry(self):
        frame = self.event()['frames'][0]
        self.assertEqual(composite_to_source(frame, 'GAZE_RGB', 250, 78/256*1000), (.5, .5, 0))
        frame['routing']['GAZE_RGB']['slots'][0]['bbox_source_xyxy'] = [60,20,100,60]
        self.assertEqual(composite_to_source(frame, 'GAZE_RGB', 750, 78/256*1000), (.8, .4, 1))
        with self.assertRaisesRegex(ValueError, 'header'):
            composite_to_source(frame, 'GAZE_RGB', 250, 0)

    def test_letterboxing_cannot_become_a_source_observation(self):
        frame = self.event()['frames'][0]
        frame['source_size'] = [120, 80]
        frame['routing']['GAZE_RGB']['slots'][0]['bbox_source_xyxy'] = [20, 10, 60, 50]
        with self.assertRaisesRegex(ValueError, 'letterboxing'):
            panel_to_source(frame, 'GAZE_RGB', 1, 0, 500)
        self.assertAlmostEqual(panel_to_source(frame, 'GAZE_RGB', 1, 500, 500)[0], 1/3)

    def test_exact_frame_avoids_two_frame_position_average(self):
        event = self.event()
        event['frames'][1]['geometry'] = {'clock_available': True, 'reference_pose':
            {'position_world': [2, 0, 0], 'rotation_xyzw': [0, 0, 0, 1], 'fov_vertical_degrees': 90}}
        result = assemble(event, self.checked(event, [self.row()]), lexical=False)
        self.assertEqual(result['selected_object_names'], ['chair1'])
        self.assertEqual(result['mention_decisions']['m0']['decisions'][0]['source_frame_ids'], ['V001'])

    def test_omitted_mention_does_not_clear_other_valid_predictions(self):
        event = self.event()
        checked = self.checked(event, [self.row()])
        self.assertFalse(checked['coverage_complete'])
        self.assertEqual(assemble(event, checked, lexical=False)['selected_object_names'], ['chair1'])

    def test_unique_language_identity_is_logged_without_visibility_assertion(self):
        event = self.event()
        result = assemble(event, self.checked(event, []))
        self.assertEqual(result['selected_object_names'], ['cup'])
        self.assertEqual(result['mention_decisions']['m2']['selection_source'], 'unique_language_compatible_identity')
        self.assertFalse(result['mention_decisions']['m2']['visual_visibility_asserted'])
        self.assertEqual(result['mention_decisions']['m0']['selected'], [])

    def test_explicit_language_id_survives_missing_visual_output(self):
        event = self.event()
        event['mention_hints'][0]['explicit_ids'] = ['chair2']
        result = assemble(event, self.checked(event, []))
        self.assertEqual(result['mention_decisions']['m0']['selected'], ['chair2'])
        self.assertEqual(result['mention_decisions']['m0']['selection_source'], 'explicit_language_id')

    def test_vague_location_does_not_fall_back_to_an_arbitrary_scene_id(self):
        event = self.event()
        event['mention_hints'][2].update(candidate_ids=[], kind='location',
            candidate_mapping_reason='contextual_location_unresolved')
        self.assertEqual(assemble(event, self.checked(event, []))['selected_object_names'], [])

    def test_deictic_location_requires_a_real_visual_choice_not_a_lexical_guess(self):
        event = self.event()
        event['mention_hints'][0].update(candidate_ids=[], kind='location',
            candidate_mapping_reason='contextual_location_unresolved')
        self.assertEqual(assemble(event, self.checked(event, []))['mention_decisions']['m0']['selected'], [])
        result = assemble(event, self.checked(event, [self.row('m0')]))
        self.assertEqual(result['mention_decisions']['m0']['selected'], ['chair1'])
        self.assertEqual(result['mention_decisions']['m0']['selection_source'], 'visual_concrete_resolution_of_deictic_location')
        self.assertIn('m0', recovery_mentions(event, result))

    def test_singular_two_id_conflict_is_not_unioned(self):
        event = self.event()
        row = self.row()
        row['instances'].append({**row['instances'][0], 'x': 375})
        result = assemble(event, self.checked(event, [row]), lexical=False)
        self.assertEqual(result['selected_object_names'], [])
        self.assertTrue(result['mention_decisions']['m0']['cardinality_conflict'])
        event['mention_hints'][0]['count'] = 2
        self.assertEqual(assemble(event, self.checked(event, [row]), lexical=False)['selected_object_names'], ['chair1', 'chair2'])

    def test_repeated_mentions_can_select_same_identity(self):
        event = self.event()
        result = assemble(event, self.checked(event, [self.row('m0'), self.row('m1')]), lexical=False)
        self.assertEqual(result['selected_object_names'], ['chair1'])
        self.assertEqual(result['mention_decisions']['m1']['selected'], ['chair1'])

    def test_component_coreference_is_explicit_not_a_same_literal_rule(self):
        event = self.event()
        event['mention_hints'][1]['parent_mention_id'] = 'm0'
        result = assemble(event, self.checked(event, [self.row('m0')]))
        self.assertEqual(result['mention_decisions']['m1']['selected'], ['chair1'])
        self.assertEqual(result['mention_decisions']['m1']['selection_source'], 'verified_language_component_or_pronoun_coreference')
        del event['mention_hints'][1]['parent_mention_id']
        self.assertEqual(assemble(event, self.checked(event, [self.row('m0')]))['mention_decisions']['m1']['selected'], [])

    def test_unsupported_frames_duplicate_ids_and_bad_numbers_reject(self):
        event = self.event()
        self.assertFalse(validate_choices({'mentions': [self.row()]}, event, 'GAZE_RGB', {'V002'})['schema_valid'])
        self.assertFalse(self.checked(event, [self.row(), self.row()])['schema_valid'])
        for value in [float('nan'), True, -1, 1001]:
            row = self.row(); row['instances'][0]['x'] = value
            self.assertFalse(self.checked(event, [row])['schema_valid'])

    def test_temporal_occurrences_have_distinct_soft_phases(self):
        event = self.event()
        specs = event['temporal_search']['mentions']
        self.assertNotEqual(specs['m0']['primary_range'], specs['m1']['primary_range'])
        self.assertIn('inspect [CURRENT MENTION: that chair]', specs['m1']['highlighted_instruction'])
        self.assertTrue(event['temporal_search']['soft_prior_not_restriction'])

    def test_recovery_retains_global_time_coverage_and_original_ids(self):
        event = self.event()
        frames = recovery_frames(event, ['m1'])
        self.assertEqual(len(frames), 8)
        self.assertEqual(frames[0]['frame_id'], 'V001')
        self.assertEqual(frames[-1]['frame_id'], 'V012')
        self.assertEqual([f['time_seconds'] for f in frames], sorted(f['time_seconds'] for f in frames))
        self.assertEqual(len(set(f['frame_id'] for f in frames)), 8)

    def test_recovery_cannot_modify_untargeted_occurrences(self):
        event = self.event()
        initial = self.checked(event, [self.row('m0'), self.row('m1')])
        later = self.checked(event, [self.row('m0', x=750), self.row('m1', x=750)])
        merged = merge_recovery(initial, later, ['m1'])
        result = assemble(event, merged, lexical=False)
        self.assertEqual(result['mention_decisions']['m0']['selected'], ['chair1'])
        self.assertEqual(result['mention_decisions']['m1']['selected'], ['chair2'])
        self.assertEqual(merge_recovery(initial, self.checked(event, []), ['m1'])['rows'], initial['rows'])

    def test_label_banner_preserves_full_rgb_pixels(self):
        source = Image.fromarray(np.random.default_rng(3).integers(0,256,(100,100,3),dtype=np.uint8))
        event = self.event()
        frame = event['frames'][0]
        composite = render_rgb_evidence(source, frame['routing']['GAZE_RGB'])
        image = labelled_rgb(composite, frame)
        np.testing.assert_array_equal(np.asarray(image)[28:128,:100], source)

    def test_prompt_uses_rich_language_and_time_without_gaze_scores_or_gt(self):
        event = self.event()
        event['gt'] = ['poison_identity']
        prompt = grounding_prompt(event, 'GAZE_RGB')
        self.assertIn('soft_primary_range', prompt)
        self.assertIn('count', prompt)
        self.assertNotIn('poison_identity', prompt)
        self.assertNotIn('phase_support_seconds', prompt)
        self.assertNotIn('chair1', prompt)

    def test_seat_pronoun_schema_reuses_literal_antecedent(self):
        context = SimpleNamespace(text='Move a chair and inspect its seat.', candidates=[
            Candidate('chair1','chair1',0,0,0),Candidate('chair2','chair2',1,0,0)])
        raw = {'mentions': [
            {'text':'a chair','object_type':'chair','component':'','parent_text':'','coreference_id':'r0'},
            {'text':'its seat','object_type':'seat','component':'seat','parent_text':'a chair',
             'anchor_text':'a chair','coreference_id':'r0'}]}
        resolved = recover_component_schema(raw, context)
        self.assertEqual(resolved['mentions'][1]['parent_mention_id'], 'm0')
        self.assertEqual(resolved['mentions'][1]['candidate_ids'], ['chair1','chair2'])
        raw['mentions'][1]['coreference_id'] = 'r1'
        with self.assertRaises(ValueError):
            recover_component_schema(raw, context)


if __name__ == '__main__':
    unittest.main()
