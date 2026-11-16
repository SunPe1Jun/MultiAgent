"""Geometry invariants and explicit referent coverage, without model/GT fitting."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation

from .agents import ReferentialAgent
from .context import Candidate
from .mention_mapping import actor_coverage_checks, map_phrase, resolve_mentions
from .projection_diagnostics import project_world_anchor
from .camera_motion_alignment import evaluate_candidate


def sample(position=(0, 0, 0), yaw=0):
    return {'cameraPosition': dict(zip('xyz', position)),
            'cameraRotation': dict(zip('xyzw', Rotation.from_euler('y', yaw, degrees=True).as_quat())),
            'cameraFOV': 90}


class GeometryDiagnosticsTests(unittest.TestCase):
    def test_world_anchor_can_leave_frame_when_camera_moves_or_rotates(self):
        point = (0, 0, 3)
        self.assertTrue(project_world_anchor(point, sample(), 100, 100)['in_frustum'])
        moved = project_world_anchor(point, sample(position=(5, 0, 0)), 100, 100)
        self.assertEqual(moved['status'], 'outside_frame')
        self.assertLess(moved['normalized_uv_unclipped'][0], 0)
        self.assertEqual(project_world_anchor(point, sample(yaw=180), 100, 100)['status'], 'behind_camera')

    def test_aspect_ratio_and_top_left_origin(self):
        inside = project_world_anchor((1, 1, 2), sample(), 200, 100)
        self.assertTrue(inside['in_frustum'])
        np.testing.assert_allclose(inside['normalized_uv_unclipped'], [.625, .25])
        self.assertEqual(inside['occlusion'], 'unknown_no_depth_or_mesh')

    def test_same_translation_of_anchor_and_camera_preserves_projection(self):
        a = project_world_anchor((1, 0, 3), sample(), 100, 100)
        b = project_world_anchor((8, 2, 2), sample(position=(7, 2, -1)), 100, 100)
        np.testing.assert_allclose(a['normalized_uv_unclipped'], b['normalized_uv_unclipped'])

    def test_rgb_motion_projection_uses_camera_a_to_camera_b_rotation(self):
        world = [(x, y, 5) for x in [-1, -.5, 0, .5, 1] for y in [-.5, 0, .5]]
        a, b = sample(), sample(yaw=12)
        to_pixels = lambda pose: [project_world_anchor(p, pose, 640, 640)['pixel_xy_unclipped'] for p in world]
        pair = {'time_a': 0., 'time_b': 1., 'size_hw': [640, 640], 'recorded_tan_fov': 1.,
                'points_a': to_pixels(a), 'points_b': to_pixels(b)}
        q = np.array([[pose['cameraRotation'][k] for k in 'xyzw'] for pose in [a, b]])
        result = evaluate_candidate([pair], np.array([0., 1.]), q, 0., 1.)
        self.assertLess(result['median_pair_error_px'], 1e-9)

    def test_invalid_pose_is_not_visible(self):
        invalid = sample()
        invalid['cameraRotation'] = dict.fromkeys('xyzw', 0)
        self.assertEqual(project_world_anchor((0, 0, 3), invalid, 100, 100)['status'], 'invalid_geometry')

    def test_actor_omission_triggers_language_only_retry(self):
        ctx = SimpleNamespace(text='Ask the other person to move that truck.', scene_key='scene1', event_id='test',
                              candidates=[Candidate('person1', 'person1', 0, 0, 0), Candidate('truck1', 'truck1', 0, 0, 0)])
        missing = {'mentions': [{'text': 'that truck', 'object_type': 'truck'}]}
        complete = {'mentions': [{'text': 'the other person', 'object_type': 'person', 'role': 'actor'},
                                 {'text': 'that truck', 'object_type': 'truck'}]}
        backend = Mock()
        backend.ask.side_effect = [missing, complete]
        result = ReferentialAgent(backend).run(ctx)
        self.assertEqual(backend.ask.call_count, 2)
        self.assertEqual(result['mentions'][0]['text'], 'the other person')
        self.assertIn('Missing explicit actor', result['validation_retries'][0]['error'])

    def test_fellow_maps_only_to_local_people(self):
        local = [Candidate('person2', 'person2', 0, 0, 0), Candidate('chair7', 'chair7', 0, 0, 0)]
        self.assertEqual(map_phrase('fellow', local), ['person2'])

    def test_bare_pronoun_parent_uses_explicit_matching_antecedent(self):
        ctx = SimpleNamespace(text='Move that truck and align it beside the building.',
                              candidates=[Candidate('truck1', 'truck1', 0, 0, 0)])
        raw = {'mentions': [{'text': 'that truck', 'object_type': 'truck', 'coreference_id': 'r1'},
                            {'text': 'it', 'object_type': 'truck', 'parent_text': 'that truck',
                             'anchor_text': 'that truck', 'coreference_id': 'r1'}]}
        parsed = resolve_mentions(ctx, raw)
        self.assertEqual(parsed['mentions'][1]['parent_text'], '')
        self.assertEqual(parsed['mentions'][1]['parent_mention_id'], 'm0')
        self.assertEqual(raw['mentions'][1]['parent_text'], 'that truck')
        invalid = copy.deepcopy(raw)
        invalid['mentions'][1]['coreference_id'] = 'r2'
        with self.assertRaisesRegex(ValueError, 'parent_text'):
            resolve_mentions(ctx, invalid)

    def test_spatial_relation_does_not_suppress_its_concrete_anchor(self):
        ctx = SimpleNamespace(text='Move to the front of the side building.',
                              candidates=[Candidate('building1', 'building1', 0, 0, 0)])
        raw = {'mentions': [{'text': 'the front of the side building', 'object_type': 'location',
                             'kind': 'location', 'parent_text': 'the side building'},
                            {'text': 'the side building', 'object_type': 'building'}]}
        parsed = resolve_mentions(ctx, raw)
        self.assertEqual(len(parsed['mentions']), 2)
        self.assertEqual(parsed['suppressed_nested_mentions'], [])


if __name__ == '__main__':
    unittest.main()
