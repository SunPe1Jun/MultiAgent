"""Clock-fit validation and source-frame pose correspondence contracts."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from .frame_pose_alignment import CameraPoseTimeline, block_partition, source_frame_pose


def row(index, t, yaw, position=(0., 0., 0.), fov=90.):
    return {'source_index': index, 'telemetry_time_seconds': t, 'camera_position': list(position),
            'camera_rotation_xyzw': Rotation.from_euler('y', yaw, degrees=True).as_quat().tolist(),
            'camera_fov_degrees': fov}


class CameraPoseTimelineTests(unittest.TestCase):
    def test_pose_at_uses_linear_position_and_slerp_with_provenance(self):
        timeline = CameraPoseTimeline([row(10, 0., 0.), row(11, .1, 10., (2., 4., 6.), 100.)])
        pose = timeline.pose_at(.05)
        self.assertEqual(pose['status'], 'POSE_AVAILABLE')
        self.assertEqual(pose['pose_source_indices'], [10, 11])
        self.assertEqual(pose['position_world'], [1., 2., 3.])
        self.assertAlmostEqual(pose['fov_vertical_degrees'], 95.)
        self.assertAlmostEqual(Rotation.from_quat(pose['rotation_xyzw']).magnitude(), np.deg2rad(5.))
        self.assertFalse(pose['extrapolated'])

    def test_pose_gap_and_extrapolation_fail_closed(self):
        timeline = CameraPoseTimeline([row(0, 0., 0.), row(1, .4, 20.)], max_gap_seconds=.25)
        self.assertEqual(timeline.pose_at(.2)['status'], 'POSE_UNAVAILABLE')
        self.assertEqual(timeline.pose_at(-.01)['status'], 'POSE_UNAVAILABLE')
        self.assertEqual(timeline.pose_at(.4)['status'], 'POSE_AVAILABLE')

    def test_duplicate_camera_timestamps_rejected(self):
        with self.assertRaisesRegex(ValueError, 'strictly'):
            CameraPoseTimeline([row(0, .1, 0.), row(1, .1, 1.)])

    def test_source_contract_requires_verified_frame_and_available_clock(self):
        timeline = CameraPoseTimeline([row(0, 0., 0.), row(1, .1, 10.)])
        frame = {'status': 'VERIFIED_EXACT_SOURCE_JPEG', 'actual_container_pts_seconds': .15}
        accepted = {'status': 'VALIDATED_CANDIDATE', 'offset_seconds': .05}
        self.assertEqual(source_frame_pose(timeline, frame, accepted)['status'], 'POSE_AVAILABLE')
        rejected = {'status': 'CLOCK_UNCERTAIN', 'offset_seconds': .05,
                    'rejection_reasons': ['time_blocks_disagree']}
        self.assertEqual(source_frame_pose(timeline, frame, rejected)['status'], 'NO_CLOCK_OFFSET')
        self.assertEqual(source_frame_pose(timeline, frame, rejected, diagnostic=True)['status'], 'POSE_AVAILABLE')
        self.assertEqual(source_frame_pose(timeline, {'status': 'UNVERIFIED'}, accepted)['status'], 'FRAME_TIME_UNVERIFIED')
        sequential = {'status': 'VERIFIED_CONTAINER_PTS_SEQUENCE', 'actual_container_pts_seconds': .15}
        self.assertEqual(source_frame_pose(timeline, sequential, accepted)['status'], 'POSE_AVAILABLE')

    def test_rejected_new_fit_uses_original_clock_even_without_rgb_support_gate(self):
        timeline = CameraPoseTimeline([row(0, 0., 0.), row(1, .1, 10.), row(2, .2, 20.)])
        frame = {'status': 'VERIFIED_CONTAINER_PTS_SEQUENCE', 'actual_container_pts_seconds': .15}
        clock = {'status': 'CLOCK_UNCERTAIN', 'offset_seconds': .03,
                 'original_offset_seconds': .05,
                 'original_clock_status': 'ORIGINAL_CLOCK_UNCERTAIN',
                 'rejection_reasons': ['time_blocks_disagree']}
        pose = source_frame_pose(timeline, frame, clock)
        self.assertEqual(pose['status'], 'POSE_AVAILABLE')
        self.assertEqual(pose['clock_status'], 'ORIGINAL_CLOCK_FALLBACK_PTS_POSE')
        self.assertAlmostEqual(pose['telemetry_time_seconds'], .10)
        self.assertAlmostEqual(source_frame_pose(timeline, frame, clock, diagnostic=True)['telemetry_time_seconds'], .12)

    def test_original_clock_fallback_keeps_frame_and_pose_availability_checks(self):
        timeline = CameraPoseTimeline([row(0, 0., 0.), row(1, .4, 20.)], max_gap_seconds=.25)
        clock = {'status': 'CLOCK_UNCERTAIN', 'offset_seconds': .03,
                 'original_offset_seconds': .05}
        unverified = {'status': 'UNVERIFIED', 'actual_container_pts_seconds': .15}
        self.assertEqual(source_frame_pose(timeline, unverified, clock)['status'], 'FRAME_TIME_UNVERIFIED')
        for pts in [.25, .55]:
            with self.subTest(pts=pts):
                frame = {'status': 'VERIFIED_CONTAINER_PTS_SEQUENCE', 'actual_container_pts_seconds': pts}
                pose = source_frame_pose(timeline, frame, clock)
                self.assertEqual(pose['status'], 'POSE_UNAVAILABLE')
                self.assertEqual(pose['clock_status'], 'ORIGINAL_CLOCK_FALLBACK_PTS_POSE')
                self.assertFalse(pose['extrapolated'])


class TemporalFoldTests(unittest.TestCase):
    def test_guarded_blocks_have_no_shared_endpoint_frames(self):
        config = {'block_seconds': 1.5, 'block_guard_seconds': .1}
        train = {'time_a': .2, 'time_b': .5, 'source_frame_indices': [2, 5]}
        validation = {'time_a': 1.7, 'time_b': 2., 'source_frame_indices': [17, 20]}
        edge = {'time_a': 1.4, 'time_b': 1.6, 'source_frame_indices': [14, 16]}
        self.assertEqual(block_partition(train, config), 'train')
        self.assertEqual(block_partition(validation, config), 'validation')
        self.assertEqual(block_partition(edge, config), 'guard')
        self.assertFalse(set(train['source_frame_indices']) & set(validation['source_frame_indices']))


if __name__ == '__main__':
    unittest.main()
