"""GT-free rotation timing sensitivity invariants."""
import unittest
import hashlib
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from .head_motion_diagnostics import inspect_time, rotation_sensitivity, verify_saved_frame_pts


def record(t, yaw):
    return {'source_index': round(t * 10), 'telemetry_time_seconds': t,
            'camera_position': [0., 0., 0.],
            'camera_rotation_xyzw': Rotation.from_euler('y', yaw, degrees=True).as_quat().tolist(),
            'camera_fov_degrees': 90.}


class HeadMotionDiagnosticsTests(unittest.TestCase):
    def test_static_camera_is_insensitive_to_time_and_sample_lookup(self):
        result = inspect_time([record(0, 0), record(.1, 0), record(.2, 0)], .05, 1920, 1920)
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual(result['bracket_angular_speed_deg_s'], 0.)
        self.assertLess(result['nearest_vs_interpolated']['max_ray_displacement_px'], 1e-8)
        for perturbation in result['time_perturbations']:
            self.assertLess(perturbation['max_ray_displacement_px'], 1e-8)

    def test_fast_turn_interpolation_records_timing_quantization_without_gt(self):
        result = inspect_time([record(0, 0), record(.1, 10), record(.2, 20)], .05, 1920, 1920)
        self.assertAlmostEqual(result['bracket_angular_speed_deg_s'], 100.)
        self.assertAlmostEqual(abs(result['nearest_pose_time_delta_seconds']), .05)
        self.assertGreater(result['nearest_vs_interpolated']['median_ray_displacement_px'], 80.)
        forward = next(p for p in result['time_perturbations'] if p['delta_seconds'] == .02)
        self.assertGreater(forward['median_ray_displacement_px'], 30.)

    def test_pixels_scale_with_source_resolution(self):
        q0, q1 = [record(0, yaw)['camera_rotation_xyzw'] for yaw in (0, 2)]
        a = rotation_sensitivity(q0, q1, 640, 640, 90)
        b = rotation_sensitivity(q0, q1, 1920, 1920, 90)
        self.assertAlmostEqual(b['median_ray_displacement_px'], 3 * a['median_ray_displacement_px'])

    def test_exact_saved_frame_pts_requires_matching_jpeg_and_container_time(self):
        jpeg = b'exact frozen source image bytes'
        panel = {'video_time_seconds': 5.139, 'source_sha256': hashlib.sha256(jpeg).hexdigest()}
        decoded = SimpleNamespace(returncode=0, stdout=jpeg,
            stderr=b'[Parsed_showinfo_0] n: 0 pts: 464935 pts_time:5.165944 duration:3718\n')
        with patch('MultiAgent.head_motion_diagnostics.subprocess.run', return_value=decoded):
            result = verify_saved_frame_pts('video.mp4', panel, [5.123922, 5.165944, 5.207256], 'ffmpeg')
        self.assertEqual(result['status'], 'VERIFIED_EXACT_SOURCE_JPEG')
        self.assertEqual(result['source_frame_index'], 1)
        self.assertAlmostEqual(result['pts_minus_requested_seconds'], .026944)

    def test_nonmatching_jpeg_never_claims_exact_source_timestamp(self):
        decoded = SimpleNamespace(returncode=0, stdout=b'different image',
            stderr=b'[Parsed_showinfo_0] n: 0 pts: 464935 pts_time:5.165944 duration:3718\n')
        with patch('MultiAgent.head_motion_diagnostics.subprocess.run', return_value=decoded):
            result = verify_saved_frame_pts('video.mp4', {'video_time_seconds': 5.139, 'source_sha256': 'wrong'},
                                            [5.165944], 'ffmpeg')
        self.assertEqual(result['status'], 'UNVERIFIED')
        self.assertNotIn('actual_container_pts_seconds', result)

    def test_large_gaps_and_out_of_range_times_do_not_fabricate_pose(self):
        records = [record(0, 0), record(1, 10)]
        self.assertEqual(inspect_time(records, .5, 640, 640)['status'], 'pose_unavailable_no_extrapolation')
        self.assertEqual(inspect_time(records, 2, 640, 640)['status'], 'pose_unavailable_no_extrapolation')
        self.assertEqual(inspect_time([], 0, 640, 640)['status'], 'insufficient_camera_samples')


if __name__ == '__main__':
    unittest.main()
