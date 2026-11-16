"""Regression checks for actual frame addresses, camera contracts and clock selection."""
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .camera_geometry import CameraPoseTimeline, aligned_frame_geometry
from .frame_pose_alignment import DEFAULT_CONFIG, MotionScorer, block_partition, evaluate_fits, fit_offset, source_frame_pose
from .gaze_evidence import build_heatmap, interpolate_camera_pose, project_gaze
from .projection_diagnostics import project_world_anchor
from .video_timing import bound_frame_pts, frame_pts_table


def camera(index, t, yaw=0., fov=90.):
    return {'source_index': index, 'telemetry_time_seconds': t,
            'camera_position': [0., 0., 0.],
            'camera_rotation_xyzw': Rotation.from_euler('y', yaw, degrees=True).as_quat().tolist(),
            'camera_fov_degrees': fov, 'validity': 'valid',
            'gaze_point_world': [0., 0., 5.], 'gaze_vector_world': [0., 0., 1.],
            'timestamp': str(t), 'quality_available': False}


class SharedGeometryTests(unittest.TestCase):
    def test_nonfinite_query_is_unavailable_without_breaking_other_queries(self):
        timeline = CameraPoseTimeline([camera(0, 0.), camera(1, .1, 10.)])
        valid, matrices, fov = timeline.sample_many([float('nan'), .05, float('inf')])
        self.assertEqual(valid.tolist(), [False, True, False])
        self.assertEqual(timeline.pose_at(float('nan'))['status'], 'POSE_UNAVAILABLE')
        self.assertTrue(np.isfinite(matrices).all())

    def test_invalid_camera_values_rejected_and_gaze_does_not_bridge_the_gap(self):
        for key, value in [('camera_fov_degrees', 180.), ('camera_position', [0., float('nan'), 0.]),
                           ('camera_rotation_xyzw', [0., 0., 0., 0.]), ('telemetry_time_seconds', float('nan'))]:
            rows = [camera(0, 0.), camera(1, .1)]
            rows[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                CameraPoseTimeline(rows)
        rows = [camera(0, 0.), camera(1, .2), camera(2, .4)]
        rows[1]['camera_fov_degrees'] = float('nan')
        self.assertIsNone(interpolate_camera_pose(rows, .2))

    def test_single_sample_only_supports_its_exact_timestamp(self):
        rows = [camera(8, .4)]
        self.assertEqual(interpolate_camera_pose(rows, .4)['pose_source_indices'], [8, 8])
        self.assertIsNone(interpolate_camera_pose(rows, .400001))

    def test_both_pose_paths_use_the_same_slerp_and_fov(self):
        rows = [camera(3, 0., 170., 60.), camera(9, .1, -170., 100.)]
        fitted = CameraPoseTimeline(rows).pose_at(.05)
        gaze = interpolate_camera_pose(rows, .05)
        np.testing.assert_allclose(fitted['rotation_xyzw'], gaze['rotation_xyzw'])
        self.assertEqual(fitted['pose_source_indices'], gaze['pose_source_indices'])
        self.assertEqual(fitted['fov_vertical_degrees'], 80.)
        self.assertEqual(CameraPoseTimeline(rows).pose_at(.1)['pose_source_indices'], [9, 9])

    def test_gaze_and_anchor_projection_agree_for_aspect_rotation_and_scaled_quaternion(self):
        pose = CameraPoseTimeline([camera(0, 0., 15.)]).pose_at(0.)
        pose['position_world'] = [2., -1., .5]
        pose['rotation_xyzw'] = (np.asarray(pose['rotation_xyzw']) * 3).tolist()
        point = [3., 0., 5.]
        gaze = project_gaze({**camera(0, 0.), 'gaze_point_world': point}, pose, 800, 400)
        sample = {'cameraPosition':dict(zip('xyz', pose['position_world'])),
                  'cameraRotation':dict(zip('xyzw', pose['rotation_xyzw'])),
                  'cameraFOV':pose['fov_vertical_degrees']}
        anchor = project_world_anchor(point, sample, 800, 400)
        np.testing.assert_allclose(gaze['pixel_xy'], anchor['pixel_xy_unclipped'])
        self.assertEqual(gaze['in_frustum'], anchor['in_frustum'])

    def test_invalid_image_or_pose_has_no_finite_looking_uv(self):
        pose = CameraPoseTimeline([camera(0, 0.)]).pose_at(0.)
        for size in [(0, 100), (100, 0), (float('nan'), 100)]:
            result = project_gaze(camera(0, 0.), pose, *size)
            self.assertEqual(result['status'], 'invalid_geometry')
            self.assertNotIn('normalized_uv', result)
        pose['fov_vertical_degrees'] = float('nan')
        self.assertEqual(project_gaze(camera(0, 0.), pose, 100, 100)['status'], 'invalid_geometry')

    def test_heatmap_kernel_uses_reference_camera_fov_not_clip_median(self):
        records = [camera(0, 0., fov=120.), camera(1, .1, fov=120.), camera(2, .2, fov=30.)]
        pose = CameraPoseTimeline(records).pose_at(.2)
        heat, stats, _ = build_heatmap(None, records, frame_time_video=.2, frame_time_telemetry=.2,
            event_window=[0., .2], clock_available=True, image_size=(200, 100),
            strategy='instant', reference_pose=pose)
        expected = 100 * np.deg2rad(2.) / (2 * np.tan(np.deg2rad(30.) / 2))
        self.assertAlmostEqual(stats['kernel_sigma_pixels_at_model_input'], expected)
        self.assertEqual(stats['kernel_reference_fov_degrees'], 30.)
        self.assertGreater(heat.max(), 0.)

    def test_nonfinite_clock_does_not_get_a_reference_pose(self):
        timeline = CameraPoseTimeline([camera(0, 0.), camera(1, .1)])
        frame = {'status':'VERIFIED_CONTAINER_PTS_SEQUENCE', 'actual_container_pts_seconds': .05}
        self.assertEqual(source_frame_pose(timeline, frame,
            {'status':'VALIDATED_CANDIDATE','offset_seconds':float('nan')})['status'], 'CLOCK_UNAVAILABLE')
        frame['actual_container_pts_seconds'] = float('nan')
        self.assertEqual(source_frame_pose(timeline, frame,
            {'status':'VALIDATED_CANDIDATE','offset_seconds':0.})['status'], 'FRAME_TIME_UNVERIFIED')


class FrameContractTests(unittest.TestCase):
    def setUp(self):
        self.pose = CameraPoseTimeline([camera(0, 0.), camera(1, .1)]).pose_at(.05)
        self.pose.update(clock_offset_seconds=.2, clock_status='VALIDATED_CANDIDATE')
        self.frame = {'source_sha256':'actual-image', 'reference_camera':self.pose,
                      'frame_time':{'status':'VERIFIED_CONTAINER_PTS_SEQUENCE',
                                    'actual_container_pts_seconds':.25}}

    def test_sequential_binding_is_supported_and_uses_actual_not_requested_time(self):
        result = aligned_frame_geometry(self.frame,'actual-image',.24,0.)
        self.assertTrue(result['clock_available'])
        self.assertEqual(result['video_time_seconds'], .25)
        self.assertEqual(result['telemetry_time_seconds'], .05)

    def test_wrong_image_and_inconsistent_clock_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'SHA256'):
            aligned_frame_geometry(self.frame,'other-image',.24,0.)
        self.pose['telemetry_time_seconds'] = .1
        with self.assertRaisesRegex(ValueError,'inconsistent'):
            aligned_frame_geometry(self.frame,'actual-image',.24,0.)

    def test_uncertain_original_clock_keeps_pose_but_disables_attention(self):
        self.pose.update(clock_status='ORIGINAL_CLOCK_FALLBACK_PTS_POSE',
                         original_clock_status='ORIGINAL_CLOCK_UNCERTAIN')
        result = aligned_frame_geometry(self.frame,'actual-image',.24,0.)
        self.assertTrue(result['pose_available'])
        self.assertIsNotNone(result['reference_pose'])
        self.assertFalse(result['clock_available'])
        self.pose['original_clock_status'] = 'ORIGINAL_CLOCK_SUPPORTED'
        self.assertTrue(aligned_frame_geometry(self.frame,'actual-image',.24,0.)['clock_available'])

    def test_explicit_attention_camera_controls_time_pose_and_provenance(self):
        original = dict(self.pose, telemetry_time_seconds=.15, clock_offset_seconds=.1,
                        clock_status='ORIGINAL_CLOCK_REFERENCE_PTS_POSE')
        self.frame.update(reference_camera=original, attention_camera=self.pose)
        result = aligned_frame_geometry(self.frame, 'actual-image', .24, .1)
        self.assertEqual(result['camera_source_field'], 'attention_camera')
        self.assertEqual(result['selected_clock_offset_seconds'], .2)
        self.assertEqual(result['selected_pose_clock_source'], 'VALIDATED_CANDIDATE')
        self.assertEqual(result['telemetry_time_seconds'], .05)
        self.assertEqual(original['telemetry_time_seconds'], .15)

    def test_unavailable_attention_pose_cannot_reuse_original_at_a_different_time(self):
        self.frame['attention_camera'] = {'status': 'POSE_UNAVAILABLE',
            'clock_status': 'VALIDATED_CANDIDATE', 'clock_offset_seconds': .3,
            'telemetry_time_seconds': -.05}
        result = aligned_frame_geometry(self.frame, 'actual-image', .24, .1)
        self.assertFalse(result['clock_available'])
        self.assertFalse(result['pose_available'])
        self.assertIsNone(result['reference_pose'])
        self.assertEqual(result['camera_source_field'], 'attention_camera')
        self.assertAlmostEqual(result['telemetry_time_seconds'], -.05)

    def test_diagnostic_or_rejected_candidate_cannot_enable_attention_with_a_stale_flag(self):
        for update in [{'clock_status': 'REJECTED_CANDIDATE_DIAGNOSTIC_ONLY'},
                       {'diagnostic_only': True}, {'candidate_clock_status': 'CLOCK_UNCERTAIN'}]:
            with self.subTest(update=update):
                pose = {**self.pose, 'attention_clock_supported': True, **update}
                frame = {**self.frame, 'attention_camera': pose}
                self.assertFalse(aligned_frame_geometry(frame, 'actual-image', .24, 0.)['clock_available'])

    def test_nonfinite_attention_query_cannot_reuse_a_valid_reference_pose(self):
        with self.assertRaisesRegex(ValueError,'disagree'):
            build_heatmap(None,[camera(0,0.),camera(1,.1)],frame_time_video=.25,
                frame_time_telemetry=float('nan'),event_window=[0.,.1],clock_available=True,
                image_size=(100,100),strategy='instant',reference_pose=self.pose)

    def test_missing_actual_pts_does_not_turn_requested_time_into_verified_time(self):
        self.frame['frame_time']['actual_container_pts_seconds'] = None
        result = aligned_frame_geometry(self.frame,'actual-image',.24,.1)
        self.assertFalse(result['clock_available'])
        self.assertEqual(result['video_time_seconds'], .24)

    def test_attention_pose_and_observation_window_must_share_one_time(self):
        with self.assertRaisesRegex(ValueError,'disagree'):
            build_heatmap(None, [camera(0, 0.),camera(1,.1)], frame_time_video=.25,
                frame_time_telemetry=.1,event_window=[0.,.1],clock_available=True,
                image_size=(100,100),strategy='instant',reference_pose=self.pose)

    def test_validated_candidate_is_not_the_default_world_projection_reference(self):
        timeline=CameraPoseTimeline([camera(0,0.),camera(1,.1),camera(2,.2)])
        frame={'status':'VERIFIED_CONTAINER_PTS_SEQUENCE','actual_container_pts_seconds':.15}
        clock={'status':'VALIDATED_CANDIDATE','offset_seconds':.10,
               'original_offset_seconds':.05,'original_clock_status':'ORIGINAL_CLOCK_SUPPORTED'}
        original=source_frame_pose(timeline,frame,clock)
        candidate=source_frame_pose(timeline,frame,clock,allow_candidate=True)
        self.assertAlmostEqual(original['clock_offset_seconds'],.05)
        self.assertEqual(original['clock_status'],'ORIGINAL_CLOCK_REFERENCE_PTS_POSE')
        self.assertAlmostEqual(candidate['clock_offset_seconds'],.10)
        self.assertEqual(candidate['clock_status'],'VALIDATED_CANDIDATE')
        self.assertNotEqual(original['telemetry_time_seconds'],candidate['telemetry_time_seconds'])


class FrameAddressTests(unittest.TestCase):
    def test_vfr_nonzero_origin_preserves_original_pts(self):
        self.assertEqual(frame_pts_table([{'best_effort_timestamp_time':t} for t in ['1.0','1.041','1.109']]),
                         [1., 1.041, 1.109])

    def test_index_binding_rejects_negative_fractional_and_nonfinite_values(self):
        for index in [-1, 2, .5, True, '1']:
            with self.subTest(index=index), self.assertRaises(ValueError):
                bound_frame_pts(index, .1, [0., .1])
        with self.assertRaises(ValueError):
            bound_frame_pts(1, float('nan'), [0., .1])
        self.assertEqual(bound_frame_pts(1, .1, [0., .1]), .1)

    def test_missing_duplicate_and_nonfinite_pts_fail_instead_of_renumbering(self):
        for rows in [[{'best_effort_timestamp_time':'0'}, {}, {'best_effort_timestamp_time':'.1'}],
                     [{'best_effort_timestamp_time':'0'}, {'best_effort_timestamp_time':'0'}],
                     [{'best_effort_timestamp_time':'nan'}], []]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                frame_pts_table(rows)


class ClockCoverageTests(unittest.TestCase):
    def test_low_residual_cannot_hide_lost_original_validation_pairs(self):
        # Six easy pairs remain; losing fourteen hard pairs must not validate the clock.
        pairs = [{'fold':'validation'} for _ in range(20)]
        original, candidate = np.full(20, 2.), np.r_[np.ones(6), np.full(14, np.nan)]
        scorer = SimpleNamespace(folds=np.array(['validation'] * 20), motion=np.full(20, 4.),
                                 pairs=pairs, errors=lambda offset: original if offset == 0. else candidate)
        config = {**DEFAULT_CONFIG, 'minimum_coverage':.25}
        fit = {'offset_seconds':.1,'profile_width_seconds':.02}
        result = evaluate_fits(scorer, fit, 0., None, fit, config)
        self.assertIn('candidate_loses_original_validation_coverage', result['rejection_reasons'])
        self.assertAlmostEqual(result['original_validation_retention_fraction'], .3)
        self.assertEqual(result['status'], 'CLOCK_UNCERTAIN')

    def test_motion_scoring_reuses_errors_without_exposing_a_mutable_cache(self):
        records = [camera(0, 0.), camera(1, .1)]
        pair = {'time_a':0., 'time_b':.1, 'points_a':[[100.,100.]], 'points_b':[[100.,100.]],
                'size_hw':[200,200], 'fold':'train', 'median_observed_displacement_px':3.}
        scorer = MotionScorer([pair], CameraPoseTimeline(records))
        first = scorer.errors(0.)
        self.assertIs(first, scorer.errors(0.))
        self.assertFalse(first.flags.writeable)
        self.assertAlmostEqual(first[0], 0.)


class SyntheticClockRecoveryTests(unittest.TestCase):
    def test_known_rgb_telemetry_lag_recovered_without_scene_labels(self):
        yaw = lambda t: 13*np.sin(3*t) + 5*np.sin(7*t)
        records = [camera(i,float(t),yaw(t)) for i,t in enumerate(np.arange(0.,4.6,.02))]
        offset = .2
        points = np.array([(x,y,8.) for x in [-2.,-1.,0.,1.,2.] for y in [-1.,0.,1.]])
        config = {**DEFAULT_CONFIG,'offset_bounds_seconds':[-.4,.4], 'minimum_pairs':3}
        pairs = []
        for start in np.arange(.4,4.1,.12):
            end = start+.16
            def pixels(video_time):
                local = Rotation.from_euler('y',yaw(video_time-offset),degrees=True).inv().apply(points)
                return np.column_stack((320+320*local[:,0]/local[:,2],240-320*local[:,1]/local[:,2]))
            # Vertical FOV corresponding to fx=fy=320 for height=480.
            fov = np.rad2deg(2*np.arctan(480/640))
            a,b = pixels(start),pixels(end)
            pair = {'time_a':float(start),'time_b':float(end),'points_a':a.tolist(),
                    'points_b':b.tolist(),'size_hw':[480,640],
                    'median_observed_displacement_px':float(np.median(np.linalg.norm(a-b,axis=1)))}
            pair['fold'] = block_partition(pair,config)
            pairs.append(pair)
        for row in records:row['camera_fov_degrees']=float(fov)
        scorer = MotionScorer(pairs,CameraPoseTimeline(records))
        fit = fit_offset(scorer,'train',config)
        reverse = fit_offset(scorer,'validation',config)
        self.assertAlmostEqual(fit['offset_seconds'],offset,delta=.005)
        result = evaluate_fits(scorer,fit,0.,None,reverse,config)
        self.assertEqual(result['status'],'VALIDATED_CANDIDATE',result['rejection_reasons'])


if __name__ == '__main__':
    unittest.main()
