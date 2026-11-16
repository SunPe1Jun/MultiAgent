"""Cross-entry regressions for clock consistency and event-only referent projections."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from PIL import Image

from .alignment_projection_audit import audit_context, load_event_referents, project_referents, render_audit_frame
from .camera_geometry import CameraPoseTimeline
from .context import Candidate
from .frame_pose_alignment import source_frame_pose
from .local_video_visual import file_sha
from .projected2d_attention_study import make_condition_images, reproject_gt_manifest
from .test_alignment_projection import camera


class AlignmentProjectionAuditTests(unittest.TestCase):
    def test_audit_and_experiment_share_clock_and_reproject_only_event_referents(self):
        for status, original_status, pts, supported in [
                ('VALIDATED_CANDIDATE', 'ORIGINAL_CLOCK_UNCERTAIN', .25, True),
                ('CLOCK_UNCERTAIN', 'ORIGINAL_CLOCK_UNCERTAIN', .25, False),
                ('CLOCK_UNCERTAIN', 'ORIGINAL_CLOCK_SUPPORTED', .25, True),
                ('VALIDATED_CANDIDATE', 'ORIGINAL_CLOCK_UNCERTAIN', .05, False)]:
            with self.subTest(status=status, original_status=original_status, pts=pts), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root/'P1.jpg'
                Image.new('RGB', (100, 100), '#445566').save(source)
                labels, anchors = root/'labels.xlsx', root/'anchors.tsv'
                labels.write_text('fixture labels')
                anchors.write_text('fixture anchors')
                target = Candidate('target', 'target', 0., 0., 5.)
                other = Candidate('another_instance', 'another_instance', 1., 0., 5.)
                ctx = SimpleNamespace(scene_key='scene1', row_index=0, event_id='event1', text='Look at this.',
                    video_path='unused.mp4', api_row={'t_start': 0., 't_end': .2},
                    provenance={'gt_file': str(labels), 'anchor_table': str(anchors), 'ground_truth_loaded': False},
                    gt={'not_loaded_for_visual_only': True}, candidates=[target, other],
                    candidate=lambda name: target if name == 'target' else other if name == other.candidate_id else None)
                records = [camera(0, 0., 0.), camera(1, .1, 10.), camera(2, .2, 20.)]
                timeline = CameraPoseTimeline(records)
                clock = {'status': status, 'offset_seconds': .2, 'original_offset_seconds': .1,
                         'original_clock_status': original_status}
                frame_time = {'status': 'VERIFIED_CONTAINER_PTS_SEQUENCE', 'actual_container_pts_seconds': pts}
                original = source_frame_pose(timeline, frame_time, clock)
                attention = source_frame_pose(timeline, frame_time, clock, allow_candidate=True)
                selected_frame = {'frame_id': 'P1', 'source_frame_index': 5, 'time_seconds': pts,
                    'path': str(source), 'sha256': file_sha(source), 'width': 100, 'height': 100,
                    'camera_pose': original, 'attention_camera_pose': attention, 'frame_pts_binding': frame_time}
                fitted = []
                def bind(*args):
                    self.assertFalse(ctx.provenance['ground_truth_loaded'])
                    self.assertTrue(ctx.gt['not_loaded_for_visual_only'])
                    fitted.append(True)
                    return {'candidate_clock_status': status, 'original_offset_seconds': .1}
                def read_labels(*args):
                    self.assertEqual(fitted, [True], 'GT must be read after fitting')
                    return [{'instruction': ctx.text, 'mapped_referents': ['target', 'target'],
                             'unmapped_referents': [], 'excel_row_index': 2}]
                evaluator = SimpleNamespace(load_anchor_alias_map=Mock(return_value=({}, [])),
                                            load_gt_rows=Mock(side_effect=read_labels))
                with patch('MultiAgent.alignment_projection_audit.probe_duration', return_value=.5), \
                     patch('MultiAgent.alignment_projection_audit.extract_dense_frames', return_value={'frames': [selected_frame]}), \
                     patch('MultiAgent.alignment_projection_audit.bind_pose_contract', side_effect=bind), \
                     patch('MultiAgent.alignment_projection_audit.module', return_value=evaluator), \
                     patch('MultiAgent.alignment_projection_audit.gaze_records', return_value=records):
                    audit = audit_context(ctx, root/'audit', frames=2)
                panel = audit['frames'][0]
                self.assertEqual(audit['referents'], ['target'])
                self.assertEqual([p['anchor_id'] for p in panel['referent_projections']], ['target'])
                self.assertNotIn('all_scene_anchor_projections', panel)
                self.assertNotIn('another_instance', json.dumps(audit))
                self.assertEqual(panel['attention_clock_supported'], supported)
                self.assertEqual(ctx.gt, {'not_loaded_for_visual_only': True})
                study = root/'study'
                contract_path = study/'pose_alignment/scene1/0/frame_pose_contract.json'
                contract_path.parent.mkdir(parents=True)
                contract_path.write_text(json.dumps({'clock': clock, 'frames': [{
                    'frame_id': 'P1', 'source_sha256': file_sha(source), 'frame_time': frame_time,
                    'reference_camera': original, 'attention_camera': attention}]}))
                row = {'scene': 'scene1', 'row_index': '0', 'panel_id': 'P1', 'frame_path': str(source),
                    'video_frame_time': str(pts - .01), 'video_time_offset_seconds': '.1', 'referent_name': 'target',
                    'image_width': '100', 'image_height': '100', 'anchor_x': '0', 'anchor_y': '0', 'anchor_z': '5',
                    'gt_x': '50', 'gt_y': '50', 'gt_u_norm': '.5', 'gt_v_norm': '.5', 'projection_valid': 'True'}
                visual = SimpleNamespace(expand_panel_context_images=lambda key, paths, meta, out, args:
                    (paths, [{'crop_bbox_norm': '0,0,1,1', 'crop_source': 'test'}]))
                with patch('MultiAgent.projected2d_attention_study.load_module', return_value=visual), \
                     patch('MultiAgent.projected2d_attention_study.gaze_records', return_value=records):
                    inputs = make_condition_images([row], {('scene1', 0): ctx}, study, {'gaze': {}})
                meta = inputs[('scene1', 0)]['panel_meta'][0]
                self.assertEqual(meta['attention_clock_supported'], supported)
                self.assertEqual(meta['selected_clock_offset_seconds'], panel['pose']['clock_offset_seconds'])
                self.assertEqual(meta['selected_pose_clock_source'], panel['pose']['clock_status'])
                self.assertEqual(meta['telemetry_time_seconds'], panel['gaze_stats']['source_frame_time_telemetry_seconds'])
                with np.load(study/'inputs/scene1/0/P1/attention_audit.npz') as heat:
                    self.assertAlmostEqual(float(heat['history_H_unnormalized'].sum()),
                                           panel['gaze_stats']['unnormalized_heat_sum'], places=6)
                    self.assertEqual(bool(heat['history_H_unnormalized'].max() > 0), supported)
                rewritten = reproject_gt_manifest([row], study)
                projected = panel['referent_projections'][0]
                self.assertEqual(len(rewritten), 1)
                self.assertEqual(rewritten[0]['referent_name'], 'target')
                self.assertEqual(rewritten[0]['selected_clock_offset_seconds'], meta['selected_clock_offset_seconds'])
                if projected['in_frustum']:
                    self.assertAlmostEqual(float(rewritten[0]['gt_u_norm']), projected['normalized_uv_unclipped'][0], places=6)
                else:
                    self.assertEqual(rewritten[0]['gt_u_norm'], '')
                self.assertEqual(row['gt_u_norm'], '.5')
                self.assertEqual(rewritten[0]['original_gt_u_norm'], '.5')

    def test_missing_and_outside_referents_are_retained_without_invented_pixels(self):
        pose = CameraPoseTimeline([camera(0, 0.)]).pose_at(0.)
        targets = [{'referent_name': 'outside', 'anchor_id': 'outside', 'point_world': [20., 0., 5.], 'mapping_status': 'mapped'},
                   {'referent_name': 'missing', 'anchor_id': None, 'point_world': None, 'mapping_status': 'unmapped_referent'}]
        projected = project_referents(targets, pose, pose, 100, 100)
        self.assertEqual([r['status'] for r in projected], ['outside_frame', 'unmapped_referent'])
        self.assertGreater(projected[0]['normalized_uv_unclipped'][0], 1.)
        self.assertIsNone(projected[1]['pixel_xy_unclipped'])
        unavailable = project_referents(targets[:1], {}, pose, 100, 100)
        self.assertEqual(unavailable[0]['status'], 'pose_unavailable')
        self.assertIsNone(unavailable[0]['pixel_xy_unclipped'])

    def test_selected_clock_marker_is_drawn_when_original_projection_is_outside(self):
        target = {'referent_name': 'target', 'in_frustum': True, 'status': 'in_frustum_visibility_unknown',
                  'pixel_xy_unclipped': [50., 50.], 'original_projection': {'in_frustum': False, 'status': 'outside_frame'}}
        geometry = {'video_time_seconds': .25, 'selected_pose_clock_source': 'VALIDATED_CANDIDATE',
                    'selected_clock_offset_seconds': .2, 'clock_available': True}
        rendered = render_audit_frame(Image.new('RGB', (100, 100)), np.zeros((100, 100)), [target],
                                      scene='scene1', row_index=0, frame_id='P1', geometry=geometry)
        self.assertEqual(rendered.getpixel((50, 50)), (255, 152, 0))

    def test_mismatched_gt_row_cannot_silently_project_other_event_targets(self):
        ctx = SimpleNamespace(scene_key='scene1', row_index=0, event_id='event', text='This interaction.',
                              provenance={'gt_file': 'labels.xlsx', 'anchor_table': 'anchors.tsv'})
        evaluator = SimpleNamespace(load_anchor_alias_map=Mock(return_value=({}, [])),
            load_gt_rows=Mock(return_value=[{'instruction': 'Another interaction.', 'mapped_referents': ['wrong']}]))
        with patch('MultiAgent.alignment_projection_audit.module', return_value=evaluator), \
             self.assertRaisesRegex(ValueError, 'does not match'):
            load_event_referents(ctx)


if __name__ == '__main__':
    unittest.main()
