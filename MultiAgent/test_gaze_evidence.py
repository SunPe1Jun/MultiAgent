"""Checks against geometric and observation-protocol failure modes."""
import copy
import unittest
import numpy as np
from PIL import Image
from .gaze_evidence import (project_gaze, select_gaze_observations, render_presentation,
                            interpolate_camera_pose, DEFAULTS)
from .telemetry import repair_telemetry_json
from .qwen_backend import QwenBackend
import json


def row(index, t, point=(0., 0., 5.)):
    return {'source_index':index, 'telemetry_time_seconds':t, 'validity':'valid',
            'gaze_point_world':point, 'gaze_vector_world':(0.,0.,1.),
            'camera_position':(0.,0.,0.),'camera_rotation_xyzw':(0.,0.,0.,1.),
            'camera_fov_degrees':90.}


POSE={'position_world':(0.,0.,0.),'rotation_xyzw':(0.,0.,0.,1.),'fov_vertical_degrees':90.}


class GeometryTests(unittest.TestCase):
    def test_vertical_fov_aspect_and_top_left(self):
        p=project_gaze(row(0,0,point=(2.,1.,4.)),POSE,800,400)
        self.assertEqual(p['status'],'projected_frustum_only')
        self.assertAlmostEqual(p['pixel_xy'][0],500)
        self.assertAlmostEqual(p['pixel_xy'][1],150)

    def test_camera_translation_applies_to_point_not_vector(self):
        pose={**POSE,'position_world':(2.,0.,0.)}
        point=project_gaze(row(0,0),pose,400,400)
        direction=row(0,0);direction['gaze_point_world']=None
        vector=project_gaze(direction,pose,400,400)
        self.assertLess(point['pixel_xy'][0],200)
        self.assertEqual(vector['pixel_xy'],[200.,200.])

    def test_camera_rotation_and_behind_camera(self):
        pose={**POSE,'rotation_xyzw':(0.,2**-.5,0.,2**-.5)}
        self.assertEqual(project_gaze(row(0,0,point=(5.,0.,0.)),pose,400,400)['status'],
                         'projected_frustum_only')
        self.assertEqual(project_gaze(row(0,0,point=(0.,0.,-1.)),POSE,400,400)['status'],'behind_camera')

    def test_outside_is_not_clamped_and_occlusion_is_not_inferred(self):
        p=project_gaze(row(0,0,point=(100.,0.,1.)),POSE,400,400)
        self.assertEqual(p['status'],'outside_frame');self.assertGreater(p['pixel_xy'][0],400)
        p=project_gaze(row(0,0),POSE,400,400)
        self.assertEqual(p['visibility'],'visibility_unknown')
        p=project_gaze(row(0,0),POSE,400,400,lambda *_:'verified_occluded')
        self.assertEqual(p['status'],'verified_occluded')

    def test_long_gap_pose_is_missing_not_extrapolated(self):
        self.assertIsNone(interpolate_camera_pose([row(0,0),row(1,5)],2.5))


class ObservationTests(unittest.TestCase):
    def test_history_excludes_future_and_caps_gap_duration(self):
        rows=[row(0,0),row(1,0.1),row(2,3),row(3,4)]
        selected=select_gaze_observations(rows,strategy='local_history',frame_time=3,
                 event_window=[0,4],clock_available=True)
        self.assertEqual([r['record']['source_index'] for r in selected['selected']],[2])
        self.assertLessEqual(sum(r['weight_seconds'] for r in selected['selected']),.25)

    def test_instant_missing_has_no_center_gaze(self):
        result=select_gaze_observations([row(0,0)],strategy='instant',frame_time=3,
                 event_window=[0,4],clock_available=True)
        self.assertEqual(result['selected'],[])
        rgb=Image.new('RGB',(60,40),(17,90,221));heat=np.zeros((40,60),np.float32)
        for method in ('H_local','F_local'):
            result,_=render_presentation(rgb,heat,method)
            np.testing.assert_array_equal(result,rgb)

    def test_unreliable_clock_disables_gaze(self):
        selected=select_gaze_observations([row(0,0)],strategy='utterance_window',frame_time=0,
                 event_window=[0,4],clock_available=False)
        self.assertEqual(selected['selected'],[])

    def test_fixed_scale_does_not_normalize_weak_observation_to_strong_peak(self):
        rgb=Image.new('RGB',(20,20),(120,120,120))
        weak=np.ones((20,20),np.float32)*.001;strong=weak*100
        a,_=render_presentation(rgb,weak,'H_local');b,_=render_presentation(rgb,strong,'H_local')
        self.assertGreater(np.asarray(b)[0,0,0],np.asarray(a)[0,0,0])

    def test_foveation_uses_multiscale_attention_and_keeps_dimensions(self):
        yy,xx=np.indices((64,64));checker=((xx+yy)%2*255).astype(np.uint8)
        rgb=Image.fromarray(np.repeat(checker[:,:,None],3,axis=2))
        heat=np.zeros((64,64),np.float32);heat[16:48,16:48]=1.
        rendered,stats=render_presentation(rgb,heat,'F_local')
        pixels=np.asarray(rendered)
        center_contrast=abs(int(pixels[31,31,0])-int(pixels[31,32,0]))
        edge_contrast=abs(int(pixels[2,2,0])-int(pixels[2,3,0]))
        self.assertEqual(rendered.size,rgb.size)
        self.assertEqual(stats['downsample_factors'],[2,4,8])
        self.assertGreater(center_contrast,edge_contrast)


class TransportAndTelemetryTests(unittest.TestCase):
    def test_missing_outer_array_and_final_bracket(self):
        self.assertEqual(len(json.loads(repair_telemetry_json('{"x":1},\n{"x":2}'))),2)
        self.assertEqual(json.loads(repair_telemetry_json('[{"x":"[text]"}')),[{'x':'[text]'}])
        with self.assertRaises(ValueError):json.loads(repair_telemetry_json('{"x":1}{"x":2}'))

    def test_base_url_has_one_v1_and_never_requires_local_gpu(self):
        for base in ('http://example.test:8000','http://example.test:8000/v1/'):
            client=QwenBackend('/tmp/unused-gaze-test','Qwen3-VL-8B-Instruct',base_url=base)
            self.assertEqual(client.base_url,'http://example.test:8000/v1')
            self.assertIsNone(client.runner)


if __name__=='__main__':unittest.main()
