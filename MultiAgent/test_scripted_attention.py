import unittest
import numpy as np
from PIL import Image
from .scripted_attention import spatial_foveation,crop_at,summarize_projected_history


class SpatialApertureTests(unittest.TestCase):
    def test_supported_focus_preserves_actual_source_detail(self):
        yy,xx=np.indices((160,160));pixels=np.repeat((((xx+yy)%2)*255).astype('uint8')[...,None],3,axis=2)
        source=Image.fromarray(pixels)
        rendered,stats=spatial_foveation(source,[80.,80.],clock_supported=True,vertical_fov=90.)
        np.testing.assert_array_equal(np.asarray(rendered)[75:85,75:85],pixels[75:85,75:85])
        self.assertFalse(np.array_equal(np.asarray(rendered)[:10,:10],pixels[:10,:10]))
        self.assertEqual(stats['maximum_original_weight'],1.)
        self.assertFalse(stats['evidence_amplitude_used'])

    def test_unsupported_clock_or_invalid_gaze_is_exact_rgb(self):
        source=Image.new('RGB',(80,60),(17,103,211))
        for point,supported in [([40,30],False),(None,True),([81,30],True),([float('nan'),30],True)]:
            rendered,stats=spatial_foveation(source,point,clock_supported=supported,vertical_fov=90.)
            np.testing.assert_array_equal(rendered,source)
            self.assertFalse(stats['aperture_enabled'])

    def test_edge_crop_keeps_query_point_and_reports_padding(self):
        image=Image.new('RGB',(100,100),'red')
        view,meta=crop_at(image,[3.,5.],.4)
        self.assertGreater(meta['padding_fraction'],0.)
        self.assertEqual(view.getpixel(tuple(map(int,meta['point_in_view']))),(255,0,0))
        self.assertEqual(meta['bbox_source_pixels'][0]+meta['point_in_view'][0],3.)


class ScriptedHistoryTests(unittest.TestCase):
    def record(self,time,point,weight=1.,index=0):
        return {'status':'projected_frustum_only','source_index':index,
                'gaze_time_seconds':time,'weight_seconds':weight,'pixel_xy':point}

    def test_bins_preserve_time_and_actual_observations(self):
        rows=[self.record(8.,[10,10],.3,0),self.record(8.5,[80,80],.1,1),
              self.record(9.,[20,20],.2,2),self.record(10.,[90,10],.4,3)]
        summary=summarize_projected_history(rows,query_time=10.,image_size=(100,100))
        bins=summary['bins_old_to_new']
        self.assertEqual([r['projected_sample_count'] for r in bins],[2,1,1])
        self.assertEqual(bins[0]['representative_observed_point']['pixel_xy'],[10.,10.])
        self.assertEqual(bins[2]['representative_observed_point']['gaze_time_seconds'],10.)
        for b in bins:
            self.assertAlmostEqual(float(np.sum(b['cell_duration_fraction'])),1.)
            self.assertAlmostEqual(float(np.sum(b['cell_duration_seconds'])),b['frustum_supported_duration_seconds'])
        self.assertNotIn('object_id',summary)

    def test_future_or_out_of_window_projection_is_rejected(self):
        for t in [7.9,10.1]:
            with self.assertRaisesRegex(ValueError,'past-only'):
                summarize_projected_history([self.record(t,[10,10])],query_time=10.,image_size=(100,100))

    def test_unsupported_and_zero_duration_do_not_create_target(self):
        rows=[{'status':'outside_frame'},self.record(9.,[10,10],0.)]
        result=summarize_projected_history(rows,query_time=10.,image_size=(100,100))
        self.assertEqual(result['excluded_projection_status_counts'],{'outside_frame':1})
        for b in result['bins_old_to_new']:
            self.assertIsNone(b['representative_observed_point'])
            self.assertIsNone(b['cell_duration_fraction'])


class RegionSelectionTests(unittest.TestCase):
    def test_stationary_sequence_keeps_three_distinct_frames(self):
        from .scripted_region_study import select_frames
        rows=[{'frame_id':f'V{i:03d}','pts':float(i),'point':[10,10],
               'clock_supported':True,'angular_speed_deg_s':0.} for i in range(5)]
        selected=select_frames(rows)
        self.assertEqual(len({r['frame_id'] for r in selected}),3)
        self.assertEqual([r['pts'] for r in selected],sorted(r['pts'] for r in selected))


if __name__=='__main__':unittest.main()
