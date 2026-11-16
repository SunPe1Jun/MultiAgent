import unittest
import numpy as np
from PIL import Image
from .attention_volume import build_volume,support_intervals,mass_region,render_volume,heat_guided_crop


def record(t,x=0.,valid=True,index=0):
    return {'telemetry_time_seconds':t,'source_index':index,'validity':'valid' if valid else 'invalid_sample',
            'gaze_point_world':[x,0.,10.],'gaze_vector_world':None}


def pose(query,angle=0.):
    from scipy.spatial.transform import Rotation
    return {'telemetry_time_seconds':query,'position_world':[0,0,0],
            'rotation_xyzw':Rotation.from_euler('y',angle,degrees=True).as_quat().tolist(),
            'fov_vertical_degrees':90.}


class AttentionVolumeTests(unittest.TestCase):
    def test_mass_is_duration_and_gap_stays_empty(self):
        rows=[record(t,index=i) for i,t in enumerate([0.,.1,.2,1.8,1.9,2.])]
        maps,stats=build_volume(rows,query=2.,pose=pose(2.),image_size=(100,80),clock_supported=True)
        self.assertAlmostEqual(float(maps.sum()),stats['total_frustum_duration'],places=6)
        self.assertLess(stats['total_frustum_duration'],.8)
        self.assertEqual(stats['bins'][1]['duration'],0.)

    def test_future_gaze_never_changes_volume(self):
        rows=[record(.5,index=0),record(.6,index=1)]
        a,sa=build_volume(rows,query=.6,pose=pose(.6),image_size=(80,80),clock_supported=True)
        b,sb=build_volume(rows+[record(.61,3.,index=2)],query=.6,pose=pose(.6),image_size=(80,80),clock_supported=True)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(sa,sb)

    def test_invalid_sample_removes_support_and_clock_falls_back(self):
        rows=[record(.3,index=0),record(.4,valid=False,index=1),record(.5,index=2)]
        supported=sum(r['weight_seconds'] for r in support_intervals(rows,.5))
        all_valid=sum(r['weight_seconds'] for r in support_intervals([dict(r,validity='valid') for r in rows],.5))
        self.assertLess(supported,all_valid)
        maps,s=build_volume(rows,query=.5,pose=None,image_size=(80,80),clock_supported=False)
        rgb=Image.new('RGB',(80,80),(43,81,122))
        np.testing.assert_array_equal(render_volume(rgb,maps,s,'CONTOURS'),rgb)

    def test_two_modes_are_preserved_without_midpoint(self):
        rows=[record(t,-4 if i<3 else 4,index=i) for i,t in enumerate([1.5,1.6,1.7,1.8,1.9,2.])]
        _,s=build_volume(rows,query=2.,pose=pose(2.),image_size=(120,120),clock_supported=True)
        modes=s['bins'][2]['modes']
        self.assertEqual(len(modes),2)
        self.assertTrue(all(m['observed_representative']['pixel_xy'][0] in [36.,84.] for m in modes))

    def test_head_compensation_projects_same_world_point_to_same_current_pixel(self):
        rows=[record(t,2.,index=i) for i,t in enumerate([1.,1.1,1.2,1.8,1.9,2.])]
        _,s=build_volume(rows,query=2.,pose=pose(2.,20),image_size=(100,100),clock_supported=True)
        points=[o['pixel_xy'] for b in s['bins'] for o in b['observations']]
        np.testing.assert_allclose(points,np.repeat([points[0]],len(points),axis=0))
        self.assertLess(points[0][0],50.)

    def test_contour_keeps_core_and_heat_crop_keeps_query(self):
        rows=[record(t,index=i) for i,t in enumerate([1.5,1.6,1.7,1.8,1.9,2.])]
        maps,stats=build_volume(rows,query=2.,pose=pose(2.),image_size=(100,100),clock_supported=True)
        rgb=Image.new('RGB',(100,100),(81,101,121))
        output=render_volume(rgb,maps,stats,'CONTOURS')
        self.assertEqual(output.getpixel((50,50)),rgb.getpixel((50,50)))
        view,meta=heat_guided_crop(rgb,[50,50],maps)
        self.assertEqual(meta['bbox_source_pixels'][0]+meta['point_in_view'][0],50.)
        self.assertTrue(.18<=meta['ratio']<=.5)
        self.assertGreaterEqual(float(maps[2][mass_region(maps[2],.8)].sum()),float(maps[2].sum())*.8-1e-7)


    def test_edge_kernel_preserves_duration_and_sparse_support_stays_sparse(self):
        rows=[record(1.,9.9,index=0)]
        maps,stats=build_volume(rows,query=1.,pose=pose(1.),image_size=(100,100),clock_supported=True)
        self.assertAlmostEqual(float(maps.sum()),stats['total_frustum_duration'],places=6)
        self.assertLess(stats['bins'][2]['display_support'],1.)
        self.assertAlmostEqual(stats['total_frustum_duration'],.125)

    def test_time_reversal_changes_channels_but_keeps_total_spatial_mass(self):
        rows=[record(t,x,index=i) for i,(t,x) in enumerate([(0.2,-5.),(1.,0.),(1.9,5.)])]
        maps,_=build_volume(rows,query=2.,pose=pose(2.),image_size=(100,100),clock_supported=True)
        reverse=maps[::-1].copy()
        np.testing.assert_allclose(maps.sum(axis=0),reverse.sum(axis=0),rtol=1e-6,atol=1e-8)
        self.assertFalse(np.array_equal(maps,reverse))


if __name__=='__main__':unittest.main()
