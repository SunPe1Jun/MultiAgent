import unittest
import numpy as np
from .spherical_attention import build_spherical_volume,direction_to_angles,angles_to_directions,render_spherical_debug
from .test_attention_volume import record,pose


def source(t,ray,index,rotation=(0,0,0,1)):
    r=dict(record(t,index=index),camera_position=[0,0,0],camera_rotation_xyzw=list(rotation),camera_fov_degrees=90.)
    r['gaze_vector_world']=list(ray);r['gaze_point_world']=None
    return r


class SphericalAttentionTests(unittest.TestCase):
    def test_angles_roundtrip_and_full_fov_is_separate_from_pixel_canvas(self):
        directions=np.array([[0,0,1],[1,0,0],[0,1,0]],dtype=float)
        for d in directions:
            yaw,pitch=direction_to_angles(d)
            np.testing.assert_allclose(angles_to_directions(np.array(yaw),np.array(pitch)),d,atol=1e-6)

    def test_head_rotation_unifies_world_ray_in_reference_sphere(self):
        from scipy.spatial.transform import Rotation
        q=Rotation.from_euler('y',90,degrees=True).as_quat().tolist()
        # Same world ray is seen from two head orientations; canonical coordinates agree.
        rows=[source(.1,[0,0,1],0),source(.2,[0,0,1],1,q)]
        maps,coverage,stats=build_spherical_volume(rows,query_time=.2,canonical_pose=pose(.2),
            image_size=(100,100),clock_supported=True,history_seconds=.2,grid_size=(360,180))
        self.assertAlmostEqual(float(maps.sum()),stats['total_duration_seconds'],places=5)
        peak=np.unravel_index(np.argmax(maps.sum(axis=0)),maps.shape[1:])
        self.assertLess(abs(peak[1]-180),4)
        self.assertEqual(stats['status'],'SPHERICAL_VOLUME')

    def test_outside_current_frame_remains_on_sphere_and_coverage_is_distinct(self):
        rows=[source(.1,[1,0,0],0),source(.2,[0,0,1],1)]
        maps,coverage,stats=build_spherical_volume(rows,query_time=.2,canonical_pose=pose(.2),
            image_size=(100,100),clock_supported=True,history_seconds=.2,grid_size=(360,180))
        self.assertGreater(float(maps[:, :, :].sum()),0.)
        self.assertLess(stats['current_fov_fraction'],1.)
        self.assertGreater(int(np.count_nonzero(coverage)),0)
        self.assertAlmostEqual(stats['total_duration_seconds'],.2,places=5)
        self.assertGreater(stats['union_view_solid_angle_sr'],0.)

    def test_future_sample_does_not_change_spherical_volume(self):
        rows=[source(.1,[0,0,1],0),source(.2,[1,0,0],1)]
        kwargs=dict(query_time=.2,canonical_pose=pose(.2),image_size=(100,100),clock_supported=True,history_seconds=.2,grid_size=(360,180))
        a,ca,sa=build_spherical_volume(rows,**kwargs)
        b,cb,sb=build_spherical_volume(rows+[source(.21,[0,1,0],2)],**kwargs)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(ca,cb)
        self.assertEqual(sa,sb)

    def test_debug_render_keeps_unknown_sphere_dark(self):
        rows=[source(.1,[0,0,1],0),source(.2,[0,0,1],1)]
        maps,coverage,stats=build_spherical_volume(rows,query_time=.2,canonical_pose=pose(.2),image_size=(100,100),clock_supported=True,history_seconds=.2,grid_size=(360,180))
        image=render_spherical_debug(maps,coverage,stats)
        self.assertEqual(image.size,(900,450))
        self.assertEqual(image.getpixel((0,0)),(22,22,22))


    def test_camera_coverage_survives_invalid_eye_tracking(self):
        rows=[source(.1,[0,0,1],0),source(.2,[0,0,1],1)]
        for r in rows:r['validity']='invalid_sample'
        maps,coverage,stats=build_spherical_volume(rows,query_time=.2,canonical_pose=pose(.2),image_size=(100,100),clock_supported=True,history_seconds=.2)
        self.assertEqual(float(maps.sum()),0.)
        self.assertGreater(float(coverage.sum()),0.)

    def test_seam_and_solid_angle_are_not_distorted(self):
        from .spherical_attention import solid_angle_weights
        self.assertAlmostEqual(float(solid_angle_weights(360,180).sum()),4*np.pi,places=9)
        rows=[source(.1,[0,0,-1],0)]
        maps,_,stats=build_spherical_volume(rows,query_time=.2,canonical_pose=pose(.2),image_size=(100,100),clock_supported=True,history_seconds=.2)
        total=maps.sum(axis=0)
        self.assertGreater(float(total[:,:4].sum()),0.)
        self.assertAlmostEqual(float(total[:,:4].sum()),float(total[:,-4:].sum()),places=6)

    def test_stable_reference_does_not_recenter_with_current_head(self):
        rows=[source(.1,[0,0,1],0),source(.2,[0,0,1],1)]
        options=dict(query_time=.2,canonical_pose=pose(.2),image_size=(100,100),clock_supported=True,history_seconds=.2)
        a,_,sa=build_spherical_volume(rows,current_pose=pose(.2),**options)
        b,_,sb=build_spherical_volume(rows,current_pose=pose(.2,100),**options)
        np.testing.assert_array_equal(a,b)
        self.assertGreater(sa['current_fov_fraction'],sb['current_fov_fraction'])


    def test_atlas_keeps_unknown_regions_and_source_ownership(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        from .spherical_attention import angular_rgb_atlas,render_cube_faces
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.png';Image.new('RGB',(100,100),(128,64,32)).save(path)
            frame={'source_path':str(path),'geometry':{'reference_pose':pose(0.)}}
            atlas,owner=angular_rgb_atlas([frame],pose(0.),grid_size=(120,60))
            pixels=np.asarray(atlas)
            self.assertTrue(np.any(owner<0))
            np.testing.assert_array_equal(pixels[owner<0],np.repeat([[22,22,22]],np.sum(owner<0),axis=0))
            np.testing.assert_array_equal(pixels[owner==0],np.repeat([[128,64,32]],np.sum(owner==0),axis=0))
            cube=render_cube_faces(atlas,np.zeros((3,60,120),np.float32),face_size=64)
            self.assertEqual(cube.size,(192,176))


    def test_six_face_preview_preserves_left_and_right_directions(self):
        from PIL import Image
        from .spherical_attention import render_cube_faces
        atlas=np.zeros((60,120,3),np.uint8)
        atlas[:,:60]=[210,35,25]  # negative yaw
        atlas[:,60:]=[25,35,210]  # positive yaw
        cube=render_cube_faces(Image.fromarray(atlas),np.zeros((3,60,120),np.float32),face_size=40)
        self.assertEqual(cube.getpixel((8,44)),(210,35,25))
        self.assertEqual(cube.getpixel((32,44)),(25,35,210))


if __name__=='__main__':unittest.main()
