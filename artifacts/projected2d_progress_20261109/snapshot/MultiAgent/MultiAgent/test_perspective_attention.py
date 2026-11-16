import tempfile
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
from .perspective_attention import (cube_owner,face_grid,select_faces,face_mass_audit,
    sample_native_face,choose_native_source,transform_density)
from .spherical_attention import spherical_grid,angles_to_directions
from .test_attention_volume import pose


class PerspectiveAttentionTests(unittest.TestCase):
    def test_faces_partition_entire_sphere_and_each_window_is_correct(self):
        yg,pg=spherical_grid(120,60);owner=cube_owner(angles_to_directions(yg,pg))
        self.assertEqual(set(np.unique(owner)),set(range(6)))
        self.assertEqual(sum(np.sum(owner==i) for i in range(6)),120*60)
        for face in range(6):self.assertTrue(np.all(cube_owner(face_grid(face,24))==face))

    def test_region_selection_depends_on_camera_coverage_and_mass_is_accounted(self):
        coverage=np.zeros((3,60,120),np.float32);coverage[:,:,45:75]=1
        selected,stats=select_faces(coverage)
        self.assertEqual(selected[0],0)
        maps=np.ones_like(coverage)
        audit=face_mass_audit(maps,list(range(6)))
        np.testing.assert_allclose(audit['retained_duration_fraction'],1.)
        self.assertIn('gaze maps not used',stats['selection'])

    def test_controls_keep_support_and_never_change_source_arrays(self):
        tiles=np.arange(3*3*4*4).reshape(3,3,4,4).astype(float);copy=tiles.copy()
        for condition in ['TRUE_HEAT','REVERSE_TIME','SHIFT_SPACE']:
            self.assertEqual(float(transform_density(tiles,condition).sum()),float(tiles.sum()))
        self.assertEqual(float(transform_density(tiles,'NO_HEAT').sum()),0.)
        np.testing.assert_array_equal(tiles,copy)
        np.testing.assert_array_equal(transform_density(tiles,'REVERSE_TIME')[0],tiles[2])

    def test_native_source_never_uses_future_frame_and_preserves_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.png';Image.new('RGB',(80,80),(130,70,35)).save(path)
            frames=[{'source_path':str(path),'pts':t,'geometry':{'reference_pose':pose(t)}} for t in [.5,1.5,2.5]]
            frame,meta=choose_native_source(frames,pose(0.),0,[0.,1.],0.)
            self.assertEqual(frame['pts'],.5);self.assertTrue(meta['used_in_bin_frame'])
            image,mask=sample_native_face(frame,pose(0.),2,20)
            self.assertFalse(mask.any());self.assertTrue(np.all(np.asarray(image)==22))


    def test_wrong_model_region_is_not_replaced_by_the_strongest_region(self):
        from .perspective_attention import observation_for_selected_face
        observations=[{'source_index':0,'world_ray':[0,0,1],'duration_seconds':10.},
                      {'source_index':1,'world_ray':[1,0,0],'duration_seconds':.1}]
        density=np.zeros((20,20));density[10,10]=1
        selected=observation_for_selected_face(observations,pose(0.),1,density)
        self.assertEqual(selected['source_index'],1)
        self.assertIsNone(observation_for_selected_face(observations,pose(0.),2,density))


if __name__=='__main__':unittest.main()
