import unittest
import numpy as np
from PIL import Image
from .full_interaction_referring import control_maps, render_full_frame, validate_prediction


class FullInteractionReferringTests(unittest.TestCase):
    def test_controls_preserve_mass_and_do_not_modify_source(self):
        a=np.zeros((3,60,84),np.float32);a[0,8,10]=.2;a[1,12,30]=.4;a[2,20,42]=.1
        original=a.copy()
        for condition in ('TRUE_HEAT','SPATIAL_MIRROR','TIME_REVERSE'):
            b=control_maps(a,condition);np.testing.assert_allclose(a.sum(dtype=np.float64),b.sum(dtype=np.float64),rtol=1e-7,atol=1e-9)
        np.testing.assert_array_equal(a,original)
        np.testing.assert_array_equal(control_maps(a,'SPATIAL_MIRROR')[:,:,::-1],a)
        np.testing.assert_array_equal(control_maps(a,'TIME_REVERSE')[::-1],a)
        self.assertFalse(control_maps(a,'NO_ADDED_HEAT').any())

    def test_full_rgb_pixels_and_aspect_are_preserved_across_conditions(self):
        rng=np.random.default_rng(3);pixels=rng.integers(0,256,(60,84,3),dtype=np.uint8)
        rgb=Image.fromarray(pixels);maps=np.zeros((3,60,84));maps[:,4:8,6:10]=.03
        for condition in ('NO_ADDED_HEAT','TRUE_HEAT','SPATIAL_MIRROR','TIME_REVERSE'):
            image=render_full_frame(rgb,maps,condition)
            self.assertEqual(image.size,(168,176))
            np.testing.assert_array_equal(np.asarray(image)[28:88,:84],pixels)

    def test_invalid_choice_or_union_is_never_silently_repaired(self):
        p={'selected_object_names':['chair2'],'decisions':[{'mention_text':'the chair','candidate_ids':['chair2'],'view_ids':['G002']}],'unresolved_mentions':[]}
        self.assertTrue(validate_prediction(p,['chair1','chair2'],{'G001','G002'},'Move the chair.')['schema_valid'])
        p['selected_object_names']=['chair1']
        result=validate_prediction(p,['chair1','chair2'],{'G001','G002'},'Move the chair.')
        self.assertIn('selected_union_mismatch',result['errors']);self.assertEqual(result['selected_object_names'],[])
        p['selected_object_names']=['chair2'];p['decisions'][0]['candidate_ids']=['other_scene_id']
        self.assertIn('invalid_decision_ids',validate_prediction(p,['chair1','chair2'],{'G002'},'Move the chair.')['errors'])


if __name__=='__main__': unittest.main()
