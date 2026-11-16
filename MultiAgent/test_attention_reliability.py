import unittest
import numpy as np
from .attention_reliability import (distinct_masks,interior_point,injected_history,pair_attention,
                                    graphical_pair_heat,cluster_paired_interval,heat_display_policy)


class AttentionReliabilityTests(unittest.TestCase):
    def masks(self):
        m=np.zeros((2,24,30),bool);m[0,3:10,2:10]=True;m[1,12:21,20:28]=True;return m

    def test_pair_geometry_and_interior_point_do_not_depend_on_ref_or_gaze(self):
        masks=self.masks();self.assertTrue(distinct_masks(*masks));self.assertFalse(distinct_masks(masks[0],masks[0]))
        for mask in masks:
            x,y=map(int,interior_point(mask));self.assertTrue(mask[y,x])
        with self.assertRaises(ValueError):interior_point(np.zeros((5,5)))

    def test_injected_counterfactual_changes_temporal_identity_with_mass_constant(self):
        a=injected_history(self.masks());b=injected_history(self.masks(),reverse=True)
        self.assertEqual(a[0].argmax(),0);self.assertEqual(a[2].argmax(),1)
        self.assertEqual(b[0].argmax(),1);self.assertEqual(b[2].argmax(),0)
        self.assertEqual(a.sum(),b.sum());np.testing.assert_array_equal(b,a[::-1])

    def test_real_pair_swap_keeps_residual_and_input_unchanged(self):
        masks=self.masks();maps=np.zeros((3,24,30));maps[:,4,4]=[1,2,3];maps[:,0,0]=[.2,.3,.4];before=maps.copy()
        mass,residual,links=pair_attention(maps,masks,'TRUE_HEAT')
        swapped,r2,_=pair_attention(maps,masks,'SWAP_PAIR')
        np.testing.assert_array_equal(mass[:,::-1],swapped);np.testing.assert_array_equal(residual,r2)
        np.testing.assert_array_equal(maps,before)
        np.testing.assert_allclose(mass.sum(1)+residual.sum((1,2)),maps.sum((1,2)))
        zero,r0,_=pair_attention(maps,masks,'NO_HEAT');self.assertFalse(zero.any());self.assertFalse(r0.any())

    def test_tiny_support_is_faint_and_null_is_dark(self):
        masks=self.masks();residual=np.zeros((3,24,30));mass=injected_history(masks)
        orange,blue,meta=graphical_pair_heat(mass,masks,residual)
        small,_,m=graphical_pair_heat(mass*1e-4,masks,residual)
        self.assertLess(float(small.max()),float(orange.max())*.01)
        self.assertLess(m['display_support_gate'],.01)
        o,b,_=graphical_pair_heat(np.zeros((3,2)),masks,residual);self.assertFalse(o.any());self.assertFalse(b.any())

    def test_display_gate_never_turns_duration_into_object_selection(self):
        self.assertEqual(heat_display_policy(0.),'NO_HEAT')
        self.assertEqual(heat_display_policy(.249),'NO_HEAT')
        self.assertEqual(heat_display_policy(.25),'TRUE_HEAT')
        with self.assertRaises(ValueError):heat_display_policy(float('nan'))
        with self.assertRaises(ValueError):heat_display_policy(-.1)
        with self.assertRaises(ValueError):heat_display_policy(.1,0.)

    def test_cluster_bootstrap_keeps_all_variants_of_interaction_together(self):
        rows=[{'cluster':'a','baseline':0,'treatment':1}]*8+[{'cluster':'b','baseline':1,'treatment':0}]*2
        result=cluster_paired_interval(rows,replicates=1000)
        self.assertEqual(result['clusters'],2);self.assertEqual(result['n'],10);self.assertAlmostEqual(result['gain'],.6)
        self.assertEqual(result['interval_95'],[-1.,1.])
        same=cluster_paired_interval([{'cluster':'x','baseline':.5,'treatment':.5}]);self.assertEqual(same['interval_95'],[0.,0.])
        self.assertIsNone(cluster_paired_interval([])['gain'])


if __name__=='__main__':unittest.main()
