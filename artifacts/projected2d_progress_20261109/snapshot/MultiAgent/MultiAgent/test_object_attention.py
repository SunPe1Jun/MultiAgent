import unittest
import numpy as np
from .object_attention import (instruction_categories,detector_query,deduplicate_detections,
    mask_attention_links,match_masks_one_to_one)


class ObjectAttentionTests(unittest.TestCase):
    def test_language_queries_preserve_references_components_and_ignore_instance_ids(self):
        categories=instruction_categories('Tell that colleague to secure that van hood and move that file beside the desk.')
        self.assertEqual(set(categories),{'person','van','hood','file','desk'})
        self.assertIn('document',detector_query(categories));self.assertIn('folder',detector_query(categories))
        self.assertEqual(instruction_categories('there'),[])

    def test_nms_does_not_remove_nested_different_categories_or_use_gaze(self):
        rows=[{'label':'chair','score':.9,'box':[0,0,10,10]},
              {'label':'chair','score':.8,'box':[0,0,10,10]},
              {'label':'desk','score':.7,'box':[0,0,10,10]}]
        kept=deduplicate_detections(rows,20,20)
        self.assertEqual([r['category'] for r in kept],['chair','desk'])

    def test_overlapping_masks_split_mass_and_never_double_count(self):
        maps=np.zeros((3,4,5));maps[:,1,2]=[1,2,3];maps[:,3,4]=[.5,1,1.5]
        masks=np.zeros((2,4,5),bool);masks[:,1,2]=True
        audit,display,unassigned=mask_attention_links(maps,masks)
        np.testing.assert_allclose(audit['mass_seconds_by_phase_object'],[[.5,.5],[1,1],[1.5,1.5]])
        np.testing.assert_allclose(np.array(audit['assigned_seconds'])+audit['unassigned_seconds'],audit['total_seconds'])
        np.testing.assert_allclose(display.sum((1,2))+unassigned.sum((1,2)),maps.sum((1,2)))
        np.testing.assert_allclose(audit['ambiguous_seconds'],[1,2,3])

    def test_empty_masks_and_empty_attention_keep_unknown_explicit(self):
        maps=np.ones((3,4,5));masks=np.zeros((0,4,5),bool)
        a,d,u=mask_attention_links(maps,masks)
        self.assertEqual(a['assigned_seconds'],[0,0,0]);np.testing.assert_array_equal(u,maps)
        self.assertEqual(a['mass_seconds_by_phase_object'],[[],[],[]]);self.assertFalse(d.any())
        with self.assertRaises(ValueError):mask_attention_links(-maps,np.zeros((1,4,5),bool))

    def test_matching_is_class_compatible_one_to_one_and_abstains(self):
        a=np.zeros((10,10),bool);a[:5,:5]=True
        b=np.zeros((10,10),bool);b[5:,5:]=True
        matches=match_masks_one_to_one([a,b],[b,a],['chair','desk'],['desk','chair'])
        self.assertEqual([(m['first'],m['second']) for m in matches],[(0,1),(1,0)])
        self.assertEqual(match_masks_one_to_one([a],[a],['chair'],['desk']),[])
        self.assertEqual(match_masks_one_to_one([],[]),[])


if __name__=='__main__':unittest.main()
