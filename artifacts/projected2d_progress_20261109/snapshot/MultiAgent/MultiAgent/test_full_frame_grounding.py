import copy
import unittest
from .full_frame_grounding import decode_grounding,grounding_prompt,validate_grounding


class FullFrameGroundingTests(unittest.TestCase):
    def event(self):
        pose={'position_world':[0,0,0],'rotation_xyzw':[0,0,0,1],'fov_vertical_degrees':90}
        frames=[{'frame_id':f'V{i:03d}','source_size':[100,100],'geometry':{'clock_available':True,'reference_pose':pose}} for i in (1,2)]
        return {'instruction':'Move that chair.','frames':frames,
            'qwen_temporal_groups':[{'view_id':'G001','frame_ids':['V001','V002']}],
            'scene_candidate_table':[{'candidate_id':'chair1','position_world':[-1,0,2],'aliases':[]},
                                     {'candidate_id':'chair2','position_world':[1,0,2],'aliases':[]},
                                     {'candidate_id':'desk1','position_world':[1,0,2],'aliases':[]}]}

    def test_decode_uses_full_type_compatible_scene_not_event_referents(self):
        event=self.event();point={'mention':'that chair','category':'chair','view_id':'G001','x':.75,'y':.5}
        self.assertEqual(decode_grounding([point],event)['selected_object_names'],['chair2'])
        event['gt']=['chair1'];event['scene_candidate_table'].reverse()
        self.assertEqual(decode_grounding([point],event)['selected_object_names'],['chair2'])
        self.assertNotIn('position_world',grounding_prompt(event));self.assertNotIn('chair2',grounding_prompt(event))

    def test_ambiguity_distance_and_unsupported_camera_abstain(self):
        e=self.event();point={'mention':'that chair','category':'chair','view_id':'G001','x':.5,'y':.5}
        self.assertEqual(decode_grounding([point],e)['selected_object_names'],[])
        point['x']=.75;e['scene_candidate_table'][0]['position_world']=[1,0,2]
        self.assertEqual(decode_grounding([point],e)['decisions'][0]['reason'],'ambiguous_same_category_anchor_centers')
        for f in e['frames']:f['geometry']['clock_available']=False
        self.assertEqual(decode_grounding([point],e)['selected_object_names'],[])
        point['category']='desk';self.assertEqual(decode_grounding([point],e)['selected_object_names'],['desk1'])

    def test_literal_mention_and_rgb_coordinates_are_validated(self):
        e=self.event();p={'referents':[{'mention':'that chair','category':'chair','view_id':'G001','x':.75,'y':.5}],'unresolved':[]}
        self.assertTrue(validate_grounding(p,e)['schema_valid']);p['referents'][0]['x']=750
        self.assertIn('invalid_coordinate',validate_grounding(p,e)['errors'])


if __name__=='__main__':unittest.main()
