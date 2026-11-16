import unittest
from .full_frame_grounding_v3 import validate_composite_grounding
from .test_full_frame_grounding import FullFrameGroundingTests


class CompositeCoordinateTests(unittest.TestCase):
    def event(self):
        e=FullFrameGroundingTests().event()
        for f in e['frames']:f['source_size']=[672,672]
        e['mention_hints']=[{'text':'that chair'}]
        return e

    def test_whole_image_coordinates_are_converted_by_script(self):
        e=self.event();p={'referents':[{'mention':'that chair','category':'chair','view_id':'G001','x':375,'y':260}],'unresolved':[]}
        r=validate_composite_grounding(p,e);self.assertTrue(r['schema_valid'])
        self.assertAlmostEqual(r['referents'][0]['x'],.75);self.assertAlmostEqual(r['referents'][0]['y'],.5)
        self.assertEqual(r['unaccounted_mention_hints'],[])
        self.assertEqual(p['referents'][0]['x'],375)

    def test_malformed_unresolved_is_recorded_as_schema_failure_without_crash(self):
        e=self.event();p={'referents':[],'unresolved':[{'mention':'that chair','x':0,'y':0}]}
        r=validate_composite_grounding(p,e)
        self.assertTrue(r['schema_valid']);self.assertIn('invalid_unresolved',r['errors'])
        self.assertEqual(len(r['rejected_unresolved']),1)
        self.assertEqual(r['unaccounted_mention_hints'],['that chair'])

    def test_repeated_mentions_require_repeated_accounting(self):
        e=self.event();e['mention_hints']=[{'text':'that chair'},{'text':'that chair'}]
        p={'referents':[{'mention':'that chair','category':'chair','view_id':'G001','x':375,'y':260}],'unresolved':[]}
        r=validate_composite_grounding(p,e)
        self.assertEqual(r['unaccounted_mention_hints'],['that chair'])

    def test_point_into_heat_panel_is_rejected_and_missing_mention_recorded(self):
        e=self.event();p={'referents':[{'mention':'that chair','category':'chair','view_id':'G001','x':700,'y':260}],'unresolved':[]}
        bad=validate_composite_grounding(p,e)
        self.assertIn('point_outside_original_rgb_panel',bad['errors']);self.assertEqual(bad['referents'],[])
        r=validate_composite_grounding({'referents':[],'unresolved':[]},e)
        self.assertTrue(r['schema_valid']);self.assertEqual(r['unaccounted_mention_hints'],['that chair'])


if __name__=='__main__':unittest.main()
