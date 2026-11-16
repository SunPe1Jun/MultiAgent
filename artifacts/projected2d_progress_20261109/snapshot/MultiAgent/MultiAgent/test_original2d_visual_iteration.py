"""Semantic safeguards for restoring the original visual experiment."""
import unittest
import numpy as np
from tools.iterate_original2d_visual import heat_bbox

class HeatCropTests(unittest.TestCase):
    def test_empty_evidence_never_creates_attention(self):
        self.assertIsNone(heat_bbox({'gaze_u_norm':.5,'gaze_v_norm':.5},np.zeros((3,128,128),np.float32),(1920,1920)))

    def test_gaze_support_defines_bounded_source_crop_at_all_edges(self):
        for x,y in [(0,0),(64,64),(127,127),(12,112)]:
            maps=np.zeros((3,128,128),np.float32);maps[-1,y,x]=.5
            box=heat_bbox({'gaze_u_norm':x/128,'gaze_v_norm':y/128},maps,(1920,1920))
            left,top,right,bottom=box
            self.assertTrue(0<=left<right<=1920)
            self.assertTrue(0<=top<bottom<=1920)
            self.assertEqual(right-left,bottom-top)
            self.assertTrue(.21*1920<=right-left<=.45*1920)

    def test_change_in_recent_attention_changes_the_source_crop(self):
        a=np.zeros((3,128,128),np.float32);b=a.copy()
        a[-1,40,30]=.5;b[-1,90,100]=.5
        self.assertNotEqual(heat_bbox({},a,(1920,1920)),heat_bbox({},b,(1920,1920)))

    def test_unsupported_earlier_history_does_not_replace_recent_attention(self):
        a=np.zeros((3,128,128),np.float32);a[0,15,15]=2.
        self.assertIsNone(heat_bbox({},a,(1920,1920)))

if __name__=='__main__':unittest.main()
