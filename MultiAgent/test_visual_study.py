import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from .visual_study import fixed_video_anchor_frames, paired_comparison


class FixedFrameStudyTests(unittest.TestCase):
    def test_fixed_video_anchors_ignore_mention_and_gaze_annotations(self):
        plan={'records':[
            {'time':0.1,'reasons':['event_context'],'mention_ids':[]},
            {'time':0.3,'reasons':['instruction_phase'],'mention_ids':['m0']},
            {'time':0.5,'reasons':['event_context'],'mention_ids':[]},
            {'time':0.7,'reasons':['gaze_stable'],'mention_ids':[]},
            {'time':0.9,'reasons':['event_context'],'mention_ids':[]},
        ]}
        selected=fixed_video_anchor_frames(plan,3)
        self.assertEqual([r['time'] for r in selected],[0.1,0.5,0.9])
        plan['records'][0]['gaze_point']=[100,200,300]
        self.assertEqual([r['time'] for r in fixed_video_anchor_frames(plan,3)],[0.1,0.5,0.9])

    def test_paired_audit_separates_source_frames_from_visual_variant_pixels(self):
        common={'scene_key':'scene1','event_id':'scene1_1','gt_referents_mapped':['chair'],
                'source_frame_ids':['F1'],'source_rgb_frame_sha256':['rgb-hash'],
                'coordinator_image_sha256':['rgb-hash'],'image_sizes':[[672,672]],
                'selected_object_names':['chair']}
        raw={**common,'input_image_sha256':['rgb-hash']}
        heat={**common,'input_image_sha256':['heat-hash']}
        with TemporaryDirectory() as directory:
            audit=paired_comparison({'R':[raw],'H_local':[heat]},Path(directory))
        self.assertTrue(audit['same_source_frames_for_all_events_and_groups'])
        self.assertEqual(audit['events_with_changed_visual_pixels_vs_R']['H_local'],1)
        self.assertEqual(audit['per_group_transitions_vs_R']['H_local']['any_hit_unchanged'],1)


if __name__=='__main__':unittest.main()
