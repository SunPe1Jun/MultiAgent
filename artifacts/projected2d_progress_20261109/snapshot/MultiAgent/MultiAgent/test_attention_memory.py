import unittest
from .attention_memory import build_memory_proposals
from .test_attention_volume import record,pose


def source(t,x,index):
    return dict(record(t,x,index=index),camera_position=[0,0,0],camera_rotation_xyzw=[0,0,0,1],camera_fov_degrees=90.)


class NativeMemoryTests(unittest.TestCase):
    def test_offscreen_history_is_not_deleted(self):
        rows=[source(t,0.,i) for i,t in enumerate([.1,.2,.3,.4])]
        result=build_memory_proposals(rows,query=.4,current_pose=pose(.4,100),image_size=(100,100),clock_supported=True)
        self.assertEqual(len(result['proposals']),1)
        p=result['proposals'][0]
        self.assertEqual(p['outside_current_frustum_count'],4)
        self.assertEqual(p['representative']['native_pixel_xy'],[50.,50.])
        self.assertGreater(p['supported_duration_seconds'],0.)

    def test_distinct_world_rays_make_separate_segments_without_merging_objects(self):
        rows=[source(t,-4 if i<3 else 4,i) for i,t in enumerate([.1,.2,.3,.4,.5,.6])]
        result=build_memory_proposals(rows,query=.6,current_pose=pose(.6),image_size=(100,100),clock_supported=True)
        self.assertEqual(len(result['proposals']),2)
        self.assertNotIn('object_id',result['proposals'][0])
        self.assertLess(result['proposals'][0]['representative']['gaze_time_seconds'],result['proposals'][1]['representative']['gaze_time_seconds'])

    def test_future_samples_and_missing_clock_do_not_add_memory(self):
        rows=[source(.1,0,0),source(.2,0,1)]
        kwargs=dict(query=.2,current_pose=pose(.2),image_size=(100,100),clock_supported=True)
        a=build_memory_proposals(rows,**kwargs);b=build_memory_proposals(rows+[source(.21,5,2)],**kwargs)
        self.assertEqual(a,b)
        self.assertEqual(build_memory_proposals(rows,**dict(kwargs,clock_supported=False))['proposals'],[])


if __name__=='__main__':unittest.main()
