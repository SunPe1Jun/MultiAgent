"""C-mode evidence coverage must survive absent or wrong speech recognition."""
import copy
import unittest
from types import SimpleNamespace

from .preprocess import plan_frames
from .asr import guard_clock
from .mention_mapping import resolve_mentions


class HybridTimingTests(unittest.TestCase):
    def setUp(self):
        self.ctx = SimpleNamespace(text='Move that chair, then inspect that chair.',
                                   api_row={'t_start':0, 't_end':10, 't_peak':5},
                                   timed_samples=[], candidates=[])
        self.mentions = resolve_mentions(self.ctx, {'mentions':[
            {'text':'that chair', 'object_type':'chair'},
            {'text':'that chair', 'object_type':'chair'}]})['mentions']
        self.alignment = {'clock':{'telemetry_to_video':0}, 'mentions':{},
                          'instruction_coverage':0}

    def test_repeated_mentions_keep_distinct_occurrence_and_phase(self):
        self.assertNotEqual(self.mentions[0]['span'], self.mentions[1]['span'])
        p = plan_frames(self.ctx,self.mentions,self.alignment,11)
        self.assertLess(p['mention_windows']['m0']['phase_hint'][0],
                        p['mention_windows']['m1']['phase_hint'][0])

    def test_asr_cannot_evict_base_frames_or_narrow_search(self):
        base = plan_frames(self.ctx,self.mentions,self.alignment,11,timed=False)
        a = copy.deepcopy(self.alignment)
        a['instruction_coverage'] = 1
        a['mentions']['m0'] = {'matched':True, 'alignment_quality':'phrase',
                              'confidence':1, 'video_clip':[.8,1.2]}
        hybrid = plan_frames(self.ctx,self.mentions,a,11)
        self.assertTrue(set(base['base_frame_times']) <= {r['time'] for r in hybrid['records']})
        self.assertEqual(hybrid['mention_windows']['m0']['video_range'],[0,10])
        self.assertTrue(hybrid['asr_added_frame_times'])

    def test_wrong_transcript_and_head_only_match_cannot_add_frames(self):
        a = copy.deepcopy(self.alignment)
        a['mentions']['m0'] = {'matched':True, 'alignment_quality':'phrase',
                              'confidence':1, 'video_clip':[.8,1.2]}
        p = plan_frames(self.ctx,self.mentions,a,11)
        self.assertFalse(p['asr_timing_used'])
        a['instruction_coverage'] = 1
        a['mentions']['m0']['alignment_quality'] = 'head_only'
        self.assertFalse(plan_frames(self.ctx,self.mentions,a,11)['asr_timing_used'])

    def test_direct_mode_ignores_even_valid_asr(self):
        a = copy.deepcopy(self.alignment)
        a.update(instruction_coverage=1, mentions={'m0':{'matched':True,
            'alignment_quality':'phrase','confidence':1,'video_clip':[1,2]}})
        p = plan_frames(self.ctx,self.mentions,a,11,timed=False)
        self.assertFalse(p['asr_timing_used'])
        self.assertTrue(all(r['telemetry_time'] is not None for r in p['records']))

    def test_large_clock_offset_falls_back_to_full_video(self):
        clock = guard_clock({'telemetry_to_video': 8.0, 'audio_to_video': 0.0,
                             'wall_audio_to_video': 8.0}, 10.0)
        self.assertTrue(clock['audio_timing_rejected'])
        p = plan_frames(self.ctx, self.mentions,
                        {'clock': clock, 'mentions': {}, 'instruction_coverage': 1},
                        11)
        self.assertEqual(p['timing_policy'], 'full_video_clock_fallback')
        self.assertEqual(p['alignment']['telemetry_to_video'], None)
        self.assertTrue(all(r['telemetry_time'] is None for r in p['records']))

    def test_guard_uses_absolute_offset_and_strict_threshold(self):
        self.assertFalse(guard_clock({'telemetry_to_video': -3.0}, 10)['audio_timing_rejected'])
        self.assertTrue(guard_clock({'telemetry_to_video': -3.01}, 10)['audio_timing_rejected'])
        self.assertTrue(guard_clock({'telemetry_to_video': 0, 'wall_audio_to_video': -4}, 10)['audio_timing_rejected'])

    def test_degenerate_window_also_falls_back_under_offset_threshold(self):
        self.ctx.api_row.update(t_start=9.88, t_end=10.0)
        p = plan_frames(self.ctx,self.mentions,self.alignment,10)
        self.assertEqual(p['timing_policy'], 'full_video_clock_fallback')
        self.assertGreater(p['records'][-1]['time']-p['records'][0]['time'], 8)
        self.assertTrue(all(r['telemetry_time'] is None for r in p['records']))


if __name__ == '__main__': unittest.main()
