from __future__ import annotations

import tempfile
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image

from .context import Candidate
from .local_video_visual import (detect_visual_content_start, startup_frame_metrics, extract_dense_frames, qwen_temporal_groups,
                                 remap_window_evidence, run_visual_calls, temporal_windows,
                                 validate_batched_visual_response, validate_visual_response)
from .mention_mapping import resolve_mentions


class LocalVideoVisualTests(unittest.TestCase):
    def setUp(self):
        self.ctx = SimpleNamespace(
            scene_key='scene2', event_id='scene2_test', text='Slide that chair to there.',
            candidates=[Candidate('chair5', 'chair5', 0, 0, 0),
                        Candidate('desk8', 'desk8', 1, 0, 1)])

    def test_deictic_location_does_not_expand_mention_candidates(self):
        result = resolve_mentions(self.ctx, {'mentions': [
            {'text': 'there', 'object_type': 'location', 'kind': 'location'}]})
        self.assertEqual(result['mentions'][0]['candidate_ids'], [])
        self.assertEqual(result['mentions'][0]['candidate_mapping_reason'],
                         'contextual_location_unresolved')

    def test_visual_response_cannot_escape_scene_and_mention_scope(self):
        mention = {'mention_id': 'm0', 'candidate_ids': ['chair5']}
        frames = {'qwen_temporal_groups': [{'frame_id': 'G001', 'view_id': 'G001',
                  'source_frame_ids': ['frame_01', 'frame_02'],
                  'source_time_range': [1.20, 1.30], 'time_seconds': 1.25}]}
        checked = validate_visual_response({'evidence': [
            {'mention_id': 'm0', 'candidate_id': 'chair5', 'support_score': .8,
             'confidence': .7, 'frame_id': 'G001'},
            {'mention_id': 'm0', 'candidate_id': 'desk8', 'support_score': .8,
             'confidence': .7, 'frame_id': 'G001'},
            {'mention_id': 'm0', 'candidate_id': 'lamp1', 'support_score': .8,
             'confidence': .7, 'frame_id': 'G001'},
        ]}, self.ctx, mention, frames)
        self.assertEqual([r['candidate_id'] for r in checked['evidence']], ['chair5'])
        self.assertEqual([r['reason'] for r in checked['rejected_evidence']],
                         ['outside_mention_closed_set', 'outside_scene_anchor_table'])
        self.assertEqual(checked['evidence'][0]['source_frame_ids'], ['frame_01', 'frame_02'])
        self.assertEqual(checked['evidence'][0]['source_time_range'], [1.20, 1.30])

    def test_deictic_can_only_resolve_inside_current_scene_table(self):
        mention = {'mention_id': 'm0', 'candidate_ids': []}
        frames = {'qwen_temporal_groups': [{'frame_id': 'G001', 'view_id': 'G001',
                  'source_frame_ids': ['frame_01', 'frame_02'],
                  'source_time_range': [1.20, 1.30], 'time_seconds': 1.25}]}
        checked = validate_visual_response({'evidence': [
            {'mention_id': 'm0', 'candidate_id': 'desk8', 'support_score': .8,
             'confidence': .7, 'frame_id': 'G001'},
            {'mention_id': 'm0', 'candidate_id': 'warehousea', 'support_score': .8,
             'confidence': .7, 'frame_id': 'G001'},
        ]}, self.ctx, mention, frames)
        self.assertEqual(checked['evidence'], [])
        self.assertEqual(len(checked['rejected_evidence']), 2)
        self.assertTrue(all(item['reason'] == 'object_evidence_for_unresolved_spatial'
                            for item in checked['rejected_evidence']))

    def test_deictic_visual_output_is_a_spatial_observation_without_anchor_id(self):
        mention = {'mention_id': 'm0', 'candidate_ids': []}
        frames = {'qwen_temporal_groups': [{'frame_id': 'G001', 'view_id': 'G001',
                  'source_frame_ids': ['frame_01', 'frame_02'],
                  'source_time_range': [1.20, 1.30], 'time_seconds': 1.25}]}
        checked = validate_visual_response({'evidence': [], 'visual_observations': [{
            'mention_id': 'm0', 'frame_id': 'G001',
            'description': 'destination beside the table', 'confidence': .6,
        }]}, self.ctx, mention, frames)
        self.assertEqual(len(checked['visual_observations']), 1)
        self.assertEqual(checked['visual_observations'][0]['source_frame_ids'], ['frame_01', 'frame_02'])
        self.assertNotIn('candidate_id', checked['visual_observations'][0])

    def test_visual_observation_cannot_smuggle_candidate_id_as_a_description(self):
        mention = {'mention_id': 'm0', 'candidate_ids': ['person1']}
        frames = {'qwen_temporal_groups': [{'frame_id': 'G001', 'view_id': 'G001',
                  'source_frame_ids': ['frame_01', 'frame_02'],
                  'source_time_range': [.1, .2], 'time_seconds': .15}]}
        ctx = SimpleNamespace(scene_key='scene2', candidates=self.ctx.candidates + [
            Candidate('person1', 'person1', 2, 0, 0)])
        checked = validate_visual_response({'evidence': [], 'visual_observations': [{
            'mention_id': 'm0', 'frame_id': 'G001', 'description': 'person1 near a desk',
            'confidence': .7}]}, ctx, mention, frames)
        self.assertEqual(checked['visual_observations'], [])
        self.assertEqual(checked['rejected_evidence'][0]['reason'],
                         'candidate_id_text_requires_candidate_specific_evidence')

    def test_same_anchor_name_is_loaded_from_each_context_only(self):
        from .local_video_visual import _candidate_scope, visual_prompt
        left = SimpleNamespace(scene_key='scene1', candidates=[Candidate('chair1', 'chair1', 0, 0, 0)])
        right = SimpleNamespace(scene_key='scene2', candidates=[Candidate('chair1', 'chair1', 9, 8, 7)])
        mention = {'mention_id': 'm0', 'candidate_ids': ['chair1'], 'text': 'chair'}
        self.assertEqual(_candidate_scope(left, mention)[1][0]['aliases'], [])
        self.assertEqual(_candidate_scope(right, mention)[1][0]['aliases'], [])
        self.assertEqual(_candidate_scope(right, mention)[1][0]['candidate_id'], 'chair1')
        self.assertNotIn('x_world', _candidate_scope(left, mention)[1][0])

    def test_video_prompt_uses_only_mention_candidates_and_no_gaze_numbers(self):
        from .local_video_visual import visual_prompt
        ctx = SimpleNamespace(scene_key='scene2', text='Move the chair over there.',
                              candidates=self.ctx.candidates)
        mention = {'mention_id': 'm0', 'text': 'the chair', 'span': [5, 14],
                   'candidate_ids': ['chair5']}
        manifest = {'qwen_temporal_groups': [{'frame_id': 'G001', 'view_id': 'G001',
                                'source_frame_ids': ['frame_000001', 'frame_000002'],
                                'source_time_range': [.20, .30], 'time_seconds': .25}]}
        manifest['frames'] = [{'frame_id': 'V001', 'view_id': 'V001',
                               'source_frame_id': 'frame_000001', 'time_seconds': .25}]
        prompt = visual_prompt(ctx, {}, mention, manifest)
        self.assertIn('mention_scene_local_candidate_table', prompt)
        self.assertNotIn('desk8', prompt)
        self.assertNotIn('gazePoint', prompt.split('Instruction: ', 1)[1])

    def test_batched_outputs_preserve_independent_candidate_sets_and_there_stays_unmapped(self):
        from .local_video_visual import visual_event_prompt
        mentions = [
            {'mention_id': 'm0', 'text': 'that chair', 'span': [5, 15],
             'kind': 'object', 'candidate_ids': ['chair5']},
            {'mention_id': 'm1', 'text': 'there', 'span': [20, 25],
             'kind': 'location', 'candidate_ids': []},
        ]
        ctx = SimpleNamespace(scene_key='scene2', text='Move that chair to there.',
                              candidates=self.ctx.candidates)
        groups = [{'view_id': 'G001', 'frame_id': 'G001', 'time_seconds': .5,
                   'source_time_range': [.4, .6],
                   'source_frame_ids': ['frame_1', 'frame_2']}]
        manifest = {'qwen_temporal_groups': groups}
        prompt = visual_event_prompt(ctx, {'mentions': mentions}, manifest)
        self.assertIn('mention_scene_local_candidate_table', prompt)
        self.assertIn('unresolved_spatial_observation', prompt)
        self.assertNotIn('desk8', prompt)
        self.assertNotIn('"support_score":0.0', prompt)
        self.assertIn('G001 is not a default answer', prompt)
        raw = {'mentions': [
            {'mention_id': 'm0', 'evidence': [
                {'candidate_id': 'chair5', 'support_score': .8, 'confidence': .7,
                 'frame_id': 'G001'}], 'visual_observations': [],
             'identity_status': 'supported'},
            {'mention_id': 'm1', 'evidence': [
                {'candidate_id': 'desk8', 'support_score': .8, 'confidence': .7,
                 'frame_id': 'G001'}], 'visual_observations': [],
             'identity_status': 'unmapped'},
        ]}
        checked, rejected = validate_batched_visual_response(raw, ctx, {'mentions': mentions}, manifest)
        self.assertIn('m0', checked)
        self.assertEqual(checked['m0']['evidence'][0]['source_frame_ids'], ['frame_1', 'frame_2'])
        self.assertEqual(checked['m1']['evidence'], [])
        self.assertEqual(len(checked['m1']['rejected_evidence']), 1)
        self.assertEqual(checked['m1']['rejected_evidence'][0]['reason'],
                         'object_evidence_for_unresolved_spatial')
        self.assertFalse(rejected)

    def test_qwen_temporal_groups_use_measured_pts_and_reject_odd_padding(self):
        metadata = [{'frame_id': f'V{i+1:03d}', 'source_frame_id': f'frame_{i:06d}',
                     'time_seconds': t}
                    for i, t in enumerate([1.01, 1.13, 1.26, 1.37])]
        groups = qwen_temporal_groups(metadata)
        self.assertEqual(len(groups), 2)
        self.assertAlmostEqual(groups[0]['time_seconds'], .5 * (1.01 + 1.13))
        self.assertEqual(groups[1]['source_frame_ids'], ['frame_000002', 'frame_000003'])
        self.assertEqual(groups[0]['view_id'], 'G001')
        self.assertEqual(groups[0]['source_time_range'], [1.01, 1.13])
        with self.assertRaisesRegex(ValueError, 'even-sized'):
            qwen_temporal_groups(metadata[:-1])

    def test_temporal_windows_cover_exact_source_frames_and_remap_late_citations(self):
        frames = [{'frame_id': f'V{i+1:03d}', 'source_frame_id': f'frame_{i:06d}',
                   'source_frame_index': i, 'time_seconds': 4. + i * .125, 'path': f'{i}.jpg'}
                  for i in range(10)]
        manifest = {'frames': frames, 'frame_count': len(frames),
                    'qwen_temporal_groups': qwen_temporal_groups(frames)}
        windows = temporal_windows(manifest, 2)
        self.assertEqual([window['frame_count'] for window in windows], [4, 4, 2])
        self.assertEqual([frame['source_frame_id'] for window in windows for frame in window['frames']],
                         [frame['source_frame_id'] for frame in frames])
        checked = {'evidence': [{'frame_id': 'G001', 'view_id': 'G001',
                                'source_frame_ids': ['frame_000004', 'frame_000005'],
                                'source_time_range': [4.5, 4.625]}]}
        mapped = remap_window_evidence(checked, windows[1])['evidence'][0]
        self.assertEqual(mapped['view_id'], 'W002')
        self.assertEqual(mapped['request_local_view_id'], 'G001')
        self.assertEqual(mapped['model_cited_view_id'], 'G003')
        self.assertEqual(mapped['citation_granularity'], 'temporal_window')
        self.assertEqual(mapped['source_frame_ids'], ['frame_000004', 'frame_000005',
                                                       'frame_000006', 'frame_000007'])
        self.assertEqual(mapped['source_time_range'], [4.5, 4.875])
        self.assertEqual(checked['evidence'][0]['view_id'], 'G001')
        self.assertEqual(windows[1]['qwen_temporal_groups'][0]['request_relative_time_seconds'], .0625)

    def test_failed_video_window_is_retained_without_dropping_later_frames(self):
        mention = {'mention_id': 'm0', 'text': 'that chair', 'span': [6, 16],
                   'candidate_ids': ['chair5']}
        frames = [{'frame_id': f'V{i+1:03d}', 'source_frame_id': f'frame_{i:06d}',
                   'source_frame_index': i, 'time_seconds': i * .125, 'path': f'{i}.jpg'}
                  for i in range(10)]
        manifest = {'frames': frames, 'frame_count': len(frames),
                    'qwen_temporal_groups': qwen_temporal_groups(frames)}
        ctx = SimpleNamespace(**vars(self.ctx), video_path='source.mp4')
        response = {'evidence': [], 'visual_observations': [{
            'mention_id': 'm0', 'frame_id': 'G001', 'description': 'A chair is visible.',
            'confidence': .5}], 'identity_status': 'ambiguous_instances'}
        backend = Mock(video_max_pixels=8_388_608)
        backend.ask_video.side_effect = [response, RuntimeError('window unavailable'), response]
        with tempfile.TemporaryDirectory() as temp:
            results, failures, calls, windows = run_visual_calls(
                ctx, {'mentions': [mention]}, manifest, backend, Path(temp),
                granularity='mention', window_groups=2)
        self.assertEqual(calls, 3)
        self.assertEqual(results[0]['status'], 'PARTIAL')
        self.assertEqual(results[0]['windows_observed'], 2)
        self.assertEqual(failures[0]['window_id'], 'W002')
        self.assertEqual([r['view_id'] for r in results[0]['visual']['visual_observations']],
                         ['W001', 'W003'])
        self.assertEqual([path for call in backend.ask_video.call_args_list for path in call.args[3]],
                         [frame['path'] for frame in frames])
        self.assertNotIn('desk8', backend.ask_video.call_args_list[0].args[2])

    def test_plain_category_word_in_observation_is_not_a_numbered_anchor_claim(self):
        ctx = SimpleNamespace(candidates=[Candidate('crate', 'crate', 0, 0, 0),
                                           Candidate('truck', 'truck', 0, 0, 0)])
        mention = {'mention_id': 'm0', 'candidate_ids': ['crate']}
        frames = {'qwen_temporal_groups': [{'view_id': 'G001', 'time_seconds': .2,
                   'source_time_range': [.1, .3], 'source_frame_ids': ['a', 'b']}]}
        output = validate_visual_response({'evidence': [], 'visual_observations': [{
            'mention_id': 'm0', 'frame_id': 'G001', 'description': 'A crate beside a truck.',
            'confidence': .5}]}, ctx, mention, frames)
        self.assertEqual(len(output['visual_observations']), 1)

    def test_startup_scan_skips_black_unity_splash_and_keeps_first_scene_pts(self):
        rows=[]
        for i,t in enumerate([0.,.25,.5,.75,1.,1.25,1.5,1.75]):
            rows.append({'source_frame_index':i*6,'pts_seconds':t})
        frame={'requested_start_seconds':0.}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);black=root/'black';content=root/'content'
            black.mkdir();content.mkdir()
            for i in range(8):
                image=Image.new('RGB',(160,160),(40+i*5,60,90))
                for y in range(160):
                    for x in range(160):
                        if (x//8+y//8)%2:image.putpixel((x,y),(180,80,30))
                if i < 2:
                    Image.new('RGB',(160,160),(0,0,0)).save(black/f'frame_{i:04d}.png')
                else:
                    image.save(black/f'frame_{i:04d}.png')
                image.save(content/f'frame_{i:04d}.png')
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','4',
                '-i',str(black/'frame_%04d.png'),'-f','lavfi','-i','sine=frequency=400:duration=2',
                '-shortest','-c:v','mpeg4','-pix_fmt','yuv420p',str(root/'splash.mp4')],check=True)
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','4',
                '-i',str(content/'frame_%04d.png'),'-f','lavfi','-i','sine=frequency=400:duration=2',
                '-shortest','-c:v','mpeg4','-pix_fmt','yuv420p',str(root/'scene.mp4')],check=True)
            splash_rows=[{'source_frame_index':i,'pts_seconds':i*.25} for i in range(8)]
            for i in range(2):splash_rows[i]['pts_seconds']=i*.25
            # Two splash frames, then two consecutive actual-scene frames.
            detected=detect_visual_content_start(root/'splash.mp4',splash_rows,0.,2.)
            self.assertEqual(detected['status'],'STARTUP_SPLASH_SKIPPED')
            self.assertGreaterEqual(detected['content_start_seconds'],.5)
            self.assertEqual(detect_visual_content_start(root/'scene.mp4',splash_rows,0.,2.)['status'],
                             'SCENE_CONTENT_AT_START')

    def test_startup_frame_metrics_identify_splash_but_keep_dark_scene_detail(self):
        import numpy as np
        blank=np.zeros((160,160,3),dtype=np.uint8)
        textured=np.zeros((160,160,3),dtype=np.uint8)
        for y in range(160):
            for x in range(160):
                if (x//8+y//8)%2:textured[y,x]=[45,55,65]
        self.assertFalse(startup_frame_metrics(blank)['usable_scene_content'])
        self.assertTrue(startup_frame_metrics(textured)['usable_scene_content'])

    def test_dense_video_extraction_supplies_multiple_ordered_real_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / 'clip.mp4'
            source = root / 'source'
            source.mkdir()
            for index in range(120):
                Image.new('RGB', (96, 96), (index % 255, 20, 40)).save(
                    source / f'frame_{index:04d}.png')
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-framerate', '12', '-i', str(source / 'frame_%04d.png'),
                '-c:v', 'mpeg4', '-pix_fmt', 'yuv420p', str(video)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = extract_dense_frames(video, root / 'frames', 1.0, 4.0,
                                          fps=4.0, min_frames=8, max_frames=32)
            self.assertGreaterEqual(result['frame_count'], 8)
            self.assertLessEqual(result['frame_count'], 32)
            self.assertEqual(result['frame_count'] % 2, 0)
            times = [f['time_seconds'] for f in result['frames']]
            self.assertEqual(times, sorted(times))
            self.assertEqual(len({f['source_frame_index'] for f in result['frames']}), result['frame_count'])
            self.assertTrue(all(f['timestamp_source'] == 'ffprobe_best_effort_pts'
                                for f in result['frames']))
            self.assertEqual(len(qwen_temporal_groups(result['frames'])), result['frame_count'] // 2)
            self.assertEqual(result['qwen_temporal_groups'][0]['source_frame_ids'],
                             [result['frames'][0]['source_frame_id'], result['frames'][1]['source_frame_id']])
            for frame in result['frames']:
                with Image.open(frame['path']) as image:
                    self.assertEqual(image.size, (96, 96))


if __name__ == '__main__':
    unittest.main()
