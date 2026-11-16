"""Synthetic color-readout controls, separate from the real-gaze pilot.

The arbitrary colored Gaussian centers are NOT gaze or object/target labels.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from MultiAgent.heatmap_temporal_pilot import prompt, save, sha, score_response
from MultiAgent.qwen_backend import QwenBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, default=Path('.local-sync/heatmap_temporal_pilot_20261105'))
    parser.add_argument('--infer', action='store_true')
    args = parser.parse_args()
    pilot = args.pilot.resolve()
    out = pilot/'synthetic_color_controls'
    manifest_path = out/'manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    else:
        main_manifest = json.loads((pilot/'stimulus_manifest.json').read_text())
        event = main_manifest['events'][1]  # Fixed scene2/411, not selected by model output.
        conditions = ['SYNTHETIC_FORWARD', 'SYNTHETIC_REVERSE']
        frames = []
        for index, frame in enumerate(event['frames']):
            with Image.open(pilot/frame['conditions']['RGB']['path']) as im:
                image = im.convert('RGB')
            # Deliberately large, opaque Gaussian color regions at unrelated locations.
            positions = [[100.,140.], [340.,140.], [100.,380.], [340.,380.]]
            old = positions[index % 4]
            new = positions[(index + 2) % 4]
            yy, xx = np.indices((image.height, image.width))
            blue = np.exp(-((xx-old[0])**2+(yy-old[1])**2)/(2*20.**2))
            red = np.exp(-((xx-new[0])**2+(yy-new[1])**2)/(2*20.**2))
            maps = [blue, red]
            total = blue + red
            alpha = np.minimum(total, 1.)*.95
            metadata = {}
            for condition in conditions:
                reverse = condition == 'SYNTHETIC_REVERSE'
                colors = np.array([[40,120,255], [255,70,0]], np.float32)
                if reverse:
                    colors = colors[::-1]
                color = np.einsum('khw,kc->hwc', np.array(maps), colors)/np.maximum(total[...,None],1e-12)
                rendered = Image.fromarray(np.clip(np.array(image)*(1-alpha[...,None])
                                           + color*alpha[...,None],0,255).astype('uint8'))
                path = out/'inputs'/condition/f'{frame["frame_id"]}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                rendered.save(path)
                older, newer = (new, old) if reverse else (old, new)
                metadata[condition] = {'path': str(path.relative_to(pilot)), 'sha256': sha(path),
                    'image_size': list(image.size), 'expected_focus_xy': newer,
                    'expected_older_xy': older, 'expected_newer_xy': newer}
            frames.append({**{k: frame[k] for k in ['frame_id','motion_bin','clock_supported']},
                           'conditions': metadata})
        synthetic_event = {'key': event['key'], 'instruction': event['instruction'], 'frames': frames}
        manifest = {'synthetic': True, 'real_gaze': False, 'object_labels_used': False,
                    'scope': 'diagnose ability to read large arbitrary colored regions; not task utility',
                    'sigma_pixels':20., 'peak_alpha':.95, 'event': synthetic_event,
                    'conditions': conditions, 'prompt': prompt(event)}
        save(manifest_path, manifest)
    if args.infer:
        backend = QwenBackend(out/'model_cache', model='/ai/data/Qwen3-VL-8B-Instruct',
                              local_only=True, namespace='synthetic_color_readout_v1', trace_visual_inputs=True)
        event = manifest['event']
        results = []
        for condition in manifest['conditions']:
            result_path = out/f'{condition}.json'
            if result_path.is_file():
                results.append(json.loads(result_path.read_text()))
                continue
            paths = [pilot/r['conditions'][condition]['path'] for r in event['frames']]
            for path, frame in zip(paths, event['frames']):
                assert sha(path) == frame['conditions'][condition]['sha256']
            print(f'INFER {condition}',flush=True)
            parsed = backend.ask(event['key'],condition,manifest['prompt'],paths,max_new_tokens=2600)
            scored, valid = score_response(parsed,event,condition)
            request=json.loads(Path(backend.requests[-1]).read_text())
            result={'synthetic':True,'condition':condition,'parsed':parsed,'schema_valid':valid,
                    'scores':scored,'request_path':backend.requests[-1],
                    'vision_verified':request['visual_tensor_trace']['input_consumption_verified'],
                    'processor_audit':request['processor_audit']}
            save(result_path,result)
            results.append(result)
        # If the chronological-image task is unreadable even with these large
        # synthetic marks, isolate a single image with an explicitly simpler task.
        single_path = out/'SINGLE_FRAME_DIAGNOSTIC.json'
        endpoint_hits = sum(score['both_endpoints_success'] for result in results for score in result['scores'])
        endpoint_frames = sum(len(result['scores']) for result in results)
        if endpoint_hits < endpoint_frames / 2 and not single_path.is_file():
            image_path = pilot/event['frames'][6]['conditions']['SYNTHETIC_FORWARD']['path']
            single_prompt = ('This image contains two added Gaussian color marks. Locate the BLUE mark '
                             'and the ORANGE/RED mark separately. Return strict JSON only: '
                             '{"blue_xy":[x,y],"orange_xy":[x,y]}. Coordinates are integers 0 to 1000, '
                             'upper-left origin, using the WHOLE image including the header. '
                             'Return null only if the specified mark is not visible. Do not identify objects.')
            parsed = backend.ask('single_frame','SYNTHETIC_FORWARD',single_prompt,[image_path],max_new_tokens=300)
            save(single_path,{'synthetic':True,'different_prompt':True,'image_path':str(image_path),
                              'parsed':parsed,'request_path':backend.requests[-1],
                              'expected_pixels':event['frames'][6]['conditions']['SYNTHETIC_FORWARD']})
        save(out/'summary.json',{'synthetic':True,'results':[
            {'condition':r['condition'],'schema_valid':r['schema_valid'],
             'focus_hits':sum(s['focus_xy']['success_at_10pct'] for s in r['scores']),
             'older_hits':sum(s['older_xy']['success_at_10pct'] for s in r['scores']),
             'newer_hits':sum(s['newer_xy']['success_at_10pct'] for s in r['scores']),
             'frames':len(r['scores'])} for r in results]})


if __name__ == '__main__':
    main()
