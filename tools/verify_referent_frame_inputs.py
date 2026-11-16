#!/usr/bin/env python3
"""Independently rebuild processor tensors for the referent frame-selection probes."""
import argparse
import json
from pathlib import Path

from tools.verify_projected2d_main_inputs import record


def verify(outputs):
    from PIL import Image
    from transformers import AutoProcessor
    from MultiAgent.projected2d_attention_study import save_json
    from MultiAgent.projected2d_main_image import verify_lock
    from MultiAgent.projected2d_routed_rgb import MODEL
    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True)
    for output in outputs:
        verify_lock(output)
        manifest = json.loads((output / 'input_manifest.json').read_text())
        verified = []
        for path in sorted((output / 'responses').glob('*.json')):
            response = json.loads(path.read_text())
            request = json.loads(Path(response['request_path']).read_text())
            captions, image_paths = request['image_captions'], request['image_paths']
            if len(captions) != len(image_paths):
                raise ValueError('Image-caption count differs')
            content, images = [], []
            for caption, image_path in zip(captions, image_paths):
                content.extend([{'type': 'text', 'text': caption}, {'type': 'image'}])
                with Image.open(image_path) as image:
                    images.append(image.convert('RGB'))
            content.append({'type': 'text', 'text': request['prompt']})
            messages = [{'role': 'system', 'content': 'Return only valid compact JSON. Base claims on the supplied evidence; mark missing evidence explicitly.'},
                        {'role': 'user', 'content': content}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=images, return_tensors='pt')
            for image in images:
                image.close()
            trace = request['visual_tensor_trace']
            if not trace['input_consumption_verified']:
                raise ValueError('Vision forward did not consume expected input')
            for key in ('input_ids', 'image_grid_thw', 'pixel_values'):
                if record(inputs[key]) != trace['processor_tensors'][key]:
                    raise ValueError('Processor tensor differs: ' + str(path) + ':' + key)
            decoded = processor.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
            positions = [decoded.index(caption) for caption in captions]
            if positions != sorted(positions):
                raise ValueError('Incorrect image-caption binding')
            verified.append({'response': path.name, 'images': len(images),
                             'processor_hashes_match': True, 'caption_binding_verified': True,
                             'vision_forward_verified': True})
        conditions = len(manifest.get('conditions', [])) or 1
        expected = len(manifest['events']) * conditions
        if len(verified) != expected:
            raise ValueError(f'Incomplete calls: {len(verified)}/{expected}')
        save_json(output / 'processor_reconstruction_verification.json',
                  {'calls_verified': len(verified), 'checks': verified,
                   'method': 'Exact input_ids, pixel_values and image_grid_thw reconstructed from request; captions checked on decoded input; actual vision-forward trace verified.'})
        print('PROCESSOR_RECONSTRUCTED', output.name, len(verified), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('outputs', type=Path, nargs='+'); a = p.parse_args()
    verify([o.resolve() for o in a.outputs])


if __name__ == '__main__':
    main()
