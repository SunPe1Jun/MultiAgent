#!/usr/bin/env python3
"""Reproduce actual processor tensors and caption order for completed main-image calls."""
from pathlib import Path
import hashlib
import json
import sys


def record(tensor):
    import torch
    value = tensor.detach().cpu().contiguous()
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
            'bytes': value.numel() * value.element_size()}


def verify(output):
    from PIL import Image
    from transformers import AutoProcessor
    from MultiAgent.projected2d_attention_study import save_json
    from MultiAgent.projected2d_main_image import verify_lock
    verify_lock(output)
    manifest = json.loads((output / 'input_manifest.json').read_text())
    processor = AutoProcessor.from_pretrained(manifest['model'], local_files_only=True)
    verified = []
    for path in sorted((output / 'responses').glob('*.json')):
        response = json.loads(path.read_text())
        if response['status'] == 'NOT_RUN':
            continue
        if 'request_path' not in response:
            raise ValueError('Completed call lacks processor provenance: ' + str(path))
        request = json.loads(Path(response['request_path']).read_text())
        captions = request['image_captions']
        content = []
        images = []
        for caption, image_path in zip(captions, request['image_paths']):
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
                raise ValueError('Reconstructed processor tensor differs: ' + str(path) + ':' + key)
        decoded = processor.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
        positions = [decoded.index(caption) for caption in captions]
        if positions != sorted(positions):
            raise ValueError('Frame captions appeared in the wrong image order')
        target = output / 'actual_model_inputs' / (path.stem + '.txt')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(decoded)
        verified.append({'response': path.name, 'processor_tensor_hashes_reproduced': True,
                         'caption_order_verified': True, 'vision_input_verified': True})
    result = {'calls_verified': len(verified), 'checks': verified,
              'method': 'Rebuilt input_ids, image_grid_thw and pixel_values from actual request; exact hashes match frozen processor trace. Caption order checked on decoded actual input_ids.'}
    save_json(output / 'processor_reconstruction_verification.json', result)
    print('VERIFIED_PROCESSOR_CALLS', len(verified), flush=True)


if __name__ == '__main__':
    verify(Path(sys.argv[1]).resolve())
