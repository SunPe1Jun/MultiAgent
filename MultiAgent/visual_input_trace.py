"""Opt-in tensor provenance at processor output and the real vision forward call."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def tensor_record(tensor):
    import torch
    value = tensor.detach().cpu().contiguous()
    data = value.view(torch.uint8).numpy().tobytes()
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def restore_video_pixels(pixels, grids, *, patch_size, temporal_patch_size,
                         merge_size, image_mean, image_std):
    """Invert Qwen3-VL patch packing for a reviewable pre-normalization image."""
    import torch
    outputs = []
    offset = 0
    for grid in grids:
        t, h, w = map(int, grid)
        count = t * h * w
        part = pixels[offset:offset + count].detach().cpu().float()
        offset += count
        part = part.reshape(t, h // merge_size, w // merge_size, merge_size,
                            merge_size, 3, temporal_patch_size, patch_size, patch_size)
        frames = part.permute(0, 6, 5, 1, 3, 7, 2, 4, 8).reshape(
            t * temporal_patch_size, 3, h * patch_size, w * patch_size)
        mean = torch.tensor(image_mean).reshape(1, 3, 1, 1)
        std = torch.tensor(image_std).reshape(1, 3, 1, 1)
        outputs.append(((frames * std + mean) * 255).round().clamp(0, 255).byte())
    if offset != len(pixels):
        raise ValueError('Unused processor pixel rows while restoring Qwen inputs')
    return outputs


class VisualInputTrace:
    def __init__(self, runner, inputs, request_path, *, kind):
        import torch
        from PIL import Image
        self.kind = kind
        self.pixel_key = 'pixel_values_videos' if kind == 'video' else 'pixel_values'
        self.grid_key = 'video_grid_thw' if kind == 'video' else 'image_grid_thw'
        self.path = Path(request_path).with_suffix('.tensor_trace')
        self.path.mkdir(parents=True, exist_ok=True)
        self.inputs = inputs
        self.vision = runner.model.model.visual
        processor = (runner.processor.video_processor if kind == 'video'
                     else runner.processor.image_processor)
        tensors = {key: value.detach().cpu().contiguous()
                   for key, value in inputs.items() if torch.is_tensor(value)}
        self.data = {'kind': kind, 'capture_scope': 'processor tensors + real vision forward input',
                     'processor_tensors': {key: tensor_record(value) for key, value in tensors.items()},
                     'processor_inputs_path': str(self.path / 'processor_inputs.pt'),
                     'vision_forward_calls': [], 'restored_frame_paths': [],
                     'input_consumption_verified': False}
        torch.save(tensors, self.path / 'processor_inputs.pt')
        restored = restore_video_pixels(
            tensors[self.pixel_key], tensors[self.grid_key].tolist(),
            patch_size=int(processor.patch_size),
            temporal_patch_size=int(processor.temporal_patch_size),
            merge_size=int(processor.merge_size),
            image_mean=processor.image_mean, image_std=processor.image_std)
        for media_index, frames in enumerate(restored):
            # Static images are repeated along the temporal patch axis by Qwen.
            # Save the first copy; preserve every real frame for video inputs.
            for index, frame in enumerate(frames if kind == 'video' else frames[:1]):
                target = self.path / f'processor_rgb_{media_index:02d}_{index:03d}.png'
                Image.fromarray(frame.permute(1, 2, 0).numpy()).save(target)
                self.data['restored_frame_paths'].append(str(target))
        self.data['expected_vision_tensor'] = tensor_record(
            tensors[self.pixel_key].to(dtype=self.vision.dtype))
        self.handle = self.vision.register_forward_pre_hook(self._capture, with_kwargs=True)
        self.finish()

    def _capture(self, module, args, kwargs):
        import torch
        value = args[0] if args else kwargs['hidden_states']
        grid = kwargs.get('grid_thw')
        row = {'input': tensor_record(value), 'grid': tensor_record(grid),
               'input_matches_processor_after_dtype_cast': (
                   tensor_record(value) == self.data['expected_vision_tensor']),
               'grid_matches_processor': tensor_record(grid) == tensor_record(self.inputs[self.grid_key])}
        self.data['vision_forward_calls'].append(row)
        self.finish()

    def finish(self):
        calls = self.data['vision_forward_calls']
        self.data['input_consumption_verified'] = bool(calls) and all(
            row['input_matches_processor_after_dtype_cast'] and row['grid_matches_processor']
            for row in calls)
        (self.path / 'trace.json').write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + '\n')
        return {'trace_path': str(self.path / 'trace.json'), **self.data}

    def close(self):
        self.handle.remove()
        return self.finish()
