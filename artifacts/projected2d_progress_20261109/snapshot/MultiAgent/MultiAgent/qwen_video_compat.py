"""Narrow compatibility repair for Qwen3-VL's timestamp-separated video RoPE.

Transformers 5.3.0 omits video-grid splitting present in upstream Qwen3-VL
v4.57.1 and upstream main. Only the RoPE call receives the split grid; the
processor/vision forward grid and pixel tensors are never modified.
"""
from functools import wraps
import hashlib
import inspect

CONTRACT = 'qwen3vl_timestamp_video_rope_grid_split_v1'
UPSTREAM = 'https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py'


def ensure_timestamp_video_rope(model):
    inner = model.model
    prior = getattr(inner, '_multiagent_video_rope_audit', None)
    if prior is not None: return prior
    original = inner.get_rope_index
    source = inspect.getsource(original)
    audit = {'contract': CONTRACT, 'original_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
             'upstream_reference': UPSTREAM, 'vision_grid_and_pixels_unchanged': True,
             'runtime_repair_applied': False}
    if 'repeat_interleave(video_grid_thw' not in source:
        import torch
        @wraps(original)
        def corrected(input_ids, mm_token_type_ids, image_grid_thw=None,
                      video_grid_thw=None, attention_mask=None, **kwargs):
            if video_grid_thw is not None:
                video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
                video_grid_thw[:, 0] = 1
            return original(input_ids, mm_token_type_ids, image_grid_thw=image_grid_thw,
                            video_grid_thw=video_grid_thw, attention_mask=attention_mask, **kwargs)
        inner.get_rope_index = corrected
        audit['runtime_repair_applied'] = True
    inner._multiagent_video_rope_audit = audit
    return audit
