"""Full-field multi-focus foveation from causal gaze-duration maps.

The rendering field encodes where supported gaze history accumulated. It never
selects an object, center, crop, or identity, and invalid support returns RGB.
"""
from __future__ import annotations

import math
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label

VERSION = 'multi_focus_gaze_history_foveation_v1'


def foveate_from_history(image, maps, *, clock_supported, vertical_fov_degrees,
                         min_rgb_weight=0.55, dilation_degrees=4.0,
                         support_saturation_seconds=0.25):
    """Apply a continuous multi-aperture resolution field without cropping.

    All temporal planes contribute as measured duration. Multiple gaze modes
    remain separate. Sparse support weakens the effect, and absent clock/map
    support gives an exact RGB fallback. The entire original field is retained.
    """
    source = image.convert('RGB')
    original = np.asarray(source, dtype=np.float32)
    maps = np.asarray(maps, dtype=np.float32)
    if maps.ndim != 3 or maps.shape[0] != 3 or maps.shape[1:] != original.shape[:2]:
        raise ValueError('Expected three gaze planes matching the full RGB field')
    if not np.all(np.isfinite(maps)) or np.any(maps < 0):
        raise ValueError('Gaze duration maps must be finite and nonnegative')
    if not (0 <= min_rgb_weight <= 1 and dilation_degrees >= 0
            and support_saturation_seconds > 0):
        raise ValueError('Invalid foveation parameters')
    if not clock_supported or float(maps.sum()) <= 0:
        return source.copy(), {'status': 'RGB_FALLBACK', 'foveation_enabled': False,
            'total_supported_seconds': float(maps.sum()), 'foci': 0}
    if not (math.isfinite(vertical_fov_degrees) and 0 < vertical_fov_degrees < 180):
        raise ValueError('Supported gaze requires a finite camera field of view')

    height, width = original.shape[:2]
    density = maps.sum(axis=0, dtype=np.float64).astype(np.float32)
    peak = float(density.max())
    if peak <= 0:
        return source.copy(), {'status': 'RGB_FALLBACK', 'foveation_enabled': False,
            'total_supported_seconds': float(maps.sum()), 'foci': 0}

    support_seconds = float(maps.sum())
    support_gate = min(1., support_seconds / support_saturation_seconds)
    field = np.sqrt(np.clip(density / peak, 0., 1.)) * support_gate
    focal_pixels = height / (2. * math.tan(math.radians(vertical_fov_degrees) / 2.))
    dilation_pixels = focal_pixels * math.tan(math.radians(dilation_degrees))
    if dilation_pixels >= 0.5:
        # Smoothly broaden each observed focus to a modest aperture while
        # preserving distant modes and their relative measured-duration mass.
        field = gaussian_filter(field, sigma=max(0.5, dilation_pixels), mode='nearest')
        field *= support_gate / max(float(field.max()), 1e-12)
    field = np.clip(field, 0., 1.)

    levels = []
    for factor in (2, 4, 8):
        small = source.resize((max(1, width // factor), max(1, height // factor)), Image.Resampling.BOX)
        levels.append(np.asarray(small.resize(source.size, Image.Resampling.BICUBIC), dtype=np.float32))
    position = (1. - field) * (len(levels) - 1)
    low = np.floor(position).astype(np.intp)
    high = np.minimum(low + 1, len(levels) - 1)
    mix = (position - low)[..., None]
    stack = np.stack(levels)
    yy, xx = np.indices(field.shape)
    locally_resolved = stack[low, yy, xx] * (1. - mix) + stack[high, yy, xx] * mix
    original_weight = min_rgb_weight + (1. - min_rgb_weight) * field
    pixels = original * original_weight[..., None] + locally_resolved * (1. - original_weight[..., None])
    result = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    return result, {
        'status': 'FOVEATED_FULL_FIELD', 'foveation_enabled': True,
        'version': VERSION, 'total_supported_seconds': support_seconds,
        'support_gate': support_gate, 'dilation_degrees': dilation_degrees,
        'dilation_pixels': dilation_pixels, 'minimum_original_rgb_weight': min_rgb_weight,
        'focus_field_mean': float(field.mean()), 'focus_field_max': float(field.max()),
        'focus_field_sharp_fraction': float(np.mean(field >= 0.8)),
        'focus_components_at_10pct_peak': int(label(density >= peak * 0.1)[1]),
        'semantics': 'multi-focus rendering field derived from past measured gaze duration; not object identity or target probability',
    }
