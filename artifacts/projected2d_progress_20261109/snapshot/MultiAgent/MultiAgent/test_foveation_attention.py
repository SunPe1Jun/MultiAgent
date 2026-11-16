import numpy as np
from PIL import Image

from MultiAgent.foveation_attention import foveate_from_history


def test_no_clock_or_empty_support_is_exact_rgb_fallback():
    image = Image.fromarray(np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3))
    zeros = np.zeros((3, 48, 64), np.float32)
    for supported, maps in [(False, np.ones_like(zeros)), (True, zeros)]:
        rendered, meta = foveate_from_history(image, maps, clock_supported=supported,
            vertical_fov_degrees=90.)
        assert np.array_equal(np.asarray(rendered), np.asarray(image))
        assert meta['status'] == 'RGB_FALLBACK'


def test_supported_foveation_keeps_full_dimensions_and_multiple_foci_sharp():
    yy, xx = np.indices((120, 160))
    image = Image.fromarray(np.stack([xx * 255 // 159, yy * 255 // 119,
        ((xx // 4 + yy // 4) % 2) * 255], axis=-1).astype(np.uint8))
    maps = np.zeros((3, 120, 160), np.float32)
    for x, y, k, seconds in [(35, 40, 0, .5), (125, 82, 2, .5)]:
        maps[k] = np.exp(-((xx-x)**2 + (yy-y)**2) / (2 * 2.**2)) * seconds
    rendered, meta = foveate_from_history(image, maps, clock_supported=True,
        vertical_fov_degrees=90.)
    arr = np.asarray(rendered)
    original = np.asarray(image)
    assert rendered.size == image.size
    assert meta['status'] == 'FOVEATED_FULL_FIELD'
    assert np.abs(arr[40, 35].astype(int) - original[40, 35].astype(int)).max() <= 1
    assert np.abs(arr[82, 125].astype(int) - original[82, 125].astype(int)).max() <= 1
    assert not np.array_equal(arr[60, 80], original[60, 80])
    assert meta['focus_field_sharp_fraction'] > 0
