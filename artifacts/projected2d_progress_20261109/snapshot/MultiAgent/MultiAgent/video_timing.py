"""Strict decoded-frame timing: missing PTS must not silently shift frame indices."""
import math


def frame_pts_table(rows):
    if not rows:
        raise ValueError('No decoded video frames')
    pts = []
    for index, row in enumerate(rows):
        try:
            value = float(row['best_effort_timestamp_time'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Missing/invalid decoded frame PTS at source index {index}') from exc
        if not math.isfinite(value) or (pts and value <= pts[-1]):
            raise ValueError(f'Nonfinite or non-increasing decoded frame PTS at source index {index}')
        pts.append(value)
    return pts


def bound_frame_pts(index, supplied_pts, pts):
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(pts):
        raise ValueError(f'Invalid source frame index: {index}')
    try:
        supplied = float(supplied_pts)
        expected = float(pts[index])
    except (TypeError, ValueError) as exc:
        raise ValueError('Missing/invalid supplied frame PTS') from exc
    if not math.isfinite(supplied) or not math.isfinite(expected) or abs(expected - supplied) > 1e-6:
        raise ValueError(f'Frame-index/PTS binding mismatch: {expected} != {supplied}')
    return expected
