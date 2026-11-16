"""Shared camera sampling and unclipped pinhole projection; no labels or model calls."""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

VERSION = 'camera_geometry_v2'
ATTENTION_CLOCK_POLICY = 'validated_candidate_or_rgb_supported_original_v1'


def camera_values(position, quaternion, fov):
    position = np.asarray(position, dtype=float)
    quaternion = np.asarray(quaternion, dtype=float)
    fov = float(fov)
    if (position.shape != (3,) or quaternion.shape != (4,)
            or not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion))
            or not math.isfinite(fov) or not 0 < fov < 180):
        raise ValueError('Invalid camera position, xyzw quaternion or vertical FOV')
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError('Invalid camera quaternion norm')
    return position, quaternion / norm, fov


class CameraPoseTimeline:
    """Linear positions/FOV and shortest-path SLERP, without gap bridging/extrapolation."""
    def __init__(self, records, max_gap_seconds=.25):
        if not records:
            raise ValueError('At least one valid camera sample required')
        if not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
            raise ValueError('max_gap_seconds must be finite and positive')
        self.records = records
        self.times = np.asarray([r['telemetry_time_seconds'] for r in records], dtype=float)
        if not np.all(np.isfinite(self.times)) or not np.all(np.diff(self.times) > 0):
            raise ValueError('Camera timestamps must be finite and increase strictly')
        values = [camera_values(r['camera_position'], r['camera_rotation_xyzw'],
                                r['camera_fov_degrees']) for r in records]
        self.positions = np.array([v[0] for v in values])
        self.quaternions = np.array([v[1] for v in values])
        self.fovs = np.array([v[2] for v in values])
        self.max_gap = max_gap_seconds
        rotations = Rotation.from_quat(self.quaternions)
        self.rotations = Slerp(self.times, rotations) if len(records) > 1 else None

    def sample_many(self, seconds):
        seconds = np.asarray(seconds, dtype=float)
        if seconds.ndim != 1:
            raise ValueError('Camera query times must be one-dimensional')
        if not len(seconds):
            return np.empty(0, bool), np.empty((0, 3, 3)), np.empty(0)
        finite = np.isfinite(seconds)
        clamped = np.clip(np.where(finite, seconds, self.times[0]), self.times[0], self.times[-1])
        if len(self.times) == 1:
            valid = finite & (seconds == self.times[0])
            matrices = np.repeat(Rotation.from_quat(self.quaternions).as_matrix(), len(seconds), axis=0)
            return valid, matrices, np.full(len(seconds), self.fovs[0])
        right = np.clip(np.searchsorted(self.times, clamped, side='left'), 1, len(self.times) - 1)
        left = right - 1
        exact = (clamped == self.times[right]) | (clamped == self.times[left])
        valid = (finite & (seconds >= self.times[0]) & (seconds <= self.times[-1])
                 & ((self.times[right] - self.times[left] <= self.max_gap) | exact))
        return valid, self.rotations(clamped).as_matrix(), np.interp(clamped, self.times, self.fovs)

    def pose_at(self, telemetry_seconds):
        valid, matrices, fovs = self.sample_many([telemetry_seconds])
        if not valid[0]:
            return {'status': 'POSE_UNAVAILABLE',
                    'reason': 'nonfinite_time_outside_camera_range_or_large_sample_gap',
                    'telemetry_time_seconds': float(telemetry_seconds), 'extrapolated': False}
        right = min(len(self.times) - 1, int(np.searchsorted(self.times, telemetry_seconds)))
        left = right if self.times[right] == telemetry_seconds else max(0, right - 1)
        return {'status': 'POSE_AVAILABLE', 'telemetry_time_seconds': float(telemetry_seconds),
                'position_world': [float(np.interp(telemetry_seconds, self.times, self.positions[:, axis]))
                                   for axis in range(3)],
                'rotation_xyzw': Rotation.from_matrix(matrices[0]).as_quat().tolist(),
                'fov_vertical_degrees': float(fovs[0]),
                'pose_source_indices': [self.records[left]['source_index'], self.records[right]['source_index']],
                'pose_source_times': [float(self.times[left]), float(self.times[right])],
                'pose_bracket_gap_seconds': float(self.times[right] - self.times[left]),
                'pose_interpolation': 'linear_position_slerp_rotation', 'extrapolated': False,
                'geometry_version': VERSION}


def interpolate_pose(records, target_time, max_gap_seconds=.25):
    """Sample gaze/camera records with missing camera fields; retain actual source indices."""
    if target_time is None:
        return None
    clean = []
    for row in records:
        try:
            seconds = float(row['telemetry_time_seconds'])
            position, quaternion, fov = camera_values(row['camera_position'], row['camera_rotation_xyzw'],
                                                       row['camera_fov_degrees'])
            if not math.isfinite(seconds) or (clean and seconds <= clean[-1]['telemetry_time_seconds']):
                continue
            clean.append({**row, 'telemetry_time_seconds': seconds, 'camera_position': position,
                          'camera_rotation_xyzw': quaternion, 'camera_fov_degrees': fov})
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    if not clean:
        return None
    pose = CameraPoseTimeline(clean, max_gap_seconds).pose_at(target_time)
    return pose if pose['status'] == 'POSE_AVAILABLE' else None


def project_world(value, pose, width, height, *, direction=False):
    """Unity camera-to-world xyzw, +z forward, vertical FOV, square pixels, top-left UV."""
    result = {'in_frustum': False, 'occlusion': 'unknown_no_depth_or_mesh',
              'pixel_calibration': 'unverified_recorded_camera_vs_video',
              'projection_model': 'vertical_fov_square_pixels_ideal_pinhole',
              'geometry_version': VERSION}
    try:
        width, height = float(width), float(height)
        if not all(math.isfinite(v) and v > 0 for v in (width, height)):
            raise ValueError('Image dimensions must be finite and positive')
        position, quaternion, fov = camera_values(pose['position_world'], pose['rotation_xyzw'],
                                                 pose['fov_vertical_degrees'])
        value = np.asarray(value, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError('A finite three-dimensional point/direction is required')
        if direction and np.linalg.norm(value) < 1e-8:
            raise ValueError('World direction must be nonzero')
        local = Rotation.from_quat(quaternion).inv().apply(value if direction else value - position)
        if not np.all(np.isfinite(local)):
            raise ValueError('Camera transformation produced nonfinite coordinates')
        focal = height / (2 * math.tan(math.radians(fov) / 2))
        if not math.isfinite(focal) or focal <= 0:
            raise ValueError('Invalid camera focal length')
        result.update(camera_xyz=local.tolist(), camera_position_world=position.tolist(),
                      camera_rotation_xyzw=quaternion.tolist(), vertical_fov_degrees=fov,
                      intrinsics={'fx': focal, 'fy': focal, 'cx': width / 2, 'cy': height / 2})
        if local[2] <= 1e-6:
            return {**result, 'status': 'behind_camera', 'normalized_uv_unclipped': None,
                    'pixel_xy_unclipped': None}
        u = .5 + focal * local[0] / (local[2] * width)
        v = .5 - focal * local[1] / (local[2] * height)
        if not math.isfinite(u) or not math.isfinite(v):
            raise ValueError('Projection produced nonfinite UV')
        inside = 0 <= u < 1 and 0 <= v < 1
        return {**result, 'status': 'in_frustum_visibility_unknown' if inside else 'outside_frame',
                'in_frustum': bool(inside), 'normalized_uv_unclipped': [float(u), float(v)],
                'pixel_xy_unclipped': [float(u * width), float(v * height)],
                'outside_edges': [edge for edge, outside in [('left', u < 0), ('right', u >= 1),
                                                            ('top', v < 0), ('bottom', v >= 1)] if outside]}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return {**result, 'status': 'invalid_geometry', 'error': str(exc)}


def aligned_frame_geometry(frame, source_sha256, requested_video_time, fallback_offset):
    """Select the same attention/projection camera for audits and experiments.

    An explicit attention camera is authoritative even when its pose is unavailable;
    never silently substitute the original camera at a different telemetry time.
    Older contracts containing only reference_camera remain supported.
    """
    camera_field = 'attention_camera' if 'attention_camera' in frame else 'reference_camera'
    camera = frame.get(camera_field) or {}
    timing = frame.get('frame_time') or {}
    actual = timing.get('actual_container_pts_seconds')
    verified = timing.get('status') in {'VERIFIED_EXACT_SOURCE_JPEG', 'VERIFIED_CONTAINER_PTS_SEQUENCE'}
    try:
        actual = float(actual)
        verified = verified and math.isfinite(actual)
    except (TypeError, ValueError):
        verified = False
    video_time = actual if verified else float(requested_video_time)
    result = {'video_time_seconds': video_time,
              'telemetry_time_seconds': video_time - float(fallback_offset),
              'clock_available': False, 'reference_pose': None, 'pose_available': False,
              'camera_source_field': camera_field, 'clock_policy': ATTENTION_CLOCK_POLICY,
              'selected_clock_offset_seconds': camera.get('clock_offset_seconds'),
              'selected_pose_clock_source': camera.get('clock_status'),
              'pose_status': camera.get('status', 'FRAME_POSE_CONTRACT_MISSING'),
              'pose_reasons': camera.get('reasons', camera.get('reason')),
              'status': 'FRAME_TIME_UNVERIFIED' if not verified else 'POSE_UNAVAILABLE'}
    if not verified:
        return result
    if frame.get('source_sha256') != source_sha256:
        raise ValueError('Frame pose contract does not match the actual source image SHA256')
    try:
        query = float(camera['telemetry_time_seconds'])
        offset = float(camera['clock_offset_seconds'])
        if not math.isfinite(query) or not math.isfinite(offset):
            return result
    except (KeyError, TypeError, ValueError, OverflowError):
        return result
    if abs(query + offset - video_time) > 1e-6:
        raise ValueError('Frame PTS, telemetry query and clock offset are inconsistent')
    result['telemetry_time_seconds'] = query
    if camera.get('status') != 'POSE_AVAILABLE':
        return result
    try:
        camera_values(camera['position_world'], camera['rotation_xyzw'], camera['fov_vertical_degrees'])
    except (KeyError, TypeError, ValueError, OverflowError):
        return result
    clock_status = camera.get('clock_status')
    candidate = (clock_status in {'VALIDATED_CANDIDATE', 'VALIDATED_CANDIDATE_NO_ORIGINAL_CLOCK'}
                 and camera.get('candidate_clock_status', 'VALIDATED_CANDIDATE') == 'VALIDATED_CANDIDATE')
    original = (clock_status == 'ORIGINAL_CLOCK_SUPPORTED'
                or (clock_status in {'ORIGINAL_CLOCK_FALLBACK_PTS_POSE', 'ORIGINAL_CLOCK_REFERENCE_PTS_POSE'}
                    and camera.get('original_clock_status') == 'ORIGINAL_CLOCK_SUPPORTED'))
    supported = (not camera.get('diagnostic_only', False)
                 and camera.get('attention_clock_supported') is not False
                 and (candidate or original))
    return {**result, 'clock_available': supported, 'reference_pose': camera,
            'telemetry_time_seconds': query, 'pose_available': True,
            'status': 'VERIFIED_FRAME_POSE_CONTRACT' if supported else 'VERIFIED_POSE_CLOCK_UNCERTAIN'}
