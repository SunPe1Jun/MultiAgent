"""Check recorded world/camera gaze coordinate consistency without target labels."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from .baseline import module
from .context import load_contexts
from .local_video_visual import save


def vec(value):
    try:
        result = tuple(float(value[key]) for key in ('x', 'y', 'z'))
        return result if all(math.isfinite(v) for v in result) else None
    except (KeyError, TypeError, ValueError):
        return None


def norm(value):
    return math.sqrt(sum(v * v for v in value))


def angle(left, right):
    lengths = norm(left) * norm(right)
    if lengths < 1e-10:
        return None
    return math.degrees(math.acos(max(-1., min(1., sum(a * b for a, b in zip(left, right)) / lengths))))


def stats(values):
    clean = sorted(value for value in values if value is not None and math.isfinite(value))
    return {'count': len(clean), 'median': statistics.median(clean) if clean else None,
            'p95': clean[min(len(clean) - 1, int(.95 * len(clean)))] if clean else None,
            'max': max(clean) if clean else None}


def audit(manifest):
    api = module('api')
    results = []
    for ctx in load_contexts(manifest, load_ground_truth=False):
        metrics = {key: [] for key in (
            'gazeVector_vs_rotated_cameraGazeDirection_angle_deg',
            'gazeVector_vs_unrotated_cameraGazeDirection_angle_deg',
            'gazeVector_vs_world_intersection_ray_angle_deg',
            'cameraHitPoint_vs_inverse_pose_gazePoint_distance_world_units',
            'cameraGazeOrigin_vs_inverse_pose_gazeOrigin_distance_world_units')}
        invalid = 0
        for _, sample in ctx.timed_samples:
            eye = sample.get('eyeGaze') or {}
            position, point, origin = (vec(sample.get('cameraPosition')), vec(eye.get('gazePoint')),
                                       vec(eye.get('gazeOrigin')))
            direction, camera_direction = vec(eye.get('gazeVector')), vec(eye.get('cameraGazeDirection'))
            rotation = sample.get('cameraRotation')
            if (position is None or not isinstance(rotation, dict) or
                    any(key not in rotation for key in ('x', 'y', 'z', 'w'))):
                invalid += 1
                continue
            q = tuple(float(rotation[key]) for key in ('x', 'y', 'z', 'w'))
            if any(not math.isfinite(v) for v in q) or norm(q) < 1e-8:
                invalid += 1
                continue
            q = tuple(v / norm(q) for v in q)
            inverse = api.quaternion_conjugate(q)
            if direction is not None and camera_direction is not None:
                transformed = api.rotate_vector_by_quaternion(camera_direction, q)
                metrics['gazeVector_vs_rotated_cameraGazeDirection_angle_deg'].append(angle(direction, transformed))
                metrics['gazeVector_vs_unrotated_cameraGazeDirection_angle_deg'].append(angle(direction, camera_direction))
            if direction is not None and point is not None and origin is not None and norm(point) > 1e-8:
                ray = tuple(a - b for a, b in zip(point, origin))
                metrics['gazeVector_vs_world_intersection_ray_angle_deg'].append(angle(direction, ray))
            for name, world, camera in (
                    ('cameraHitPoint_vs_inverse_pose_gazePoint_distance_world_units', point, vec(eye.get('cameraHitPoint'))),
                    ('cameraGazeOrigin_vs_inverse_pose_gazeOrigin_distance_world_units', origin, vec(eye.get('cameraGazeOrigin')))):
                if world is None or camera is None or (world is point and norm(world) < 1e-8):
                    continue
                projected = api.rotate_vector_by_quaternion(tuple(a - b for a, b in zip(world, position)), inverse)
                metrics[name].append(norm(tuple(a - b for a, b in zip(projected, camera))))
        results.append({'scene_key': ctx.scene_key, 'event_id': ctx.event_id,
                        'telemetry_samples': len(ctx.timed_samples), 'invalid_pose_samples': invalid,
                        'metrics': {key: stats(values) for key, values in metrics.items()}})
    return {'protocol': 'CPU numerical consistency audit, not RGB pixel calibration',
            'ground_truth_loaded': False, 'ground_truth_used': False,
            'coordinate_hypothesis': {
                'gazePoint': 'finite world intersection', 'gazeVector': 'world direction',
                'cameraHitPoint': 'finite camera-coordinate intersection',
                'cameraGazeDirection': 'camera direction',
                'rotation': 'camera-to-world quaternion xyzw; inverse for world-to-camera'},
            'limitations': ['Does not calibrate video/telemetry time alignment',
                            'Does not establish depth, visibility, anchor identity or recorded-camera equivalence'],
            'events': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(json.loads(Path(args.manifest).read_text()))
    save(Path(args.output), result)
    print(json.dumps({'events': len(result['events']), 'output': args.output}, ensure_ascii=False))


if __name__ == '__main__':
    main()
