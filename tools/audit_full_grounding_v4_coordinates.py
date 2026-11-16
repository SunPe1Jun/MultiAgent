#!/usr/bin/env python3
"""Verify the frozen V4 composite-to-RGB coordinate contract without GT."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / '.local-sync/full_interaction_referring_20261108/grounding_v4'


def read(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit(run: Path):
    manifest = read(run / 'manifest.json')
    protocol = read(run / 'protocol_lock.json')
    runtime = read(run / 'runtime_lock.json')
    frozen = read(run / 'predictions_frozen.json')
    parent_manifest = ROOT / '.local-sync/full_interaction_referring_20261106/grounding_v3/manifest.json'

    assert sha256(run / 'manifest.json') == protocol['manifest_sha256']
    assert sha256(parent_manifest) == manifest['parent_manifest_sha256']
    assert frozen['before_gt_evaluation'] is True

    runtime_paths = {
        'MultiAgent/full_frame_grounding.py': run / 'source_validated_runtime/MultiAgent/full_frame_grounding.py',
        'MultiAgent/full_frame_grounding_v4.py': run / 'source_validated_runtime/MultiAgent/full_frame_grounding_v4.py',
        'MultiAgent/test_full_frame_grounding_v4.py': run / 'source_validated_runtime/MultiAgent/test_full_frame_grounding_v4.py',
        'tools/run_full_frame_grounding_v4.py': run / 'source_validated_runtime/tools/run_full_frame_grounding_v4.py',
        'docs/FULL_GROUNDING_V4_PILOT_20261108_zh.md': run / 'source_validated_runtime/docs/FULL_GROUNDING_V4_PILOT_20261108_zh.md',
    }
    runtime_source_matches = {
        name: sha256(path) == runtime['runtime_source_sha256'][name]
        for name, path in runtime_paths.items()
    }
    assert all(runtime_source_matches.values())

    response_hash_matches = 0
    vision_verified = 0
    processor_sizes = set()
    for request in manifest['requests']:
        path = run / 'responses' / f"{request['id']}.json"
        response = read(path)
        assert sha256(path) == frozen['response_sha256'][request['id']]
        response_hash_matches += 1
        processor = response.get('processor_audit', {})
        trace = processor.get('visual_tensor_trace', {})
        assert response.get('vision_verified') is True
        assert trace.get('input_consumption_verified') is True
        assert trace.get('vision_forward_calls')
        assert all(call['input_matches_processor_after_dtype_cast']
                   and call['grid_matches_processor']
                   for call in trace['vision_forward_calls'])
        assert processor.get('processor_sampling_disabled') is True
        assert processor.get('processor_resampling_performed') is False
        vision_verified += 1
        processor_sizes.add(tuple(processor['processor_effective_frame_size_px']))

    source_frames = composite_images = composite_hash_matches = 0
    composite_size_matches = top_left_rgb_matches = 0
    grounded_raw_points = raw_points_in_rgb = 0
    accepted_rows = validator_mapping_matches = 0
    decoder_rows = decoder_mapping_matches = 0
    seen_images = set()

    for event in manifest['events']:
        for frame in event['frames']:
            source_frames += 1
            assert frame['source_size'] == [672, 672]
            source_path = Path(frame['source_path'])
            assert sha256(source_path) == frame['source_sha256']
            with Image.open(source_path) as source_image:
                source_rgb = source_image.convert('RGB')

            for metadata in frame['images'].values():
                image_path = (run / metadata['path']).resolve()
                if image_path in seen_images:
                    continue
                seen_images.add(image_path)
                assert sha256(image_path) == metadata['sha256']
                composite_hash_matches += 1
                with Image.open(image_path) as image:
                    composite = image.convert('RGB')
                    assert composite.size == tuple(metadata['size']) == (1344, 1400)
                    composite_size_matches += 1
                    assert composite.crop((0, 28, 672, 700)).tobytes() == source_rgb.tobytes()
                    top_left_rgb_matches += 1

    for path in sorted((run / 'responses').glob('*.json')):
        response = read(path)
        raw_by_id = {row.get('mention_id'): row for row in response.get('parsed', {}).get('mentions', [])
                     if isinstance(row, dict)}
        for raw in raw_by_id.values():
            if raw.get('status') != 'grounded':
                continue
            grounded_raw_points += 1
            x, y = raw['x'], raw['y']
            assert isinstance(x, (int, float)) and isinstance(y, (int, float))
            in_rgb = 0 <= x <= 500 and 20 <= y <= 500
            raw_points_in_rgb += in_rgb

        for row in response.get('grounding_validation', {}).get('grounded_referents', []):
            raw = raw_by_id[row['mention_id']]
            expected_x = raw['x'] / 1000 * 1344 / 672
            expected_y = (raw['y'] / 1000 * 1400 - 28) / 672
            accepted_rows += 1
            validator_mapping_matches += (
                abs(row['x'] - expected_x) < 1e-12
                and abs(row['y'] - expected_y) < 1e-12
            )

        for decision in response.get('decoder', {}).get('decisions', []):
            raw = raw_by_id[decision['visual_choice']['mention_id']]
            expected_x = raw['x'] / 1000 * 1344 / 672
            expected_y = (raw['y'] / 1000 * 1400 - 28) / 672
            visual = decision['visual_choice']
            decoder_rows += 1
            decoder_mapping_matches += (
                abs(visual['x'] - expected_x) < 1e-12
                and abs(visual['y'] - expected_y) < 1e-12
            )

    synthetic_points = []
    for x, y in ((0, 20), (250, 260), (500, 500)):
        mapped = [x / 1000 * 1344 / 672, (y / 1000 * 1400 - 28) / 672]
        expected = [(0, 0), (.5, .5), (1, 1)][len(synthetic_points)]
        assert all(abs(actual - target) < 1e-12 for actual, target in zip(mapped, expected))
        synthetic_points.append({'composite_0_to_1000': [x, y], 'rgb_local_0_to_1': mapped})

    assert response_hash_matches == len(manifest['requests']) == 32
    assert vision_verified == 32
    assert source_frames == 96
    assert composite_hash_matches == composite_size_matches == top_left_rgb_matches == 384
    assert grounded_raw_points == raw_points_in_rgb == 76
    assert accepted_rows == validator_mapping_matches == 70
    assert decoder_rows == decoder_mapping_matches == 70
    assert processor_sizes == {(1408, 1344)}

    return {
        'version': 'full_grounding_v4_coordinate_contract_audit_v1',
        'gt_read': False,
        'run_manifest_sha256': sha256(run / 'manifest.json'),
        'parent_manifest_sha256': sha256(parent_manifest),
        'manifest_protocol_lock_match': True,
        'runtime_source_hash_matches': runtime_source_matches,
        'frozen_responses': len(manifest['requests']),
        'response_hash_matches': response_hash_matches,
        'vision_input_consumption_verified': vision_verified,
        'source_frame_count': source_frames,
        'source_frame_hash_matches': source_frames,
        'composite_input_files': len(seen_images),
        'composite_hash_matches': composite_hash_matches,
        'composite_size_matches': composite_size_matches,
        'top_left_rgb_exact_pixel_matches': top_left_rgb_matches,
        'composite_size_px': [1344, 1400],
        'rgb_panel_rect_px_xyxy': [0, 28, 672, 700],
        'source_rgb_size_px': [672, 672],
        'coordinate_mapping': {
            'x_rgb': '(x_composite / 1000 * 1344) / 672',
            'y_rgb': '(y_composite / 1000 * 1400 - 28) / 672',
            'valid_composite_bounds': {'x': [0, 500], 'y': [20, 500]},
        },
        'synthetic_mapping_checks': synthetic_points,
        'grounded_raw_points': grounded_raw_points,
        'raw_points_inside_rgb_panel': raw_points_in_rgb,
        'accepted_rows': accepted_rows,
        'accepted_rows_matching_independent_mapping': validator_mapping_matches,
        'decoder_visual_rows': decoder_rows,
        'decoder_rows_matching_independent_mapping': decoder_mapping_matches,
        'processor': {
            'effective_frame_size_px_counts': {str(size): 32 for size in sorted(processor_sizes)},
            'sampling_disabled': True,
            'resampling_reported': False,
            'whole_video_vision_input_consumption_verified': True,
        },
        'limit': 'This verifies coordinate geometry and provenance, not point-on-object accuracy, anchor visibility, or occlusion.',
        'next_step': 'Freeze a genuinely unseen interaction cohort before any new model inference; do not tune this decoder on the same eight events.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.run.resolve())
    output = args.output or args.run / 'coordinate_audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
