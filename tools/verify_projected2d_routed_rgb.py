#!/usr/bin/env python3
"""Independently reconstruct the frozen RGB panels from saved source and boxes."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from MultiAgent.gaze_rgb_routing import render_rgb_evidence
from MultiAgent.projected2d_attention_study import save_json, sha_file
from MultiAgent.projected2d_routed_rgb import GROUPS


def main(output):
    manifest = json.loads((output / 'input_manifest.json').read_text())
    counts, failures = Counter(), []
    for event in manifest['events']:
        for panel in event['panels']:
            source = Path(panel['source_path'])
            if sha_file(source) != panel['source_sha256']:
                failures.append({'event': event['id'], 'panel': panel['frame_id'], 'reason': 'source_hash'})
                continue
            with Image.open(source) as opened:
                rgb = opened.convert('RGB')
                rgb.thumbnail((512, 512), Image.Resampling.LANCZOS)
            hashes = []
            for group in GROUPS:
                target = Path(panel['inputs'][group]['path'])
                route = {'image_size': list(rgb.size),
                         'slots': [{'bbox_source_xyxy': box} for box in panel['boxes'][group]]}
                expected = np.asarray(render_rgb_evidence(rgb, route))
                with Image.open(target) as img:
                    actual = np.asarray(img.convert('RGB'))
                matches = actual.shape == expected.shape and np.array_equal(actual, expected)
                correct_sha = sha_file(target) == panel['inputs'][group]['sha256']
                counts['images'] += 1
                counts['pixels_match'] += int(matches)
                counts['hashes_match'] += int(correct_sha)
                if not matches or not correct_sha:
                    failures.append({'event': event['id'], 'panel': panel['frame_id'], 'group': group,
                                     'reason': 'image_or_hash_mismatch'})
                hashes.append(panel['inputs'][group]['sha256'])
            if not panel['clock_supported']:
                counts['unsupported_panels'] += 1
                same = len(set(hashes)) == 1
                counts['unsupported_identical'] += int(same)
                if not same:
                    failures.append({'event': event['id'], 'panel': panel['frame_id'], 'reason': 'fallback_differs'})
    result = {'counts': dict(counts), 'failures': failures,
              'all_verified': not failures, 'heatmap_model_input': False}
    save_json(output / 'input_reconstruction_verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
