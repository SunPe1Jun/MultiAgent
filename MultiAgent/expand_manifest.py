"""Build a larger fixed diagnostic manifest while preserving an existing pilot."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from .baseline import scene_paths

SCENES = ['scene1', 'scene2', 'scene3', 'scene4_room1', 'scene4_room2',
          'scene4_room3', 'scene4_room4', 'scene5']


def load_rows(scene):
    with scene_paths(scene)['input'].open(encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def proportional_quotas(scene_sizes, total):
    raw = {scene: scene_sizes[scene] * total / sum(scene_sizes.values()) for scene in SCENES}
    quotas = {scene: int(raw[scene]) for scene in SCENES}
    remainder = total - sum(quotas.values())
    order = sorted(SCENES, key=lambda scene: (raw[scene] - quotas[scene], scene), reverse=True)
    for scene in order[:remainder]:
        quotas[scene] += 1
    return quotas


def build_manifest(base_manifest, target_total, seed, base_path=None):
    base_samples = list(base_manifest.get('samples', []))
    if target_total < len(base_samples):
        raise ValueError(f'target_total={target_total} is smaller than base sample count={len(base_samples)}')
    rows = {scene: load_rows(scene) for scene in SCENES}
    quotas = proportional_quotas({scene: len(rows[scene]) for scene in SCENES}, target_total)
    rng = random.Random(seed)
    selected = {(item['scene_key'], int(item['row_index'])) for item in base_samples}
    by_scene = {scene: [item for item in base_samples if item['scene_key'] == scene]
                for scene in SCENES}
    for scene in SCENES:
        if len(by_scene[scene]) > quotas[scene]:
            raise ValueError(f'base samples exceed quota for {scene}: {len(by_scene[scene])}>{quotas[scene]}')
        available = [index for index in range(len(rows[scene]))
                     if (scene, index) not in selected]
        rng.shuffle(available)
        needed = quotas[scene] - len(by_scene[scene])
        for index in available[:needed]:
            row = rows[scene][index]
            item = {'scene_key': scene, 'row_index': index,
                    'event_id': row['event_id'],
                    'instruction': row.get('instruction_text', ''),
                    'selection_tags': ['random_stratified_supplement']}
            by_scene[scene].append(item)
            selected.add((scene, index))
    samples = []
    for scene in SCENES:
        samples.extend(sorted(by_scene[scene], key=lambda item: int(item['row_index'])))
    if len(samples) != target_total:
        raise AssertionError(f'expected {target_total} samples, got {len(samples)}')
    return {
        'seed': seed,
        'split': 'existing evaluation pool (no supplied train/val/test split)',
        'selection': 'pilot20 preserved; proportional scene quotas with seeded random supplements; no prediction-based filtering',
        'base_manifest': str(Path(base_path).resolve()) if base_path else None,
        'scene_quotas': quotas,
        'samples': samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--target-total', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261021)
    args = parser.parse_args()
    base = json.loads(Path(args.base_manifest).read_text())
    manifest = build_manifest(base, args.target_total, args.seed, args.base_manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'samples': len(manifest['samples']),
                      'scene_quotas': manifest['scene_quotas']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
