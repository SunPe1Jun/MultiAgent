#!/usr/bin/env python3
"""Measure the actual spatial intervention without reading referent GT."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from MultiAgent.heatmap_temporal_pilot import save, sha
from tools.run_gaze_rgb_routing import OUTPUT, check_lock


def audit(output):
    manifest = check_lock(output)
    events = []
    totals = Counter()
    for event in manifest['events']:
        counts = Counter()
        contrasts = []
        for frame in event['frames']:
            with np.load(output / frame['maps_path']) as archive:
                actual = archive['maps']
            for index, true in enumerate(frame['routing']['GAZE_RGB']['slots']):
                uniform = frame['routing']['UNIFORM_RGB']['slots'][index]
                mirrored = frame['routing']['MIRRORED_GAZE_RGB']['slots'][index]
                counts['all_slots'] += 1
                if true['fallback_reason'] is not None:
                    counts['fallback_slots'] += 1
                    continue
                counts['active_slots'] += 1
                counts['true_vs_uniform_rectangles_different'] += true['bbox_source_xyxy'] != uniform['bbox_source_xyxy']
                counts['true_vs_mirrored_rectangles_different'] += true['bbox_source_xyxy'] != mirrored['bbox_source_xyxy']
                mass = actual[index]
                fraction = lambda slot: float(mass[slot['bbox_source_xyxy'][1]:slot['bbox_source_xyxy'][3],
                                                   slot['bbox_source_xyxy'][0]:slot['bbox_source_xyxy'][2]].sum() / mass.sum())
                contrasts.append({'frame_id': frame['frame_id'], 'phase': index,
                    'true_mass_fraction': fraction(true), 'uniform_mass_fraction': fraction(uniform),
                    'mirrored_mass_fraction': fraction(mirrored)})
        totals.update(counts)
        events.append({'event': event['event'], 'counts': dict(counts), 'slots': contrasts})
    all_slots = [slot for event in events for slot in event['slots']]
    result = {'manifest_sha256': sha(output / 'manifest.json'), 'gt_read': False,
              'counts': dict(totals), 'events': events,
              'mean_actual_attention_fraction_in_active_windows': {
                  key: float(np.mean([slot[key] for slot in all_slots])) if all_slots else None
                  for key in ('true_mass_fraction', 'uniform_mass_fraction', 'mirrored_mass_fraction')},
              'interpretation': 'Checks that routing changes RGB regions and internal density coverage. No object identity, referent correctness, visibility or gaze accuracy claim.'}
    save(output / 'routing_intervention_audit.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'events'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    audit(parser.parse_args().output.resolve())
