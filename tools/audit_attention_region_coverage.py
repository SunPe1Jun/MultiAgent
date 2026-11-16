#!/usr/bin/env python3
"""Equal representative-count comparison of cube windows and spherical regions.

This reuses actual gaze observations, no model/objects/GT. It measures angular
observation-duration coverage only. Ready component routes are used; the coarse
comparison is a script oracle, not a new model result.
"""
import json
import numpy as np
from MultiAgent.attention_regions import angular_coverage
from MultiAgent.perspective_attention import observation_for_selected_face, cube_owner
from MultiAgent.perspective_attention_study import SOURCE, OUTPUT as PREVIOUS, read
from MultiAgent.spherical_attention import spherical_grid, angles_to_directions
from MultiAgent.heatmap_temporal_pilot import save, sha
from tools.run_attention_region_study import OUTPUT


def audit(output=OUTPUT):
    m = read(output / 'manifest.json')
    sphere = {e['id']: e for e in read(SOURCE / 'manifest.json')['events']}
    previous = {e['id']: e for e in read(PREVIOUS / 'manifest.json')['events']}
    rows = []
    for event in m['events']:
        old = previous[event['id']]
        tiles = np.load(PREVIOUS / 'events' / event['event'] / 'perspective_density.npz')['density']
        mass = np.load(event['source_volume_path'])['duration_maps']
        yg, pg = spherical_grid(mass.shape[2], mass.shape[1])
        owner = cube_owner(angles_to_directions(yg, pg))
        for phase in (0, 2):
            obs = sphere[event['id']]['volume_stats']['bins'][phase]['observations']
            face_rows = []
            # Both methods rank by supported mass. A face must contain an actual
            # observation; Gaussian tails alone cannot manufacture a proposal.
            order = sorted(range(6), key=lambda j: (-float(mass[phase][owner == old['selected_faces'][j]].sum()), j))
            for j in order:
                representative = observation_for_selected_face(obs, event['canonical_pose'], old['selected_faces'][j], tiles[phase, j])
                if representative is not None: face_rows.append(representative)
            components = sorted([r for r in event['regions'] if r['phase'] == phase and r['status'] == 'RGB_EVIDENCE_READY'],
                                key=lambda r: (-r['mass_seconds'], r['component']))
            count = min(2, len(face_rows), len(components))
            for method, representatives in [('coarse_faces', face_rows[:count]),
                    ('continuous_regions', [r['representative_observation'] for r in components[:count]])]:
                for radius in (4, 8, 12):
                    rows.append({'event': event['event'], 'phase': phase, 'method': method, 'radius_degrees': radius,
                        'representative_count': count, 'source_indices': [r['source_index'] for r in representatives],
                        **angular_coverage(obs, event['canonical_pose'], representatives, radius)})
    summary = []
    for method in ['coarse_faces', 'continuous_regions']:
        for radius in (4, 8, 12):
            chosen = [r for r in rows if r['method'] == method and r['radius_degrees'] == radius]
            total = sum(r['total_seconds'] for r in chosen); covered = sum(r['covered_seconds'] for r in chosen)
            summary.append({'method': method, 'radius_degrees': radius, 'intervals': len(chosen),
                'representatives': sum(r['representative_count'] for r in chosen),
                'covered_seconds': covered, 'total_seconds': total, 'coverage_fraction': covered / total})
    result = {'manifest_sha256': sha(output / 'manifest.json'), 'code_sha256': sha(__file__),
        'comparison': 'Per interval equal count min(2, supported face count, ready component count). Both rank by mass; representative nearest local peak.',
        'scope': 'Offline angular geometry diagnostic; no model, no object/reference accuracy and no head-motion causal attribution.',
        'summary': summary, 'rows': rows}
    save(output / 'matched_angular_coverage.json', result)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': audit()
