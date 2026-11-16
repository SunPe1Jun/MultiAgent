"""Spatial attention proposals from a complete angular density, before display cuts.

These are connected attention envelopes, not objects, fixations or referents.
Per-cell input is supported seconds. Only the proposal envelope is reduced;
the original volume and the explicit residual remain available to reviewers.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.ndimage import label
from scipy.spatial.transform import Rotation
from .spherical_attention import spherical_grid, angles_to_directions, solid_angle_weights
from .perspective_attention import sphere_indices

VERSION = 'spherical_mass_envelope_regions_v1'


def spherical_components(mask):
    """Eight-connected grid components with periodic longitude and joined poles.

    Polar row cells meet at the pole on the spherical cell tessellation. There
    are no cube-face cuts in this topology. Labels are deterministic, positive;
    zero is background. This measures connected support, not distinct objects.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or min(mask.shape) < 2:
        raise ValueError('Expected a spherical H by W mask')
    labs, count = label(mask, np.ones((3, 3), dtype=np.uint8))
    parent = list(range(count + 1))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        if a and b:
            a, b = find(int(a)), find(int(b))
            parent[max(a, b)] = min(a, b)

    for y in range(mask.shape[0]):
        for other in range(max(0, y - 1), min(mask.shape[0], y + 2)):
            union(labs[y, 0], labs[other, -1])
    for row in (labs[0], labs[-1]):
        occupied = np.unique(row[row > 0])
        if len(occupied):
            for other in occupied[1:]:
                union(occupied[0], other)
    roots = {i: find(i) for i in range(1, count + 1)}
    contiguous = {v: i + 1 for i, v in enumerate(sorted(set(roots.values())))}
    lookup = np.array([0] + [contiguous[roots[i]] for i in range(1, count + 1)], dtype=np.int32)
    return lookup[labs], len(contiguous)


def mass_envelope(mass, retained_fraction=.9):
    """Highest-density cells enclosing at least the requested supported seconds.

    Equal-density boundary ties are all retained. The threshold is on seconds
    per steradian, not equirectangular cell mass, nor nonlinear display RGB.
    """
    mass = np.asarray(mass, dtype=np.float64)
    if mass.ndim != 2 or min(mass.shape) < 2 or not np.all(np.isfinite(mass)) or np.any(mass < 0):
        raise ValueError('Expected finite nonnegative spherical cell mass')
    if not math.isfinite(retained_fraction) or not 0 < retained_fraction <= 1:
        raise ValueError('retained_fraction must be in (0, 1]')
    h, w = mass.shape
    density = mass / solid_angle_weights(w, h)
    total = float(mass.sum())
    if total == 0:
        return np.zeros_like(mass, dtype=bool), {'threshold_seconds_per_sr': None,
            'total_seconds': 0., 'retained_seconds': 0., 'retained_fraction': None, 'residual_seconds': 0.}
    order = np.argsort(-density.ravel(), kind='stable')
    cumulative = np.cumsum(mass.ravel()[order])
    position = min(int(np.searchsorted(cumulative, total * retained_fraction)), len(order) - 1)
    threshold = float(density.ravel()[order[position]])
    mask = (density >= threshold) & (mass > 0)
    kept = float(mass[mask].sum())
    return mask, {'threshold_seconds_per_sr': threshold, 'total_seconds': total,
        'retained_seconds': kept, 'retained_fraction': kept / total,
        'residual_seconds': max(0., total - kept)}


def extract_regions(maps, retained_fraction=.9):
    """Keep every connected envelope component in each time bin, without top-k.

    Sorting by mass is for deterministic auditing. Model IDs and screen order
    must be independently permuted; no scores need enter model text.
    """
    maps = np.asarray(maps, dtype=np.float64)
    if maps.ndim != 3:
        raise ValueError('Expected time by H by W cell mass')
    _, h, w = maps.shape
    yg, pg = spherical_grid(w, h)
    rays = angles_to_directions(yg, pg)
    density = maps / solid_angle_weights(w, h)[None]
    labels = np.zeros(maps.shape, dtype=np.int32)
    regions, audits = [], []
    for phase, mass in enumerate(maps):
        mask, audit = mass_envelope(mass, retained_fraction)
        component, n = spherical_components(mask)
        ranked = sorted(range(1, n + 1), key=lambda i: (-float(mass[component == i].sum()), i))
        for local_id, component_id in enumerate(ranked, start=1):
            selection = component == component_id
            values = np.where(selection, density[phase], -1.)
            y, x = np.unravel_index(np.argmax(values), values.shape)
            seconds = float(mass[selection].sum())
            centroid = (rays[selection] * mass[selection, None]).sum(axis=0)
            norm = np.linalg.norm(centroid)
            centroid = centroid / norm if norm > 1e-12 else rays[y, x]
            regions.append({'phase': phase, 'component': local_id,
                'mass_seconds': seconds, 'fraction_of_phase': seconds / audit['total_seconds'],
                'cell_count': int(selection.sum()), 'peak_yx': [int(y), int(x)],
                'peak_direction': rays[y, x].tolist(), 'centroid_direction': centroid.tolist(),
                'peak_density_seconds_per_sr': float(density[phase, y, x]),
                'solid_angle_sr': float(solid_angle_weights(w, h)[selection].sum())})
            labels[phase, selection] = local_id
        audits.append({**audit, 'phase': phase, 'components': n})
    return labels, regions, audits


def observation_cells(observations, canonical_pose, shape):
    if not observations:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    rays = Rotation.from_quat(canonical_pose['rotation_xyzw']).inv().apply(
        np.asarray([o['world_ray'] for o in observations], dtype=float))
    return sphere_indices(rays, shape[1], shape[0])


def region_observations(observations, canonical_pose, label_map, component):
    x, y = observation_cells(observations, canonical_pose, label_map.shape)
    return [o for o, xx, yy in zip(observations, x, y) if label_map[yy, xx] == component]


def representative_observation(observations, canonical_pose, region):
    """Only observations already assigned to this region are eligible."""
    if not observations:
        return None
    reference = Rotation.from_quat(canonical_pose['rotation_xyzw'])
    ray = np.asarray(region['peak_direction'])
    return max(observations, key=lambda o: (float(reference.inv().apply(o['world_ray']) @ ray),
                                            float(o['duration_seconds']), -int(o['source_index'])))


def angular_coverage(observations, canonical_pose, representatives, radius_degrees):
    """Duration near routed representatives; no claim about object coverage."""
    total = sum(float(o['duration_seconds']) for o in observations)
    if not representatives or total == 0:
        return {'covered_seconds': 0., 'total_seconds': total, 'fraction': 0. if total else None}
    rays = np.asarray([o['world_ray'] for o in observations], dtype=float)
    selected = np.asarray([o['world_ray'] for o in representatives], dtype=float)
    # World rotations cancel in these angular distances. Input rays must be unit.
    cosine = (rays @ selected.T).max(axis=1)
    covered = sum(float(o['duration_seconds']) for o, v in zip(observations, cosine)
                  if v >= math.cos(math.radians(radius_degrees)) - 1e-12)
    return {'covered_seconds': covered, 'total_seconds': total, 'fraction': covered / total}
