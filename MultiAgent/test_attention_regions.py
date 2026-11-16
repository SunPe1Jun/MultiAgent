import math
import unittest
import numpy as np
from .attention_regions import (spherical_components, mass_envelope, extract_regions,
    region_observations, representative_observation, angular_coverage)
from .spherical_attention import spherical_grid, angles_to_directions, solid_angle_weights


def gaussian(yaw, pitch=0., seconds=1.):
    yg, pg = spherical_grid(360, 180)
    ray = angles_to_directions(np.array(math.radians(yaw)), np.array(math.radians(pitch)))
    rays = angles_to_directions(yg, pg)
    angle = np.arccos(np.clip(rays @ ray, -1, 1))
    m = np.exp(-.5 * (angle / math.radians(2)) ** 2) * solid_angle_weights(360, 180)
    return m / m.sum() * seconds


class AttentionRegionTests(unittest.TestCase):
    def test_continuous_hotspot_crossing_cube_seam_is_one_component(self):
        maps = gaussian(45.)[None]
        labs, regions, audits = extract_regions(maps)
        self.assertEqual(len(regions), 1)
        self.assertGreaterEqual(audits[0]['retained_fraction'], .9)
        self.assertTrue(np.any(labs[0, :, :225] > 0))
        self.assertTrue(np.any(labs[0, :, 225:] > 0))

    def test_two_hotspots_inside_one_cube_face_are_kept(self):
        labs, regions, _ = extract_regions((gaussian(-20) + gaussian(20))[None])
        self.assertEqual(len(regions), 2)
        self.assertEqual(set(np.unique(labs)), {0, 1, 2})

    def test_longitude_wrap_and_polar_cell_topology(self):
        labs, regions, _ = extract_regions(gaussian(180)[None])
        self.assertEqual(len(regions), 1)
        self.assertGreater(int(labs[0, :, 0].max()), 0)
        self.assertEqual(int(labs[0, :, 0].max()), int(labs[0, :, -1].max()))
        mask = np.zeros((10, 20), bool)
        mask[0, 2] = mask[0, 13] = True
        labels, n = spherical_components(mask)
        self.assertEqual(n, 1)
        self.assertEqual(labels[0, 2], labels[0, 13])

    def test_density_not_cell_mass_determines_envelope_and_ties_survive(self):
        area = solid_angle_weights(12, 6)
        mass = area * 3.
        mask, audit = mass_envelope(mass, .9)
        self.assertTrue(mask.all())
        self.assertAlmostEqual(audit['retained_fraction'], 1.)
        self.assertAlmostEqual(audit['threshold_seconds_per_sr'], 3.)

    def test_supported_mass_and_residual_conserve_input_no_topk(self):
        maps = np.stack([gaussian(-70) + gaussian(0) + gaussian(70), gaussian(10)])
        before = maps.copy()
        labs, regions, audits = extract_regions(maps)
        self.assertEqual(len([r for r in regions if r['phase'] == 0]), 3)
        for k, audit in enumerate(audits):
            self.assertAlmostEqual(audit['retained_seconds'] + audit['residual_seconds'], float(maps[k].sum()))
            self.assertAlmostEqual(sum(r['mass_seconds'] for r in regions if r['phase'] == k), audit['retained_seconds'])
        np.testing.assert_array_equal(maps, before)

    def test_empty_and_invalid_inputs(self):
        labs, regions, audits = extract_regions(np.zeros((3, 10, 20)))
        self.assertFalse(labs.any()); self.assertEqual(regions, [])
        self.assertIsNone(audits[0]['retained_fraction'])
        for mass in (np.full((10, 20), -1.), np.full((10, 20), np.nan)):
            with self.assertRaises(ValueError): mass_envelope(mass)
        for fraction in (0., 1.1, float('nan')):
            with self.assertRaises(ValueError): mass_envelope(np.zeros((10, 20)), fraction)

    def test_observation_assignment_never_substitutes_stronger_other_region(self):
        pose = {'rotation_xyzw': [0, 0, 0, 1]}
        labs, regions, _ = extract_regions((gaussian(-20) + gaussian(20, seconds=.3))[None])
        observations = [{'world_ray': [math.sin(math.radians(y)), 0., math.cos(math.radians(y))],
                         'duration_seconds': d, 'source_index': i} for i, (y, d) in enumerate([(-20, 10.), (20, .1)])]
        region = next(r for r in regions if r['peak_direction'][0] > 0)
        subset = region_observations(observations, pose, labs[0], region['component'])
        self.assertEqual([o['source_index'] for o in subset], [1])
        self.assertEqual(representative_observation(subset, pose, region)['source_index'], 1)
        self.assertIsNone(representative_observation([], pose, region))
        covered = angular_coverage(observations, pose, subset, 8.)
        self.assertAlmostEqual(covered['covered_seconds'], .1)




class NativeRegionGeometryTests(unittest.TestCase):
    def test_native_pixel_lookup_matches_known_camera_rotation(self):
        from tools.run_attention_region_study import native_angular_lookup
        from scipy.spatial.transform import Rotation
        reference = {'rotation_xyzw': [0., 0., 0., 1.]}
        camera = {'rotation_xyzw': Rotation.from_euler('y', 35, degrees=True).as_quat().tolist(),
                  'fov_vertical_degrees': 90.}
        rays = native_angular_lookup(camera, reference, (101, 101))
        expected = Rotation.from_euler('y', 35, degrees=True).apply([0, 0, 1])
        np.testing.assert_allclose(rays[50, 50], expected, atol=1e-12)
        self.assertLess(float(rays[50, 0] @ expected), 1.)
        np.testing.assert_allclose(np.linalg.norm(rays, axis=-1), 1., atol=1e-12)

    def test_crop_keeps_component_extent_and_has_no_phantom_empty_crop(self):
        from tools.run_attention_region_study import component_crop
        mask = np.zeros((100, 120), bool); mask[2:40, 3:80] = True
        box = component_crop(mask, [4, 5], (120, 100))
        self.assertLessEqual(box[0], 3); self.assertLessEqual(box[1], 2)
        self.assertGreaterEqual(box[2], 80); self.assertGreaterEqual(box[3], 40)
        self.assertGreaterEqual(box[0], 0); self.assertGreaterEqual(box[1], 0)
        self.assertLessEqual(box[2], 120); self.assertLessEqual(box[3], 100)
        self.assertIsNone(component_crop(np.zeros_like(mask), [4, 5], (120, 100)))


if __name__ == '__main__': unittest.main()
