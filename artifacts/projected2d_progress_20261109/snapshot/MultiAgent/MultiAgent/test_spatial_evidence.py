import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .context import Candidate
from .spatial_evidence import (build_directional_panorama, direction_uv, pixel_rays,
                               spherical_foveation, spherical_heatmap)
from .visual_organization_audit import _prompt


class SpatialEvidenceTests(unittest.TestCase):
    def test_pixel_ray_uses_same_projection_as_camera_model(self):
        uv = np.array([[.21, .67], [.8, .3], [.5, .5]])
        rays = pixel_rays(uv, (800, 400), 80)
        tangent = np.tan(np.radians(40))
        recovered = np.stack((.5+rays[:, 0]/(2*rays[:, 2]*tangent*2),
                              .5-rays[:, 1]/(2*rays[:, 2]*tangent)), axis=-1)
        np.testing.assert_allclose(recovered, uv, atol=1e-7)
        rotation = Rotation.from_euler('xyz', [15, -30, 5], degrees=True)
        np.testing.assert_allclose(rotation.inv().apply(rotation.apply(rays)), rays, atol=1e-7)

    def test_spherical_kernel_wraps_and_scales_longitude_by_latitude(self):
        seam = spherical_heatmap([[0, 0, -1]], [1.], (64, 128), 8)
        np.testing.assert_allclose(seam[:, 0], seam[:, -1], atol=1e-6)
        pole = spherical_heatmap([[0, 1, 0]], [1.], (64, 128), 8)
        np.testing.assert_allclose(pole[0], np.repeat(pole[0, 0], 128), atol=1e-6)
        self.assertAlmostEqual(float(direction_uv([0, 0, 1])[0]), .5)

    def test_no_gaze_keeps_rgb_in_foveation(self):
        image = Image.new('RGB', (128, 64), (20, 80, 160))
        heat = np.zeros((64, 128), dtype=np.float32)
        rendered, stats = spherical_foveation(image, heat, heat > 0, {})
        np.testing.assert_array_equal(rendered, image)
        self.assertEqual(stats['status'], 'raw_rgb_fallback_no_gaze')

    def test_failed_panorama_keeps_all_source_frames_in_atlas(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = []
            for index, seconds in enumerate([-1., .1]):
                path = root / f'{index}.png'
                Image.new('RGB', (64, 64), (40+index, 100, 170)).save(path)
                frames.append({'view_id': f'V{index:03d}', 'frame_id': f'V{index:03d}',
                               'source_frame_id': f'frame_{index:06d}',
                               'time_seconds': seconds, 'path': str(path)})
            records = [{'telemetry_time_seconds': seconds, 'camera_position': [0, 0, 0],
                        'camera_rotation_xyzw': [0, 0, 0, 1], 'camera_fov_degrees': 90,
                        'source_index': index}
                       for index, seconds in enumerate([0., .2])]
            geometry = build_directional_panorama(frames, records, root / 'out',
                offset=0., config={}, shape=(64, 128), gaze_observations=[])
            self.assertEqual(geometry['status'], 'FALLBACK_SPATIAL_ATLAS')
            self.assertEqual(geometry['missing_pose_view_ids'], ['V000'])
            atlas = json.loads((root / 'out' / 'spatial_atlas_index.json').read_text())
            self.assertEqual([row['view_id'] for row in atlas['views']], ['V000', 'V001'])
            self.assertIsNone(atlas['views'][0]['camera_pose'])
            self.assertFalse(atlas['is_panorama'])

    def test_prompt_cannot_receive_audit_centers_or_numeric_gaze(self):
        ctx = SimpleNamespace(scene_key='scene2', text='Move that chair.', candidates=[
            Candidate('chair1', 'chair1', 12, 2, 17), Candidate('other1', 'other1', 3, 5, 7)])
        mention = {'mention_id': 'm0', 'text': 'that chair', 'candidate_ids': ['chair1'],
                   'span': [5, 15], 'kind': 'object'}
        prompt = _prompt(ctx, mention, 'A-H', [
            {'view_id': 'V001', 'expected_center': [.123456, .987654],
             'heatmap_sha256': 'fake_hash', 'source_frame_ids': ['frame_000001'],
             'time_range': [1., 1.]}], 'image')
        self.assertNotIn('expected_center', prompt)
        self.assertNotIn('.123456', prompt)
        self.assertNotIn('fake_hash', prompt)
        self.assertNotIn('other1', prompt)
        self.assertNotIn('x_world', prompt)
        self.assertIn('standalone grayscale heatmap', prompt)


if __name__ == '__main__':
    unittest.main()
