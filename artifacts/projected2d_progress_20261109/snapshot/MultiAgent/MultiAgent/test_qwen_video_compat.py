import types
import unittest
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from .qwen_video_compat import ensure_timestamp_video_rope


class VideoRopeCompatibilityTests(unittest.TestCase):
    def fixture(self):
        inner=types.SimpleNamespace(config=types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=2)))
        inner.get_vision_position_ids=types.MethodType(Qwen3VLModel.get_vision_position_ids,inner)
        inner.get_rope_index=types.MethodType(Qwen3VLModel.get_rope_index,inner)
        model=types.SimpleNamespace(model=inner)
        ids=torch.tensor([[10,20,20,20,20,11,20,20,20,20,12]])
        mm=torch.tensor([[0,2,2,2,2,0,2,2,2,2,0]])
        grid=torch.tensor([[2,4,4]])
        return model,ids,mm,grid

    def test_two_timestamp_groups_use_one_grid_each_without_changing_vision_grid(self):
        model,ids,mm,grid=self.fixture();before=grid.clone();original=model.model.get_rope_index
        expanded=torch.tensor([[1,4,4],[1,4,4]])
        expected=original(ids,mm,video_grid_thw=expanded)
        audit=ensure_timestamp_video_rope(model)
        positions,deltas=model.model.get_rope_index(ids,mm,video_grid_thw=grid)
        self.assertEqual(tuple(positions.shape),(3,1,11));torch.testing.assert_close(positions,expected[0]);torch.testing.assert_close(deltas,expected[1])
        torch.testing.assert_close(grid,before);self.assertTrue(audit['vision_grid_and_pixels_unchanged'])

    def test_install_is_idempotent(self):
        model,ids,mm,grid=self.fixture();a=ensure_timestamp_video_rope(model);method=model.model.get_rope_index
        self.assertEqual(a,ensure_timestamp_video_rope(model));self.assertIs(method,model.model.get_rope_index)

    def test_image_only_grid_is_not_split(self):
        model,ids,mm,grid=self.fixture();ids=ids[:,:5];mm=torch.tensor([[0,1,1,1,1]])
        original=model.model.get_rope_index;expected=original(ids,mm,image_grid_thw=torch.tensor([[1,4,4]]))
        ensure_timestamp_video_rope(model);actual=model.model.get_rope_index(ids,mm,image_grid_thw=torch.tensor([[1,4,4]]))
        torch.testing.assert_close(actual[0],expected[0]);torch.testing.assert_close(actual[1],expected[1])


if __name__=='__main__':unittest.main()
