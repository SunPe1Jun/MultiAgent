"""Reuse original loaders, projection, model loader and evaluator, without editing them."""
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path("/workspace/usr3/TriModal-Referring")

@lru_cache(None)
def module(name):
    paths = {
        'api': 'scripts/grounding/run_qwen3vl_plus_api_single_event_3d.py',
        'local': 'scripts/grounding/run_qwen3vl_local_keyframe_grounding.py',
        'single': 'scripts/grounding/run_qwen3vl_local_single_event_3d.py',
        'eval': 'scripts/eval/evaluate_local_3d_object_match.py',
    }
    spec = importlib.util.spec_from_file_location('multiagent_baseline_' + name, ROOT / paths[name])
    obj = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = obj
    spec.loader.exec_module(obj)
    return obj

GT_FILES = {'scene1': 'scene1_cleaned_v3.xlsx', 'scene2': 'scene2_cleaned_v2.xlsx'}

def scene_paths(scene):
    data = ROOT / 'data'
    return {'input': data / f'{scene}_api_input.csv', 'anchor': data / f'{scene}_anchor_table.tsv',
            'gt': data / GT_FILES.get(scene, f'{scene}.xlsx'),
            'baseline': data / f'{scene}_local_3d_outputs_qwen3vl30b_mention_first_v3'}


def local_path(value):
    """Relocate only the known historical workspace prefix."""
    return Path(str(value).replace('/ai/data/', '/workspace/usr3/'))
