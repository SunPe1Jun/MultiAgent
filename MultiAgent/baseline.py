"""Reuse original loaders, projection, model loader and evaluator, without editing them."""
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path

WORKSPACE_ROOT = Path(os.environ.get("SPJ_WORKSPACE_ROOT", str(Path(__file__).resolve().parents[2])))
MULTIAGENT_ROOT = Path(os.environ.get("SPJ_MULTIAGENT_ROOT", str(WORKSPACE_ROOT / "Multiagent")))
ROOT = Path(os.environ.get("SPJ_TRIMODAL_ROOT", str(WORKSPACE_ROOT / "Trimodal-Referring")))
V3DMD_ROOT = Path(os.environ.get("SPJ_V3DMD_ROOT", str(WORKSPACE_ROOT / "V3dMD")))
DATA_ROOT = Path(os.environ.get("SPJ_PREPARED_DATA_ROOT", str(ROOT / "data")))

LEGACY_ROOTS = {
    "/workspace/usr3/TriModal-Referring": ROOT,
    "/workspace/usr3/MultiAgent": MULTIAGENT_ROOT,
    "/workspace/usr3/V3dMD": WORKSPACE_ROOT / "V3dMD",
    "/workspace/usr3/Qwen3-VL-30B-A3B-Instruct": WORKSPACE_ROOT / "Qwen3-VL-30B-A3B-Instruct",
    "/workspace/usr3/Qwen3-VL-8B-Instruct": WORKSPACE_ROOT / "Qwen3-VL-8B-Instruct",
    "/workspace/usr3/InternVL3-38B-Instruct": WORKSPACE_ROOT / "InternVL3-38B-Instruct",
    "/workspace/usr3/grounding-dino-base": WORKSPACE_ROOT / "grounding-dino-base",
    "/workspace/usr3/sam-base": WORKSPACE_ROOT / "sam-base",
    "/workspace/usr3/ASR": WORKSPACE_ROOT / "ASR",
}

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
    data = DATA_ROOT
    anchor = data / f'{scene}_anchor_table.tsv'
    if not anchor.is_file():
        anchor = V3DMD_ROOT / 'anchor_tables' / f'{scene}_anchor_table.tsv'
    return {'input': data / f'{scene}_api_input.csv', 'anchor': anchor,
            'gt': data / GT_FILES.get(scene, f'{scene}.xlsx'),
            'baseline': data / f'{scene}_local_3d_outputs_qwen3vl30b_mention_first_v3'}


def local_path(value):
    """Relocate released paths into the current workspace while preserving relative paths."""
    text = str(value).replace('/ai/data/', '/workspace/usr3/')
    for legacy, current in sorted(LEGACY_ROOTS.items(), key=lambda item: len(item[0]), reverse=True):
        if text == legacy or text.startswith(legacy + '/'):
            suffix = text[len(legacy):].lstrip('/')
            return current / suffix
    return Path(text)
