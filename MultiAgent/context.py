from __future__ import annotations
import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
from .baseline import module, local_path, scene_paths

@dataclass
class Candidate:
    candidate_id: str
    object_name: str
    x_world: float
    y_world: float
    z_world: float
    # Aliases are loaded from the current scene's anchor table.  Keeping them
    # on the candidate prevents a global vocabulary rule from leaking between
    # scenes with different object schemas.
    aliases: tuple = ()

    def world_row(self):
        return {k: v for k, v in asdict(self).items()
                if k not in ('candidate_id', 'aliases')}

@dataclass
class Evidence:
    sample_id: str
    mention_id: str
    candidate_id: str
    source: str
    support_score: float
    confidence: float
    evidence_type: str
    explanation: str = ''
    temporal_range: list | None = None
    point_2d: list | None = None
    point_3d: list | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        import math
        for value in (self.support_score, self.confidence):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Evidence scores must be finite and in [0,1]')

    def to_dict(self): return asdict(self)

@dataclass
class AgentContext:
    event_id: str
    sample_id: str
    scene_id: str
    scene_key: str
    row_index: int
    instruction_text: str
    utterance_text: str
    video_path: str
    json_path: str
    event_json_path: str
    api_row: dict
    event: dict
    candidates: list[Candidate]
    timed_samples: list
    temporal_partitions: dict
    gt: dict
    provenance: dict

    @property
    def text(self): return self.instruction_text or self.utterance_text
    @property
    def peak_spatial(self): return self.event.get('peak_spatial', {})
    @property
    def gt_referents_mapped(self): return self.gt['mapped_referents']
    def candidate(self, name): return next((c for c in self.candidates if c.candidate_id == name), None)
    def to_dict(self): return asdict(self)


def vrtriref_sample_to_agent_context(sample):
    """Single adapter: existing CSV row plus its original scene/row address.

    GT is isolated from inference payloads. No target_description is used as language,
    because in other input profiles that field can contain annotation-derived targets.
    """
    import json
    row = dict(sample)
    scene = row.pop('_scene_key')
    index = int(row.pop('_row_index'))
    paths = scene_paths(scene)
    api, ev = module('api'), module('eval')
    anchors = api.load_scene_anchor_table(paths['anchor'])
    aliases, _ = ev.load_anchor_alias_map(paths['anchor'])
    gt = ev.load_gt_rows(paths['gt'], aliases)[index]
    event_path = local_path(row['event_json_path'])
    event = json.loads(event_path.read_text())
    raw_path = local_path(row['json_path'])
    raw = api.load_multimodal_samples(raw_path)
    timed = api.collect_timed_samples(raw)
    start, end = float(row['t_start']), float(row['t_end'])
    parts = {'before': [], 'during': [], 'after': []}
    for i, (t, _) in enumerate(timed):
        parts['before' if t < start else 'after' if t > end else 'during'].append(i)
    return AgentContext(
        str(row['event_id']), str(row['sample_id']), str(row['scene_id']), scene, index,
        str(row.get('instruction_text') or ''), str(row.get('utterance_text') or ''),
        str(local_path(row['video_path'])), str(raw_path), str(event_path), row, event,
        [Candidate(a['object_name'], a['object_name'], float(a['x_world']),
                   float(a['y_world']), float(a['z_world']),
                   tuple(a.get('aliases') or ())) for a in anchors],
        timed, parts, gt,
        {'input_csv': str(paths['input']), 'anchor_table': str(paths['anchor']), 'gt_file': str(paths['gt']),
         'split': 'existing evaluation pool; no official train/val/test split found',
         'time_alignment': 'metadata anchored: telemetry-to-video offset derived from first telemetry and audio metadata when available',
         'instruction_window': 'existing t_start/t_end event window; exact speech boundaries unavailable',
         'sequence_loader': 'baseline load_multimodal_samples + collect_timed_samples'})


def load_contexts(manifest):
    """Load only selected real events from their existing CSVs."""
    contexts, rows_by_scene = [], {}
    for item in manifest['samples']:
        scene = item['scene_key']
        if scene not in rows_by_scene:
            with scene_paths(scene)['input'].open(encoding='utf-8-sig') as f:
                rows_by_scene[scene] = list(csv.DictReader(f))
        row = dict(rows_by_scene[scene][item['row_index']])
        assert row['event_id'] == item['event_id'], 'Manifest event/row mismatch'
        row.update(_scene_key=scene, _row_index=item['row_index'])
        contexts.append(vrtriref_sample_to_agent_context(row))
    return contexts
