from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

from .agents import VisualAgent, sample_key
from .coordinator import weighted_fusion


@dataclass
class ExperimentConfig:
    name: str
    enabled_agents: tuple[str, ...]
    weights: dict[str, float]


UNIFIED_V3 = ExperimentConfig(
    'V3_unified_mention_asr_gaze_hand', ('visual',), {'visual': 1.0})

# A single canonical experiment replaces the former modality-weight ablations.
# Gaze is an image attention view and Hand is a Visual conditioning input; they
# are therefore not independent evidence streams to weight in this coordinator.
EXPERIMENTS = {'V3': UNIFIED_V3}


def run_experiment(contexts, config, *, visual_predictions=None, qwen_mentions=None,
                   hand_visual_cues=None, monolithic_predictions=None, output_dir=None):
    agents = {'visual': VisualAgent(visual_predictions)}
    results = []
    for ctx in contexts:
        key = sample_key(ctx)
        parse = (qwen_mentions or {}).get(key)
        if not parse or parse.get('source') != 'instruction_only_qwen':
            raise ValueError(f'Missing instruction-only Referential output: {key}')
        ev = []
        for mention in parse['mentions']:
            ev.extend(agents['visual'].run(ctx, mention))
        pred = weighted_fusion(ctx, ev, config.weights, parse['mentions'])
        coordinator = (monolithic_predictions or {}).get(key)
        if coordinator and (coordinator.get('selected_object_names') or coordinator.get('temporal_focus')):
            names = [n for n in coordinator['selected_object_names']
                     if n in {c.candidate_id for c in ctx.candidates}]
            resolved = [ctx.candidate(n).world_row() for n in names]
            pred['selected_object_names'] = names
            pred['selected_object_name'] = names[0] if names else ''
            pred['resolved_object_rows'] = resolved
            pred['resolved_object_row'] = resolved[0] if resolved else None
            pred['global_coordinator'] = coordinator
        row = {'experiment': config.name, 'scene_key': ctx.scene_key, 'event_id': ctx.event_id,
               'sample_id': ctx.sample_id, 'row_index': ctx.row_index, 'scene_id': ctx.scene_id,
               'instruction': ctx.text, 'referential': parse, **pred,
               'gt_referents_mapped': ctx.gt_referents_mapped, 'provenance': ctx.provenance,
               'response_status': 'ok',
               'adjusted_response': {'selected_object_names': pred['selected_object_names'],
                                     'selected_object_name': pred['selected_object_name']}}
        results.append(row)
        if output_dir:
            out = Path(output_dir) / ctx.scene_key
            out.mkdir(parents=True, exist_ok=True)
            (out / f'row_{ctx.row_index}.json').write_text(json.dumps(row, ensure_ascii=False, indent=2))
    return results
