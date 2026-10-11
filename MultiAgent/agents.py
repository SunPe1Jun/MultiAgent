"""Mention-first capabilities. No hand ray-hit coordinates are consumed here."""
from __future__ import annotations

import math
import re

from .context import Evidence
from .mention_mapping import PROMPT as REFERENTIAL_PROMPT_V4, resolve_mentions


def sample_key(ctx):
    return f'{ctx.scene_key}/{ctx.event_id}'


def object_type(name):
    return re.sub(r'(?:\d+|[A-Z])$', '', name).lower().replace('_', ' ')


def norm(text):
    return re.sub(r'[^a-z0-9]', '', str(text).lower())


def family(text):
    value = norm(text)
    aliases = {'workers': 'person', 'worker': 'person', 'people': 'person',
               'colleague': 'person', 'assistant': 'person', 'volunteer': 'person',
               'curtains': 'curtain', 'files': 'file', 'chairs': 'chair',
               'warehouses': 'warehouse', 'buildings': 'building', 'drones': 'drone',
               'desks': 'desk', 'trucks': 'truck', 'persons': 'person'}
    return aliases.get(value, value)


def evidence(ctx, mention, candidate, source, score=0., confidence=0., **kwargs):
    return Evidence(sample_id=ctx.sample_id, mention_id=mention['mention_id'],
                    candidate_id=candidate, source=source, support_score=score,
                    confidence=confidence, evidence_type=kwargs.pop('evidence_type', source), **kwargs)


# V4 keeps the original response contract but adds explicit component/parent
# fields required by the deterministic mapping layer.
REFERENTIAL_PROMPT = REFERENTIAL_PROMPT_V4


class ReferentialAgent:
    """Instruction-only Qwen parser, followed by a separate closed-set type matcher."""
    def __init__(self, backend=None, outputs=None):
        self.backend, self.outputs = backend, outputs or {}

    def run(self, ctx):
        retries = []
        prompt = REFERENTIAL_PROMPT + ctx.text
        for attempt in range(3 if self.backend else 1):
            try:
                if self.backend:
                    stage = 'referential' if attempt == 0 else f'referential_validation_retry_{attempt}'
                    parsed = self.backend.ask(sample_key(ctx), stage, prompt)
                else:
                    parsed = self.outputs.get(sample_key(ctx))
                result = self._resolve(ctx, parsed)
                if retries:
                    result['validation_retries'] = retries
                return result
            except ValueError as exc:
                if not self.backend or attempt == 2:
                    raise
                retries.append({'attempt': attempt + 1, 'error': str(exc)})
                prompt = (REFERENTIAL_PROMPT + ctx.text + '\nValidation feedback: ' + str(exc) +
                          '\nReturn the complete mentions object again. Every text value must be copied '
                          'literally from the instruction, preserving misspellings. A normalized category '
                          'may appear only in object_type or synonyms. Do not add objects or candidate IDs. '
                          'Check each text against the original instruction before responding.')

    def _resolve(self, ctx, parsed):
        return resolve_mentions(ctx, parsed)


class GazeAgent:
    """Deprecated compatibility shell; gaze is consumed inside Visual inputs."""
    def run(self, ctx, mention):
        return []


class VisualAgent:
    """Use actual per-mention RGB-model scores; never import monolithic selections."""
    source = 'visual'

    def __init__(self, outputs=None):
        self.outputs = outputs or {}

    def run(self, ctx, mention):
        payload = self.outputs.get(sample_key(ctx), {})
        rows = [r for r in payload.get('evidence', []) if r.get('mention_id') == mention['mention_id']]
        out = []
        for name in mention['candidate_ids']:
            matches = [r for r in rows if r.get('candidate_id') == name]
            row = max(matches, key=lambda r: float(r.get('support_score', 0)) * float(r.get('confidence', 0)), default={})
            score, confidence = float(row.get('support_score', 0)), float(row.get('confidence', 0))
            if not all(math.isfinite(x) and 0 <= x <= 1 for x in (score, confidence)):
                raise ValueError('Nonfinite/out-of-range visual evidence')
            out.append(evidence(ctx, mention, name, self.source, score, confidence,
                                evidence_type=f'{self.source}_visual_cue',
                                temporal_range=row.get('temporal_range'),
                                explanation=row.get('explanation', 'No candidate-specific visual evidence'),
                                metadata={'status': 'observed' if row else 'missing',
                                          'cue_type': row.get('cue_type'),
                                          'rightIndexFingerRayHitPoint_used': False}))
        return out


class HandAgent(VisualAgent):
    """Deprecated compatibility shell; Hand cues condition Visual and do not vote."""
    def run(self, ctx, mention):
        return []


class SpatialAgent:
    """Deprecated compatibility shell; spatial geometry is not an evidence stream."""
    def run(self, ctx, mention, mentions=()):
        return []
