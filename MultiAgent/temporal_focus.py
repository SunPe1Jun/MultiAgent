"""Local-first mention views with explicit, auditable full-event recovery."""
import re
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


VERSION = 'mention_focus_v2_labelled_frames'
MIN_SUPPORT = 0.35


def focus_spec(text, mention, window, frames, clock_rejected=False):
    start, end = mention['span']
    # Preserve character positions: repeated identical phrases remain distinct.
    bounds = [0] + [m.end() for m in re.finditer(r'[,;]|\bthen\b', text, re.I)] + [len(text)]
    left, right = next(((a, b) for a, b in zip(bounds, bounds[1:])
                        if a <= start < b), (0, len(text)))
    hint = window.get('asr_hint') if not clock_rejected else None
    lo, hi = (hint['video_range'] if hint else
              window.get('phase_hint', window['video_range']))
    source = 'asr_hint' if hint else 'soft_instruction_phase'
    primary = [f for f in frames if lo <= f['time'] <= hi]
    if not primary and frames:
        # No extracted frame in the proposed interval: do not pretend the
        # nearest observation is a measured speech-aligned frame.
        primary = [min(frames, key=lambda f: abs(f['time'] - (lo + hi) / 2))]
        source += '_nearest_available'
    if len(primary) > 5:
        primary = [primary[round(i * (len(primary)-1) / 4)] for i in range(5)]
    primary_ids = {f['frame_id'] for f in primary}
    outside = [f for f in frames if f['frame_id'] not in primary_ids]
    # At most one preceding and one following contextual observation.
    before = [f for f in outside if f['time'] < lo]
    after = [f for f in outside if f['time'] > hi]
    context = (before[-1:] + after[:1])
    return {'version': VERSION, 'mention_id': mention['mention_id'],
            'highlighted_instruction': text[:start] + '[CURRENT MENTION: ' + text[start:end] + ']' + text[end:],
            'action_context': text[left:right].strip(),
            'action_span': [left, right], 'primary_range': [lo, hi],
            'time_source': source, 'clock_rejected': clock_rejected,
            'primary_frame_ids': [f['frame_id'] for f in primary],
            'context_frame_ids': [f['frame_id'] for f in context]}


def labelled_paths(frames, source_paths, spec, stage, directory):
    """Image order changes per mention: render actual IDs on the images."""
    result = {}
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        fid = frame['frame_id']
        role = 'PRIMARY' if fid in spec['primary_frame_ids'] else 'CONTEXT'
        source = Path(source_paths[fid])
        digest = hashlib.sha256(source.read_bytes() + json.dumps(
            [VERSION, spec['mention_id'], fid, role, stage], sort_keys=True).encode()).hexdigest()[:16]
        target = directory / f'{digest}.jpg'
        if not target.exists():
            with Image.open(source) as opened:
                rgb = opened.convert('RGB')
            view = Image.new('RGB', (rgb.width, rgb.height + 28), 'white')
            view.paste(rgb, (0, 28))
            ImageDraw.Draw(view).text((8, 8),
                f'{fid} | t={frame["time"]:.3f}s | {role} | mention={spec["mention_id"]} | {stage}',
                fill='black')
            view.save(target)
        result[fid] = str(target)
    return result


def focused_visual(callback, backend, ctx, referential, paths, frames, plan,
                   cues, use_gaze, use_regions, mention_paths):
    rows, rejected, audits, warnings = [], [], {}, []
    paths_by_id = dict(zip([f['frame_id'] for f in frames], paths))
    for mention in referential['mentions']:
        mid = mention['mention_id']
        spec = focus_spec(ctx.text, mention, plan['mention_windows'][mid], frames,
                          plan.get('alignment', {}).get('audio_timing_rejected', False))
        audits[mid] = {**spec, 'passes': []}
        if not mention['candidate_ids']:
            audits[mid]['status'] = 'no_scene_candidates'
            continue
        chosen_ids = set(spec['primary_frame_ids'] + spec['context_frame_ids'])
        local = [f for f in frames if f['frame_id'] in chosen_ids]
        for stage in ('local', 'expanded'):
            supplied = local if stage == 'local' else frames
            supplied_ids = {f['frame_id'] for f in supplied}
            labelled = None
            if isinstance(getattr(backend, 'cache', None), Path):
                sources = {f['frame_id']: (mention_paths or {}).get(f['frame_id'], f['image_rgb'])
                           for f in supplied}
                labelled = labelled_paths(supplied, sources, spec, stage,
                                          backend.cache/'focus_images')
            scoped_cues = cues
            if isinstance(cues, dict):
                scoped_cues = {**cues, 'cues': [c for c in cues.get('cues', [])
                               if mid in c.get('mention_ids', []) and c.get('frame_id') in supplied_ids]}
            result = callback(backend, ctx, {**referential, 'mentions': [mention]},
                              [paths_by_id[f['frame_id']] for f in supplied], supplied, plan,
                              cues=scoped_cues, use_gaze=use_gaze, use_regions=use_regions,
                              isolate_mentions=True, mention_paths=labelled or mention_paths,
                              focus_context={**spec, 'stage': stage})
            audits[mid]['passes'].append({'stage': stage,
                'supplied_frame_ids': [f['frame_id'] for f in supplied],
                'accepted_rows': len(result['evidence']),
                'rejected_rows': len(result['rejected_evidence'])})
            rejected.extend(result['rejected_evidence'])
            warnings.extend(result.get('parse_warning') or [])
            supported = any(e['support_score'] * e['confidence'] >= MIN_SUPPORT
                            for e in result['evidence'])
            if stage == 'local' and not supported:
                audits[mid]['expansion_reason'] = 'no_valid_local_support_above_0.35'
                continue
            rows.extend(result['evidence'])
            audits[mid]['status'] = stage + ('_supported' if supported else '_uncertain')
            break
    return {'evidence': rows, 'rejected_evidence': rejected, 'frames': frames,
            'parse_warning': warnings or None, 'source': VERSION,
            'world_distance_gaze_used': False, 'hand_vote_used': False,
            'mention_isolation': True, 'temporal_focus': True, 'focus_audit': audits}
