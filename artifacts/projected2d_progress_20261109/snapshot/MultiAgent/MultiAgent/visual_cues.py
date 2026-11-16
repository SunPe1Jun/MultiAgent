"""Instruction-conditioned RGB/hand evidence, with actual frames and frame times."""
import json
from pathlib import Path

from .agents import sample_key
from .asr import OfflineASR, align_mentions, clock_alignment
from .baseline import module
from .visualize import _frame


def _unique_records(records, limit=8):
    out = []
    seen = set()
    for row in sorted(records, key=lambda x: (x['video_time'], x.get('reason', ''))):
        key = round(float(row['video_time']), 3)
        if key in seen:
            continue
        seen.add(key); out.append(row)
    if len(out) <= limit:
        return out
    # Keep one midpoint for every mention first, then distribute context and
    # add mention boundaries. This prevents a long instruction from consuming
    # all eight image slots before the model sees the scene context.
    mention = [x for x in out if x.get('mention_id')]
    context = [x for x in out if not x.get('mention_id')]
    mids = [x for x in mention if x.get('reason') == 'mention_mid']
    chosen = mids[:limit]
    remaining = [x for x in context if x not in chosen]
    slots = min(max(0, limit-len(chosen)), len(remaining))
    if slots:
        for idx in range(slots):
            chosen.append(remaining[round(idx*(len(remaining)-1)/max(1, slots-1))])
    for row in mention:
        if len(chosen) >= limit: break
        if row not in chosen: chosen.append(row)
    return sorted(chosen, key=lambda x: x['video_time'])


def frame_plan(ctx, referential, asr=None, asr_cache=None, use_gaze_preprocess=False):
    start, end = float(ctx.api_row['t_start']), float(ctx.api_row['t_end'])
    clock = clock_alignment(ctx)
    # Event bounds are telemetry-relative; convert them to screen-recording time.
    telemetry_to_video = -clock.get('audio_to_telemetry', 0.0) + clock.get('audio_to_video', 0.0)
    event_video_start = start + telemetry_to_video
    event_video_end = end + telemetry_to_video
    alignment = align_mentions(ctx, referential, asr, asr_cache) if asr else {
        'status': 'disabled', 'clock': clock, 'mentions': {}
    }
    records = []
    mention_windows = {}
    for mention_id, match in alignment.get('mentions', {}).items():
        mention_windows[mention_id] = match
        clip = match.get('video_clip') if match.get('matched') else None
        if not clip or match.get('confidence', 0.0) < 0.45:
            continue
        v0, v1 = clip
        t0 = match.get('telemetry_clip', [None, None])[0]
        t1 = match.get('telemetry_clip', [None, None])[1]
        records.extend([
            {'video_time': v0, 'telemetry_time': t0, 'mention_id': mention_id, 'reason': 'mention_start'},
            {'video_time': (v0 + v1) / 2, 'telemetry_time': (t0 + t1) / 2, 'mention_id': mention_id, 'reason': 'mention_mid'},
            {'video_time': v1, 'telemetry_time': t1, 'mention_id': mention_id, 'reason': 'mention_end'},
        ])
    # Keep event context even when ASR misses a word or audio is unavailable.
    for i in range(4):
        v = event_video_start + (event_video_end-event_video_start)*(i+.5)/4
        records.append({'video_time': v, 'telemetry_time': start + (end-start)*(i+.5)/4,
                        'reason': 'event_context'})
    visual_records = _unique_records(records)
    tracked = [t for t,s in ctx.timed_samples if start <= t <= end and
               (s.get('handData',{}).get('isLeftHandTracked') or s.get('handData',{}).get('isRightHandTracked'))]
    hand_records = []
    for t in ([tracked[int((len(tracked)-1)*q)] for q in [0,.25,.5,.75,1]] if tracked else []):
        hand_records.append({'video_time': t + telemetry_to_video, 'telemetry_time': t, 'reason': 'hand_tracked'})
    hand_records = _unique_records(hand_records + visual_records[::2])
    # Each mention records its own gaze proposals. No hand-ray hits are read.
    return {'visual_times': [x['video_time'] for x in visual_records],
            'hand_times': [x['video_time'] for x in hand_records],
            'visual_records': visual_records, 'hand_records': hand_records,
            'mention_windows': mention_windows, 'asr': alignment,
            'use_gaze_preprocess': bool(use_gaze_preprocess),
            'alignment': {**clock, 'telemetry_to_video': telemetry_to_video,
                          'description': 'audio mention clips plus event context; low-confidence matches fall back to context'}}


def _gaze_overlay(image, sample, pose, api):
    """Render gaze as a soft 2-D attention prior; never use it as an object label."""
    point = api.sample_gaze_point(sample)
    if point is None:
        return image, None
    uv = api.project_world_to_image(pose, {'x_world': point[0], 'y_world': point[1], 'z_world': point[2]})
    u, v, ok = uv
    if not ok or u is None or not 0 <= u <= 1 or not 0 <= v <= 1:
        return image, None
    from PIL import Image, ImageDraw
    rgba = image.convert('RGBA')
    glow = Image.new('RGBA', rgba.size, (0, 0, 0, 0)); draw = ImageDraw.Draw(glow)
    cx, cy = int(u * image.width), int(v * image.height)
    for radius, alpha in [(110, 18), (75, 30), (42, 50), (18, 85)]:
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=(255, 40, 30, alpha))
    return Image.alpha_composite(rgba, glow).convert('RGB'), [round(float(u), 4), round(float(v), 4)]


def prepare_frames(ctx, records, directory, use_gaze=False, segmenter=None, labels=()):
    paths, metadata = [], []
    api = module('api')
    duration = module('local').probe_video_duration(Path(ctx.video_path), None)
    for i, row in enumerate(records):
        video_t = float(row['video_time'])
        if duration is not None:
            video_t = min(max(0., video_t), max(0., duration-.1))
        path = directory / f'{i}_{video_t:.3f}.jpg'
        region_rows = []
        if not path.exists():
            if not _frame(ctx.video_path,video_t,path): continue
            from PIL import Image
            image = Image.open(path).convert('RGB'); image.thumbnail((672,672))
            telemetry_t = row.get('telemetry_time')
            _, raw = min(ctx.timed_samples, key=lambda s: abs(s[0] - telemetry_t)) if telemetry_t is not None else min(ctx.timed_samples, key=lambda s: abs(s[0]-float(ctx.api_row['t_peak'])))
            pose = {'camera_position':raw.get('cameraPosition'), 'camera_rotation':raw.get('cameraRotation'),
                    'camera_fov':raw.get('cameraFOV')}
            gaze_uv = None
            if use_gaze:
                image, gaze_uv = _gaze_overlay(image, raw, pose, api)
            if segmenter is not None:
                image, region_rows = segmenter.annotate(image, list(labels))
            image.save(path)
        _, raw = min(ctx.timed_samples, key=lambda s:abs(s[0]-(row.get('telemetry_time') if row.get('telemetry_time') is not None else float(ctx.api_row['t_peak']))))
        pose = {'camera_position':raw.get('cameraPosition'), 'camera_rotation':raw.get('cameraRotation'),
                'camera_fov':raw.get('cameraFOV')}
        points = {}
        for c in ctx.candidates:
            u,v,ok = api.project_world_to_image(pose,c.world_row())
            if ok and 0<=u<=1 and 0<=v<=1: points[c.candidate_id]=[round(u,3),round(v,3)]
        metadata.append({'frame':len(paths), 'time':round(video_t,3),
                         'telemetry_time': row.get('telemetry_time'), 'mention_id': row.get('mention_id'),
                         'reason': row.get('reason'), 'approx_candidate_uv':points,
                         'gaze_preprocessed': bool(use_gaze), 'region_count': len(region_rows) if segmenter is not None else 0,
                         'regions': region_rows if segmenter is not None else []})
        paths.append(path)
    if not paths: raise RuntimeError(f'No real video frames for {sample_key(ctx)}')
    return paths,metadata


def visual_call(backend, ctx, referential, plan, directory, source, segmenter=None, prepared=None):
    records = plan['hand_records' if source=='hand' else 'visual_records']
    use_gaze = source == 'visual' and plan.get('use_gaze_preprocess', False)
    labels = [m.get('object_type', '') for m in referential.get('mentions', [])]
    if prepared is None:
        paths, frames = prepare_frames(ctx, records, directory/source, use_gaze=use_gaze,
                                       segmenter=segmenter if source == 'visual' else None, labels=labels)
    else:
        paths, frames = prepared
    mentions = referential['mentions']
    allowed = {n for m in mentions for n in m['candidate_ids']}
    frames = [{**f, 'approx_candidate_uv':{n:p for n,p in f['approx_candidate_uv'].items() if n in allowed}} for f in frames]
    gaze_clause = '''A soft red gaze halo may be present. It is only an attention prior: it can fall on
an obstacle, background or an adjacent object. Never treat it as a target label or object center.
''' if use_gaze else '''Ignore green gaze markers and do not infer object identity from gaze telemetry.
'''
    region_clause = '''Some frames include category-level GroundingDINO/SAM region overlays. They are
visible-region proposals, not candidate IDs; use RGB appearance and language to verify them.
''' if segmenter is not None and source == 'visual' else ''
    common = '''The frames show a real VR interaction in chronological order. For EACH language mention,
score compatible candidates independently. Include actors, action targets and concrete spatial anchors.
A region with no matching candidate remains unresolved. Do not output all same-class instances merely
because uncertain. Approximate candidate projected centers are identity hints, NOT reliable boxes;
if they disagree with pixels, lower confidence. u goes left-to-right and v top-to-bottom in [0,1].
Choose a relevant supplied frame/time separately for each mention; different objects may occur at
different times. Output JSON {"evidence":[{"mention_id":"m0","candidate_id":"allowed name",
"support_score":0.0,"confidence":0.0,"temporal_range":[0.0,0.0],
"cue_type":"...","explanation":"brief observable evidence"}]}.
Only use supplied mention IDs, compatible candidate IDs, and supplied frame times.
Unobserved evidence has score=0 and confidence=0. Do not include GT or invent objects.
'''
    common = gaze_clause + region_clause + common
    if source=='hand':
        task = '''You are the Hand Agent. Use the VISIBLE HAND as a referential attention cue.
Assess visible pointing, manipulation/contact, source interaction, destination indication,
or weak/irrelevant gestures, for each language-mentioned object. Determine left/right only
if visually supported. Do not compute a 3D target from hand pose, hand joints or ray-hit points.
Ignore green gaze markers for scoring; instruction and RGB are supplied to understand hand cues.
If a hand is absent or no gesture-object association is visible, return zero confidence.
A hand cue is auxiliary: it cannot suppress other explicit language mentions.
'''
    else:
        task = '''You are the Visual Agent. Judge whether each candidate matches the mention's visible
category, attributes and scene relationships. Focus on object appearance. Ignore gaze markers
and hand gestures as evidence for this score; they have dedicated agents. Raw recordings may
contain these overlays. Give distinct candidate scores when pixels support disambiguation.
'''
    prompt = task + common + json.dumps({'instruction':ctx.text,'mentions':mentions,'frames':frames},ensure_ascii=False)
    payload = backend.ask(sample_key(ctx),source,prompt,paths)
    by_id = {m['mention_id']:m for m in mentions}
    valid_evidence = []
    rejected = []
    for row in payload.get('evidence',[]):
        mid = row.get('mention_id'); name = row.get('candidate_id')
        if mid not in by_id or name not in by_id[mid]['candidate_ids']:
            rejected.append({'row': row, 'error': 'candidate_or_mention_not_in_closed_set'})
            continue
        window = row.get('temporal_range')
        if window is not None:
            ft=[f['time'] for f in frames]
            if len(window)!=2 or window[0]>window[1] or any(not any(abs(t-v)<=.02 for v in ft) for t in window):
                row['rejected_temporal_range'] = window
                row['temporal_range'] = None
                # A model may use [0, 0] as a schema placeholder. The frame
                # itself is still valid evidence; only the invented timestamp
                # is discarded.
                row['validation_error'] = 'timestamp_not_in_supplied_frames_discarded'
        valid_evidence.append(row)
    return {**payload,'evidence': valid_evidence, 'rejected_evidence': rejected,
            'source':f'qwen_rgb_{source}','frames':frames,
            'rightIndexFingerRayHitPoint_used':False}
