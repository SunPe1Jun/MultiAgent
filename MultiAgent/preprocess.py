"""Mention-timed RGB, gaze attention views and optional category masks."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .agents import family, object_type
from .baseline import module
from .visualize import _frame
from .asr import guard_clock


VERSION = 'V3_scene_local_coreference_clock_guard_mention_focus'


def _cue_times(ctx, start, end, offset):
    """Select gaze-stable and hand-tracked times, without target IDs or hits."""
    samples = [(t, raw) for t, raw in ctx.timed_samples if start <= t+offset <= end]
    proposals = []
    for index, (t, raw) in enumerate(samples):
        if index % 3: continue
        eye = raw.get('eyeGaze', {})
        vector = eye.get('gazeVector', {})
        try:
            v = np.array([vector[k] for k in ('x', 'y', 'z')], dtype=float)
            v /= np.linalg.norm(v)
            nearby = []
            for tt, rr in samples[max(0,index-3):index+4]:
                if abs(tt-t) > .3 or rr.get('eyeGaze', {}).get('isEyeOpen') is False: continue
                vv = rr.get('eyeGaze', {}).get('gazeVector', {})
                w = np.array([vv[k] for k in ('x','y','z')], dtype=float)
                if np.isfinite(w).all() and np.linalg.norm(w) > 1e-6:
                    nearby.append(float(np.dot(v, w/np.linalg.norm(w))))
            if eye.get('isEyeOpen') is not False and np.isfinite(v).all() and len(nearby) >= 3:
                proposals.append((float(np.mean(nearby)), t+offset, 'gaze_stable'))
        except (KeyError, TypeError, ValueError):
            pass
    result = []
    if proposals:
        _, t, reason = max(proposals)
        result.append((t, reason))
    tracked = [t+offset for t, raw in samples if
               raw.get('handData', {}).get('isRightHandTracked') or
               raw.get('handData', {}).get('isLeftHandTracked')]
    if tracked:
        # Tracking is only an availability cue; it does not assert pointing.
        t = max(tracked, key=lambda t: min([abs(t-v) for v,_ in result] or [1.]))
        result.append((t, 'hand_tracked'))
    return result


def plan_frames(ctx, mentions, alignment, duration, timed=True):
    """C policy: ASR may add frames but can never remove base evidence."""
    clock = guard_clock(alignment['clock'], duration)
    offset = clock.get('telemetry_to_video') or 0.
    limit = max(0., duration-.1)
    start = min(limit, max(0., float(ctx.api_row['t_start'])+offset))
    end = min(limit, float(ctx.api_row['t_end'])+offset)
    if clock.get('audio_timing_rejected') or end-start < min(.5, duration*.1):
        if not clock.get('audio_timing_rejected'):
            clock = {**clock, 'raw_clock':dict(clock), 'audio_timing_rejected':True,
                     'telemetry_to_video':None, 'audio_to_video':None,
                     'audio_to_telemetry':None, 'clock_available':False,
                     'calibration_status':'rejected_degenerate_event_window',
                     'fallback_reason':'event_window_too_short_after_clipping'}
        start, end, offset = 0., limit, 0.
        timed = False
    records = {}
    def add(t, reason, mids=()):
        t = round(min(limit, max(0., t)), 3)
        row = records.setdefault(t, {'time':t, 'mention_ids':[], 'reasons':[]})
        row['mention_ids'] = sorted(set(row['mention_ids']) | set(mids))
        if reason not in row['reasons']: row['reasons'].append(reason)

    # A fixed coverage floor independent of transcript and number of mentions.
    for q in (.05,.25,.5,.75,.95): add(start+(end-start)*q, 'event_context')
    peak = ((start+end)/2 if clock.get('audio_timing_rejected') else
            float(ctx.api_row.get('t_peak', (start+end)/2-offset))+offset)
    if start <= peak <= end and all(abs(t-peak) > .25 for t in records):
        add(peak, 'event_peak')
    # Clauses give only a soft order prior, never claimed word timestamps.
    boundaries = [0] + [m.end() for m in re.finditer(r'[,;]|\bthen\b', ctx.text, re.I)]
    boundaries = sorted(set(boundaries))
    clauses = [(a,b) for a,b in zip(boundaries, boundaries[1:]+[len(ctx.text)])
               if re.search(r'\w',ctx.text[a:b]) and ctx.text[a:b].strip().lower() != 'then']
    windows = {}
    phases = {}
    for mention in mentions:
        mid = mention['mention_id']
        pos = mention.get('span', [0])[0]
        phase = next((i for i,(a,b) in enumerate(clauses) if a <= pos < b), 0)
        count = max(1,len(clauses))
        lo = start + (end-start)*max(0., (phase-.25)/count)
        hi = start + (end-start)*min(1., (phase+1.25)/count)
        windows[mid] = {'video_range':[start,end], 'phase_hint':[lo,hi],
                        'status':'instruction_event', 'phase_is_soft_prior':True,
                        'asr_hint':None, 'clock_status':clock.get('calibration_status')}
        phases.setdefault(phase,[]).append(mid)
    for phase,mids in phases.items():
        add(start+(end-start)*(phase+.5)/max(1,len(clauses)), 'instruction_phase', mids)
    for t,reason in ([] if clock.get('audio_timing_rejected') else _cue_times(ctx,start,end,offset)):
        if all(abs(t-old) > .25 for old in records): add(t,reason)
    base_times = sorted(records)

    # At most two extra ASR frames. Weak head-only guesses and unrelated
    # transcripts never constrain a mention or evict its base observations.
    hints = []
    reliable = alignment.get('instruction_coverage', 0.) >= .65
    for mention in mentions:
        mid = mention['mention_id']
        match = alignment.get('mentions',{}).get(mid,{})
        clip = match.get('video_clip')
        accepted = (timed and reliable and match.get('matched') and
                    match.get('alignment_quality') == 'phrase' and
                    match.get('confidence',0.) >= .75 and clip and
                    clip[0] < end and clip[1] > start)
        if not accepted: continue
        lo,hi = max(start,clip[0]),min(end,clip[1])
        windows[mid]['asr_hint'] = {'video_range':[lo,hi], 'confidence':match['confidence'],
                                   'role':'supplement_only'}
        hints.append(((lo+hi)/2,mid))
    added = []
    for t,mid in hints:
        near = min(records,key=lambda old:abs(old-t))
        if abs(near-t) <= .25:
            add(near,'asr_hint',[mid])
        elif len(added)<2:
            add(t,'asr_hint',[mid]);added.append(round(t,3))
    rows = sorted(records.values(),key=lambda r:r['time'])
    for row in rows:
        row['reason'] = '+'.join(row['reasons'])
        row['telemetry_time'] = row['time']-offset if clock.get('telemetry_to_video') is not None else None
    return {'version':VERSION, 'records':rows, 'mention_windows':windows,
            'alignment':clock, 'asr_timing_used':any('asr_hint' in r['reasons'] for r in rows),
            'base_frame_times':base_times, 'asr_added_frame_times':added,
            'timing_policy':('full_video_clock_fallback' if clock.get('audio_timing_rejected')
                             else 'instruction_event_plus_asr_hint' if timed else 'instruction_event_only')}


def gaze_heatmap(ctx, frame_time, raw, width, height):
    """Reproject nearby valid gaze rays to the current frame; no hit depth is used."""
    api = module('api')
    rotation = raw.get('cameraRotation', {})
    fov = raw.get('cameraFOV')
    heat = np.zeros((height,width), dtype=np.float32)
    try:
        quat = tuple(float(rotation[k]) for k in ['x','y','z','w'])
        tan = math.tan(math.radians(float(fov))/2)
        if tan <= 0: raise ValueError('invalid FOV')
    except (KeyError, TypeError, ValueError):
        return heat, {'status':'invalid_camera', 'valid_samples':0}
    yy,xx = np.mgrid[0:height,0:width]
    points = []
    for t,s in ctx.timed_samples:
        if abs(t-frame_time) > .3: continue
        eye = s.get('eyeGaze',{})
        if eye.get('isEyeOpen') is False: continue
        direction = api.point3_from_mapping(eye.get('gazeVector'))
        if direction is None: continue
        x,y,z = api.rotate_vector_by_quaternion(direction, api.quaternion_conjugate(quat))
        if z <= 1e-6: continue
        u,v = .5+x/(2*z*tan*(width/height)), .5-y/(2*z*tan)
        if not (0<=u<=1 and 0<=v<=1): continue
        weight = math.exp(-abs(t-frame_time)/.2)
        sigma = max(5.,min(width,height)*.07)
        heat += (weight*np.exp(-((xx-u*width)**2+(yy-v*height)**2)/(2*sigma*sigma))).astype(np.float32)
        points.append([round(u,4),round(v,4)])
    if heat.max()>0: heat /= heat.max()
    return heat, {'status':'valid' if points else 'no_valid_ray', 'valid_samples':len(points),
                  'uv_points':points, 'source':'world_gazeVector_reprojected',
                  'window_seconds':.6, 'sigma_fraction':.07,
                  'projection_status':'pinhole_FOV_estimate_not_pixel_calibrated'}


def attention_view(image, heat):
    if not heat.any(): return image.copy(), None
    rgb = np.asarray(image).astype(np.float32)
    alpha = (heat*.20)[...,None]
    overlay = Image.fromarray((rgb*(1-alpha)+np.array([255,40,35])*alpha).astype(np.uint8))
    yy,xx = np.nonzero(heat>=.25)
    pad = int(min(image.size)*.12)
    box = [max(0,int(xx.min())-pad),max(0,int(yy.min())-pad),
           min(image.width,int(xx.max())+pad+1),min(image.height,int(yy.max())+pad+1)]
    crop = image.crop(box)
    side_width = image.width//2
    view = Image.new('RGB',(image.width+side_width,image.height),'white')
    view.paste(image,(0,0))
    overlay.thumbnail((side_width,image.height//2))
    crop.thumbnail((side_width,image.height//2))
    view.paste(overlay,(image.width,0));view.paste(crop,(image.width,image.height//2))
    return view, box


def candidate_crop_panel(image, projected_uv, candidate_ids, tile_size=(156, 128), columns=4):
    """Render small, explicitly labelled candidate crops for same-class comparison.

    The projection is only a proposal for where to look in RGB.  The label is a
    bookkeeping aid for the visual model and is never treated as an identity
    assertion by the resolver.
    """
    entries = [(name, projected_uv[name]) for name in candidate_ids if name in projected_uv]
    if not entries:
        return None, []
    tw, th = tile_size
    rows = (len(entries) + columns - 1) // columns
    panel = Image.new('RGB', (columns * tw, rows * th), 'white')
    draw = ImageDraw.Draw(panel)
    pad = max(8, int(min(image.size) * .10))
    for index, (name, uv) in enumerate(entries):
        try:
            u, v = float(uv[0]), float(uv[1])
        except (TypeError, ValueError, IndexError):
            continue
        cx, cy = u * image.width, v * image.height
        box = (max(0, int(cx - pad)), max(0, int(cy - pad)),
               min(image.width, int(cx + pad)), min(image.height, int(cy + pad)))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        crop = image.crop(box)
        crop.thumbnail((tw - 4, th - 25))
        x, y = (index % columns) * tw, (index // columns) * th
        panel.paste(crop, (x + (tw - crop.width) // 2, y + 22 + (th - 22 - crop.height) // 2))
        draw.rectangle((x, y, x + tw - 1, y + th - 1), outline=(30, 30, 30), width=2)
        draw.rectangle((x + 1, y + 1, x + tw - 2, y + 20), fill=(245, 245, 245))
        draw.text((x + 4, y + 4), str(name), fill=(0, 0, 0))
    return panel, [name for name, _ in entries]


def prepare_isolated_views(frames, directory):
    """Create mention-neutral RGB views without the shared candidate panel."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for frame in frames:
        image = Image.open(frame['image_rgb']).convert('RGB')
        if frame.get('gaze', {}).get('uv_points'):
            marker = ImageDraw.Draw(image)
            for u, v in frame['gaze']['uv_points']:
                cx, cy = int(u * image.width), int(v * image.height)
                radius = max(5, int(min(image.size) * .012))
                marker.ellipse((cx-radius, cy-radius, cx+radius, cy+radius),
                               outline=(30, 220, 70), width=max(2, radius // 3))
        path = directory / f"{frame['frame_id']}.jpg"
        image.save(path)
        result[frame['frame_id']] = str(path)
    return result


def prepare_views(ctx, referential, plan, directory, use_gaze=False, segmenter=None):
    directory = Path(directory)
    signature = {'version':VERSION, 'plan':plan, 'use_gaze':use_gaze, 'regions':segmenter is not None,
                 'mention_types':[m['object_type'] for m in referential['mentions']],
                 'candidate_ids':[m['candidate_ids'] for m in referential['mentions']]}
    digest = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:16]
    directory = directory/digest
    directory.mkdir(parents=True,exist_ok=True)
    cache = directory/'views.json'
    if cache.exists():
        cached = json.loads(cache.read_text())
        if all(Path(p).exists() for p in cached['paths']): return cached['paths'],cached['frames']
    api = module('api')
    paths,frames=[],[]
    labels=[m['object_type'] for m in referential['mentions']]
    for idx,row in enumerate(plan['records']):
        frame_dir=directory/f'frame{idx}';frame_dir.mkdir(exist_ok=True)
        raw_path=frame_dir/'rgb.jpg'
        if not _frame(ctx.video_path,row['time'],raw_path):
            raise RuntimeError(f'Unable to extract real frame {ctx.video_path}@{row["time"]}')
        im=Image.open(raw_path).convert('RGB');im.thumbnail((672,672));im.save(raw_path)
        frame={**row,'frame_id':f'F{idx+1}', 'image_rgb':str(raw_path), 'approx_candidate_uv':{}}
        heat=np.zeros((im.height,im.width),np.float32)
        telemetry_t=row['telemetry_time']
        if telemetry_t is not None and ctx.timed_samples:
            t,raw=min(ctx.timed_samples,key=lambda s:abs(s[0]-telemetry_t))
            frame['telemetry_gap_seconds']=abs(t-telemetry_t)
            if abs(t-telemetry_t)<=.5:
                pose={'camera_position':raw.get('cameraPosition'),'camera_rotation':raw.get('cameraRotation'),
                      'camera_fov':raw.get('cameraFOV')}
                allowed={n for m in referential['mentions'] for n in m['candidate_ids']}
                for c in ctx.candidates:
                    if c.candidate_id not in allowed: continue
                    u,v,ok=api.project_world_to_image(pose,c.world_row())
                    if ok and 0<=u<=1 and 0<=v<=1:
                        frame['approx_candidate_uv'][c.candidate_id]=[round(u,3),round(v,3)]
                if use_gaze:
                    heat,gaze=gaze_heatmap(ctx,telemetry_t,raw,im.width,im.height)
                    frame['gaze']=gaze
            else: frame['gaze']={'status':'telemetry_unavailable_at_frame'}
        if use_gaze:
            np.savez_compressed(frame_dir/'heatmap.npz',heatmap=heat)
            Image.fromarray((heat*255).astype(np.uint8)).save(frame_dir/'heatmap.png')
            frame['heatmap_path']=str(frame_dir/'heatmap.npz')
        frame['regions']=[]
        if segmenter is not None:
            # Detect and segment the untouched RGB, then render attention views.
            annotated,regions=segmenter.annotate(im,labels,frame_dir,heat if use_gaze else None)
            annotated.save(frame_dir/'regions.jpg')
            for reg in regions:
                x0,y0,x1,y1=reg['box'];options=[]
                for name,(u,v) in frame['approx_candidate_uv'].items():
                    if family(object_type(name)) == family(reg['label']) and x0<=u*im.width<=x1 and y0<=v*im.height<=y1:
                        options.append(name)
                reg['candidate_options']=options
                reg['identity_status']='unverified_projection_hint'
            frame['regions']=regions
        display_im = im.copy()
        if use_gaze and frame.get('gaze', {}).get('uv_points'):
            # Match the original trimodal storyboard convention with a small
            # green gaze marker while retaining the soft heatmap crop. This is
            # an attention cue, never a candidate identity assertion.
            marker = ImageDraw.Draw(display_im)
            for u, v in frame['gaze']['uv_points']:
                cx, cy = int(u * display_im.width), int(v * display_im.height)
                radius = max(5, int(min(display_im.size) * .012))
                marker.ellipse((cx-radius, cy-radius, cx+radius, cy+radius),
                               outline=(30, 220, 70), width=max(2, radius // 3))
        view,box=attention_view(display_im,heat) if use_gaze else (display_im,None)
        frame['gaze_crop_box']=box
        if segmenter is not None:
            annotated.thumbnail((im.width//2,im.height))
            combined=Image.new('RGB',(view.width+im.width//2,im.height),'white')
            combined.paste(view,(0,0));combined.paste(annotated,(view.width,0));view=combined
        allowed = sorted({n for m in referential['mentions'] for n in m['candidate_ids']})
        panel, panel_names = candidate_crop_panel(im, frame['approx_candidate_uv'], allowed)
        if panel is not None:
            frame['candidate_panel_candidates'] = panel_names
            canvas = Image.new('RGB', (max(view.width, panel.width), view.height + panel.height), 'white')
            canvas.paste(view, (0, 0))
            canvas.paste(panel, (0, view.height))
            view = canvas
        path=frame_dir/'view.jpg';view.save(path);paths.append(str(path));frames.append(frame)
    associate_regions(frames)
    cache.write_text(json.dumps({'paths':paths,'frames':frames},indent=2))
    return paths,frames


def associate_regions(frames):
    """Conservative adjacent-frame Hungarian matching; ambiguous IDs stay unknown."""
    from scipy.optimize import linear_sum_assignment
    previous=[]; next_id=0
    for frame in frames:
        current=frame['regions']
        for row in current: row['association_status']='new_or_unmatched';row['track_id']=None
        if previous and current:
            cost=np.ones((len(previous),len(current)),dtype=float)*10
            for i,a in enumerate(previous):
                for j,b in enumerate(current):
                    if a['label']!=b['label']:continue
                    x0,y0=max(a['box'][0],b['box'][0]),max(a['box'][1],b['box'][1])
                    x1,y1=min(a['box'][2],b['box'][2]),min(a['box'][3],b['box'][3])
                    inter=max(0,x1-x0)*max(0,y1-y0)
                    aa=(a['box'][2]-a['box'][0])*(a['box'][3]-a['box'][1])
                    bb=(b['box'][2]-b['box'][0])*(b['box'][3]-b['box'][1])
                    cost[i,j]=1-inter/max(1,aa+bb-inter)
            ri,ci=linear_sum_assignment(cost)
            for i,j in zip(ri,ci):
                alternatives=sorted(cost[:,j])
                if cost[i,j]<.6 and (len(alternatives)<2 or alternatives[1]-alternatives[0]>.15):
                    current[j]['track_id']=previous[i]['track_id']
                    current[j]['association_status']='provisional_adjacent_bbox_match'
        for row in current:
            if row['track_id'] is None:
                row['track_id']=f'T{next_id}';next_id+=1
        previous=current
