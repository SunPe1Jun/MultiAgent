#!/usr/bin/env python3
"""Prompt-only temporal plus gaze alignment control on frozen 2D full-RGB panels."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from MultiAgent.projected2d_attention_study import CSV_COLUMNS, evaluator_adapter, save_json, sha_file, write_csv
from MultiAgent.projected2d_main_image import normalize, verify_lock
from MultiAgent.projected2d_routed_rgb import MODEL, ROOT

SOURCE = ROOT / '.local-sync/projected2d_main_rgb32_20261109'
PRIOR = ROOT / '.local-sync/referent_rgb_expansion32_20261109_repaired/manifest.json'
OUTPUT = ROOT / '.local-sync/referent_temporal_gaze_marked_prompt_v5_20261109'
EVENT_IDS = {'scene1_495','scene2_369','scene2_412','scene3_161','scene4_room1_1','scene4_room2_19','scene4_room3_108','scene4_room4_72','scene5_226','scene2_482'}


def prepare(out: Path):
    """Render a visible gaze marker on the same single full RGB image per panel."""
    import csv
    from PIL import Image, ImageDraw
    if (out / 'input_manifest.json').exists():
        raise ValueError('Refuse to overwrite frozen probe')
    verify_lock(SOURCE)
    src = json.loads((SOURCE / 'input_manifest.json').read_text())
    prior = {e['id']: e for e in json.loads(PRIOR.read_text())['events']}
    gaze_rows = {}
    with (SOURCE / 'rebuilt_manifest_all.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            scene = row.get('scene') or row.get('\ufeffscene')
            if not scene or not row.get('panel_id'):
                continue
            key = (scene, int(row['row_index']), row['panel_id'])
            if row.get('gaze_projection_valid') == 'True' and row.get('gaze_u_norm') and row.get('gaze_v_norm'):
                gaze_rows[key] = {'u': float(row['gaze_u_norm']), 'v': float(row['gaze_v_norm']),
                                  'source': row.get('gaze_source') or 'gazePoint'}
    events = []
    for original in src['events']:
        if original['id'] not in EVENT_IDS:
            continue
        e = json.loads(json.dumps(original)); pe = prior[e['id']]; specs = {}
        for mention in e['mentions']:
            mid = mention['mention_id']; temporal = pe['temporal_search']['mentions'][mid]; lo, hi = temporal['primary_range']
            phase_panels = [p['frame_id'] for p in e['panels'] if lo <= p['requested_video_time'] <= hi]
            context_panels = [p['frame_id'] for p in e['panels'] if p['frame_id'] not in phase_panels]
            specs[mid] = {'highlighted_instruction': temporal['highlighted_instruction'], 'action_context': temporal['action_context'],
                'soft_phase_range': [lo, hi], 'phase_panel_ids': phase_panels, 'context_panel_ids': context_panels,
                'phase_coverage': 'covered' if phase_panels else 'no_supplied_panel_in_soft_phase', 'time_source': temporal['time_source']}
        e['temporal_alignment_specs'] = specs
        for panel in e['panels']:
            key = (e['scene_key'], int(e['row_index']), panel['frame_id'])
            gaze = gaze_rows.get(key)
            if gaze is None:
                raise ValueError('No measured gaze coordinate for ' + repr(key))
            src_path = Path(panel['inputs']['MAIN_RGB_768']['path'])
            target = out / 'inputs' / e['id'] / panel['frame_id'] / 'MAIN_RGB_768_GAZE_MARKED.png'
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src_path).convert('RGB') as image:
                marked = image.copy(); draw = ImageDraw.Draw(marked)
                cx, cy = round(gaze['u'] * (marked.width - 1)), round(gaze['v'] * (marked.height - 1))
                radius = max(10, round(min(marked.size) * .020))
                # Hollow region cue: a thin high-contrast boundary with a transparent center.
                draw.ellipse((cx-radius-2, cy-radius-2, cx+radius+2, cy+radius+2), outline=(0,0,0), width=4)
                draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=(40,240,80), width=3)
                tick = max(5, radius // 3)
                draw.line((cx-radius-2, cy, cx-radius-tick, cy), fill=(40,240,80), width=2)
                draw.line((cx+radius+2, cy, cx+radius+tick, cy), fill=(40,240,80), width=2)
                draw.line((cx, cy-radius-2, cx, cy-radius-tick), fill=(40,240,80), width=2)
                draw.line((cx, cy+radius+2, cx, cy+radius+tick), fill=(40,240,80), width=2)
                marked.save(target)
            panel['inputs']['MAIN_RGB_768_GAZE_MARKED']={'path':str(target), 'sha256':sha_file(target), 'size':[768,768]}
            panel['gaze_marker']={'u':gaze['u'],'v':gaze['v'],'source':gaze['source'],'image_path':str(target)}
        events.append(e)
    if len(events) != 10:
        raise ValueError('Expected fixed 10-event development subset')
    out.mkdir(parents=True, exist_ok=True)
    manifest={'version':'referent_temporal_gaze_marked_prompt_v5','events':events,
      'source_manifest_sha256':sha_file(SOURCE/'input_manifest.json'),'prior_manifest_sha256':sha_file(PRIOR),
      'same_visual_content_as':'projected2d_main_rgb32_20261109 MAIN_RGB_768, plus a thin hollow gaze-region ring on each panel',
      'change':'One full RGB image per frame with a measured gaze marker; no collage, crop, heatmap, target label or GT referent passed to model.',
      'sample_status':'Previously seen development events; not held out.','ground_truth_in_prompt':False,
      'gaze_source':'rebuilt manifest gazePoint projection; one coordinate per source panel, independently of referent labels'}
    save_json(out/'input_manifest.json',manifest)
    files=[Path(__file__).resolve(),out/'input_manifest.json',SOURCE/'input_manifest.json',SOURCE/'protocol_lock.json',
      SOURCE/'rebuilt_manifest_all.csv',PRIOR,ROOT/'MultiAgent/projected2d_main_image.py',ROOT/'MultiAgent/qwen_backend.py',
      ROOT/'MultiAgent/projected2d_attention_study.py',ROOT/'tools/run_full_frame_grounding_v4.py',
      ROOT.parent/'TriModal-Referring/exam2/evaluate_2d_point_grounding.py',MODEL/'config.json']
    save_json(out/'protocol_lock.json',{'files':{str(p):sha_file(p) for p in files}})
    print('FROZEN_TEMPORAL_GAZE_MARKED_PROMPT_V5',len(events),'events',sum(len(e['mentions']) for e in events),'mentions',flush=True)

def prompt_for(event):
    frames=[{'frame_id':p['frame_id'],'video_time_seconds':p['requested_video_time']} for p in event['panels']]
    tasks=[]
    for m in event['mentions']:
        s=event['temporal_alignment_specs'][m['mention_id']]
        tasks.append({**m,'temporal_alignment':s})
    return '''Solve 2D temporal referring-point localization for every mention in the complete instruction.
Each supplied image is one COMPLETE RGB video frame. The caption immediately before it gives its real
frame_id and video time. Each mention is a separate occurrence and must be assigned its own best
evidence frame; do not make all mentions share the first frame merely because an object is visible there.
Do not force different mentions onto different frames when the same frame truly supports them.

For each mention, the temporal_alignment block gives a rough clause-phase prior derived from the
instruction and identifies which supplied panels fall inside that rough range. It is not ground truth,
not a speech timestamp, and not a hard filter. Use it as a prior about which panel can support the
current expression. A panel is strong evidence only when it supports BOTH the referent's identity or
spatial relation AND the current mention's action/phase. Mere visibility in an earlier panel is not
sufficient temporal evidence. If several phase panels support the same mention, choose the one with
the clearest current action, state, or relation; do not choose the earliest visible panel by default.
If no supplied panel lies in the rough phase, explicitly recognize that coverage is missing and use
only independently visible evidence; do not pretend the requested action already happened.
The instruction is a command or description, not proof that the commanded action occurred in the RGB.
Use only what is visibly supported by each frame. For static anchors, use the panel that best supports
the stated relation, and if all are equivalent say so in evidence rather than silently defaulting to P1.


Each supplied image contains one hollow green gaze-region ring added by the input protocol at that panel's measured gaze location. Its center is transparent so the RGB target remains visible. It is a weak, frame-specific attention cue. It is not the answer,
not a guaranteed annotation, and must never be used as the output point. However, different panels
can have the gaze dot over different objects, so compare it separately for every mention and every
panel. A frame where the referent is visible but the gaze dot is clearly directed at another object
is weaker evidence for that mention than a frame where the dot is near the referent or its stated
relation. Use gaze alignment together with identity, relation, action phase and image evidence; do
not let the dot alone decide between similar objects. If the marker is absent, ambiguous, or noisy, mark
its contribution as unclear and continue using the RGB evidence. The selected frame's evidence must
state whether the visible gaze cue supports, conflicts with, or is unclear for the current mention.

For every mention_id return exactly one row. A localized physical target needs one supplied frame_id
and point_xy=[x,y] on the visible target in the entire corresponding RGB image, scale 0..1000,
origin top-left. For an ambiguous or not_visible mention return no point. For a component, locate
the indicated whole parent. The selected frame must be one of the supplied frame IDs.
Return JSON only:
{"mentions":[{"mention_id":"...","status":"localized|ambiguous|not_visible",
"predictions":[{"frame_id":"supplied_frame_id","point_xy":[x,y]}],
"gaze_alignment":"supports|conflicts|unclear",
"evidence":"identity/relationship, temporal-phase, and gaze-alignment reason"}]}
''' + json.dumps({'instruction':event['instruction'],'source_frames':frames,'localization_tasks':tasks},ensure_ascii=False)


def infer(out: Path):
    from MultiAgent.qwen_backend import QwenBackend
    from tools.run_full_frame_grounding_v4 import clean_visual_trace
    verify_lock(out); m=json.loads((out/'input_manifest.json').read_text())
    backend=QwenBackend(out/'model_cache',str(MODEL),local_only=True,namespace='referent_temporal_gaze_marked_prompt_v5',trace_visual_inputs=True)
    for i,e in enumerate(m['events'],1):
        target=out/'responses'/f"{e['id']}.json"; images=[p['inputs']['MAIN_RGB_768_GAZE_MARKED'] for p in e['panels']]
        captions=[f"Frame {p['frame_id']} | video_time={p['requested_video_time']:.3f} seconds" for p in e['panels']]
        prompt=prompt_for(e); fingerprint=hashlib.sha256(json.dumps([prompt,captions,[x['sha256'] for x in images]],ensure_ascii=False).encode()).hexdigest()
        if target.exists():
            if json.loads(target.read_text())['fingerprint']!=fingerprint: raise ValueError('Fingerprint mismatch')
            continue
        for x in images:
            if sha_file(Path(x['path']))!=x['sha256']: raise ValueError('Input RGB changed')
        req=None
        try:
            raw=backend.ask(e['id'],'TEMPORAL_GAZE_MARKED_PROMPT_V5',prompt,[x['path'] for x in images],max_new_tokens=2304,image_captions=captions)
            req=Path(backend.requests[-1]); record=json.loads(req.read_text())
            parsed,errors,conversions=normalize(raw,e)
            if raw.get('_parse_warning'): errors.append(raw['_parse_warning'])
            result={'status':'COMPLETE' if not errors else 'PARSE_WITH_WARNINGS','parsed':parsed,'warnings':errors,
              'conversions':conversions,'raw_response':record['response'],'request_path':str(req),
              'vision_verified':bool((record.get('visual_tensor_trace') or {}).get('input_consumption_verified')),
              'processor_audit':record.get('processor_audit')}
            if not result['vision_verified']: raise ValueError('Vision input not verified')
        except Exception as exc:
            result={'status':'FAILED','parsed':{'mentions':[]},'error':f'{type(exc).__name__}: {exc}'}
            if backend.requests:req=Path(backend.requests[-1])
        finally:
            if req: clean_visual_trace(str(req))
        result.update(event=e['id'],fingerprint=fingerprint,prompt=prompt,input_images=images,image_captions=captions)
        save_json(target,result); print('TEMPORAL_GAZE_MARKED_PROMPT_V5',i,'/',len(m['events']),e['id'],result['status'],flush=True)


def report(out: Path):
    verify_lock(out);m=json.loads((out/'input_manifest.json').read_text()); rows=[]; counts=Counter(); frame_counts=Counter()
    for e in m['events']:
        r=json.loads((out/'responses'/f"{e['id']}.json").read_text());counts[r['status']]+=1;counts['vision_verified']+=r.get('vision_verified',False)
        for row in r['parsed']['mentions']:
            s=e['temporal_alignment_specs'][row['mention_id']]
            for p in row['predictions']:
                frame_counts[p['frame_id']]+=1
                rows.append({'event':e['id'],'mention_id':row['mention_id'],'mention':row['mention'],
                  'selected_frame':p['frame_id'],'selected_time':next(x['requested_video_time'] for x in e['panels'] if x['frame_id']==p['frame_id']),
                  'phase_panel_ids':s['phase_panel_ids'],'phase_coverage':s['phase_coverage'],
                  'selected_in_phase':p['frame_id'] in s['phase_panel_ids'],'evidence':row['evidence']})
    save_json(out/'predictions_frozen.json',{'responses':{str(p):sha_file(p) for p in sorted((out/'responses').glob('*.json'))}})
    save_json(out/'summary.json',{'counts':dict(counts),'frame_choices':dict(frame_counts),'rows':rows,
      'interpretation':'Prompt-only control on same full RGB panels; selected_in_phase uses a soft instruction-phase prior, not ground truth.'})
    print('REPORTED_TEMPORAL_GAZE_MARKED_PROMPT_V5',dict(counts),dict(frame_counts),flush=True)


def evaluate(out: Path):
    verify_lock(out);m=json.loads((out/'input_manifest.json').read_text()); rows=[]
    for e in m['events']:
        r=json.loads((out/'responses'/f"{e['id']}.json").read_text());pred=evaluator_adapter(r['parsed'])
        rows.append({'scene':e['scene_key'],'row_index':e['row_index'],'event_id':e['event_id'],'instruction':e['instruction'],
          'model_input_image':';'.join(x['path'] for x in r['input_images']),'model_raw_output':r.get('raw_response',''),
          'parsed_json':json.dumps(pred,ensure_ascii=False),'parse_ok':str(r['status'] in ('COMPLETE','PARSE_WITH_WARNINGS')),
          'prediction_count':len(pred['referents']),'error_message':r.get('error','')})
    pred=out/'predictions.csv';write_csv(pred,rows,CSV_COLUMNS)
    import subprocess,sys
    target=out/'evaluation';subprocess.run([sys.executable,str(ROOT.parent/'TriModal-Referring/exam2/evaluate_2d_point_grounding.py'),
      '--manifest',str(SOURCE/'rebuilt_manifest_all.csv'),'--pred_csv',str(pred),'--output_dir',str(target),'--coordinate_mode','panel'],check=True,capture_output=True,text=True)
    save_json(out/'evaluation_summary.json',json.loads((target/'2d_eval_summary.json').read_text()))
    print((target/'2d_eval_summary.json').read_text(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','infer','report','evaluate']);p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    {'prepare':prepare,'infer':infer,'report':report,'evaluate':evaluate}[a.stage](a.output.resolve())
if __name__=='__main__':main()
