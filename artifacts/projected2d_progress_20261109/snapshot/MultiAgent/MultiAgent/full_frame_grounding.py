"""Full-field visual referent choices with GT-free geometric ID decoding.

The model chooses category and image location from the whole interaction. A
fixed decoder compares against ALL type-compatible public scene anchors, never
against the event's annotated referents. No candidate projection is rendered or
retained as an all-anchor audit. Only the chosen/runner-up distances are logged.
"""
from __future__ import annotations
import json
import math
import re
import numpy as np
from .camera_geometry import project_world
from .mention_mapping import category

VERSION='full_frame_grounding_v2'
MAX_DISTANCE=.15
MIN_MARGIN=.02


def normalized_category(text):
    return ' '.join(category(str(text)).split())


def categories_for_candidate(candidate):
    return {normalized_category(x) for x in [candidate['candidate_id'],*candidate.get('aliases',[])]}


PROMPT='''Select the concrete objects referred to by the ORIGINAL INSTRUCTION from
this entire chronological first-person interaction. Inspect ALL frames and use
visible actions, hand pointing/contact, appearance and spatial relations. Include
actors, acted-on objects, concrete destinations and reference objects. Resolve
repeated occurrences individually. Do not return only the last target or every
looked-at object. For a part with no separate scene category, locate its parent
object. A vague here/there is not automatically a concrete object.

Each video frame has FOUR full-field panels: TOP LEFT original RGB, TOP RIGHT
EARLIER attention, BOTTOM LEFT MIDDLE attention, BOTTOM RIGHT RECENT attention.
The three maps cover the exact same full camera field as RGB, and summarize the
two seconds before that RGB frame. Orange is weak measured gaze density. It is
not a target label; it may be on background, hands, reference or other objects.
Black means missing added gaze, not negative evidence. Some graphics are control
conditions. Original RGB recorder marks remain. Judge RGB and language first.

Your task is VISUAL GROUNDING: choose a supporting temporal view and a visible
point ON EACH INTENDED OBJECT in that view's TOP LEFT RGB panel. Coordinates
are normalized ONLY within that original RGB panel: top-left (0,0), bottom-right
(1,1), independent of the surrounding composite. Do not point into a heat panel.
A Gxxx video view contains the two listed source frames; choose a stable visible
object in that temporal group. Do not guess a world anchor number. Scripts will
perform scene-ID matching after your visual choice without referent annotations.
Use category names from the entire scene vocabulary below, choosing a compatible
specific category where possible. This vocabulary is not filtered to targets.
Copy mention as an exact substring of the original instruction. If an intended
object is absent or impossible to locate, put its literal phrase in unresolved.

Return JSON only with exactly two keys:
{"referents":[{"mention":"literal phrase","category":"scene category",
"view_id":"Gxxx","x":0.0,"y":0.0}],"unresolved":[]}
The numbers above only illustrate the format, never default coordinates. x,y
must be finite in [0,1]. Return one row per intended visible instance/mention;
multiple mentions may refer to the same object. Never output attention readout,
world positions, candidate IDs or explanations.

INPUT: '''


def grounding_prompt(event):
    from .local_video_visual import prompt_temporal_groups
    vocab=sorted(set().union(*(categories_for_candidate(c) for c in event['scene_candidate_table'])))
    return PROMPT+json.dumps({'instruction':event['instruction'],'scene_category_vocabulary':vocab,
        'video_views':prompt_temporal_groups(event['qwen_temporal_groups'])},ensure_ascii=False)


def validate_grounding(parsed,event):
    errors=[]
    if not isinstance(parsed,dict) or set(parsed)!={'referents','unresolved'}:
        return {'schema_valid':False,'errors':['wrong_keys'],'referents':[]}
    rows=parsed['referents'];views={g['view_id'] for g in event['qwen_temporal_groups']}
    vocab=set().union(*(categories_for_candidate(c) for c in event['scene_candidate_table']))
    if not isinstance(rows,list):errors.append('referents_not_array');rows=[]
    for row in rows:
        if not isinstance(row,dict) or set(row)!={'mention','category','view_id','x','y'}:
            errors.append('invalid_referent');continue
        if not isinstance(row['mention'],str) or not row['mention'] or row['mention'] not in event['instruction']:errors.append('nonliteral_mention')
        if not isinstance(row['category'],str) or normalized_category(row['category']) not in vocab:errors.append('unknown_category')
        if row['view_id'] not in views:errors.append('unknown_view')
        for key in ('x','y'):
            v=row[key]
            if not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or not 0<=v<=1:errors.append('invalid_coordinate')
    if not isinstance(parsed['unresolved'],list) or not all(isinstance(t,str) and t and t in event['instruction'] for t in parsed['unresolved']):errors.append('invalid_unresolved')
    return {'schema_valid':not errors,'errors':sorted(set(errors)),'referents':rows if not errors else []}


def decode_grounding(rows,event,max_distance=MAX_DISTANCE,min_margin=MIN_MARGIN):
    """Same decoder for all conditions. No event GT, gaze or prior response read.

    Distances compare the model's chosen RGB point with category-compatible
    scene anchor centers. Project each public candidate into both source frames
    of the chosen temporal patch and average its available in-frustum positions.
    No unsupported camera is substituted. Large distance / close runner-up
    remains unresolved; exact duplicate centers never force a tie selection.
    """
    by_frame={f['frame_id']:f for f in event['frames']}
    groups={g['view_id']:g for g in event['qwen_temporal_groups']}
    selected=[];decisions=[]
    for row in rows:
        family=normalized_category(row['category'])
        candidates=[c for c in event['scene_candidate_table'] if family in categories_for_candidate(c)]
        frames=[by_frame[fid] for fid in groups[row['view_id']]['frame_ids']]
        scored=[];usable_frames=sum(f['geometry']['clock_available'] for f in frames)
        for candidate in candidates:
            positions=[]
            for frame in frames:
                geometry=frame['geometry']
                if not geometry['clock_available']:continue
                result=project_world(candidate['position_world'],geometry['reference_pose'],*frame['source_size'])
                if result['in_frustum']:
                    uv=result['normalized_uv_unclipped'];positions.append(uv)
            if positions:
                uv=np.mean(positions,axis=0)
                distance=float(np.linalg.norm(uv-np.array([row['x'],row['y']])))
                scored.append((distance,candidate['candidate_id'],len(positions)))
        scored.sort();choice=None;reason='no_supported_camera_or_in_frustum_candidate'
        # A visually chosen category with a single scene identity can resolve
        # without a camera; the model must still have emitted a visible point.
        if usable_frames==0 and len(candidates)==1:
            choice=candidates[0]['candidate_id'];reason='visually_selected_unique_scene_category'
        elif scored:
            gap=scored[1][0]-scored[0][0] if len(scored)>1 else None
            if scored[0][0]>max_distance:reason='anchor_center_far_from_selected_rgb_point'
            elif gap is not None and gap<min_margin:reason='ambiguous_same_category_anchor_centers'
            else:choice=scored[0][1];reason='geometric_scene_id_match'
        decisions.append({'visual_choice':row,'selected_candidate':choice,'reason':reason,
            'compatible_scene_candidates':len(candidates),'supported_camera_frames':usable_frames,
            'best_distance':scored[0][0] if scored else None,
            'runner_up_distance':scored[1][0] if len(scored)>1 else None,
            'source_frame_ids':groups[row['view_id']]['frame_ids'],
            'identity_and_occlusion_verified':False})
        if choice and choice not in selected:selected.append(choice)
    return {'selected_object_names':selected,'decisions':decisions,'gt_used':False,
            'max_normalized_distance':max_distance,'min_runner_up_margin':min_margin,
            'scope':'Model visual choices plus fixed approximate public-anchor decoder; not pure model anchor-ID reasoning.'}
