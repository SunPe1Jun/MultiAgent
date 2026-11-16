"""Explicit whole-image coordinates and shared language-only mention coverage."""
import copy
import json
import math
from .full_frame_grounding import categories_for_candidate,validate_grounding,normalized_category
from .local_video_visual import prompt_temporal_groups

VERSION='full_frame_grounding_v3_composite_coordinates'
PROMPT='''Ground EVERY listed language mention in this complete first-person interaction.
Use the entire chronological sequence, visible pointing/contact/actions and spatial
relations. Include actors, manipulated objects and concrete reference/destination
objects. Mention hints come only from instruction parsing, not object annotations.
A component resolves to its visible parent category when no component category exists.

Each displayed video frame contains four full camera fields: TOP LEFT is the
original RGB. The other three panels are earlier / middle / recent attention
from the preceding two seconds. Orange means weak gaze density; black means
missing added evidence. Attention can be on background/reference/other objects.
Some maps are controls. Choose the actual referred RGB object, not a heat blob.

For each identifiable mention, return a point ON THE OBJECT in the TOP LEFT RGB
panel. IMPORTANT: x and y use the WHOLE DISPLAYED COMPOSITE IMAGE on the standard
0-to-1000 scale. (0,0) is its extreme top-left, (1000,1000) its extreme bottom-right.
Do not mentally rescale coordinates to the RGB subpanel. Scripts perform that
conversion. A valid point in the top-left RGB panel has x below 500 and y between
20 and 500 for this layout. Return the actual object position, not a default or
panel center. Never place a point in the three heat panels.

Cite the Gxxx temporal view in which that visible point is supported. Prefer a
view listed as having camera support when the same object is visible there;
this lets scripts match your point to scene IDs. Still inspect all views, and
retain unresolved mentions if the object cannot be located. Each Gxxx view
contains two source frames. Do not assume all actions happen in G001.

Choose a compatible category from the complete scene vocabulary. Scripts use
fixed geometry to decode identities; do not guess arbitrary anchor numbers.
Return JSON only with exactly {"referents":[{"mention":"exact original phrase",
"category":"scene category","view_id":"Gxxx","x":0,"y":0}],"unresolved":[]}.
The example coordinates are format placeholders, not answers. Ground each listed
mention occurrence independently, copying its literal text exactly. Do not invent
hidden objects or extra targets; uncertain object identity may remain unresolved.

INPUT: '''


def grounding_prompt(event):
    vocab=sorted(set().union(*(categories_for_candidate(c) for c in event['scene_candidate_table'])))
    by_id={f['frame_id']:f for f in event['frames']}
    availability=[{'view_id':g['view_id'],'supported_source_frames':sum(by_id[f]['geometry']['clock_available'] for f in g['frame_ids'])}
                  for g in event['qwen_temporal_groups']]
    return PROMPT+json.dumps({'instruction':event['instruction'],'mention_hints':event['mention_hints'],
        'scene_category_vocabulary':vocab,'video_views':prompt_temporal_groups(event['qwen_temporal_groups']),
        'camera_support_for_point_decoding':availability},ensure_ascii=False)


def validate_composite_grounding(parsed,event):
    """Keep individually valid visible choices; reject bad rows without repairing them."""
    if (not isinstance(parsed,dict) or set(parsed)!={'referents','unresolved'}
            or not isinstance(parsed.get('referents'),list)
            or not isinstance(parsed.get('unresolved'),list)):
        return {'schema_valid':False,'errors':['wrong_grounding_structure'],'referents':[],
                'unresolved':[],'rejected_referents':[],'rejected_unresolved':[],
                'coordinate_contract':'whole_composite_0_to_1000'}
    width,height=event['frames'][0]['source_size'];total_width=width*2;total_height=(height+28)*2
    views={g['view_id'] for g in event['qwen_temporal_groups']}
    vocab=set().union(*(categories_for_candidate(c) for c in event['scene_candidate_table']))
    accepted=[];rejected=[];errors=[]
    for index,raw in enumerate(parsed['referents']):
        row_errors=[]
        if not isinstance(raw,dict) or set(raw)!={'mention','category','view_id','x','y'}:
            rejected.append({'index':index,'reason':['invalid_referent_shape']});errors.append('invalid_referent_shape');continue
        row=dict(raw)
        if not isinstance(row['mention'],str) or not row['mention'] or row['mention'] not in event['instruction']:row_errors.append('nonliteral_mention')
        if not isinstance(row['category'],str) or normalized_category(row['category']) not in vocab:row_errors.append('unknown_category')
        if row['view_id'] not in views:row_errors.append('unknown_view')
        point_ok=True
        for key in ('x','y'):
            value=row[key]
            if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or not 0<=value<=1000:
                row_errors.append('invalid_composite_coordinate');point_ok=False
        if point_ok:
            row['x']=row['x']/1000*total_width/width
            row['y']=(row['y']/1000*total_height-28)/height
            if not 0<=row['x']<=1 or not 0<=row['y']<=1:row_errors.append('point_outside_original_rgb_panel')
        if row_errors:
            reasons=sorted(set(row_errors));rejected.append({'index':index,'mention':raw.get('mention'),'reason':reasons});errors.extend(reasons)
        else:accepted.append(row)
    unresolved=[];rejected_unresolved=[]
    for index,value in enumerate(parsed['unresolved']):
        if isinstance(value,str) and value and value in event['instruction']:unresolved.append(value)
        else:
            rejected_unresolved.append({'index':index,'value':value,'reason':'not_literal_phrase'});errors.append('invalid_unresolved')
    from collections import Counter
    expected=[h['text'] for h in event['mention_hints'] if isinstance(h,dict) and isinstance(h.get('text'),str)]
    accounted=[row['mention'] for row in accepted]+unresolved
    remaining=Counter(expected)-Counter(accounted)
    return {'schema_valid':True,'errors':sorted(set(errors)),'referents':accepted,'unresolved':unresolved,
            'rejected_referents':rejected,'rejected_unresolved':rejected_unresolved,
            'unaccounted_mention_hints':list(remaining.elements()),
            'coordinate_contract':'whole_composite_0_to_1000',
            'failure_policy':'Use only individually valid literal/category/view/point rows. Preserve rejected raw rows; never infer missing referents.'}
