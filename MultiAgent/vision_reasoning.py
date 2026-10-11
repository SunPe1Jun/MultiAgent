"""Hand observations condition visual disambiguation rather than voting on IDs."""
import json
import math

from .agents import sample_key


def _coordinator_candidate_table(ctx):
    return [{'candidate_id': c.candidate_id, 'x_world': c.x_world,
             'y_world': c.y_world, 'z_world': c.z_world} for c in ctx.candidates]


def monolithic_coordinator(backend, ctx, referential, paths, frames, visual, use_spatial_cues=True):
    """Global closed-set decision layer modeled on the original trimodal protocol.

    Mention extraction and visual evidence remain separately logged. This stage only
    resolves conflicts between same-class candidates and merges mention decisions.
    """
    peak = (ctx.peak_spatial or {}) if use_spatial_cues else {}
    cues = {
        'gazePoint': peak.get('gaze_point'), 'gazeVector': peak.get('gaze_vector'),
        'gazeOrigin': peak.get('gaze_origin'), 'cameraPosition': peak.get('camera_position'),
        'cameraRotation': peak.get('camera_rotation'), 'cameraFOV': peak.get('camera_fov'),
    }
    evidence = [{k: row.get(k) for k in ('mention_id', 'candidate_id', 'support_score',
                                         'confidence', 'frame_id', 'explanation')}
                for row in (visual or {}).get('evidence', [])]
    prompt = '''You are the final global referent coordinator for one VR interaction event.
Read every explicit mention in the instruction, compare same-class candidate instances, and
return the union of intended canonical object IDs. The candidate list is closed: never invent
an ID. A component mention (door/window/flap) resolves to its parent object. A vague location
such as here/there is not itself an object; include a concrete scene anchor only when the
instruction's relation and the supplied visual/spatial context support it. Do not default to
the first numbered candidate. Use RGB frames as primary evidence; gaze/world values and
projected candidate centers are weak priors. Current visual evidence rows are suggestions,
not ground truth, and may conflict. Compare candidates explicitly before selecting. A mention
with count=1 or a singular noun must select at most ONE same-class candidate. Never union all
positive rows for a singular mention. If a singular mention cannot be disambiguated, leave it
unselected. Use gazePoint and spatial_context_text as tie-breakers when they agree with RGB.
Return only JSON: {"selected_object_names":["candidate_id",...],
"decisions":[{"mention_id":"m0","candidate_id":"candidate_id","reason":"brief"}],
"confidence":0.0}.
'''
    focus_enabled = bool((visual or {}).get('temporal_focus'))
    if focus_enabled:
        prompt += ('\nTemporal focus is enabled. Each decision must use evidence for that exact '
                   'mention occurrence; do not transfer evidence from an earlier identical phrase. '
                   'Only choose candidates supported by that mention\'s visual evidence, explicit ID, '
                   'or a unique compatible candidate. Do not create a new identity from global RGB.\n')
    payload = {'instruction': ctx.text, 'mentions': referential['mentions'],
               'candidate_world_table': _coordinator_candidate_table(ctx),
               'weak_spatial_cues': cues,
               'gaze_summary': ctx.api_row.get('gaze_summary', '') if use_spatial_cues else '',
               'spatial_context_text': ctx.api_row.get('spatial_context_text', '') if use_spatial_cues else '',
               'spatial_cues_enabled': use_spatial_cues,
               'frames': [{'frame_id': f.get('frame_id'), 'time': f.get('time'),
                           'mention_ids': f.get('mention_ids', []),
                           'approx_candidate_uv': f.get('approx_candidate_uv', {})}
                          for f in frames],
               'visual_evidence': evidence}
    if focus_enabled:
        payload['mention_focus_audit'] = visual.get('focus_audit', {})
    raw = backend.ask(sample_key(ctx), 'v3_global_coordinator', prompt + json.dumps(payload, ensure_ascii=False), paths)
    allowed = {c.candidate_id for c in ctx.candidates}
    selected = raw.get('selected_object_names') if isinstance(raw, dict) else []
    if not isinstance(selected, list):
        selected = []
    selected = [name for name in selected if isinstance(name, str) and name in allowed]
    decisions = raw.get('decisions', []) if isinstance(raw, dict) else []
    # Models sometimes return one positive row per same-class candidate even
    # when the language is singular. Enforce the mention cardinality here so
    # the global layer cannot turn a singular "that forklift/chair" into a
    # category prediction. Ties use the event gaze point only as a weak tie
    # breaker; the RGB evidence remains the source of candidate scores.
    gaze = peak.get('gaze_point') or {}
    gp = [gaze.get(k) for k in ('x', 'y', 'z')]
    by_mid = {m['mention_id']: m for m in referential.get('mentions', [])}
    visual_scores = {}
    for row in evidence:
        try:
            visual_scores[(row.get('mention_id'), row.get('candidate_id'))] = max(
                visual_scores.get((row.get('mention_id'), row.get('candidate_id')), 0.0),
                float(row.get('support_score', 0)) * float(row.get('confidence', 0)))
        except (TypeError, ValueError):
            pass
    chosen_by_mid = {}
    if focus_enabled:
        # Language-determined identities must survive a coordinator omission.
        # Their inclusion asserts a linguistic referent, not visible evidence.
        for mid, mention in by_mid.items():
            explicit = mention.get('explicit_ids', [])
            if explicit:
                chosen_by_mid[mid] = list(explicit)
            elif len(mention.get('candidate_ids', [])) == 1 and mention.get('kind') not in ('location', 'region'):
                chosen_by_mid[mid] = list(mention['candidate_ids'])
    rejected_decisions = []
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        mid, name = decision.get('mention_id'), decision.get('candidate_id')
        if mid not in by_mid or name not in by_mid[mid].get('candidate_ids', []):
            continue
        if (focus_enabled and visual_scores.get((mid, name), 0) <= 0
                and name not in by_mid[mid].get('explicit_ids', [])
                and len(by_mid[mid]['candidate_ids']) != 1):
            rejected_decisions.append({**decision, 'rejection_reason': 'no_evidence_for_this_occurrence'})
            continue
        chosen_by_mid.setdefault(mid, []).append(name)
    for mid, names in chosen_by_mid.items():
        mention = by_mid[mid]
        count = mention.get('count')
        if count == 1 and len(names) > 1:
            unique = list(dict.fromkeys(names))
            def rank(name):
                score = visual_scores.get((mid, name), 0.0)
                candidate = ctx.candidate(name)
                if candidate is None or not all(isinstance(v, (int, float)) for v in gp):
                    distance = float('inf')
                else:
                    distance = sum((getattr(candidate, axis) - value) ** 2
                                   for axis, value in zip(('x_world', 'y_world', 'z_world'), gp)) ** .5
                return (-score, distance, name)
            chosen_by_mid[mid] = [min(unique, key=rank)]
        elif isinstance(count, int) and count > 0:
            chosen_by_mid[mid] = list(dict.fromkeys(names))[:count]
    if chosen_by_mid or focus_enabled:
        selected = list(dict.fromkeys(name for names in chosen_by_mid.values() for name in names))
    if focus_enabled:
        decisions = [d for d in decisions if isinstance(d, dict)
                     and d.get('candidate_id') in chosen_by_mid.get(d.get('mention_id'), [])]
    return {'selected_object_names': list(dict.fromkeys(selected)),
            'decisions': decisions,
            'temporal_focus': focus_enabled, 'rejected_decisions': rejected_decisions,
            'confidence': raw.get('confidence', 0) if isinstance(raw, dict) else 0,
            'raw': raw, 'source': 'instruction_visual_spatial_global_coordinator'}


def hand_cues(backend, ctx, referential, paths, frames):
    prompt = '''Read the real chronological first-person RGB frames. Describe ONLY visible hand cues:
pointing direction in the image, contact/manipulation, or no visible association. Match a cue to
language mention IDs only when visually plausible. No candidate selection and no world coordinates.
Never use gaze markers or telemetry ray-hit points. A pointing hand does not prove a target identity.
Return JSON {"cues":[{"frame_id":"F1", "mention_ids":["m0"], "cue_type":"pointing|contact|none",
"confidence":0.0, "explanation":"brief visible evidence"}]}. Use only supplied frame/mention IDs.
''' + json.dumps({'instruction':ctx.text,'mentions':referential['mentions'],
                 'frames':[{'frame_id':f['frame_id'],'time':f['time']} for f in frames]},ensure_ascii=False)
    raw=backend.ask(sample_key(ctx),'v3_hand_observation',prompt,paths)
    allowed_frames={f['frame_id'] for f in frames}; allowed_mentions={m['mention_id'] for m in referential['mentions']}
    valid=[];rejected=[]
    for cue in raw.get('cues',[]):
        if cue.get('frame_id') not in allowed_frames or not isinstance(cue.get('mention_ids'),list):
            rejected.append(cue);continue
        try: confidence=float(cue.get('confidence',0))
        except (TypeError,ValueError): rejected.append(cue);continue
        if not math.isfinite(confidence) or not 0<=confidence<=1:
            rejected.append(cue);continue
        valid.append({**cue,'mention_ids':[m for m in cue['mention_ids'] if m in allowed_mentions]})
    # Hand is an auxiliary observation for the visual reasoner.  It is
    # deliberately not emitted as candidate evidence, so the hand stream
    # cannot create an independent target vote.
    return {'cues':valid,'evidence':[],'rejected':rejected,
            'rightIndexFingerRayHitPoint_used':False,
            'role':'visual_conditioning_only'}


def visual_evidence(backend,ctx,referential,paths,frames,plan,cues=None,use_gaze=False,use_regions=False,
                    isolate_mentions=False, mention_paths=None, focus_context=None):
    task = '''Resolve EVERY language mention against its compatible closed-set candidates using the
chronological first-person video views. Each input image preserves the original full RGB on the left.
Any additional panels are attention views or visible region proposals. Evaluate all language objects,
including actors, targets, destinations and concrete relation anchors. Do not restrict the entire
instruction to one target. Distinguish count from numeric candidate suffixes.
The instruction is the semantic source. Event/phase windows are soft search hints, NOT
speech timestamps or restrictions. ASR hints only add observations; consider ALL supplied
frames for every mention. Repeated "that chair/desk" phrases may denote different instances.
For here/there, identify a destination region from the preceding action, temporal visual
context, gaze and visible gestures, then select a scene anchor only if that region supports
it. Its broad candidate pool is not evidence that any specific object is the destination.
Absence of a visible hand does not exclude a target supported by other visual/gaze evidence.
Compare same-class instances explicitly; identical category-level descriptions cannot
establish a numbered instance. Provide at most two discriminative evidence rows per candidate,
not repeated identical rows for every frame. Keep explanations short.
Approximate projected candidate centers describe coordinates in the ORIGINAL RGB panel, not the full
composite image. They are unverified identity hints: verify appearance and relations; do not assume
 nearest projection equals identity. Use projected centers only to compare same-class
instances after checking RGB evidence; never select a candidate solely from a coordinate
distance. Missing evidence should remain unresolved.
Some views include a bottom strip of projected candidate crops labelled with canonical IDs. This strip
is a comparison aid only: verify each crop against the full RGB view and treat projection errors as
possible. It is especially useful for distinguishing repeated chairs, desks, PCs and buildings.
Return JSON {"evidence":[{"mention_id":"m0","candidate_id":"allowed ID",
"support_score":0.0,"confidence":0.0,"frame_id":"F1 or null","explanation":"brief observable basis"}]}.
Use ONLY supplied candidate IDs and frame IDs. Positive support MUST cite a supplied frame.
Do not fabricate IDs for unsupported types. Negative/missing observations use score=confidence=0.
'''
    if use_gaze:
        task += '''Gaze heatmaps and crops are SOFT attention hints. Looking at an obstacle or background
is possible. Zero heatmap overlap never excludes a language object. Preserve full-frame context.
'''
    else:
        task += 'Ignore any baked-in gaze markers; no gaze evidence is enabled.\n'
    if use_regions:
        task += '''Region boxes and SAM masks are category-level visible proposals, not candidate IDs.
Track IDs are provisional adjacent-frame matches; disappearance is not proof of occlusion or identity.
Use multiple frames when obstacles hide an object; never invent its hidden appearance.
'''
    if cues is not None:
        task += '''Hand observations are auxiliary visual cues, not ground truth or an independent vote.
Verify each observation in RGB and combine it with language, appearance and relation context.
'''
    else:
        task += 'Ignore hand gestures for this ablation.\n'
    if focus_context:
        task = task.replace('ASR hints only add observations; consider ALL supplied\nframes for every mention.',
                            'Use the temporal_focus roles to distinguish primary evidence from context.')
        task = task.replace('"explanation":"brief observable basis"',
                            '"explanation":"brief observable basis","temporal_justification":"basis for linking the frame to this occurrence"')
        task += ('\nEach image has an ID banner. Cite that ID, never the ordinal position in this call. '
                 'Projected UVs are relative to the RGB below the banner. A time hint does not '
                 'prove that a depicted action occurs during speech. Do not invent visible manipulation.\n')
    frame_meta=[]
    for f in frames:
        frame_meta.append({k:f[k] for k in ['frame_id','time','mention_ids','approx_candidate_uv',
                                             'candidate_panel_candidates','regions'] if k in f})
        if use_gaze: frame_meta[-1]['gaze_status']=f.get('gaze',{}).get('status','unavailable')
    by_id={m['mention_id']:m for m in referential['mentions']};by_frame={f['frame_id']:f for f in frames}
    payloads = []
    if isolate_mentions:
        # One closed-set call per mention. The model sees only this mention's
        # candidate table and projected UVs, preventing unrelated instances
        # from competing inside the same decision.
        for mention in referential['mentions']:
            mid = mention['mention_id']
            allowed = set(mention['candidate_ids'])
            isolated_frames = []
            for frame in frames:
                isolated_frames.append({
                    'frame_id': frame.get('frame_id'), 'time': frame.get('time'),
                    'temporal_role': ('primary' if frame.get('frame_id') in focus_context['primary_frame_ids']
                                      else 'context') if focus_context else 'event',
                    'mention_ids': frame.get('mention_ids', []),
                    'approx_candidate_uv': {n: uv for n, uv in frame.get('approx_candidate_uv', {}).items()
                                            if n in allowed},
                    'gaze_status': frame.get('gaze', {}).get('status', 'unavailable')})
            table = [{'candidate_id': c.candidate_id, 'x_world': c.x_world,
                      'y_world': c.y_world, 'z_world': c.z_world}
                     for c in ctx.candidates if c.candidate_id in allowed]
            hand = cues
            if isinstance(cues, dict):
                hand = {**cues, 'cues': [cue for cue in cues.get('cues', [])
                                         if mid in cue.get('mention_ids', [])]}
            isolated_prompt = task + ('\nThis is an isolated decision for exactly one mention. '
                'Ignore all other mentions and candidates. The candidate table and projected UV '
                'fields below contain only this mention\'s closed-set candidates. Respect the '
                'mention count; a singular mention has at most one selected candidate.\n')
            if focus_context:
                isolated_prompt += ('The supplied images are a local-first view. Treat primary '
                    'frames as the strongest temporal evidence and context frames as secondary. '
                    'Do not use a context frame to override a clear primary-frame observation. '
                    'Use highlighted_instruction and action_context to identify the CURRENT occurrence, '
                    'not another identical phrase. Soft phase hints are not measured speech times.\n')
                if focus_context['stage'] == 'local':
                    isolated_prompt += ('Positive evidence must cite a PRIMARY frame. Context frames '
                        'help interpret the scene but cannot establish the current target. Return no '
                        'positive evidence if primary frames are insufficient; a full-event recovery follows.\n')
                else:
                    isolated_prompt += ('Local evidence was insufficient. Search the full event now. '
                        'For positive evidence outside primary frames, include temporal_justification '
                        'explaining why it belongs to THIS occurrence rather than another action.\n')
            isolated_prompt += json.dumps({
                'instruction': ctx.text, 'mention': mention,
                'frames': isolated_frames,
                'mention_window': plan['mention_windows'].get(mid),
                'temporal_focus': focus_context,
                'hand_observations': hand,
                'candidate_world_table': table,
            }, ensure_ascii=False)
            image_paths = [mention_paths.get(f['frame_id'], f['image_rgb']) for f in frames] if mention_paths else paths
            payloads.append(backend.ask(sample_key(ctx), f'v3_visual_mention_{mid}', isolated_prompt, image_paths))
    else:
        prompt=task+json.dumps({'instruction':ctx.text,'mentions':referential['mentions'],
                                'frames':frame_meta,'mention_windows':plan['mention_windows'],
                                'hand_observations':cues},ensure_ascii=False)
        payloads.append(backend.ask(sample_key(ctx),'v3_visual',prompt,paths))
    rows=[];rejected=[]
    for payload in payloads:
      for row in payload.get('evidence',[]):
        mid,name=row.get('mention_id'),row.get('candidate_id')
        if mid not in by_id or name not in by_id[mid]['candidate_ids']:
            rejected.append({'row':row,'reason':'outside_closed_candidate_set'});continue
        try: score,conf=float(row.get('support_score',0)),float(row.get('confidence',0))
        except (ValueError,TypeError):
            rejected.append({'row':row,'reason':'invalid_score'});continue
        if not all(math.isfinite(v) and 0<=v<=1 for v in [score,conf]):
            rejected.append({'row':row,'reason':'invalid_score'});continue
        fid=row.get('frame_id');frame=by_frame.get(fid)
        if score*conf>0 and frame is None:
            rejected.append({'row':row,'reason':'positive_evidence_missing_frame'});continue
        if focus_context and score*conf > 0 and fid not in focus_context['primary_frame_ids']:
            if focus_context['stage'] == 'local':
                rejected.append({'row':row,'reason':'context_only_local_evidence'});continue
            if not isinstance(row.get('temporal_justification'), str) or not row['temporal_justification'].strip():
                rejected.append({'row':row,'reason':'expanded_evidence_without_occurrence_justification'});continue
        rows.append({**row,'support_score':score,'confidence':conf,
                     'temporal_range':[frame['time'],frame['time']] if frame else None,
                     'cue_type':'conditioned_visual_mention_isolated' if isolate_mentions else 'conditioned_visual'})
    warnings = [p.get('_parse_warning') for p in payloads if p.get('_parse_warning')]
    return {'evidence':rows,'rejected_evidence':rejected,'frames':frames,
            'parse_warning':warnings or None,
            'source':'instruction_event_conditioned_visual','world_distance_gaze_used':False,
            'hand_vote_used':False,'mention_isolation':isolate_mentions}
