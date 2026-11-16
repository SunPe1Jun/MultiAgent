#!/usr/bin/env python3
"""Post-freeze exact-set diagnosis and paired uncertainty for the 32-event study."""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from MultiAgent.full_interaction_referring import paired_set_metrics
from MultiAgent.camera_geometry import project_world
from MultiAgent.heatmap_temporal_pilot import save, sha
from MultiAgent.referent_rgb_selection import assemble
from tools.run_referent_rgb_expansion import OUTPUT, check_study, read


def counts(gt, prediction):
    gt, prediction = set(gt), set(prediction)
    return [len(gt & prediction), len(prediction - gt), len(gt - prediction), int(gt == prediction)]


def paired_stats(before, after, indices):
    before, after = np.asarray(before), np.asarray(after)
    def score(rows):
        total = rows.sum(axis=-2)
        denominator = 2*total[...,0] + total[...,1] + total[...,2]
        micro = np.divide(2*total[...,0], denominator, out=np.zeros_like(denominator,dtype=float), where=denominator>0)
        exact = rows[...,3].mean(axis=-1)
        return micro, exact
    b, a = score(before), score(after)
    bs, ats = score(before[indices]), score(after[indices])
    return {metric: {'difference':float(av-bv),
                     'interval_95':np.quantile(sample_after-sample_before,[.025,.975]).tolist()}
            for metric,bv,av,sample_before,sample_after in zip(['micro_f1','exact_set'],b,a,bs,ats)}



def transition_counts(events, condition, before, after):
    result = Counter()
    changes = []
    for event in events:
        old = event['predictions'][before][condition]
        new = event['predictions'][after][condition]
        old_exact = set(old) == set(event['gt'])
        new_exact = set(new) == set(event['gt'])
        result['exact_gained'] += int(new_exact and not old_exact)
        result['exact_lost'] += int(old_exact and not new_exact)
        result['set_changed'] += int(set(old) != set(new))
        if set(old) != set(new):
            changes.append({'id': event['id'], 'before': old, 'after': new,
                            'gt': event['gt'], 'exact_before': old_exact, 'exact_after': new_exact})
    return {**dict(result), 'changed_events': changes}


def visual_audit(responses, manifest):
    by_event = {event['id']: event for event in manifest['events']}
    output = {}
    for condition in manifest['conditions']:
        sources, reasons, panels, unresolved = Counter(), Counter(), Counter(), Counter()
        totals = Counter()
        repeated = []
        for request in manifest['requests']:
            if request['condition'] != condition:
                continue
            response = responses[request['id']]
            event = by_event[request['event_id']]
            frames = {frame['frame_id']: frame for frame in event['frames']}
            mentions = {mention['mention_id']: mention for mention in event['mention_hints']}
            seen = {}
            for row in response['final_validation']['rows']:
                if row['status'] == 'unresolved':
                    unresolved[row['unresolved_reason']] += 1
                    totals['non_concrete_reason_on_lexically_concrete_mention'] += int(
                        row['unresolved_reason'] == 'non_concrete_reference' and bool(mentions[row['mention_id']].get('candidate_ids')))
                for instance in row['instances']:
                    totals['legal_visual_observations'] += 1
                    panels[str(instance['source_panel'])] += 1
                    totals['observation_in_unsupported_source_frame'] += int(not frames[instance['frame_id']]['geometry']['clock_available'])
                    point = (instance['frame_id'], instance['x'], instance['y'])
                    seen.setdefault(point, set()).add(row['mention_id'])
            for point, mids in seen.items():
                if len(mids) > 1:
                    repeated.append({'event': event['id'], 'frame_x_y': list(point), 'mention_ids': sorted(mids)})
            for decision in response['stages']['recovered']['mention_decisions'].values():
                sources[decision['selection_source']] += 1
                totals['mention_rows'] += 1
                totals['cardinality_conflict'] += int(decision['cardinality_conflict'])
                totals['unresolved_after_assembly'] += int(not decision['selected'])
                for decoded in decision['decisions']:
                    reasons[decoded['reason']] += 1
        output[condition] = {'totals': dict(totals), 'source_panels': dict(panels),
                             'selection_sources': dict(sources), 'decode_reasons': dict(reasons),
                             'model_unresolved_reasons': dict(unresolved),
                             'same_point_across_mentions': repeated,
                             'point_correctness': 'Not assessed: shared points may be legitimate coreference; no independent pixel/object labels.'}
    return output


def report(output):
    m = check_study(output)
    a = read(output/'analysis.json')
    frozen = read(output/'predictions_frozen.json')
    responses = {r['id']:read(output/'responses'/f"{r['id']}.json") for r in m['requests']}
    for key in responses:
        if sha(output/'responses'/f'{key}.json') != frozen['response_sha256'][key]:
            raise ValueError('Frozen responses changed')
    events = a['events']
    by_event = {e['id']:e for e in m['events']}
    rng=np.random.default_rng(202611091)
    indices=rng.integers(0,len(events),size=(5000,len(events)))
    contrasts={}
    for stage in m['stages']:
        treatment=[counts(e['gt'],e['predictions'][stage]['GAZE_RGB']) for e in events]
        contrasts[stage]={control:paired_stats(
            [counts(e['gt'],e['predictions'][stage][control]) for e in events],treatment,indices)
            for control in ['FULL_RGB','UNIFORM_RGB','MIRRORED_GAZE_RGB']}
    recovery_contrasts={condition:paired_stats(
        [counts(e['gt'],e['predictions']['language_assembled'][condition]) for e in events],
        [counts(e['gt'],e['predictions']['recovered'][condition]) for e in events],indices)
        for condition in m['conditions']}
    language_rows=[]
    failures={}
    per_scene={}
    sources={}
    rows_review=[]
    for condition in m['conditions']:
        failures[condition]=Counter()
        sources[condition]=Counter()
        for event in events:
            prediction=event['predictions']['recovered'][condition]
            tp,fp,fn,exact=counts(event['gt'],prediction)
            category='exact' if exact else 'missing_only' if fp==0 else 'extra_only' if fn==0 else 'missing_and_extra'
            failures[condition][category]+=1
            response=responses[f"{event['id']}_{condition}"]
            sources[condition].update(d['selection_source'] for d in response['stages']['recovered']['mention_decisions'].values())
        per_scene[condition]={}
        for scene in ['scene1','scene2','scene3','scene4','scene5']:
            subset=[e for e in events if by_event[e['id']]['scene_key'].split('_')[0]==scene]
            per_scene[condition][scene]=paired_set_metrics([{'gt':e['gt'],'prediction':e['predictions']['recovered'][condition]} for e in subset])
    for event in events:
        definition=by_event[event['id']]
        prior=assemble(definition,{'rows':[]},lexical=True)
        language_rows.append({'gt':event['gt'],'prediction':prior['selected_object_names']})
        union=set(n for mention in definition['mention_hints'] for n in mention.get('candidate_ids',[]))
        candidates = {c['candidate_id']: c for c in definition['scene_candidate_table']}
        identity_support = {}
        for name in event['gt']:
            candidate = candidates.get(name)
            projected_frames = []
            if candidate is not None:
                for frame in definition['frames']:
                    if frame['geometry']['clock_available']:
                        point = project_world(candidate['position_world'], frame['geometry']['reference_pose'], *frame['source_size'])
                        if point['in_frustum']:
                            projected_frames.append(frame['frame_id'])
            identity_support[name] = {
                'in_full_scene_candidate_table': candidate is not None,
                'literal_compatible_mentions': [h['mention_id'] for h in definition['mention_hints'] if name in h.get('candidate_ids', [])],
                'supported_in_frustum_anchor_frames': projected_frames,
                'visibility_or_pixel_correctness_verified': False}
        rows_review.append({'id':event['id'],'gt':event['gt'],'language_prior':prior['selected_object_names'],
            'posthoc_gt_identity_support': identity_support,
            'gt_outside_lexical_union':sorted(set(event['gt'])-union),
            'has_visual_only_deictic_scope':any(not h.get('candidate_ids') and h.get('kind') in ('location','region') for h in definition['mention_hints'])})
    summary={'manifest_sha256':sha(output/'manifest.json'),'paired_gaze_contrasts':contrasts,
        'paired_recovery_contrasts':recovery_contrasts,'language_prior_set_metrics':paired_set_metrics(language_rows),
        'exact_failure_types':{c:dict(v) for c,v in failures.items()},'per_scene_recovered':per_scene,
        'selection_sources':{c:dict(v) for c,v in sources.items()},'language_audit':rows_review,
        'bootstrap':{'replicates':5000,'unit':'interaction, conditions paired','scope':'Exploratory local holdout addresses; small scene-stratified sample, not population equivalence.'}}
    summary['stage_transitions'] = {
        condition: {
            'language_assembly': transition_counts(events, condition, 'visual_only', 'language_assembled'),
            'temporal_recovery': transition_counts(events, condition, 'language_assembled', 'recovered')}
        for condition in m['conditions']}
    summary['support_strata_recovered'] = {
        condition: {label: paired_set_metrics([
            {'gt': e['gt'], 'prediction': e['predictions']['recovered'][condition]}
            for e in events if bool(e['supported_frames']) == supported])
            for label, supported in [('supported', True), ('fallback', False)]}
        for condition in m['conditions']}
    summary['visual_output_audit'] = visual_audit(responses, m)
    canonical = [responses[r['id']] for r in m['requests'] if r['id'] == r['canonical_request_id']]
    summary['cost'] = {
        stage: {'records': len(calls), 'model_calls_attempted': sum('supplied_frame_ids' in call for call in calls),
                'seconds': sum(call.get('seconds', 0) for call in calls),
                'expanded_video_tokens': sum(call.get('processor_audit', {}).get('expanded_video_token_count', 0) for call in calls),
                'failures': sum(bool(call.get('failure')) for call in calls),
                'skipped_parser_failure': sum(bool((call.get('failure') or {}).get('parser_failure')) for call in calls)}
        for stage in ['initial', 'recovery']
        for calls in [[r[stage] for r in canonical if r.get(stage)]]}
    summary['logical_trial_cost_per_condition'] = {
        condition: {
            'initial_model_trials': sum('supplied_frame_ids' in r['initial'] for r in trials),
            'recovery_model_trials': sum(r['recovery'] is not None for r in trials),
            'initial_video_tokens': sum(r['initial'].get('processor_audit', {}).get('expanded_video_token_count', 0) for r in trials),
            'recovery_video_tokens': sum((r.get('recovery') or {}).get('processor_audit', {}).get('expanded_video_token_count', 0) for r in trials),
            'note': 'Includes identical-input aliases; physical executed calls are in cost. Recovery budgets are equal caps, not equal realized calls.'}
        for condition in m['conditions']
        for trials in [[responses[r['id']] for r in m['requests'] if r['condition'] == condition]]}
    language_audit_by_event = {row['id']: row for row in rows_review}
    provenance = {}
    for condition in m['conditions']:
        condition_errors = []
        for event in events:
            response = responses[f"{event['id']}_{condition}"]
            decisions = response['stages']['recovered']['mention_decisions']
            predicted = set(event['predictions']['recovered'][condition])
            truth = set(event['gt'])
            definition = by_event[event['id']]
            support = language_audit_by_event[event['id']]['posthoc_gt_identity_support']
            missing = []
            for name in sorted(truth-predicted):
                mids = support[name]['literal_compatible_mentions']
                missing.append({'id': name, **support[name],
                    'compatible_mention_decisions': {mid: {
                        'status': decisions[mid].get('status'),
                        'unresolved_reason': decisions[mid].get('unresolved_reason'),
                        'selection_source': decisions[mid]['selection_source'],
                        'selected': decisions[mid]['selected'],
                        'decode_reasons': [d['reason'] for d in decisions[mid]['decisions']]}
                        for mid in mids},
                    'additional_deictic_search_available': language_audit_by_event[event['id']]['has_visual_only_deictic_scope']})
            extra = [{'id': name,
                'selected_by_mentions': [mid for mid,d in decisions.items() if name in d['selected']],
                'selection_sources': sorted({d['selection_source'] for d in decisions.values() if name in d['selected']})}
                for name in sorted(predicted-truth)]
            if missing or extra:
                condition_errors.append({'event': event['id'], 'parser_valid': definition['parser_valid'],
                                         'missing': missing, 'extra': extra})
        provenance[condition] = condition_errors
    summary['error_provenance'] = provenance
    save(output/'summary.json',summary)
    lines=['# Referent-aware RGB / Exact-set — 2026-11-09','',
        f"{len(events)} selected interactions; {len(m['requests'])} condition results; {a['unique_initial_inputs']} unique initial inputs. All predictions frozen before GT.",'',
        '| Stage | Condition | TP/FP/FN | Micro-F1 | Exact-set | Any-hit |',
        '|---|---|---|---:|---:|---:|']
    for stage,cs in a['stages'].items():
        for condition,r in cs.items():
            q=r['set_metrics']
            lines.append(f"| {stage} | {condition} | {q['tp']}/{q['fp']}/{q['fn']} | {100*q['micro_f1']:.2f}% | {q['exact_set_count']}/{q['events']} | {q['any_hit_count']}/{q['events']} |")
    prior=summary['language_prior_set_metrics']
    lines+=['',f"Instruction/scene-language prior without RGB localization: Micro-F1 {100*prior['micro_f1']:.2f}%; Exact {prior['exact_set_count']}/{prior['events']}. This uses the shared model language parse and explicit/unique lexical identities; it is not a zero-model baseline.",'',
        'Final GAZE_RGB minus paired controls (5000 interaction resamples):','']
    for control,r in contrasts['recovered'].items():
        parts=[]
        for metric,q in r.items():
            lo,hi=q['interval_95'];parts.append(f"{metric} {100*q['difference']:+.2f} pp [{100*lo:.2f}, {100*hi:.2f}]")
        lines.append('- '+control+': '+'; '.join(parts))
    lines+=['','Exact-set failure types:','','```json',json.dumps(summary['exact_failure_types'],indent=2),'```','',
        'The visual_only stage still uses the same language/phase prompt and lexical compatible pools; its name means a legal visual point and geometric identity are required. language_assembled adds deterministic old language rules. recovered has one extra model pass and its cost is recorded separately. These stages do not isolate the causal effect of the new prompt or soft temporal hints.','',
        'Scene4 occupies half the sample. See per_scene_recovered and the language-only prior before attributing changes to visual attention. Raw recorder marks remain; absolute pixel/object identity geometry is uncalibrated; no default agent replacement.','',
        'Artifacts: [viewer](index.html), [exact-set review](exact_set_review.html), [summary](summary.json), [all stage results](analysis.json), [input verification](input_verification.json), [prediction freeze](predictions_frozen.json).','']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    # A compact post-inference page with the per-mention identity ledger and
    # source RGB points. These review images are never used as model inputs.
    page=['<!doctype html><meta charset="utf-8"><title>Exact-set review</title>',
        '<style>body{font:16px sans-serif;background:#14202b;color:#eee;margin:24px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #567}pre{white-space:pre-wrap}img{max-width:600px;width:100%}a{color:#8de}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}</style>',
        '<h1>Exact-set / per-mention review</h1><p><a href="index.html">All source frames</a> · <a href="RESULTS.md">Results</a>. Source point overlays are post-inference review only.</p>']
    for event in events:
        definition=by_event[event['id']]
        page+=['<h2>'+html.escape(event['id'])+'</h2><p>'+html.escape(definition['instruction'])+'</p>',
               '<p>Final-set GT: '+html.escape(', '.join(event['gt']))+'</p><div class=grid>']
        for condition in m['conditions']:
            response=responses[f"{event['id']}_{condition}"]
            final=response['stages']['recovered']
            page+=['<div><h3>'+condition+'</h3><p>Prediction: '+html.escape(', '.join(final['selected_object_names']))+'</p>',
                   '<p>Missing: '+html.escape(', '.join(sorted(set(event['gt'])-set(final['selected_object_names']))))+
                   ' · Extra: '+html.escape(', '.join(sorted(set(final['selected_object_names'])-set(event['gt']))))+'</p>',
                   '<table><tr><th>Mention</th><th>ID choice</th><th>Source</th></tr>']
            for mention in definition['mention_hints']:
                decision=final['mention_decisions'][mention['mention_id']]
                page+=['<tr><td>'+html.escape(mention['mention_id']+': '+mention['text'])+'</td><td>'+html.escape(', '.join(decision['selected']))+'</td><td>'+html.escape(decision['selection_source'])+'</td></tr>']
            page+=['</table>']
            points={}
            for row in response['final_validation']['rows']:
                for instance in row['instances']:
                    points.setdefault(instance['frame_id'],[]).append((row['mention_id'],instance['x_source'],instance['y_source']))
            for fid,observations in sorted(points.items())[:4]:
                frame=next(f for f in definition['frames'] if f['frame_id']==fid)
                with Image.open(frame['source_path']) as image:rgb=image.convert('RGB')
                draw=ImageDraw.Draw(rgb)
                for mid,x,y in observations:
                    px,py=x*rgb.width,y*rgb.height
                    draw.line((px-7,py,px+7,py),fill='#ffdf48',width=3)
                    draw.line((px,py-7,px,py+7),fill='#ffdf48',width=3)
                    draw.text((px+8,py+3),mid,fill='white',stroke_width=1,stroke_fill='black',font=ImageFont.load_default(size=16))
                target=output/'review_rgb'/event['id']/condition/f'{fid}.jpg'
                target.parent.mkdir(parents=True,exist_ok=True);rgb.save(target,quality=90)
                page+=['<details><summary>'+fid+' · selected source points</summary><img loading=lazy src="'+html.escape(str(target.relative_to(output)))+'"></details>']
            page+=['</div>']
        page+=['</div>']
    (output/'exact_set_review.html').write_text('\n'.join(page))
    print(json.dumps({'final':{c:r['set_metrics'] for c,r in a['stages']['recovered'].items()},
        'language_prior':summary['language_prior_set_metrics'],'contrasts':contrasts['recovered']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    report(parser.parse_args().output.resolve())
