#!/usr/bin/env python3
"""Counterfactual cue usage and conditional real referent-center diagnostics.

Known injected cues test graphical association, NOT actual referent accuracy.
Real labels are annotation-conditioned, weak center proxies, NOT pixel GT.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import random
import shutil
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import binary_erosion
from MultiAgent.attention_reliability import (VERSION,distinct_masks,interior_point,injected_history,
    pair_attention,graphical_pair_heat,cluster_paired_interval)
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.perspective_attention_study import ROOT,read
from MultiAgent.qwen_backend import QwenBackend
from tools.run_object_attention_study import OUTPUT as SOURCE

OUTPUT=ROOT/'.local-sync/attention_reliability_20261106'
LAYOUTS=['SEPARATE','REGISTERED_RGB']
CONTROL_CONDITIONS=['FORWARD','REVERSED','NO_HEAT']
REAL_CONDITIONS=['TRUE_HEAT','SWAP_PAIR','NO_HEAT']
CUE_PROMPT='''Read the three graphical attention panels EARLY, MIDDLE, LATE for the two outlined proposed object regions A and B.
The full RGB scene and enlarged A/B RGB details identify the same two proposed regions. The three smaller panels at the RIGHT use exactly that full-scene camera orientation. Printed A/B IDs are registered to their regions. Panels can have dark or RGB backgrounds. WHITE/GRAY outlines and the original RGB colors do NOT show attention. Only the added ORANGE fill shows attention assigned to a region; BLUE is unassigned gaze outside both proposals. A whole-region fill represents an attention association, not a target label or proof that every pixel was watched.
Return the region with more added ORANGE attention in EARLY and then in LATE. Return none for a time panel if there is no added orange on either region. Ignore object attractiveness and instruction guesses. IDs/positions can change. These are controlled attention graphics, not real referent labels.
Return JSON only, exactly {"early":"A","late":"B"}. Each value must be A, B or none.'''
REAL_PROMPT='''The original instruction can refer to several objects. For this question consider ONLY the TWO outlined __CATEGORY__ proposals with printed IDs A and B.
Which of these two concrete proposals is a referent or spatial reference in the instruction? The full RGB scene and enlarged A/B details identify the same proposed objects. Other mentioned categories may appear in context but are not selectable. IDs are local addresses, not object names. The candidate pair may be incomplete or visually ambiguous; choose unclear if the evidence cannot identify one.
The three panels at the RIGHT show EARLY / MIDDLE / LATE within the supported two-second gaze history preceding this frame, registered to the full RGB scene. Added ORANGE is attention assigned to each object's mask; BLUE retains gaze outside both candidates. WHITE/GRAY boundaries, printed IDs and original RGB colors are not attention. Missing or tiny heat is missing attention evidence, not evidence against the object. The graphics may include controls. Gaze can land on background, hands, references or a different object, so judge RGB and language first and use attention as a weak disambiguating cue. Do not automatically select the hottest object. Do not invent hidden objects.
Return JSON only, exactly {"selected":"A"}. selected must be A, B or unclear. No scores or explanations.
ORIGINAL INSTRUCTION: __INSTRUCTION__'''


def font(size):return ImageFont.load_default(size=size)


def crop_object(rgb,box,size=264):
    w,h=rgb.size;x0,y0,x1,y1=box;dx=.15*(x1-x0);dy=.15*(y1-y0)
    b=[max(0,int(x0-dx)),max(0,int(y0-dy)),min(w,math.ceil(x1+dx)),min(h,math.ceil(y1+dy))]
    crop=rgb.crop(b);ratio=min(size/crop.width,size/crop.height)
    crop=crop.resize((max(1,round(crop.width*ratio)),max(1,round(crop.height*ratio))),Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(size,size),'#222a30');canvas.paste(crop,((size-crop.width)//2,(size-crop.height)//2));return canvas


def render(rgb,objects,masks,mass,residual,labels,layout):
    width,height=rgb.size;image=Image.new('RGB',(1008,1008),'#15212b');draw=ImageDraw.Draw(image)
    draw.text((16,8),'OBJECT REGIONS + REGISTERED ATTENTION',fill='white',font=font(22))
    context=rgb.copy();cd=ImageDraw.Draw(context)
    for j,obj in enumerate(objects):
        cd.rectangle(obj['box'],outline='white',width=3);x,y=interior_point(masks[j])
        cd.text((x,y),labels[j],fill='white',stroke_width=2,stroke_fill='black',font=font(30))
    image.paste(context.resize((576,576),Image.Resampling.LANCZOS),(16,42))
    for label in ['A','B']:
        j=labels.index(label);x=16+288*(['A','B'].index(label))
        draw.text((x+8,632),label+' | RGB DETAIL',fill='white',font=font(22))
        image.paste(crop_object(rgb,objects[j]['box']),(x,670))
    orange,blue,meta=graphical_pair_heat(mass,masks,residual)
    for k,name in enumerate(['EARLY','MIDDLE','LATE']):
        x=660;y=62+k*312;draw.text((x,y-25),name+' ATTENTION',fill='white',font=font(18))
        base=np.asarray(rgb,dtype=float).copy() if layout=='REGISTERED_RGB' else np.full((height,width,3),20.,float)
        for mask in masks:
            edge=mask&~binary_erosion(mask);base[edge]=[95,95,95]
        a=orange[k,...,None];b=blue[k,...,None]
        base=base*(1-b)+np.array([45,160,255])*b
        base=base*(1-a)+np.array([255,85,25])*a
        panel=Image.fromarray(base.clip(0,255).astype(np.uint8));pd=ImageDraw.Draw(panel)
        for j in range(2):
            px,py=interior_point(masks[j]);pd.text((px,py),labels[j],fill='white',stroke_width=2,stroke_fill='black',font=font(28))
        image.paste(panel.resize((288,288),Image.Resampling.LANCZOS),(x,y))
    draw.text((16,974),'Only added ORANGE is assigned attention. Object identity and referent intent remain separate.',fill='white',font=font(13))
    return image,meta


def frame_source(event,frame):
    return {'event_id':event['id'],'event':event['event'],'frame_id':frame['frame_id'],
        'instruction':event['instruction'],'source_path':frame['source_path'],'source_sha256':frame['source_sha256'],
        'source_pts':frame['source_pts'],'source_npz_path':str(SOURCE/frame['npz_path']),
        'source_npz_sha256':frame['npz_sha256']}


def control_cases(parent):
    cases=[];excluded=[]
    for event in parent['events']:
        choices=[]
        for frame in event['frames']:
            data=np.load(SOURCE/frame['npz_path']);masks=data['masks']
            for category in sorted({o['category'] for o in frame['objects']}):
                eligible=[i for i,obj in enumerate(frame['objects']) if obj['category']==category and obj['score']>=.3 and masks[i].sum()>=150]
                for a in eligible:
                    for b in eligible:
                        if b<=a or not distinct_masks(masks[a],masks[b]):continue
                        score=min(int(masks[a].sum()),int(masks[b].sum()))
                        choices.append((score,frame['frame_id'],category,a,b,frame))
        if not choices:
            excluded.append({'event':event['event'],'reason':'No two disjoint same-category proposals with score>=.3 and mask area>=150 pixels.'});continue
        best=max(choices,key=lambda v:v[:5]);_,fid,category,a,b,frame=best
        cases.append({**frame_source(event,frame),'id':'cue_'+event['id'],'family':'injected_cue',
            'category':category,'indices':[a,b],'objects':[frame['objects'][a],frame['objects'][b]],
            'label_provenance':'Scripted forward/reverse attention mass; not semantic object or referent GT.',
            'annotation_conditioned_candidates':False})
    return cases,excluded


def real_cases(parent):
    audit=read(SOURCE/'referent_center_audit.json');cases=[];excluded=[]
    for event in parent['events']:
        for frame in event['frames']:
            refs=[r for r in audit['eligible_reference_frames'] if r['event']==event['event'] and r['frame_id']==frame['frame_id']]
            data=np.load(SOURCE/frame['npz_path']);masks=data['masks']
            for category in sorted({r['reference_category'] for r in refs}):
                candidates=[i for i,obj in enumerate(frame['objects']) if obj['category']==category]
                label={'event':event['event'],'frame_id':frame['frame_id'],'category':category}
                relevant=[r for r in refs if r['reference_category']==category]
                if len(relevant)!=1 or len(relevant[0]['mask_matching_ids'])!=1:
                    excluded.append({**label,'reason':'Needs exactly one in-frustum category referent and a unique same-category mask-center match.'});continue
                target=relevant[0]['mask_matching_ids'][0];a=next((i for i in candidates if frame['objects'][i]['id']==target),None)
                if a is None or masks[a].sum()<100:
                    excluded.append({**label,'reason':'Target mask absent or fewer than100 pixels.'});continue
                others=[i for i in candidates if i!=a and masks[i].sum()>=100 and distinct_masks(masks[a],masks[i])]
                if not others:
                    excluded.append({**label,'reason':'No disjoint same-category alternative with mask area>=100.'});continue
                ac=np.mean(frame['objects'][a]['box'][:2])+np.mean(frame['objects'][a]['box'][2:])
                center=lambda i:(np.array(frame['objects'][i]['box'][:2])+frame['objects'][i]['box'][2:])/2
                b=min(others,key=lambda i:(float(np.linalg.norm(center(i)-center(a))),i))
                cases.append({**frame_source(event,frame),'id':'real_'+event['id']+'_'+frame['frame_id']+'_'+category,
                    'family':'real_center_proxy','category':category,'indices':[a,b],
                    'objects':[frame['objects'][a],frame['objects'][b]],'target_physical_index':0,
                    'reference_name':relevant[0]['reference_name'],'reference_point_xy':relevant[0]['point_xy'],
                    'label_provenance':'Weak existing referent-center containment in one segmented proposal; no pixel calibration/visibility GT.',
                    'annotation_conditioned_candidates':True,
                    'candidate_policy':'Conditional closed-set diagnostic: target proposal plus nearest disjoint same-category proposal; gaze not used to choose pair.'})
    return cases,excluded


def prepare(output):
    if (output/'manifest.json').exists():raise ValueError('Never overwrite frozen inputs')
    output.mkdir(parents=True,exist_ok=True);parent=read(SOURCE/'manifest.json')
    control,cexcluded=control_cases(parent);real,rexcluded=real_cases(parent);cases=control+real;requests=[]
    for case in cases:
        source=np.load(case['source_npz_path']);masks=source['masks'][case['indices']]
        case_dir=output/'cases'/case['id'];case_dir.mkdir(parents=True,exist_ok=True)
        npz=case_dir/'pair.npz';np.savez_compressed(npz,masks=masks,raw_maps=source['maps'])
        case['pair_npz_path']=str(npz.relative_to(output));case['pair_npz_sha256']=sha(npz)
        with Image.open(case['source_path']) as im:rgb=im.convert('RGB')
        conditions=CONTROL_CONDITIONS if case['family']=='injected_cue' else REAL_CONDITIONS
        base_labels=list('AB');random.Random('attention-reliability:'+case['id']).shuffle(base_labels)
        for condition in conditions:
            if case['family']=='injected_cue':
                mass=np.zeros((3,2)) if condition=='NO_HEAT' else injected_history(masks,reverse=condition=='REVERSED')
                residual=np.zeros_like(source['maps']);links=None
            else:mass,residual,links=pair_attention(source['maps'],masks,condition)
            for permutation in (0,1):
                labels=base_labels if permutation==0 else base_labels[::-1]
                expected=({'early':'none','late':'none'} if condition=='NO_HEAT' else
                    {'early':labels[0 if condition=='FORWARD' else 1],'late':labels[1 if condition=='FORWARD' else 0]}) if case['family']=='injected_cue' else {'selected':labels[0]}
                prompt=CUE_PROMPT if case['family']=='injected_cue' else REAL_PROMPT.replace('__CATEGORY__',case['category']).replace('__INSTRUCTION__',case['instruction'])
                for layout in LAYOUTS:
                    rid=f"{case['id']}_{condition}_p{permutation}_{layout}"
                    image,meta=render(rgb,case['objects'],masks,mass,residual,labels,layout)
                    path=case_dir/f'{condition}_p{permutation}_{layout}.png';image.save(path)
                    requests.append({'id':rid,'case_id':case['id'],'event':case['event'],'family':case['family'],
                        'condition':condition,'permutation':permutation,'layout':layout,'labels_by_physical_region':labels,
                        'expected':expected,'path':str(path.relative_to(output)),'sha256':sha(path),'prompt':prompt,'display_support':meta})
        print('PREPARED',case['id'],flush=True)
    manifest={'version':VERSION,'source_manifest_sha256':sha(SOURCE/'manifest.json'),
        'source_center_audit_sha256':sha(SOURCE/'referent_center_audit.json'),'cases':cases,'requests':requests,
        'excluded_control_events':cexcluded,'excluded_real_frame_categories':rexcluded,'layouts':LAYOUTS,
        'control_scope':'Known injected attention in existing RGB proposals; tests causal cue association, not true referents or head-motion robustness.',
        'real_scope':'Annotation-conditioned same-category pairs from previously seen interactions. Weak center proxy, no human/pixel GT or unseen validation.',
        'evaluation':'Two ID/column permutations; same RGB/candidates/instruction within heat controls. Correct both directions for injected forward/reverse; null abstention. Real helpful/harmful with interaction-cluster intervals.',
        'fixed_acceptance':'Mechanism readiness: >=90% forward/reverse pairs both correct and >=90% null correct across frozen old cases. Real benefit requires independent object labels/unseen interactions; never promote on this weak proxy.',
        'no_all_anchor_projection':True,'code_sha256':sha(__file__),'core_sha256':sha(ROOT/'MultiAgent/attention_reliability.py')}
    save(output/'manifest.json',manifest);save(output/'protocol_lock.json',{'manifest_sha256':sha(output/'manifest.json'),
        'before_any_inference':True,'frozen_utc':datetime.now(timezone.utc).isoformat()})
    (output/'source').mkdir(exist_ok=True);shutil.copy2(__file__,output/'source/run_attention_reliability_study.py');shutil.copy2(ROOT/'MultiAgent/attention_reliability.py',output/'source/attention_reliability.py')
    report(output)


def valid_response(parsed,family):
    if family=='injected_cue':return set(parsed)=={'early','late'} and all(v in {'A','B','none'} for v in parsed.values())
    return set(parsed)=={'selected'} and parsed['selected'] in {'A','B','unclear'}


def infer(output,family='all'):
    m=read(output/'manifest.json');assert sha(output/'manifest.json')==read(output/'protocol_lock.json')['manifest_sha256']
    backend=QwenBackend(output/'model_cache',model='/ai/data/Qwen3-VL-8B-Instruct',local_only=True,
        namespace='paired_attention_reliability_v1',trace_visual_inputs=True)
    schedule=[r for r in m['requests'] if family=='all' or r['family']==family];random.Random(20261106).shuffle(schedule)
    for request in schedule:
        dest=output/'responses'/f"{request['id']}.json"
        if dest.exists():continue
        path=output/request['path'];assert sha(path)==request['sha256'];print('INFER',request['id'],flush=True);start=time.monotonic()
        parsed=backend.ask(request['id'],'paired_attention',request['prompt'],[path],max_new_tokens=100)
        raw=read(backend.requests[-1]);valid=valid_response(parsed,request['family'])
        physical={v:i for i,v in enumerate(request['labels_by_physical_region'])}
        physical_prediction=({k:physical.get(v,v) for k,v in parsed.items()} if valid else None)
        save(dest,{'id':request['id'],'case_id':request['case_id'],'event':request['event'],'family':request['family'],
            'condition':request['condition'],'permutation':request['permutation'],'layout':request['layout'],
            'parsed':parsed,'schema_valid':valid,'physical_prediction':physical_prediction,
            'matches_label':valid and parsed==request['expected'],'request_path':backend.requests[-1],
            'processor_audit':raw['processor_audit'],'vision_verified':raw['visual_tensor_trace']['input_consumption_verified'],
            'seconds':time.monotonic()-start})
        print('RESULT',valid,parsed,request['expected'],flush=True)
    report(output)


def report(output):
    m=read(output/'manifest.json');responses={p.stem:read(p) for p in (output/'responses').glob('*.json')}
    indexed={(r['case_id'],r['condition'],r['permutation'],r['layout']):responses.get(r['id']) for r in m['requests']}
    summary={'controlled_cases':sum(c['family']=='injected_cue' for c in m['cases']),
        'real_proxy_cases':sum(c['family']=='real_center_proxy' for c in m['cases']),
        'planned_calls':len(m['requests']),'completed_calls':len(responses),
        'all_schema_valid':all(r['schema_valid'] for r in responses.values()),
        'all_vision_verified':all(r['vision_verified'] for r in responses.values()),
        'visual_tokens':sorted({r['processor_audit']['expanded_image_token_count'] for r in responses.values()}),
        'mechanism':{},'real_weak_proxy':{},'unseen_validation':False,'final_referent_accuracy_available':False}
    for layout in LAYOUTS:
        rows=[r for r in responses.values() if r['family']=='injected_cue' and r['layout']==layout]
        groups=[];null=[];permutation_agree=[]
        for case in [c for c in m['cases'] if c['family']=='injected_cue']:
            for permutation in (0,1):
                a=indexed.get((case['id'],'FORWARD',permutation,layout));b=indexed.get((case['id'],'REVERSED',permutation,layout))
                if a and b:groups.append(a['matches_label'] and b['matches_label'])
                n=indexed.get((case['id'],'NO_HEAT',permutation,layout))
                if n:null.append(n['matches_label'])
            for condition in CONTROL_CONDITIONS:
                a=indexed.get((case['id'],condition,0,layout));b=indexed.get((case['id'],condition,1,layout))
                if a and b and a['schema_valid'] and b['schema_valid']:permutation_agree.append(a['physical_prediction']==b['physical_prediction'])
        summary['mechanism'][layout]={'calls':len(rows),'all_time_fields_correct':sum(r['matches_label'] for r in rows),
            'forward_reverse_pairs':len(groups),'both_directions_correct':sum(groups),'null_calls':len(null),'correct_null':sum(null),
            'id_permutation_pairs':len(permutation_agree),'same_physical_output':sum(permutation_agree),
            'engineering_readiness_met':bool(groups and null and np.mean(groups)>=.9 and np.mean(null)>=.9),
            'scope':'Injected display labels only, not referent accuracy.'}
        by_condition={};pair_rows=[]
        for condition in REAL_CONDITIONS:
            rr=[r for r in responses.values() if r['family']=='real_center_proxy' and r['layout']==layout and r['condition']==condition]
            by_condition[condition]={'calls':len(rr),'hits_weak_label':sum(r['matches_label'] for r in rr),
                'abstained':sum(r['parsed'].get('selected')=='unclear' for r in rr)}
        for case in [c for c in m['cases'] if c['family']=='real_center_proxy']:
            conditions={}
            for condition in REAL_CONDITIONS:
                rr=[indexed.get((case['id'],condition,p,layout)) for p in (0,1)]
                if all(rr):conditions[condition]=float(np.mean([r['matches_label'] for r in rr]))
            if len(conditions)==3:pair_rows.append({'case_id':case['id'],'cluster':case['event'],**conditions})
        contrasts={}
        for control in ['NO_HEAT','SWAP_PAIR']:
            rows=[{'cluster':r['cluster'],'baseline':r[control],'treatment':r['TRUE_HEAT']} for r in pair_rows]
            contrasts[control]={**cluster_paired_interval(rows),
                'helpful_cases':sum(r['TRUE_HEAT']>r[control] for r in pair_rows),
                'harmful_cases':sum(r['TRUE_HEAT']<r[control] for r in pair_rows),
                'unchanged_cases':sum(r['TRUE_HEAT']==r[control] for r in pair_rows)}
        summary['real_weak_proxy'][layout]={'by_condition':by_condition,'paired_cases':pair_rows,'contrasts':contrasts,
            'scope':'Selected annotation-conditioned same-category pairs, weak referent-center proxy; intervals cannot establish label truth or final accuracy.'}
    save(output/'analysis.json',summary)
    payload=json.dumps({'m':m,'responses':responses},ensure_ascii=False).replace('</','<\\/')
    template='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>热图证据可靠性</title><style>body{font:15px system-ui;background:#f3f5f7;margin:20px}header{position:sticky;top:0;background:#f3f5f7;padding:10px}select{font:inherit;margin:6px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:14px}figure{background:white;margin:0;padding:10px}img{width:100%}pre{white-space:pre-wrap}</style><header><h2>同类物体消歧：线索使用机制与真实指代弱代理分开</h2><select id="family"><option>injected_cue</option><option>real_center_proxy</option></select><select id="event"></select><select id="layout"><option>SEPARATE</option><option>REGISTERED_RGB</option></select></header><p>机制层：在真实RGB的两个提案区域注入已知早→晚注意，再反转、置空，并交换A/B编号和详情位置。它只验证模型是否使用了新增图形线索，不是用户真实referent。真实层：当前referent中心唯一落入mask的条件式二选一，候选集由标注构造，是未标定/可见性未知的弱代理，也不是完整pipeline或像素GT。所有交互均是历史样例。输出改变、格式合规都不直接记为收益。</p><p>橙色整区域填充是分配到mask的支持时长；蓝色保留未被两个候选承接的gaze。低于250ms的总承接时长会降低亮度，避免把极少噪声归一化成强线索。图形仍以图像输入，提示没有注意分数或答案。</p><main></main><script>const d=__DATA__,family=document.querySelector('#family'),q=document.querySelector('#event'),layout=document.querySelector('#layout');[...new Set(d.m.cases.map(c=>c.event))].forEach(e=>{let o=document.createElement('option');o.value=e;o.textContent=e;q.append(o)});function render(){let main=document.querySelector('main');main.replaceChildren();d.m.requests.filter(r=>r.event===q.value&&r.family===family.value&&r.layout===layout.value).forEach(r=>{let f=document.createElement('figure'),im=document.createElement('img'),t=document.createElement('pre');im.src=r.path;let z=d.responses[r.id];t.textContent=r.case_id+' '+r.condition+' p'+r.permutation+' | '+(z?JSON.stringify(z.parsed):'尚未运行')+' | 诊断标签 '+JSON.stringify(r.expected);f.append(im,t);main.append(f)})}family.onchange=render;q.onchange=render;layout.onchange=render;render();</script></html>'''
    (output/'index.html').write_text(template.replace('__DATA__',payload));print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','infer','report']);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--family',choices=['all','injected_cue','real_center_proxy'],default='all');a=p.parse_args()
    if a.action=='infer':infer(a.output.resolve(),a.family)
    else:{'prepare':prepare,'report':report}[a.action](a.output.resolve())
