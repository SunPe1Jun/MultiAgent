#!/usr/bin/env python3
"""Adaptive local-object heat layout: preserve graphical evidence, reduce lookup.

Started after preliminary full-scene layout failures. All cases are regression;
this is not untouched validation or a pure single-factor resolution ablation.
"""
import argparse
from collections import Counter
import io
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from MultiAgent.attention_reliability import injected_history,pair_attention,graphical_pair_heat,interior_point
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.perspective_attention_study import read
from MultiAgent.qwen_backend import QwenBackend
from tools.run_attention_reliability_study import OUTPUT,valid_response,CONTROL_CONDITIONS,REAL_CONDITIONS

CUE_PROMPT='''Compare the two proposed object regions A and B using their graphical attention histories.
Top: original RGB context and two enlarged A/B RGB details. Bottom: THREE ROWS of attention panels, EARLY / MIDDLE / LATE. LEFT COLUMN is region A; RIGHT COLUMN is region B. Each panel is cropped to the same native-image region as that column's RGB detail, with aspect preserved. These local panels remove the need to locate a small object in a full-scene map.
Only added ORANGE in these bottom attention panels is assigned attention. BLUE is unassigned gaze, not assigned to A or B. Printed IDs, white panel borders, and the original RGB colors in the top are not attention. The orange may fill the proposed object's silhouette; this is a graphical attention association, not a target-object label.
For the EARLY ROW, return which column's attention panel contains more orange. Then do the same for the LATE ROW. If neither panel in a requested row contains orange, return none for that row. Do not guess an object from RGB appearance or from its position in the source scene.
Return JSON only with exactly {"early":"A","late":"B"}; each value A, B or none.'''
REAL_PROMPT='''Select the intended __CATEGORY__ proposal from A and B using the ORIGINAL INSTRUCTION and the visible RGB evidence.
Top: source RGB context and enlarged A/B RGB details. Bottom: EARLY / MIDDLE / LATE rows of local graphical attention; A in the left column, B in the right. Each heat panel uses exactly the crop of that column's RGB detail. These are supported gaze observations from the two seconds before this frame. Orange is attention assigned to that object's mask; blue is unassigned gaze. No orange means missing gaze evidence, not evidence against an object. The graphics may include counterfactual controls.
The original instruction may mention other categories but only A and B are selectable here. Gaze is a weak cue: it may be on background, a reference object or another object. Use visible RGB and language relations first; use attention only to help unresolved same-category ambiguity. Do not simply select the hotter region. If neither is identifiable as the intended instance, return unclear. Do not invent objects or infer target identity from the printed ID.
Return JSON only, exactly {"selected":"A"}, value A, B or unclear.
ORIGINAL INSTRUCTION: __INSTRUCTION__'''


def font(size):return ImageFont.load_default(size=size)


def box_crop(box,size):
    w,h=size;x0,y0,x1,y1=box;dx=.15*(x1-x0);dy=.15*(y1-y0)
    return [max(0,int(x0-dx)),max(0,int(y0-dy)),min(w,int(np.ceil(x1+dx))),min(h,int(np.ceil(y1+dy)))]


def fit(image,size,background=(20,20,20)):
    canvas=Image.new('RGB',size,background);ratio=min(size[0]/image.width,size[1]/image.height)
    image=image.resize((max(1,round(image.width*ratio)),max(1,round(image.height*ratio))),Image.Resampling.LANCZOS)
    canvas.paste(image,((size[0]-image.width)//2,(size[1]-image.height)//2));return canvas


def render_local(rgb,objects,masks,mass,residual,labels):
    image=Image.new('RGB',(1008,1008),'#15212b');draw=ImageDraw.Draw(image)
    draw.text((14,8),'RGB OBJECTS + LOCAL GRAPHICAL ATTENTION',fill='white',font=font(22))
    scene=rgb.copy();sd=ImageDraw.Draw(scene)
    for j,obj in enumerate(objects):
        sd.rectangle(obj['box'],outline='white',width=3);x,y=interior_point(masks[j]);sd.text((x,y),labels[j],fill='white',stroke_width=2,stroke_fill='black',font=font(32))
    image.paste(scene.resize((320,320),Image.Resampling.LANCZOS),(16,48));draw.text((16,378),'SOURCE CONTEXT',fill='white',font=font(17))
    orange,blue,meta=graphical_pair_heat(mass,masks,residual)
    for label in ['A','B']:
        j=labels.index(label);x=370+(['A','B'].index(label))*318;box=box_crop(objects[j]['box'],rgb.size)
        draw.text((x,42),label+' | RGB DETAIL',fill='white',font=font(22))
        image.paste(fit(rgb.crop(box),(288,288),(34,42,48)),(x,76))
        draw.text((x,380),label+' | ATTENTION COLUMN',fill='white',font=font(16))
        for k in range(3):
            y=430+k*180;base=np.full((*masks.shape[1:],3),20.,float)
            # Render only the chosen object's orange; overlapping candidates cannot
            # borrow another candidate's heat when viewed through the crop.
            strength=min(1.,float(mass.sum())/.25);peak=float(mass.max())
            a=(.9*np.sqrt(float(mass[k,j])/peak)*strength)*masks[j][...,None] if peak>0 else np.zeros((*masks.shape[1:],1))
            b=blue[k,...,None]
            base=base*(1-b)+np.array([45,160,255])*b;base=base*(1-a)+np.array([255,85,25])*a
            panel=Image.fromarray(base.clip(0,255).astype(np.uint8)).crop(box)
            image.paste(fit(panel,(288,152)),(x,y));draw.rectangle((x,y,x+287,y+151),outline='#60717e')
    for k,name in enumerate(['EARLY','MIDDLE','LATE']):
        draw.text((48,470+k*180),name,fill='white',font=font(25))
    draw.text((16,972),'Dark panels can mean no added attention. Only ORANGE counts; RGB object colors do not.',fill='white',font=font(14))
    return image,meta


def prepare(parent):
    output=parent/'local_layout_v2'
    if (output/'manifest.json').exists():raise ValueError('Do not overwrite frozen local-layout inputs')
    output.mkdir(parents=True,exist_ok=True);m=read(parent/'manifest.json');requests=[]
    for case in m['cases']:
        d=np.load(parent/case['pair_npz_path']);masks=d['masks'];maps=d['raw_maps']
        with Image.open(case['source_path']) as im:rgb=im.convert('RGB')
        original=next(r for r in m['requests'] if r['case_id']==case['id'] and r['permutation']==0)
        base_labels=original['labels_by_physical_region']
        for condition in CONTROL_CONDITIONS if case['family']=='injected_cue' else REAL_CONDITIONS:
            if case['family']=='injected_cue':
                mass=np.zeros((3,2)) if condition=='NO_HEAT' else injected_history(masks,reverse=condition=='REVERSED');residual=np.zeros_like(maps)
            else:mass,residual,_=pair_attention(maps,masks,condition)
            for permutation in (0,1):
                labels=base_labels if permutation==0 else base_labels[::-1]
                expected=({'early':'none','late':'none'} if condition=='NO_HEAT' else
                    {'early':labels[int(mass[0].argmax())],'late':labels[int(mass[2].argmax())]}) if case['family']=='injected_cue' else {'selected':labels[0]}
                rid=f"{case['id']}_{condition}_p{permutation}_LOCAL";path=output/'images'/f'{rid}.png';path.parent.mkdir(exist_ok=True)
                image,meta=render_local(rgb,case['objects'],masks,mass,residual,labels);image.save(path)
                prompt=CUE_PROMPT if case['family']=='injected_cue' else REAL_PROMPT.replace('__CATEGORY__',case['category']).replace('__INSTRUCTION__',case['instruction'])
                requests.append({'id':rid,'case_id':case['id'],'event':case['event'],'family':case['family'],'condition':condition,
                    'permutation':permutation,'labels_by_physical_region':labels,'expected':expected,'prompt':prompt,
                    'path':str(path.relative_to(output)),'sha256':sha(path),'display_support':meta})
    save(output/'manifest.json',{'version':'local_object_attention_layout_v2','requests':requests,
        'parent_manifest_sha256':sha(parent/'manifest.json'),'code_sha256':sha(__file__),
        'adaptive_regression_only':True,'not_pure_resolution_ablation':True,
        'reason':'Full-scene mechanism examples showed null guessing and swapped time labels before this layout was prepared. Changes localize graphical evidence and simplify lookup/prompt; original data unchanged.'})
    save(output/'protocol_lock.json',{'manifest_sha256':sha(output/'manifest.json'),'before_v2_model':True})
    shutil.copy2(__file__,output/'source.py');report(parent)


def infer(parent):
    output=parent/'local_layout_v2';m=read(output/'manifest.json');assert sha(parent/'manifest.json')==m['parent_manifest_sha256']
    assert sha(output/'manifest.json')==read(output/'protocol_lock.json')['manifest_sha256']
    backend=QwenBackend(output/'model_cache',model='/ai/data/Qwen3-VL-8B-Instruct',local_only=True,
        namespace='local_attention_reliability_v2',trace_visual_inputs=True)
    schedule=m['requests'][:];random.Random(20261106).shuffle(schedule)
    for r in schedule:
        dest=output/'responses'/f"{r['id']}.json"
        if dest.exists():continue
        assert sha(output/r['path'])==r['sha256'];print('INFER',r['id'],flush=True);start=time.monotonic()
        parsed=backend.ask(r['id'],'paired_attention',r['prompt'],[output/r['path']],max_new_tokens=100);raw=read(backend.requests[-1])
        valid=valid_response(parsed,r['family']);physical={v:i for i,v in enumerate(r['labels_by_physical_region'])}
        save(dest,{**{k:r[k] for k in ['id','case_id','event','family','condition','permutation']},
            'parsed':parsed,'schema_valid':valid,'physical_prediction':{k:physical.get(v,v) for k,v in parsed.items()} if valid else None,
            'matches_label':valid and parsed==r['expected'],'request_path':backend.requests[-1],
            'processor_audit':raw['processor_audit'],'vision_verified':raw['visual_tensor_trace']['input_consumption_verified'],
            'seconds':time.monotonic()-start})
        print('RESULT',parsed,r['expected'],flush=True)
    report(parent)


def report(parent):
    output=parent/'local_layout_v2';m=read(output/'manifest.json');responses={p.stem:read(p) for p in (output/'responses').glob('*.json')}
    result={'planned_calls':len(m['requests']),'completed_calls':len(responses),'by_family_condition':{},
        'all_schema_valid':all(r['schema_valid'] for r in responses.values()),'all_vision_verified':all(r['vision_verified'] for r in responses.values()),
        'visual_tokens':sorted({r['processor_audit']['expanded_image_token_count'] for r in responses.values()}),
        'adaptive_regression_only':True}
    for family in ['injected_cue','real_center_proxy']:
        result['by_family_condition'][family]={}
        for condition in CONTROL_CONDITIONS if family=='injected_cue' else REAL_CONDITIONS:
            rr=[r for r in responses.values() if r['family']==family and r['condition']==condition]
            result['by_family_condition'][family][condition]={'n':len(rr),'matches_diagnostic_label':sum(r['matches_label'] for r in rr)}
    save(output/'analysis.json',result)
    data=json.dumps({'m':m,'responses':responses},ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>物体局部热图回归</title><style>body{font:15px system-ui;background:#f3f5f7;margin:20px}header{position:sticky;top:0;background:#f3f5f7;padding:10px}select{font:inherit;margin:6px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:14px}figure{margin:0;padding:10px;background:white}img{width:100%}pre{white-space:pre-wrap}</style><header><h2>对象局部热图：降低全场景查找负担</h2><select id="event"></select><select id="family"><option>injected_cue</option><option>real_center_proxy</option></select></header><p>看到前版机制失败后追加的回归版，不是独立验证。RGB细节和下方局部热图使用同一裁剪；左列A、右列B，三行为早中晚。热图仍是图像，未提供数值注意答案。脚本注入标签只测视觉关联，真实referent中心仍是条件式弱代理。版式、面积、查找方式和提示一起变动，不能当纯分辨率消融。</p><main></main><script>const d=__DATA__,q=document.querySelector('#event'),family=document.querySelector('#family');[...new Set(d.m.requests.map(r=>r.event))].forEach(e=>{let o=document.createElement('option');o.value=e;o.textContent=e;q.append(o)});function render(){let main=document.querySelector('main');main.replaceChildren();d.m.requests.filter(r=>r.event===q.value&&r.family===family.value).forEach(r=>{let f=document.createElement('figure'),im=document.createElement('img'),t=document.createElement('pre');im.src=r.path;let z=d.responses[r.id];t.textContent=r.case_id+' '+r.condition+' p'+r.permutation+' | '+(z?JSON.stringify(z.parsed):'尚未运行')+' | 诊断标签 '+JSON.stringify(r.expected);f.append(im,t);main.append(f)})}q.onchange=render;family.onchange=render;render();</script></html>'''
    (output/'index.html').write_text(page.replace('__DATA__',data));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','infer','report']);p.add_argument('--parent',type=Path,default=OUTPUT)
    a=p.parse_args();{'prepare':prepare,'infer':infer,'report':report}[a.action](a.parent.resolve())
