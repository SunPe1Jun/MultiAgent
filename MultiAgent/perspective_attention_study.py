"""Perspective windows and script-assigned time phases for visual attention routing.

This is an attention-region probe, not object or referent accuracy. All model
attention evidence remains graphical. Candidate labels and screen order are
permuted deterministically to expose position/letter shortcuts.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image,ImageDraw
from .heatmap_temporal_pilot import save,sha
from .perspective_attention import (FACES,select_faces,face_mass_audit,sample_face_density,
    sample_native_face,choose_native_source,transform_density)
from .qwen_backend import QwenBackend

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'.local-sync/spherical_attention_20261106'
OUTPUT=ROOT/'.local-sync/perspective_attention_20261106'
VERSION='perspective_attention_phase_routing_v1'
CONDITIONS=['TRUE_HEAT','REVERSE_TIME','SHIFT_SPACE','NO_HEAT']
DEV={'scene1/261','scene2/411','scene3/577','scene4_room1/72','scene4_room2/149','scene5/28'}
JOINT_PROMPT='''Use the added ORANGE HEATMAPS to choose which RGB region should be inspected first.
The top row shows six RGB perspective views of fixed angular regions. Each column has its own printed region ID, A/B/C/D/E/F. Column order and IDs can vary.
Below the RGB are three HEATMAP rows, EARLY, MIDDLE, LATE. Each heat tile uses exactly the same viewing direction as the RGB above it. Gray is the footprint of the supplied RGB context source. Heat outside that footprint remains historical direction evidence; a different source frame may show it. RGB object colors are not gaze marks.
Choose the printed REGION ID of the tile containing the BRIGHTEST ORANGE added heat in EARLY, then in LATE. If a requested heat row has no orange at all, return none for that row.
Heat is user attention, not a target-object label. Do not infer attention from object appearance. The diagrams can be controls.
Return only JSON with exactly two fields A/B/C/D/E/F/none: {"early":"A","late":"B"}.'''
PHASE_PROMPT='''Use the added ORANGE HEATMAP to choose which RGB region should be inspected first.
Six regions are arranged in a 2-by-3 grid. Each region has a printed ID A/B/C/D/E/F, a small RGB image ABOVE and a larger HEAT image BELOW. IDs and screen positions can vary.
All HEAT tiles show the SAME one time interval. Each HEAT tile uses exactly the same viewing direction as its RGB. Gray is the footprint of the supplied RGB context source. Heat outside that footprint remains historical direction evidence; a different source frame may show it. RGB object colors are not gaze marks.
Choose the printed REGION ID whose HEAT tile contains the BRIGHTEST ORANGE added heat among all six regions. If ALL six HEAT tiles have no orange, return none.
Attention is not a target-object label. The script handles chronology. Do not infer attention from RGB appearance. Diagrams can be controls.
Return only one JSON field A/B/C/D/E/F/none: {"selected":"A"}.'''


def read(path):return json.loads(Path(path).read_text())


def heat_image(plane,covered,global_peak,support):
    rgb=np.full((*plane.shape,3),20.,np.float32);rgb[covered]=[65,65,65]
    if global_peak>0:
        alpha=(np.sqrt(plane/global_peak)*.95*support)[...,None]
        rgb=rgb*(1-alpha)+np.array([255,85,25])*alpha
    return Image.fromarray(rgb.clip(0,255).astype('uint8'))


def card(rgb,heat,labels,order,mode,phase=None):
    canvas=Image.new('RGB',(1008,1008),'#15212b');draw=ImageDraw.Draw(canvas)
    if mode=='joint':
        draw.text((12,8),'RGB CONTEXT - OBJECT COLORS ARE NOT ATTENTION',fill='white')
        for col,index in enumerate(order):
            x=12+col*164;canvas.paste(rgb[index].resize((148,148)),(x,34))
            draw.rectangle((x,34,x+147,181),outline='white',width=2)
            draw.text((x+8,42),labels[index]+' RGB',fill='white',stroke_width=2,stroke_fill='black')
        for k,name in enumerate(['EARLY HEAT','MIDDLE HEAT','LATE HEAT']):
            y=236+k*226;draw.text((12,y-24),name,fill='white')
            for col,index in enumerate(order):
                x=12+col*164;canvas.paste(heat[k][index].resize((148,148)),(x,y))
                draw.rectangle((x,y,x+147,y+147),outline='white',width=2)
                draw.text((x+8,y+8),labels[index],fill='white',stroke_width=2,stroke_fill='black')
    elif mode=='phase':
        draw.text((12,8),'ONE SCRIPT-ASSIGNED INTERVAL | EACH REGION: RGB ABOVE, HEAT BELOW',fill='white')
        for col,index in enumerate(order):
            x=12+(col%3)*328;y=36+(col//3)*470
            draw.text((x+8,y),labels[index]+' | RGB',fill='white')
            canvas.paste(rgb[index].resize((152,152)),(x+80,y+22))
            draw.rectangle((x+80,y+22,x+231,y+173),outline='white',width=2)
            draw.text((x+8,y+184),labels[index]+' | HEAT',fill='white')
            canvas.paste(heat[phase][index].resize((256,256)),(x+32,y+208))
            draw.rectangle((x+32,y+208,x+287,y+463),outline='white',width=2)
            draw.text((x+42,y+220),labels[index],fill='white',stroke_width=2,stroke_fill='black')
    else:raise ValueError('Unknown layout')
    draw.text((12,985),'Only added orange in HEAT counts. No orange -> none. Angular regions are not object IDs.',fill='white')
    return canvas


def prepare(output):
    if (output/'manifest.json').exists():raise ValueError('Never overwrite prepared study')
    output.mkdir(parents=True,exist_ok=True);source=read(SOURCE/'manifest.json');events=[];requests=[]
    for e in source['events']:
        data=np.load(SOURCE/'events'/e['event']/'volume.npz');maps=data['duration_maps'];coverage=data['camera_coverage_seconds']
        selected,selection_stats=select_faces(coverage,count=6);audit=face_mass_audit(maps,selected)
        density=np.stack([sample_face_density(maps,face,224) for face in selected],axis=1)
        native=[];mid_rgb=[];mid_masks=[]
        directory=output/'events'/e['event'];directory.mkdir(parents=True,exist_ok=True)
        for region,face in enumerate(selected):
            sources=[]
            for phase,b in enumerate(e['volume_stats']['bins']):
                frame,meta=choose_native_source(e['frames'],e['canonical_pose'],face,b['window'],e['frames'][-1]['geometry']['selected_clock_offset_seconds'])
                if sha(frame['source_path'])!=frame['source_sha256']:raise ValueError('Source frame hash changed')
                rgb,mask=sample_native_face(frame,e['canonical_pose'],face,224)
                path=directory/f'native_region{region}_phase{phase}.png';rgb.save(path)
                sources.append({'phase':phase,'frame_id':frame['frame_id'],'source_path':frame['source_path'],
                    'source_sha256':frame['source_sha256'],'source_pts':frame['pts'],'rendered_path':str(path.relative_to(output)),
                    'rendered_sha256':sha(path),'selection':meta,'recorded_fraction':float(mask.mean())})
                if phase==1:mid_rgb.append(rgb);mid_masks.append(mask)
            native.append(sources)
        labels=list('ABCDEF');order=list(range(6));rng=random.Random('20261106:'+e['id']);rng.shuffle(labels);rng.shuffle(order)
        entry={'id':e['id'],'event':e['event'],'split':'development' if e['event'] in DEV else 'check',
            'selected_faces':selected,'face_names':[FACES[f][0] for f in selected],
            'selection_stats':selection_stats,'mass_audit':audit,'labels_by_region':labels,'screen_order':order,
            'native_sources_by_region':native,'requests':[],
            'model_rgb_context':'same middle-phase native source in all model cards; phase-specific sources are reviewable separately',
            'reference_source_event':e['event'],'max_rotation':e['max_head_rotation_from_reference_degrees'],
            'peak_vs_mass':{}}
        support=min(1.,float(maps.sum())/.25)
        for condition in CONDITIONS:
            transformed=transform_density(density,condition);global_peak=float(transformed.max())
            heat=[[heat_image(transformed[k,j],mid_masks[j],global_peak,support) for j in range(6)] for k in range(3)]
            expected=[];peak_values=[]
            for phase in range(3):
                peaks=[float(transformed[phase,j].max()) for j in range(6)];peak_values.append(peaks)
                expected.append('none' if max(peaks)<=0 else labels[int(np.argmax(peaks))])
            entry['peak_vs_mass'][condition]={'displayed_peaks':peak_values,'expected_by_time':expected}
            for mode,phase in [('joint',None),('phase',0),('phase',2)]:
                rid=e['id']+'_'+condition+'_'+mode+(f'_{phase}' if phase is not None else '')
                image=card(mid_rgb,heat,labels,order,mode,phase)
                path=directory/f'{condition}_{mode}_{phase if phase is not None else "all"}.png'
                image.save(path)
                request={'id':rid,'event_id':e['id'],'event':e['event'],'split':entry['split'],'condition':condition,
                    'mode':mode,'phase':phase,'path':str(path.relative_to(output)),'sha256':sha(path),'size':list(image.size),
                    'expected':{'early':expected[0],'late':expected[2]} if mode=='joint' else {'selected':expected[phase]},
                    'prompt_key':mode,'global_peak':global_peak,'peak_values_by_time':peak_values}
                requests.append(request);entry['requests'].append(rid)
        np.savez_compressed(directory/'perspective_density.npz',density=density)
        events.append(entry);print('PREPARED',e['event'],entry['face_names'],'retained',audit['retained_duration_fraction'],flush=True)
    manifest={'version':VERSION,'source_manifest_sha256':sha(SOURCE/'manifest.json'),'prompts':{'joint':JOINT_PROMPT,'phase':PHASE_PROMPT},
        'conditions':CONDITIONS,'events':events,'requests':requests,'image_size':[1008,1008],
        'dev_events':sorted(DEV),'design_frozen_before_model_outputs':True,
        'check_scope':'Preassigned remaining seven interactions; all previously seen in earlier studies, not independent unseen validation',
        'gt_loaded':False,'anchors_projected':False,'attention_in_model_prompt':'images only, no candidate scores or expected labels',
        'candidate_semantics':'all six disjoint angular faces; ordering by camera coverage only, not object IDs',
        'scope':'visual attention region prioritization; not final referring or object recognition',
        'time_budget_comparison':'joint one call per condition versus phase two calls per condition; both 1008 square, phase gets more pixels per time interval',
        'code_sha256':sha(__file__),'renderer_sha256':sha(Path(__file__).with_name('perspective_attention.py'))}
    save(output/'manifest.json',manifest);gallery(output,manifest)


def infer(output,split,modes=None):
    manifest=read(output/'manifest.json');lock={'manifest_sha256':sha(output/'manifest.json')}
    protocol=output/'protocol_lock.json'
    if protocol.exists() and read(protocol)!=lock:raise ValueError('Protocol changed')
    save(protocol,lock)
    if split=='check':
        if not (output/'development_frozen.json').exists():raise ValueError('Freeze development findings before check')
        save(output/'check_started.json',{'manifest_sha256':lock['manifest_sha256'],'development_sha256':sha(output/'development_frozen.json')})
    backend=QwenBackend(output/'model_cache',model=str(ROOT.parent/'Qwen3-VL-8B-Instruct'),local_only=True,namespace=VERSION,trace_visual_inputs=True)
    schedule=[r for r in manifest['requests'] if r['split']==split and (modes is None or r['mode'] in modes)];random.Random(20261106).shuffle(schedule)
    for r in schedule:
        path=output/r['path'];dest=output/'responses'/f'{r["id"]}.json';prompt=manifest['prompts'][r['prompt_key']]
        if sha(path)!=r['sha256']:raise ValueError('Input changed')
        if dest.exists() and read(dest).get('status')=='COMPLETE':
            saved=read(dest);req=read(saved['request_path'])
            if req['prompt']!=prompt or req['image_sha256']!=[r['sha256']]:raise ValueError('Cache mismatch')
            continue
        print('INFER',r['id'],flush=True);start=time.monotonic()
        parsed=backend.ask(r['id'],r['mode'],prompt,[path],max_new_tokens=120);request=read(backend.requests[-1])
        valid=set(parsed)==set(r['expected']) and all(v in ['A','B','C','D','E','F','none'] for v in parsed.values())
        result={**{k:r[k] for k in ['id','event_id','event','split','condition','mode','phase','expected']},
            'status':'COMPLETE','parsed':parsed,'schema_valid':valid,'correct':valid and parsed==r['expected'],
            'field_correct':{k:valid and parsed.get(k)==v for k,v in r['expected'].items()},
            'seconds':time.monotonic()-start,'request_path':backend.requests[-1],
            'processor_audit':request['processor_audit'],'vision_verified':request['visual_tensor_trace']['input_consumption_verified']}
        save(dest,result);print('RESULT',parsed,'expected',r['expected'],flush=True)
    report(output)


def report(output):
    manifest=read(output/'manifest.json');rows=[read(p) for p in (output/'responses').glob('*.json')]
    indexed={r['id']:r for r in rows};stats={}
    for split in ['development','check']:
        stats[split]={}
        for condition in CONDITIONS:
            stats[split][condition]={}
            for mode in ['joint','phase']:
                r=[r for r in rows if r['split']==split and r['condition']==condition and r['mode']==mode]
                pairs=[]
                for e in manifest['events']:
                    if e['split']!=split:continue
                    matching=[row for row in r if row['event_id']==e['id']]
                    if mode=='joint' and len(matching)==1:pairs.append(matching[0]['correct'])
                    elif mode=='phase' and len(matching)==2:pairs.append(all(row['correct'] for row in matching))
                stats[split][condition][mode]={'calls':len(r),'call_correct':sum(v['correct'] for v in r),
                    'complete_event_pairs':len(pairs),'both_times_correct':sum(pairs),
                    'fields':sum(len(v['expected']) for v in r),'fields_correct':sum(sum(v['field_correct'].values()) for v in r)}
    transitions={}
    for split in ['development','check']:
        selected=[e for e in manifest['events'] if e['split']==split and e['peak_vs_mass']['TRUE_HEAT']['expected_by_time'][0]!=e['peak_vs_mass']['TRUE_HEAT']['expected_by_time'][2]]
        transitions[split]={'events':[e['id'] for e in selected]}
        for mode in ['joint','phase']:
            for condition in ['TRUE_HEAT','REVERSE_TIME']:
                pairs=[]
                for e in selected:
                    r=[r for r in rows if r['event_id']==e['id'] and r['mode']==mode and r['condition']==condition]
                    if len(r)==(1 if mode=='joint' else 2):pairs.append(all(v['correct'] for v in r))
                transitions[split][mode+'_'+condition]={'n':len(pairs),'both_times_correct':sum(pairs)}
    analysis={'stats':stats,'transition_events':transitions,'calls':len(rows),
        'tokens':sorted({r['processor_audit']['expanded_image_token_count'] for r in rows}),
        'all_vision_verified':all(r['vision_verified'] for r in rows),'all_schema_valid':all(r['schema_valid'] for r in rows),
        'coverage':{e['id']:e['mass_audit']['retained_duration_fraction'] for e in manifest['events']},
        'scope':manifest['scope'],'final_referent_evaluator_run':False}
    save(output/'analysis.json',analysis);gallery(output,manifest,rows)
    print(json.dumps(analysis,ensure_ascii=False,indent=2))
    return analysis


def gallery(output,m,rows=None):
    payload=json.dumps({'manifest':m,'results':rows or []},ensure_ascii=False).replace('</','<\\/')
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>透视窗口中的视觉注意筛选</title><style>body{font:15px system-ui;margin:20px;background:#f2f5f7;color:#17212d}header{position:sticky;top:0;background:#f2f5f7;padding:12px}select{font:inherit;margin:8px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:14px}figure{margin:0;padding:12px;background:white}img{width:100%}p{max-width:1350px}</style><header><h2>固定方向、透视窗口、脚本绑定时间</h2><select id="event"></select><select id="condition"></select><select id="mode"><option value="joint">联合判断早晚</option><option value="phase">逐段筛选，脚本还原时间</option><option value="sources">各阶段真实RGB来源</option></select></header><p>六个不重叠的90°方向窗口完整覆盖球面，仅按相机覆盖排序，字母与显示位置打乱。热图继续以图像输入；没有数值注视先验。模型卡片在所有条件使用相同的中段原始RGB，保持注意对照一致；来源页另提供各阶段RGB。角度区域不是对象ID，旋转采样未处理平移视差和遮挡。</p><p>这13条交互都在之前实验出现过，6条开发、7条检查，不是独立未见验证。逐段方案每段图片更大、调用数翻倍，不能把差异只归因于时间绑定。</p><p id="meta"></p><main></main><script>const d=__DATA__,m=d.manifest,q=document.querySelector('#event'),c=document.querySelector('#condition'),mode=document.querySelector('#mode');m.events.forEach((e,i)=>{let o=document.createElement('option');o.value=i;o.textContent=e.event+' | '+e.split;q.append(o)});m.conditions.forEach(v=>{let o=document.createElement('option');o.value=v;o.textContent=v;c.append(o)});function render(){const e=m.events[+q.value],main=document.querySelector('main');main.replaceChildren();document.querySelector('#meta').textContent='窗口：'+e.face_names.join('/')+' | 各阶段保留热量 '+e.mass_audit.retained_duration_fraction.map(v=>(v*100).toFixed(1)+'%').join('/');if(mode.value==='sources'){e.native_sources_by_region.forEach((items,j)=>items.forEach(s=>{let f=document.createElement('figure'),text=document.createElement('figcaption'),im=document.createElement('img');text.textContent=e.labels_by_region[j]+' | 阶段 '+s.phase+' | '+s.frame_id+' PTS '+s.source_pts.toFixed(3)+' | 记录覆盖 '+(s.recorded_fraction*100).toFixed(1)+'%';im.src=s.rendered_path;f.append(text,im);main.append(f)}));return;}m.requests.filter(r=>r.event_id===e.id&&r.condition===c.value&&r.mode===mode.value).forEach(r=>{let f=document.createElement('figure'),text=document.createElement('figcaption'),im=document.createElement('img'),out=d.results.find(v=>v.id===r.id);text.textContent=(r.phase===null?'早晚联合':'阶段 '+r.phase)+' | expected '+JSON.stringify(r.expected)+(out?' | model '+JSON.stringify(out.parsed)+' | '+(out.correct?'✓':'×'):'');im.src=r.path;f.append(text,im);main.append(f)})}q.onchange=render;c.onchange=render;mode.onchange=render;render();</script></html>'''
    (output/'index.html').write_text(html.replace('__DATA__',payload))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','infer','report']);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--split',choices=['development','check'],default='development');p.add_argument('--modes',nargs='+',choices=['joint','phase'])
    a=p.parse_args();{'prepare':lambda:prepare(a.output.resolve()),'infer':lambda:infer(a.output.resolve(),a.split,a.modes),'report':lambda:report(a.output.resolve())}[a.action]()
