#!/usr/bin/env python3
"""Object proposals -> fractional gaze/mask links -> graphical object selection.

Inference never reads referent projections. A separate post-inference audit may
reuse only the event referent projections; they are not segmentation/pixel GT.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import shutil
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import binary_erosion
from MultiAgent.attention_volume import build_volume
from MultiAgent.context import load_contexts
from MultiAgent.gaze_evidence import gaze_records
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.object_attention import (VERSION,instruction_categories,detector_query,
    deduplicate_detections,mask_attention_links)
from MultiAgent.perspective_attention_study import ROOT,SOURCE,DEV,read
from MultiAgent.qwen_backend import QwenBackend
from MultiAgent.spherical_attention_study import AUDIT

OUTPUT=ROOT/'.local-sync/object_attention_research_20261106/study'
MODELROOT=ROOT/'.local-sync/models'
CONDITIONS=['NO_ADDED_HEAT','RAW_DENSITY','OBJECT_HEAT']
PROMPT='''Choose concrete visible object proposals that the ORIGINAL INSTRUCTION refers to, including actors, acted-on objects and spatial reference objects.
The large RGB scene has WHITE numbered object boundaries. Enlarged RGB details at the bottom use the same printed IDs. Only the six or fewer IDs listed below are selectable in THIS PAGE; other pages may contain further candidates. IDs are local to this image, not object names or ground truth.
The three small panels at the right show older / middle / recent attention within the supported two-second history before this frame, registered to this same frame. Gray outlines only locate proposals and are not attention. Added ORANGE is gaze evidence. Some displays distribute gaze overlap across the corresponding object's mask; blue outside proposals is unassigned gaze. The instruction, RGB and attention must agree: gaze is a soft clue, not a target label; it may point to a reference, hand, background or another object. Missing heat does not justify rejection. Original RGB may contain recorder marks.
Identify objects from RGB rather than assuming a detector label. Use attention to help distinguish visually similar alternatives, but do not choose every orange object. Retain unresolved alternatives explicitly. Do not invent objects outside the displayed proposals.
Return JSON only with exactly two disjoint arrays of printed IDs: {"selected":["O01"],"uncertain":["O02"]}. selected = concrete visible objects supported as instruction referents; uncertain = visually plausible alternatives whose intended identity remains unclear. Empty arrays are allowed. Never output coordinates, descriptions or scores.
AVAILABLE IDS: __IDS__
ORIGINAL INSTRUCTION: __INSTRUCTION__'''


def font(size):return ImageFont.load_default(size=size)


def region_crop(rgb,box,padding=.12):
    w,h=rgb.size;x0,y0,x1,y1=box;dx=(x1-x0)*padding;dy=(y1-y0)*padding
    b=[max(0,int(x0-dx)),max(0,int(y0-dy)),min(w,math.ceil(x1+dx)),min(h,math.ceil(y1+dy))]
    return rgb.crop(b),b


def show_heat(maps,masks,links,condition,page_indices):
    k,h,w=maps.shape;base=np.full((k,h,w,3),20.,float)
    for j in page_indices:
        mask=masks[j];edge=mask&~binary_erosion(mask)
        base[:,edge]=[95,95,95]
    if condition=='NO_ADDED_HEAT':return [Image.fromarray(a.astype(np.uint8)) for a in base]
    peak=float(maps.max())
    if condition=='RAW_DENSITY':
        a=(np.sqrt(maps/peak)*.95)[...,None] if peak>0 else np.zeros((*maps.shape,1))
        base=base*(1-a)+np.array([255,85,25])*a
    elif condition=='OBJECT_HEAT':
        # Whole-object graphical intensity encodes supported mask-overlap seconds,
        # not density per pixel or calibrated target probability. No text scores.
        mass=np.asarray(links['mass_seconds_by_phase_object']);maximum=float(mass.max()) if mass.size else 0.
        area=masks.sum(axis=0);strength=np.einsum('kn,nhw->khw',mass,masks.astype(float)) if len(masks) else np.zeros_like(maps)
        strength=np.divide(strength,area[None],out=np.zeros_like(strength),where=area[None]>0)
        a=(.9*np.sqrt(np.clip(strength/maximum,0,1)))[...,None] if maximum>0 else np.zeros((*maps.shape,1))
        base=base*(1-a)+np.array([255,85,25])*a
        if peak>0:
            unknown=(area==0)[None];u=(.7*np.sqrt(maps/peak)*unknown)[...,None]
            base=base*(1-u)+np.array([45,160,255])*u
    else:raise ValueError(condition)
    return [Image.fromarray(a.clip(0,255).astype(np.uint8)) for a in base]


def render_card(rgb,objects,masks,maps,links,page_indices,condition):
    image=Image.new('RGB',(1008,1008),'#15212b');draw=ImageDraw.Draw(image)
    draw.text((16,8),'VISIBLE OBJECT PROPOSALS + SCRIPT-BOUND ATTENTION',fill='white',font=font(19))
    scene=rgb.copy();sd=ImageDraw.Draw(scene);pixels=np.asarray(scene).copy()
    for j in page_indices:
        mask=masks[j];edge=mask&~binary_erosion(mask,iterations=2);pixels[edge]=[255,255,255]
    scene=Image.fromarray(pixels);sd=ImageDraw.Draw(scene)
    for j in page_indices:
        x0,y0,x1,y1=objects[j]['box'];sd.text((max(0,x0),max(0,y0-20)),objects[j]['id'],fill='white',stroke_width=2,stroke_fill='black',font=font(20))
    image.paste(scene.resize((672,672),Image.Resampling.LANCZOS),(12,42))
    panels=show_heat(maps,masks,links,condition,page_indices)
    for k,panel in enumerate(panels):
        x=756;y=62+k*216;draw.text((x,y-22),['OLDER','MIDDLE','RECENT'][k],fill='white',font=font(16))
        tile=panel.resize((192,192),Image.Resampling.BILINEAR);td=ImageDraw.Draw(tile)
        for j in page_indices:
            box=objects[j]['box'];px=(box[0]+box[2])/2/rgb.width*192;py=(box[1]+box[3])/2/rgb.height*192
            td.text((px,py),objects[j]['id'],fill='white',stroke_width=1,stroke_fill='black',font=font(10))
        image.paste(tile,(x,y))
    draw.text((16,738),'ACTUAL OBJECT DETAILS - printed IDs link to the full RGB scene',fill='white',font=font(16))
    for col,j in enumerate(page_indices):
        x=12+164*col;crop,_=region_crop(rgb,objects[j]['box'])
        # Aspect is preserved with letterboxing instead of stretching thin objects.
        tile=Image.new('RGB',(148,148),'#222a30');ratio=min(148/crop.width,148/crop.height)
        crop=crop.resize((max(1,round(crop.width*ratio)),max(1,round(crop.height*ratio))),Image.Resampling.LANCZOS)
        tile.paste(crop,((148-crop.width)//2,(148-crop.height)//2));image.paste(tile,(x,786))
        draw.text((x+4,762),objects[j]['id'],fill='white',font=font(18))
    draw.text((16,968),'Proposal IDs are not target labels. Keep uncertainty; unassigned gaze is preserved.',fill='white',font=font(14))
    return image


def prepare(output):
    if (output/'manifest.json').exists():raise ValueError('Never overwrite frozen experiment inputs')
    output.mkdir(parents=True,exist_ok=True)
    source=read(SOURCE/'manifest.json');selection=read(AUDIT/'selection.json');keys={e['event'] for e in source['events']}
    contexts=load_contexts({'samples':[r for r in selection['samples'] if f"{r['scene_key']}/{r['row_index']}" in keys]},load_ground_truth=False)
    contexts={f'{c.scene_key}/{c.row_index}':c for c in contexts}
    # Separate lightweight vision models are used without the Qwen resident.
    import torch
    from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection,Sam2Processor,Sam2Model
    dp=AutoProcessor.from_pretrained(MODELROOT/'grounding-dino-tiny',local_files_only=True)
    detector=AutoModelForZeroShotObjectDetection.from_pretrained(MODELROOT/'grounding-dino-tiny',local_files_only=True).cuda().eval()
    sp=Sam2Processor.from_pretrained(MODELROOT/'sam2.1-hiera-tiny',local_files_only=True)
    segmenter=Sam2Model.from_pretrained(MODELROOT/'sam2.1-hiera-tiny',local_files_only=True).cuda().eval()
    events=[];requests=[]
    for event in source['events']:
        ctx=contexts[event['event']];categories=instruction_categories(ctx.text);query=detector_query(categories);records=gaze_records(ctx)
        entry={'id':event['id'],'event':event['event'],'instruction':ctx.text,'split':'development' if event['event'] in DEV else 'check',
            'query_categories':categories,'detector_query':query,'frames':[]}
        frames=event['frames'];chosen=[frames[0],frames[len(frames)//2],frames[-1]]
        for frame in chosen:
            directory=output/'events'/event['event']/frame['frame_id'];directory.mkdir(parents=True,exist_ok=True)
            path=Path(frame['source_path']);assert sha(path)==frame['source_sha256']
            with Image.open(path) as source_image:rgb=source_image.convert('RGB')
            x=dp(images=rgb,text=query,return_tensors='pt').to('cuda')
            with torch.inference_mode():prediction=detector(**x)
            post=dp.post_process_grounded_object_detection(prediction,x.input_ids,threshold=.25,text_threshold=.2,target_sizes=[rgb.size[::-1]])[0]
            raw=[{'box':b.detach().float().cpu().tolist(),'score':float(s),'label':str(label)} for b,s,label in zip(post['boxes'],post['scores'],post['text_labels'])]
            objects=deduplicate_detections(raw,*rgb.size)
            if objects:
                sx=sp(images=rgb,input_boxes=[[r['box'] for r in objects]],return_tensors='pt').to('cuda')
                with torch.inference_mode():seg=segmenter(**sx,multimask_output=False)
                masks=sp.post_process_masks(seg.pred_masks.cpu(),sx['original_sizes'].cpu())[0][:,0].numpy().astype(bool)
                for i,row in enumerate(objects):row['sam_quality']=float(seg.iou_scores[0,i,0])
            else:masks=np.zeros((0,rgb.height,rgb.width),bool)
            # IDs and display order do not reveal scores, gaze or annotation order.
            rng=random.Random('object-attention-v1:'+event['id']+':'+frame['frame_id'])
            names=[f'O{i+1:02d}' for i in range(len(objects))];rng.shuffle(names);order=list(range(len(objects)));rng.shuffle(order)
            for i,obj in enumerate(objects):obj.update(id=names[i],mask_area_pixels=int(masks[i].sum()))
            geometry=frame['geometry'];query_time=geometry['telemetry_time_seconds']
            maps,stats=build_volume(records,query=query_time,pose=geometry['reference_pose'],image_size=rgb.size,
                clock_supported=geometry['clock_available'],history=2.,bins=3,sigma_degrees=2.)
            links,object_display,unassigned=mask_attention_links(maps,masks)
            npz=directory/'object_attention.npz';np.savez_compressed(npz,masks=masks,maps=maps,object_display_density=object_display,unassigned_maps=unassigned)
            row={'frame_id':frame['frame_id'],'source_path':str(path),'source_sha256':sha(path),'source_pts':frame['pts'],
                 'geometry':geometry,'objects':objects,'raw_detector_count':len(raw),'raw_detections':raw,'display_order':order,
                 'attention_stats':stats,'links':links,'npz_path':str(npz.relative_to(output)),'npz_sha256':sha(npz),
                 'inference_gt_loaded':False,'anchors_projected':False,'pages':[]}
            for page in range(max(1,math.ceil(len(order)/6))):
                indices=order[page*6:(page+1)*6];ids=[objects[j]['id'] for j in indices]
                for condition in CONDITIONS:
                    rid=f"{event['id']}_{frame['frame_id']}_page{page}_{condition}"
                    image=render_card(rgb,objects,masks,maps,links,indices,condition);dest=directory/f'page{page}_{condition}.png';image.save(dest)
                    prompt=PROMPT.replace('__IDS__',','.join(ids) or '(none)').replace('__INSTRUCTION__',ctx.text)
                    request={'id':rid,'event_id':event['id'],'event':event['event'],'split':entry['split'],'frame_id':frame['frame_id'],
                        'page':page,'indices':indices,'available_ids':ids,'condition':condition,'path':str(dest.relative_to(output)),
                        'sha256':sha(dest),'prompt':prompt}
                    requests.append(request);row['pages'].append(rid)
            entry['frames'].append(row);print('PREPARED',event['event'],frame['frame_id'],'objects',len(objects),flush=True)
        events.append(entry)
    manifest={'version':VERSION,'source_manifest_sha256':sha(SOURCE/'manifest.json'),'events':events,'requests':requests,
        'conditions':CONDITIONS,'regression_only':True,'inference_gt_loaded':False,'anchors_projected':False,
        'model_provenance':{name:read(MODELROOT/name/'download_provenance.json') for name in ['grounding-dino-tiny','sam2.1-hiera-tiny']},
        'sampling':'First, middle-index and last supported native frame for each of 13 old interactions; independent of gaze and GT.',
        'proposal_policy':'Instruction-only pilot noun dictionary, detector .25/text .2, class NMS .6, all accepted proposals retained. No gaze-based filtering.',
        'history_scope':'2-second causal gaze reprojected into native current frame; rotation/pose binding supported, dynamic correspondence unverified.',
        'code_sha256':sha(__file__),'object_code_sha256':sha(ROOT/'MultiAgent/object_attention.py')}
    save(output/'manifest.json',manifest);save(output/'protocol_lock.json',{'manifest_sha256':sha(output/'manifest.json'),'before_model_inference':True})
    (output/'source').mkdir(exist_ok=True);shutil.copy2(__file__,output/'source/run_object_attention_study.py');shutil.copy2(ROOT/'MultiAgent/object_attention.py',output/'source/object_attention.py')
    report(output)


def infer(output,split,limit=None):
    m=read(output/'manifest.json');assert sha(output/'manifest.json')==read(output/'protocol_lock.json')['manifest_sha256']
    if split=='check':
        assert (output/'development_frozen.json').exists()
        started=output/'check_started.json'
        if not started.exists():save(started,{'development_sha256':sha(output/'development_frozen.json'),'manifest_sha256':sha(output/'manifest.json')})
    backend=QwenBackend(output/'model_cache',model='/ai/data/Qwen3-VL-8B-Instruct',local_only=True,
                        namespace='object_attention_selection_v1',trace_visual_inputs=True)
    schedule=[r for r in m['requests'] if r['split']==split];random.Random(20261106).shuffle(schedule);count=0
    for request in schedule:
        dest=output/'responses'/f"{request['id']}.json"
        if dest.exists():continue
        path=output/request['path'];assert sha(path)==request['sha256']
        print('INFER',request['id'],flush=True);start=time.monotonic()
        parsed=backend.ask(request['id'],'select_objects',request['prompt'],[path],max_new_tokens=160)
        raw=read(backend.requests[-1]);valid=(set(parsed)=={'selected','uncertain'} and all(isinstance(parsed[k],list) for k in parsed))
        if valid:
            all_ids=parsed['selected']+parsed['uncertain'];valid=(all(isinstance(i,str) and i in request['available_ids'] for i in all_ids) and len(set(all_ids))==len(all_ids))
        save(dest,{'id':request['id'],'event_id':request['event_id'],'frame_id':request['frame_id'],'page':request['page'],
            'condition':request['condition'],'split':split,'parsed':parsed,'schema_valid':valid,
            'request_path':backend.requests[-1],'processor_audit':raw['processor_audit'],
            'vision_verified':raw['visual_tensor_trace']['input_consumption_verified'],'seconds':time.monotonic()-start})
        print('RESULT',valid,parsed,flush=True);count+=1
        if limit is not None and count>=limit:break
    report(output)
    expected=[r for r in m['requests'] if r['split']=='development']
    if split=='development' and all((output/'responses'/f"{r['id']}.json").exists() for r in expected):
        freeze=output/'development_frozen.json'
        if not freeze.exists():save(freeze,{'manifest_sha256':sha(output/'manifest.json'),'completed_development_calls':len(expected),'check_not_started':True})


def report(output):
    m=read(output/'manifest.json');responses={p.stem:read(p) for p in (output/'responses').glob('*.json')}
    frames=[f for e in m['events'] for f in e['frames']];objects=sum(len(f['objects']) for f in frames)
    total=sum(sum(f['links']['total_seconds']) for f in frames);assigned=sum(sum(f['links']['assigned_seconds']) for f in frames)
    by_condition={}
    for c in CONDITIONS:
        rows=[r for r in responses.values() if r['condition']==c];valid=[r for r in rows if r['schema_valid']]
        by_condition[c]={'calls':len(rows),'valid_schema':len(valid),'selected':sum(len(r['parsed']['selected']) for r in valid),
            'uncertain':sum(len(r['parsed']['uncertain']) for r in valid)}
    summary={'events':len(m['events']),'frames':len(frames),'objects':objects,'frames_without_proposals':sum(not f['objects'] for f in frames),
        'planned_model_calls':len(m['requests']),'completed_model_calls':len(responses),'by_condition':by_condition,
        'supported_frustum_seconds_over_overlapping_queries':total,'assigned_seconds':assigned,
        'assigned_fraction':assigned/total if total else None,'unassigned_seconds':total-assigned,
        'all_vision_verified':all(r['vision_verified'] for r in responses.values()),
        'visual_tokens':sorted({r['processor_audit']['expanded_image_token_count'] for r in responses.values()}),
        'ground_truth_pixel_accuracy_available':False,'regression_only':True}
    save(output/'analysis.json',summary)
    data=json.dumps({'m':m,'responses':responses},ensure_ascii=False).replace('</','<\\/')
    template='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>物体候选与注视关联</title><style>body{font:15px system-ui;background:#f3f5f7;margin:20px}header{position:sticky;top:0;background:#f3f5f7;padding:8px}select{font:inherit;margin:6px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:14px}figure{background:white;margin:0;padding:10px}img{width:100%}pre{white-space:pre-wrap}</style><header><h2>先找到可见物体，再关联注视与语言</h2><select id="event"></select><select id="condition"><option>NO_ADDED_HEAT</option><option>RAW_DENSITY</option><option>OBJECT_HEAT</option></select><p id="instruction"></p></header><p>候选由原始指令类别、Grounding DINO 与 SAM2 生成；不使用GT、anchor投影或gaze删候选。右侧三段为当前帧之前2秒的注意。OBJECT_HEAT以物体mask内分配到的支持时长作整物体着色，蓝色保留无物体承接的gaze；形状填充不表示整个物体每处都被看过。重叠mask平分质量，不强制归到唯一对象。全部检测候选分页保留，O编号只是图内地址。该版尚未验证跨帧身份。</p><p>原RGB自带绿色标记，各条件均保留；NO_ADDED_HEAT不是完全无gaze的采集基线。检测类别来自仅用于本pilot的词表，未知类别可能漏检。全部13交互均为旧样例。referent中心审计不等于像素/遮挡GT。</p><main></main><script>const d=__DATA__,q=document.querySelector('#event'),c=document.querySelector('#condition');d.m.events.forEach(e=>{let o=document.createElement('option');o.value=e.id;o.textContent=e.event;q.append(o)});function render(){let e=d.m.events.find(e=>e.id===q.value);document.querySelector('#instruction').textContent=e.instruction;let main=document.querySelector('main');main.replaceChildren();d.m.requests.filter(r=>r.event_id===q.value&&r.condition===c.value).forEach(r=>{let f=document.createElement('figure'),im=document.createElement('img'),t=document.createElement('pre');im.src=r.path;let z=d.responses[r.id];t.textContent=r.frame_id+' | 页'+(r.page+1)+' | '+(z?JSON.stringify(z.parsed):'待运行');f.append(im,t);main.append(f)})}q.onchange=render;c.onchange=render;render();</script></html>'''
    (output/'index.html').write_text(template.replace('__DATA__',data));print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','infer','report']);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--split',choices=['development','check'],default='development');p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.action=='infer':infer(a.output.resolve(),a.split,a.limit)
    else:{'prepare':prepare,'report':report}[a.action](a.output.resolve())
