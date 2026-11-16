"""Script-led region presentation study; one semantic category per model call.

Labels are a frozen assistant visual review, not human/object-instance GT.
No instructions, target labels or anchor projections are used for visual inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image,ImageDraw

from .camera_geometry import aligned_frame_geometry
from .context import load_contexts
from .gaze_evidence import build_heatmap,gaze_records,render_presentation
from .heatmap_temporal_pilot import sha,save
from .qwen_backend import QwenBackend
from .scripted_attention import spatial_foveation,crop_at,evidence_card

ROOT=Path(__file__).resolve().parents[1]
VERSION='scripted_region_semantics_v1'
DEV=[('scene1',261),('scene2',411),('scene3',577),('scene4_room1',72),('scene5',28)]
VALIDATION=[('scene2',71),('scene2',678),('scene3',747),('scene4_room1',155),('scene4_room4',25)]
CONDITIONS=['RGB_REGION','LEGACY_FOVEATION','SPATIAL_FOVEATION','DETAIL_035','HISTORY_OVERLAY','SHIFTED_DETAIL']
CATEGORIES=['person','chair','desk','table','cabinet','door','window','curtain','bed','sofa','stove','sink',
    'lamp','shelf','computer','plant','vehicle','drone','helicopter','building',
    'container','crate','barrel','tank','ground','sky','wall','ceiling','hand','other','unclear']
PROMPT=('Identify the visible physical object or surface AT THE CENTER OF THE FOUR SMALL CORNER MARKS '
        'in the RIGHT panel. The left panel is context of the same scene. '
        'Use context only to recognize the indicated local content. The right panel may show '
        'a full view or a magnified region. Ignore the image header and artificial corner marks. '
        'If a translucent VR hand overlaps the point, identify the underlying scene surface when visible; '
        'otherwise choose hand or unclear. Ground includes floors, roads and rugs. '
        'Computer includes monitors and computer towers. Vehicle includes cars, trucks, vans and forklifts. '
        'Classify the center point, not the dominant object elsewhere in the crop. '
        'Return exactly one JSON field, category, from this list: '
        +json.dumps(CATEGORIES)+'. Do not return coordinates, gaze judgments, time interpretation, or explanations. '
        'Example format: {"category":"chair"}. Use unclear only when the indicated physical content cannot be recognized.')


def read(path):return json.loads(Path(path).read_text())


def sample_id(event,frame):return event.replace('/','_')+'_'+frame


def select_frames(rows):
    eligible=[r for r in rows if r['point'] is not None and r['clock_supported'] and r['angular_speed_deg_s'] is not None]
    if len(eligible)<3:raise ValueError('Need three measured gaze regions per event')
    low=min(eligible,key=lambda r:(r['angular_speed_deg_s'],r['pts']))
    high=max([r for r in eligible if r['frame_id']!=low['frame_id']],
             key=lambda r:(r['angular_speed_deg_s'],-r['pts']))
    remaining=[r for r in eligible if r['frame_id'] not in {low['frame_id'],high['frame_id']}]
    middle=min(remaining,key=lambda r:abs(r['pts']-np.median([v['pts'] for v in eligible])))
    return sorted([low,high,middle],key=lambda r:r['pts'])


def prepare(output,audit_root):
    if output.exists():raise ValueError('Choose a new output directory')
    selection=read(audit_root/'selection.json')
    by_key={(r['scene_key'],r['row_index']):r for r in selection['samples']}
    selected=[by_key[k] for k in DEV+VALIDATION]
    contexts=load_contexts({'samples':selected},load_ground_truth=False)
    with (audit_root/'per_frame.csv').open(encoding='utf-8-sig',newline='') as f:
        motion={(r['scene'],int(r['row_index']),r['frame_id']):r for r in csv.DictReader(f)}
    output.mkdir(parents=True)
    samples=[];coverage=[]
    for ctx in contexts:
        key=f'{ctx.scene_key}/{ctx.row_index}'
        contract=read(audit_root/'pose_alignment'/key/'frame_pose_contract.json')
        if sha(ctx.json_path)!=contract['source_hashes']['telemetry_sha256']:raise ValueError('Telemetry changed')
        records=gaze_records(ctx);rows=[]
        for frame in contract['frames']:
            geometry=aligned_frame_geometry(frame,sha(frame['source_path']),frame['requested_seek_seconds'],
                                             contract['clock']['original_offset_seconds'])
            with Image.open(frame['source_path']) as im:rgb=im.convert('RGB')
            kwargs=dict(frame_time_video=geometry['video_time_seconds'],frame_time_telemetry=geometry['telemetry_time_seconds'],
                        event_window=[float(ctx.api_row['t_start']),float(ctx.api_row['t_end'])],
                        clock_available=geometry['clock_available'],image_size=rgb.size,reference_pose=geometry['reference_pose'])
            current,stats,projected=build_heatmap(ctx,records,strategy='instant',**kwargs)
            point=next((p['pixel_xy'] for p in projected if p['status']=='projected_frustum_only'),None)
            speed=motion[(ctx.scene_key,ctx.row_index,frame['frame_id'])]['angular_speed_deg_s']
            rows.append({'frame_id':frame['frame_id'],'pts':geometry['video_time_seconds'],
                         'geometry':geometry,'source_path':frame['source_path'],'source_sha256':frame['source_sha256'],
                         'point':point,'clock_supported':geometry['clock_available'],'current':current,
                         'current_stats':stats,'angular_speed_deg_s':float(speed) if speed else None})
        chosen=select_frames(rows)
        coverage.extend({'event':key,**{k:r[k] for k in ['frame_id','pts','clock_supported','angular_speed_deg_s']},
                         'gaze_projected':r['point'] is not None,
                         'selected':r['frame_id'] in {c['frame_id'] for c in chosen}} for r in rows)
        for frame in chosen:
            with Image.open(frame['source_path']) as im:rgb=im.convert('RGB')
            point=frame['point'];geometry=frame['geometry'];sid=sample_id(key,frame['frame_id'])
            legacy,_=render_presentation(rgb,frame['current'],'F_inst')
            corrected,fovstats=spatial_foveation(rgb,point,clock_supported=geometry['clock_available'],
                    vertical_fov=geometry['reference_pose']['fov_vertical_degrees'])
            history,hstats,_=build_heatmap(ctx,records,strategy='local_history',
                frame_time_video=frame['pts'],frame_time_telemetry=geometry['telemetry_time_seconds'],
                event_window=[float(ctx.api_row['t_start']),float(ctx.api_row['t_end'])],
                clock_available=True,image_size=rgb.size,reference_pose=geometry['reference_pose'])
            overlay,_=render_presentation(rgb,history,'H_local',{'overlay_max_alpha':.5})
            detail,crop=crop_at(rgb,point,.35)
            # Deliberate spatial mismatch, not a purported gaze measurement.
            wrong=[float((point[0]+rgb.width*.37)%rgb.width),float((point[1]+rgb.height*.29)%rgb.height)]
            sham,sham_crop=crop_at(rgb,wrong,.35)
            views={'RGB_REGION':(rgb,point),'LEGACY_FOVEATION':(legacy,point),
                   'SPATIAL_FOVEATION':(corrected,point),'DETAIL_035':(detail,crop['point_in_view']),
                   'HISTORY_OVERLAY':(overlay,point),'SHIFTED_DETAIL':(sham,sham_crop['point_in_view'])}
            result={'id':sid,'event':key,'frame_id':frame['frame_id'],'pts':frame['pts'],
                'split':'dev' if (ctx.scene_key,ctx.row_index) in DEV else 'validation',
                'source_path':frame['source_path'],'source_sha256':frame['source_sha256'],
                'point':point,'shifted_point':wrong,'angular_speed_deg_s':frame['angular_speed_deg_s'],
                'motion_bin':'low_lt10' if frame['angular_speed_deg_s']<10 else 'medium_10to45' if frame['angular_speed_deg_s']<45 else 'high_ge45',
                'geometry':geometry,'foveation_stats':fovstats,'crop':crop,'shifted_crop':sham_crop,
                'history_stats':hstats,'conditions':{}}
            for condition,(view,local_point) in views.items():
                # Each view explicitly identifies its own region; no cue inversion is requested.
                context_point=wrong if condition=='SHIFTED_DETAIL' else point
                card=evidence_card(rgb,view,context_point,local_point,frame_id=frame['frame_id'],pts=frame['pts'])
                path=output/'inputs'/sid/f'{condition}.png';path.parent.mkdir(parents=True,exist_ok=True);card.save(path)
                result['conditions'][condition]={'path':str(path.relative_to(output)),'sha256':sha(path),'size':list(card.size)}
            samples.append(result)
        print('PREPARED',key,flush=True)
    manifest={'version':VERSION,'categories':CATEGORIES,'prompt':PROMPT,'conditions':CONDITIONS,'samples':samples,
        'coverage':coverage,'image_size':[896,480],'selection':'GT-free supported gaze, per event min motion/max motion/middle PTS, selected before model outputs',
        'annotation_source':'assistant visual review, provisional; not human pixel GT or event referent GT',
        'gt_loaded':False,'anchors_projected':False,
        'scope':'script-selected region semantic recognition; not final referring accuracy',
        'code_sha256':sha(__file__),'renderer_sha256':sha(Path(__file__).with_name('scripted_attention.py'))}
    save(output/'manifest.json',manifest)
    save(output/'labels_template.json',{'annotation_source':manifest['annotation_source'],'frozen_before_inference':False,
        'samples':{s['id']:{'accepted':[],'shifted_accepted':[],'review_note':''} for s in samples}})
    gallery(output,manifest)
    return manifest


def gallery(output,manifest):
    for split in ['dev','validation']:
        samples=[s for s in manifest['samples'] if s['split']==split]
        # Annotation reference is raw RGB detail, blind to tested Qwen outputs.
        canvas=Image.new('RGB',(448*3,300*len(samples)),'white');draw=ImageDraw.Draw(canvas)
        for index,s in enumerate(samples):
            for col,condition in enumerate(['RGB_REGION','DETAIL_035','SHIFTED_DETAIL']):
                with Image.open(output/s['conditions'][condition]['path']) as im:
                    panel=im.crop((448,32,896,480)).resize((260,260),Image.Resampling.LANCZOS)
                x,y=col*448,index*300
                canvas.paste(panel,(x,y+36));draw.text((x+5,y+3),s['id'],fill='black')
                draw.text((x+5,y+18),condition,fill='black')
        # Five-row slices avoid requiring a huge image to inspect labels.
        for page in range(0,len(samples),5):
            path=output/f'annotation_{split}_{page//5+1}.jpg'
            canvas.crop((0,page*300,1344,min((page+5)*300,canvas.height))).save(path,quality=95)
    data=json.dumps(manifest,ensure_ascii=False).replace('</','<\\/')
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>脚本组织区域语义试验</title>
<style>body{font:15px system-ui;background:#f5f6f8;color:#17212b;margin:20px}header{position:sticky;top:0;background:#f5f6f8;padding:12px}select{font:inherit}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:14px}figure{margin:0;padding:10px;background:white}img{width:100%}p{max-width:1100px}</style>
<header><h2>脚本处理图像；模型只辨认指定区域</h2><select id="sample"></select><span id="meta"></span></header>
<p>各图尺寸、提示和视觉预算相同。右侧四个角标中间是脚本指定区域。SHIFTED_DETAIL是人为错移区域，仅检验图像内容/对应变化。无GT投影；区域类别为助手视觉复核标签，不能作最终指代准确率。</p><main id="views"></main>
<script>const d=__DATA__;const select=document.querySelector('#sample');d.samples.forEach((s,i)=>{let o=document.createElement('option');o.value=i;o.textContent=s.id+' / '+s.split;select.append(o)});function render(){const s=d.samples[+select.value];document.querySelector('#meta').textContent=' | '+s.angular_speed_deg_s.toFixed(1)+'deg/s | PTS '+s.pts.toFixed(3);const main=document.querySelector('#views');main.replaceChildren();d.conditions.forEach(c=>{if(!s.conditions[c])return;let f=document.createElement('figure'),title=document.createElement('figcaption'),img=document.createElement('img');title.textContent=c;img.src=s.conditions[c].path;f.append(title,img);main.append(f)})}select.onchange=render;render();</script></html>'''
    (output/'index.html').write_text(html.replace('__DATA__',data))


def infer(output,manifest,split,conditions,max_calls=None):
    labels_path=output/'labels.json'
    labels=read(labels_path)
    if not labels.get('frozen_before_inference'):raise ValueError('Labels must be reviewed/frozen before inference')
    for sample in manifest['samples']:
        annotation=labels['samples'][sample['id']]
        for field in ['accepted','shifted_accepted']:
            values=annotation[field]
            if not values or any(v not in manifest['categories'] for v in values):
                raise ValueError(f'Invalid frozen labels: {sample["id"]} {field}')
    label_hash=sha(labels_path)
    lock=output/'annotation_lock.json'
    if lock.exists() and read(lock)['sha256']!=label_hash:raise ValueError('Labels changed after inference began')
    save(lock,{'sha256':label_hash,'source':labels['annotation_source']})
    backend=QwenBackend(output/'model_cache',model=str(ROOT.parent/'Qwen3-VL-8B-Instruct'),local_only=True,
                        namespace=VERSION,trace_visual_inputs=True)
    schedule=[(s,c) for s in manifest['samples'] if s['split']==split for c in conditions]
    random.Random(20261105).shuffle(schedule)
    calls=0
    for sample,condition in schedule:
        destination=output/'responses'/sample['id']/f'{condition}.json'
        file=sample['conditions'][condition];path=output/file['path']
        if sha(path)!=file['sha256']:raise ValueError('Input modified')
        if destination.exists() and read(destination).get('status')=='COMPLETE':
            cached=read(destination);request=read(cached['request_path'])
            if (request['prompt']!=manifest['prompt'] or request['image_sha256']!=[file['sha256']]
                    or request['model']!=backend.model_path or request['max_new_tokens']!=120):
                raise ValueError('Cached response differs from current image/prompt/model protocol')
            continue
        if max_calls is not None and calls>=max_calls:break
        print('INFER',sample['id'],condition,flush=True);start=time.monotonic()
        try:
            parsed=backend.ask(sample['id'],condition,manifest['prompt'],[path],max_new_tokens=120)
            request=read(backend.requests[-1]);category=parsed.get('category')
            accepted=labels['samples'][sample['id']]['shifted_accepted' if condition=='SHIFTED_DETAIL' else 'accepted']
            result={'sample':sample['id'],'event':sample['event'],'split':split,'condition':condition,
                'status':'COMPLETE','parsed':parsed,'category':category,'schema_valid':set(parsed)=={'category'} and category in manifest['categories'],
                'correct':category in accepted,'accepted':accepted,'motion_bin':sample['motion_bin'],
                'seconds':time.monotonic()-start,'request_path':backend.requests[-1],
                'processor_audit':request['processor_audit'],'vision_verified':request['visual_tensor_trace']['input_consumption_verified']}
        except Exception as exc:
            result={'sample':sample['id'],'split':split,'condition':condition,'status':'FAILED',
                    'error':f'{type(exc).__name__}: {exc}','correct':False,'seconds':time.monotonic()-start}
            save(destination,result);summarize(output,manifest);raise
        save(destination,result);calls+=1
        print('RESULT',result['category'],'correct',result['correct'],f'{result["seconds"]:.1f}s',flush=True)
        summarize(output,manifest)
    return summarize(output,manifest)


def summarize(output,manifest):
    rows=[read(p) for p in (output/'responses').glob('*/*.json')]
    stats={}
    for split in ['dev','validation']:
        stats[split]={}
        for condition in sorted({r['condition'] for r in rows}):
            selected=[r for r in rows if r['split']==split and r['condition']==condition]
            stats[split][condition]={'n':len(selected),'correct':sum(r.get('correct',False) for r in selected),
                'schema_valid':sum(r.get('schema_valid',False) for r in selected),
                'by_motion':{b:{'n':sum(r.get('motion_bin')==b for r in selected),
                    'correct':sum(r.get('correct',False) and r.get('motion_bin')==b for r in selected)}
                    for b in ['low_lt10','medium_10to45','high_ge45']}}
    summary={'version':VERSION,'call_counts':dict(Counter(r['status'] for r in rows)),
             'stats':stats,'visual_token_counts':sorted({r['processor_audit']['expanded_image_token_count'] for r in rows if r['status']=='COMPLETE'}),
             'all_complete_vision_verified':all(r['vision_verified'] for r in rows if r['status']=='COMPLETE'),
             'annotation_source':manifest['annotation_source'],'final_referent_evaluator_run':False}
    save(output/'summary.json',summary)
    return summary


def add_detail(output,manifest,ratio=.5):
    condition=f'DETAIL_{round(ratio*100):03d}'
    if condition in manifest['conditions']:return manifest
    if (output/'validation_lock.json').exists():raise ValueError('No iteration after validation begins')
    manifest['conditions'].append(condition)
    for sample in manifest['samples']:
        with Image.open(sample['source_path']) as im:rgb=im.convert('RGB')
        view,meta=crop_at(rgb,sample['point'],ratio)
        card=evidence_card(rgb,view,sample['point'],meta['point_in_view'],frame_id=sample['frame_id'],pts=sample['pts'])
        path=output/'inputs'/sample['id']/f'{condition}.png';card.save(path)
        sample['conditions'][condition]={'path':str(path.relative_to(output)),'sha256':sha(path),'size':list(card.size)}
        sample.setdefault('additional_crops',{})[condition]=meta
    save(output/'manifest.json',manifest);gallery(output,manifest)
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','infer','summarize','add-large-detail','add-small-detail'])
    p.add_argument('--output',type=Path,default=ROOT/'.local-sync/scripted_region_semantics_20261105')
    p.add_argument('--audit-root',type=Path,default=ROOT/'.local-sync/projection_expansion24_20261105')
    p.add_argument('--split',choices=['dev','validation'],default='dev')
    p.add_argument('--conditions',nargs='+',choices=CONDITIONS+['DETAIL_018','DETAIL_050'])
    p.add_argument('--max-calls',type=int)
    a=p.parse_args(argv);output=a.output.resolve()
    if a.action=='prepare':prepare(output,a.audit_root.resolve());return
    manifest=read(output/'manifest.json')
    if a.action=='add-large-detail':add_detail(output,manifest,.5);return
    if a.action=='add-small-detail':add_detail(output,manifest,.18);return
    if a.action=='summarize':summarize(output,manifest);return
    conditions=a.conditions or manifest['conditions']
    if a.split=='validation':
        lock=output/'validation_lock.json'
        protocol={'conditions':conditions,'manifest_sha256':sha(output/'manifest.json'),
                  'prompt_sha256':hashlib.sha256(manifest['prompt'].encode()).hexdigest()}
        if lock.exists() and read(lock)!=protocol:raise ValueError('Validation protocol frozen')
        save(lock,protocol)
    infer(output,manifest,a.split,conditions,a.max_calls)


if __name__=='__main__':main()
