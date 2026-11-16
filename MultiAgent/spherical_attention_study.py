"""Fixed-reference spherical gaze + recorded RGB atlas and visual priority probe.

The probe asks which RGB angular region deserves inspection, not gaze coordinates
or target-object identity. Gaze enters the model only through rendered images.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from .camera_geometry import aligned_frame_geometry
from .context import load_contexts
from .gaze_evidence import gaze_records
from .heatmap_temporal_pilot import save,sha
from .qwen_backend import QwenBackend
from .spherical_attention import (build_spherical_volume,angular_rgb_atlas,overlay_on_atlas,
    render_spherical_debug,spherical_grid,angles_to_directions,_fov_mask,solid_angle_weights)

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'.local-sync/spherical_attention_20261106'
AUDIT=ROOT/'.local-sync/projection_expansion24_20261105'
VERSION='spherical_visual_priority_probe_v1'
CONDITIONS=['TRUE_HEAT','REVERSE_TIME','SHIFT_SPACE','NO_HEAT']
PROMPT='''Choose which labeled RGB region should be inspected first, based on the user's visual attention.
Three equal angular regions A, B, C are marked in the upper RGB scene and repeated as enlarged RGB strips at the bottom.
The middle rows show EARLY, MIDDLE and LATE gaze heatmaps in the exact same fixed spatial coordinates as the upper scene.
Added orange heat indicates measured user attention, not object color or a target label. Gray marks recorded camera coverage. Dark areas have no recorded RGB.
Choose the RGB region containing the BRIGHTEST added heat peak in EARLY, then in LATE. Use none when that time row has no added heat.
Use the heatmap images to prioritize inspection; do not guess from which RGB object looks interesting. The images may be controls.
Return only JSON with exactly two fields, each A/B/C/none: {"early":"A","late":"B"}. No coordinates, explanations or object labels.'''


def read(path):return json.loads(Path(path).read_text())


def unwrap_viewport(owner):
    """Choose a small angular viewport; storage stays a full fixed sphere."""
    h,w=owner.shape;seen=owner>=0
    columns=np.flatnonzero(np.any(seen,axis=0))
    if not len(columns):raise ValueError('No recorded RGB footprint')
    # Place the largest unobserved circular gap at the external seam.
    extended=np.r_[columns,columns[0]+w];gaps=np.diff(extended)
    k=int(np.argmax(gaps));start=int(extended[k+1]%w)
    rolled=np.roll(seen,-start,axis=1);ys,xs=np.nonzero(rolled)
    x0=max(0,int(xs.min())-3);x1=min(w,int(xs.max())+4)
    y0=max(0,int(ys.min())-3);y1=min(h,int(ys.max())+4)
    return {'roll_columns':-start,'bbox':[x0,y0,x1,y1],
            'yaw_span_degrees':(x1-x0)/w*360,'pitch_span_degrees':(y1-y0)/h*180}


def viewport_array(array,view):
    rolled=np.roll(array,view['roll_columns'],axis=1)
    x0,y0,x1,y1=view['bbox'];return rolled[y0:y1,x0:x1]


def make_probe_card(atlas,maps,coverage,owner,view,condition):
    rgb=viewport_array(np.asarray(atlas),view)
    # Maps/coverage are enlarged to the exact atlas grid before viewport selection.
    w,h=atlas.size
    mass=np.stack([np.asarray(Image.fromarray(plane).resize((w,h),Image.Resampling.BILINEAR)) for plane in maps])
    density=mass/solid_angle_weights(w,h)[None,...]
    planes=np.stack([viewport_array(plane,view) for plane in density])
    seen=viewport_array(owner,view)>=0
    if condition=='REVERSE_TIME':planes=planes[::-1].copy()
    elif condition=='SHIFT_SPACE':planes=np.roll(planes,planes.shape[2]//3,axis=2)
    elif condition=='NO_HEAT':planes=np.zeros_like(planes)
    elif condition!='TRUE_HEAT':raise ValueError('Unknown condition')
    height,width=rgb.shape[:2];cuts=[0,width//3,2*width//3,width]
    # Ground truth is the intended visible peak, not target identity or hidden text.
    # Keep mass-integral priorities too, so differences can be audited.
    peak_expected=[];integral_expected=[]
    for i in [0,2]:
        values=[float(planes[i,:,cuts[j]:cuts[j+1]].max()) for j in range(3)]
        integrals=[float(planes[i,:,cuts[j]:cuts[j+1]].sum()) for j in range(3)]
        peak_expected.append('none' if max(values)<=0 else 'ABC'[int(np.argmax(values))])
        integral_expected.append('none' if max(integrals)<=0 else 'ABC'[int(np.argmax(integrals))])
    # Same display scale in all temporal rows, gated by true supported seconds.
    global_peak=float(planes.max());support=min(1.,float(maps.sum())/.25)
    canvas=Image.new('RGB',(1008,1008),'#15212b');draw=ImageDraw.Draw(canvas)
    draw.text((12,8),'FIXED DIRECTION MAP | RECORDED RGB | UNKNOWN = DARK',fill='white')
    box_width=984;left=12
    def put_panel(array,y,ph,label):
        image=Image.fromarray(np.asarray(array,dtype=np.uint8)).resize((box_width,ph),Image.Resampling.BILINEAR)
        canvas.paste(image,(left,y));draw.text((left+4,y+3),label,fill='white',stroke_width=1,stroke_fill='black')
        for j,letter in enumerate('ABC'):
            x=left+box_width*j//3;draw.rectangle((x,y,x+box_width//3-1,y+ph-1),outline='white',width=2)
            draw.text((x+box_width//6-5,y+8),letter,fill='white',stroke_width=2,stroke_fill='black')
    put_panel(rgb,28,240,'RGB')
    for i,label in enumerate(['EARLY','MIDDLE','LATE']):
        base=np.full_like(rgb,20);base[seen]=[65,65,65]
        if global_peak>0:
            a=(np.sqrt(planes[i]/global_peak)*.95*support)[...,None]
            base=np.clip(base*(1-a)+np.array([255,85,25])*a,0,255).astype('uint8')
        put_panel(base,284+i*140,126,label)
    for j,letter in enumerate('ABC'):
        crop=Image.fromarray(rgb[:,cuts[j]:cuts[j+1]]).resize((320,240),Image.Resampling.BILINEAR)
        x=12+j*328;canvas.paste(crop,(x,738));draw.rectangle((x,738,x+319,977),outline='white',width=2)
        draw.text((x+10,744),letter,fill='white',stroke_width=2,stroke_fill='black')
    draw.text((12,985),'Same RGB in all controls. Heatmap stays visual. Regions are not object IDs.',fill='white')
    return canvas,{'early':peak_expected[0],'late':peak_expected[1]}, {'mass_integral_priority':{'early':integral_expected[0],'late':integral_expected[1]},
        'peak_and_integral_agree':peak_expected==integral_expected}


def prepare(output):
    if (output/'manifest.json').exists():raise ValueError('Never overwrite prepared study')
    output.mkdir(parents=True,exist_ok=True)
    selection=read(AUDIT/'selection.json');contexts=load_contexts(selection,load_ground_truth=False)
    events=[];fallback=[]
    for ctx in contexts:
        key=f'{ctx.scene_key}/{ctx.row_index}';contract=read(AUDIT/'pose_alignment'/key/'frame_pose_contract.json')
        if sha(ctx.json_path)!=contract['source_hashes']['telemetry_sha256']:raise ValueError('Telemetry hash changed')
        frames=[]
        for f in contract['frames']:
            g=aligned_frame_geometry(f,sha(f['source_path']),f['requested_seek_seconds'],contract['clock']['original_offset_seconds'])
            if g['clock_available']:
                frames.append({'frame_id':f['frame_id'],'source_path':f['source_path'],'source_sha256':f['source_sha256'],
                    'pts':g['video_time_seconds'],'geometry':g})
        if len(frames)<3:fallback.append({'event':key,'supported_frames':len(frames),'status':'RGB_FALLBACK'});continue
        frames.sort(key=lambda f:f['pts']);reference=frames[0];canonical=reference['geometry']['reference_pose']
        query=frames[-1];qg=query['geometry'];records=gaze_records(ctx)
        history=max(.1,qg['telemetry_time_seconds']-max(0.,reference['geometry']['telemetry_time_seconds']))
        with Image.open(reference['source_path']) as im:size=im.size
        maps,coverage,stats=build_spherical_volume(records,query_time=qg['telemetry_time_seconds'],canonical_pose=canonical,
            current_pose=qg['reference_pose'],image_size=size,clock_supported=True,history_seconds=history)
        atlas,owner=angular_rgb_atlas(frames,canonical)
        directory=output/'events'/key;directory.mkdir(parents=True,exist_ok=True)
        atlas.save(directory/'angular_rgb.png');overlay_on_atlas(atlas,maps).save(directory/'angular_heat_overlay.png')
        render_spherical_debug(maps,coverage,stats).save(directory/'sphere_heat_only.png')
        np.savez_compressed(directory/'volume.npz',duration_maps=maps,camera_coverage_seconds=coverage,rgb_source_index=owner)
        view=unwrap_viewport(owner)
        positions=np.array([f['geometry']['reference_pose']['position_world'] for f in frames]);motion=np.linalg.norm(positions-positions[0],axis=1)
        rotations=Rotation.from_quat(np.array([f['geometry']['reference_pose']['rotation_xyzw'] for f in frames]))
        angles=(rotations[0].inv()*rotations).magnitude()*180/math.pi
        entry={'id':key.replace('/','_'),'event':key,'frames':frames,'reference_frame_id':reference['frame_id'],
            'canonical_pose':canonical,'query_frame_id':query['frame_id'],'query_pts':query['pts'],
            'history_seconds':history,'volume_stats':stats,'viewport':view,
            'max_camera_translation_from_reference_world_units':float(motion.max()),
            'max_head_rotation_from_reference_degrees':float(angles.max()),
            'rgb_atlas':'events/'+key+'/angular_rgb.png','rgb_heat_overlay':'events/'+key+'/angular_heat_overlay.png',
            'heat_only':'events/'+key+'/sphere_heat_only.png','conditions':{}}
        for c in CONDITIONS:
            card,expected,audit=make_probe_card(atlas,maps,coverage,owner,view,c)
            path=directory/f'{c}.png';card.save(path)
            entry['conditions'][c]={'path':str(path.relative_to(output)),'sha256':sha(path),'size':list(card.size),
                'expected_visual_priority':expected,'priority_audit':audit}
        events.append(entry);print('PREPARED',key,'frames',len(frames),'history',round(history,2),'viewport',round(view['yaw_span_degrees'],1),flush=True)
    manifest={'version':VERSION,'conditions':CONDITIONS,'prompt':PROMPT,'events':events,'fallback_events':fallback,
        'gt_loaded':False,'anchors_projected':False,'instructions_in_model_prompt':False,
        'attention_input':'rendered image only; no numeric candidate weights in prompt',
        'candidate_semantics':'equal angular RGB regions, not object IDs','scope':'visual attention priority probe, not final referring evaluation',
        'code_sha256':sha(__file__),'sphere_code_sha256':sha(Path(__file__).with_name('spherical_attention.py'))}
    save(output/'manifest.json',manifest);gallery(output,manifest)


def gallery(output,m):
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>转头后的统一方向热图</title>
<style>body{font:15px system-ui;margin:20px;background:#f2f5f7;color:#17212b}header{position:sticky;top:0;background:#f2f5f7;padding:12px}select{font:inherit;margin:8px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:15px}figure{margin:0;padding:12px;background:white}img{width:100%}p{max-width:1300px}</style>
<header><h2>固定方向坐标中的注视历史</h2><select id="event"></select><select id="mode"><option value="overview">统一方向图</option><option value="probe">视觉优先级筛选对照</option><option value="sources">原始帧</option></select><span id="meta"></span></header>
<p>每段交互以第一张受支持帧为固定方向参考。单帧只是球面上的一个视野窗口，转头后各视野在同一坐标中占不同区域。灰色表示相机曾覆盖的方向，深色表示缺少RGB。RGB角度图仅由原始帧按旋转采样，不是完整三维重建；平移、遮挡、动态物体会造成错位。每个像素保留原始帧索引。</p>
<p>模型实验只看图中的热图，选择A/B/C哪个RGB区域优先检查，不接收数值注意分数。三个时间面板共用方向坐标和颜色。对照包括真热图、时间反转、空间错移、无热图。此实验尚不评价具体对象或最终referent。</p><main></main>
<script>const d=__DATA__,q=document.querySelector('#event'),mode=document.querySelector('#mode');d.events.forEach((e,i)=>{let o=document.createElement('option');o.value=i;o.textContent=e.event;q.append(o)});function render(){const e=d.events[+q.value],main=document.querySelector('main');main.replaceChildren();document.querySelector('#meta').textContent=' | '+e.frames.length+'帧 | 最大相对转角 '+e.max_head_rotation_from_reference_degrees.toFixed(1)+'° | 方向图显示跨度 '+e.viewport.yaw_span_degrees.toFixed(1)+'°';let items=mode.value==='overview'?[['记录过的RGB方向图',e.rgb_atlas],['RGB＋统一热图',e.rgb_heat_overlay],['球面注视密度和视野覆盖',e.heat_only]]:mode.value==='probe'?d.conditions.map(c=>[c,e.conditions[c].path]):e.frames.map(f=>[f.frame_id+' | PTS '+f.pts.toFixed(3)+'s',f.source_path]);items.forEach(([title,path])=>{let f=document.createElement('figure'),c=document.createElement('figcaption'),im=document.createElement('img');c.textContent=title;im.src=path;f.append(c,im);main.append(f)})}q.onchange=render;mode.onchange=render;render();</script></html>'''
    (output/'index.html').write_text(html.replace('__DATA__',json.dumps(m,ensure_ascii=False).replace('</','<\\/')))


def infer(output):
    m=read(output/'manifest.json');lock={'manifest_sha256':sha(output/'manifest.json'),'prompt_sha256':hashlib.sha256(m['prompt'].encode()).hexdigest()}
    if (output/'protocol_lock.json').exists() and read(output/'protocol_lock.json')!=lock:raise ValueError('Protocol changed')
    save(output/'protocol_lock.json',lock)
    backend=QwenBackend(output/'model_cache',model=str(ROOT.parent/'Qwen3-VL-8B-Instruct'),local_only=True,namespace=VERSION,trace_visual_inputs=True)
    schedule=[(e,c) for e in m['events'] for c in m['conditions']];random.Random(20261106).shuffle(schedule)
    for e,c in schedule:
        file=e['conditions'][c];path=output/file['path'];destination=output/'responses'/e['id']/f'{c}.json'
        if sha(path)!=file['sha256']:raise ValueError('Input changed')
        if destination.exists() and read(destination)['status']=='COMPLETE':continue
        print('INFER',e['event'],c,flush=True);start=time.monotonic()
        parsed=backend.ask(e['id'],c,m['prompt'],[path],max_new_tokens=120)
        request=read(backend.requests[-1]);expected=file['expected_visual_priority']
        valid=set(parsed)=={'early','late'} and all(parsed.get(k) in ['A','B','C','none'] for k in ['early','late'])
        save(destination,{'event':e['event'],'id':e['id'],'condition':c,'status':'COMPLETE','parsed':parsed,'schema_valid':valid,
            'expected':expected,'early_correct':valid and parsed['early']==expected['early'],
            'late_correct':valid and parsed['late']==expected['late'],'pair_correct':valid and parsed==expected,
            'seconds':time.monotonic()-start,'request_path':backend.requests[-1],
            'processor_audit':request['processor_audit'],'vision_verified':request['visual_tensor_trace']['input_consumption_verified']})
        print('RESULT',parsed,'expected',expected,flush=True)
    report(output)


def report(output):
    m=read(output/'manifest.json');rows=[read(p) for p in (output/'responses').glob('*/*.json')];stats={}
    for c in m['conditions']:
        r=[row for row in rows if row['condition']==c]
        stats[c]={'n':len(r),'early_correct':sum(s['early_correct'] for s in r),'late_correct':sum(s['late_correct'] for s in r),
                  'pair_correct':sum(s['pair_correct'] for s in r),'schema_valid':sum(s['schema_valid'] for s in r)}
    analysis={'conditions':stats,'model_calls':len(rows),'tokens':sorted({r['processor_audit']['expanded_image_token_count'] for r in rows}),
        'all_vision_verified':all(r['vision_verified'] for r in rows),'supported_events':len(m['events']),
        'rgb_fallback_events':len(m['fallback_events']),'scope':m['scope']}
    save(output/'analysis.json',analysis)
    payload=json.dumps(rows,ensure_ascii=False).replace('</','<\\/')
    html=(output/'index.html').read_text().split('<!-- RESULTS -->')[0]
    html+='<!-- RESULTS --><script>const results=__RESULTS__,baseRender=render;render=function(){baseRender();if(mode.value!=="probe")return;let e=d.events[+q.value];document.querySelectorAll("figure").forEach((f,i)=>{let r=results.find(r=>r.id===e.id&&r.condition===d.conditions[i]);if(r)f.querySelector("figcaption").textContent+=" | model "+JSON.stringify(r.parsed)+" | expected "+JSON.stringify(r.expected)});};q.onchange=render;mode.onchange=render;render();</script>'.replace('__RESULTS__',payload)
    (output/'index.html').write_text(html);print(json.dumps(analysis,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','infer','report']);p.add_argument('--output',type=Path,default=OUTPUT)
    a=p.parse_args();{'prepare':prepare,'infer':infer,'report':report}[a.action](a.output.resolve())
