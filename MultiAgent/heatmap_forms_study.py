"""Alternative attention construction, presentation and script-routed RGB memory.

Old examples are regression only. Fresh interactions are frozen before inference.
GT/referent labels are never loaded or projected.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import subprocess
import time
import numpy as np
from PIL import Image,ImageDraw

from .attention_volume import build_volume,render_volume,heat_guided_crop,VERSION as VOLUME_VERSION
from .camera_geometry import aligned_frame_geometry,interpolate_pose
from .context import load_contexts
from .gaze_evidence import build_heatmap,gaze_records,render_presentation,project_gaze
from .heatmap_temporal_pilot import save,sha
from .local_video_visual import probe_frames
from .qwen_backend import QwenBackend
from .scripted_attention import crop_at,evidence_card
from .scripted_region_study import select_frames,PROMPT,CATEGORIES

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'.local-sync/heatmap_forms_v2_20261105'
OLD=ROOT/'.local-sync/scripted_region_semantics_20261105'
AUDIT=ROOT/'.local-sync/projection_expansion24_20261105'
VERSION='heatmap_forms_and_script_routing_v2'
FRESH=[('scene3',109),('scene4_room2',149),('scene5',112)]
CONDITIONS=['RGB','LEGACY_FILLED','MASS_CONTOURS','MASS_SPOTLIGHT','CROP_018','HEAT_ADAPTIVE_CROP']
TEMPORAL_EVENTS=['scene3/577','scene2/411','scene4_room1/155','scene3/109','scene4_room2/149','scene5/112']
TCONDITIONS=['CURRENT_CROP','NATIVE_CROP','CURRENT_CONTOUR']


def read(path):return json.loads(Path(path).read_text())


def contexts_for(keys):
    selection=read(AUDIT/'selection.json')
    selected=[s for s in selection['samples'] if f"{s['scene_key']}/{s['row_index']}" in keys]
    return {f'{c.scene_key}/{c.row_index}':c for c in load_contexts({'samples':selected},load_ground_truth=False)}


def prepare(output):
    if (output/'manifest.json').exists():raise ValueError('Preparation never overwrites existing manifest')
    output.mkdir(parents=True,exist_ok=True)
    previous=read(OLD/'manifest.json');labels=read(OLD/'labels.json')
    samples=[{k:s[k] for k in ['id','event','frame_id','pts','source_path','source_sha256','point','geometry','angular_speed_deg_s','motion_bin']} for s in previous['samples']]
    for s in samples:s['split']='regression'
    keys={s['event'] for s in samples}|{f'{a}/{b}' for a,b in FRESH}
    contexts=contexts_for(keys)
    with (AUDIT/'per_frame.csv').open(encoding='utf-8-sig',newline='') as f:
        motion={(r['scene'],int(r['row_index']),r['frame_id']):r for r in csv.DictReader(f)}
    records={key:gaze_records(ctx) for key,ctx in contexts.items()}
    fresh_coverage=[]
    for scene,row_index in FRESH:
        event=f'{scene}/{row_index}';ctx=contexts[event]
        contract=read(AUDIT/'pose_alignment'/event/'frame_pose_contract.json');eligible=[]
        if sha(ctx.json_path)!=contract['source_hashes']['telemetry_sha256']:raise ValueError('Telemetry changed')
        for frame in contract['frames']:
            geometry=aligned_frame_geometry(frame,sha(frame['source_path']),frame['requested_seek_seconds'],contract['clock']['original_offset_seconds'])
            with Image.open(frame['source_path']) as im:size=im.size
            _,_,projected=build_heatmap(ctx,records[event],frame_time_video=geometry['video_time_seconds'],
                frame_time_telemetry=geometry['telemetry_time_seconds'],event_window=[0,0],
                clock_available=geometry['clock_available'],image_size=size,strategy='instant',reference_pose=geometry['reference_pose'])
            point=next((r['pixel_xy'] for r in projected if r['status']=='projected_frustum_only'),None)
            speed=motion[(scene,row_index,frame['frame_id'])]['angular_speed_deg_s']
            eligible.append({'id':event.replace('/','_')+'_'+frame['frame_id'],'event':event,'frame_id':frame['frame_id'],
                'pts':geometry['video_time_seconds'],'source_path':frame['source_path'],'source_sha256':frame['source_sha256'],
                'point':point,'geometry':geometry,'clock_supported':geometry['clock_available'],
                'angular_speed_deg_s':float(speed) if speed else None,'split':'fresh'})
        chosen=select_frames(eligible)
        fresh_coverage.extend({k:r[k] for k in ['id','point','clock_supported','angular_speed_deg_s']} for r in eligible)
        for s in chosen:
            s['motion_bin']='low_lt10' if s['angular_speed_deg_s']<10 else 'medium_10to45' if s['angular_speed_deg_s']<45 else 'high_ge45'
            samples.append(s)
    for s in samples:
        ctx=contexts[s['event']];g=s['geometry']
        contract=read(AUDIT/'pose_alignment'/s['event']/'frame_pose_contract.json')
        if sha(ctx.json_path)!=contract['source_hashes']['telemetry_sha256']:raise ValueError('Telemetry changed')
        if sha(s['source_path'])!=s['source_sha256']:raise ValueError('Source frame changed')
        with Image.open(s['source_path']) as im:rgb=im.convert('RGB')
        maps,stats=build_volume(records[s['event']],query=g['telemetry_time_seconds'],pose=g['reference_pose'],
                image_size=rgb.size,clock_supported=g['clock_available'])
        legacy,_,_=build_heatmap(ctx,records[s['event']],frame_time_video=s['pts'],frame_time_telemetry=g['telemetry_time_seconds'],
            event_window=[0,0],clock_available=g['clock_available'],image_size=rgb.size,strategy='local_history',reference_pose=g['reference_pose'])
        filled,_=render_presentation(rgb,legacy,'H_local',{'overlay_max_alpha':.5})
        contour=render_volume(rgb,maps,stats,'CONTOURS');spot=render_volume(rgb,maps,stats,'SPOTLIGHT')
        crop,meta=crop_at(rgb,s['point'],.18);adaptive,ameta=heat_guided_crop(rgb,s['point'],maps)
        views={'RGB':(rgb,s['point']),'LEGACY_FILLED':(filled,s['point']),
            'MASS_CONTOURS':(contour,s['point']),'MASS_SPOTLIGHT':(spot,s['point']),
            'CROP_018':(crop,meta['point_in_view']),'HEAT_ADAPTIVE_CROP':(adaptive,ameta['point_in_view'])}
        s['volume']=stats;s['adaptive_crop']=ameta;s['conditions']={}
        directory=output/'inputs'/s['id'];directory.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(directory/'volume.npz',duration_maps=maps)
        for c,(view,point) in views.items():
            card=evidence_card(rgb,view,s['point'],point,frame_id=s['frame_id'],pts=s['pts'])
            path=directory/f'{c}.png';card.save(path)
            s['conditions'][c]={'path':str(path.relative_to(output)),'sha256':sha(path),'size':list(card.size)}
        separate=render_volume(rgb,maps,stats,'SEPARATE_MAP');separate.save(directory/'density_only.png')
        print('PREPARED',s['id'],s['split'],'duration',round(stats['total_frustum_duration'],3),flush=True)
    manifest={'version':VERSION,'volume_version':VOLUME_VERSION,'categories':CATEGORIES,'prompt':PROMPT,
        'conditions':CONDITIONS,'samples':samples,'fresh_source_coverage':fresh_coverage,
        'design_frozen_before_fresh_model_outputs':True,'image_size':[896,480],
        'previous_manifest_sha256':sha(OLD/'manifest.json'),'gt_loaded':False,'anchors_projected':False,
        'fresh_scope':'3 unseen interactions from audited supported clocks; scene overlap with regression',
        'label_scope':'assistant review only, not human object/referent GT',
        'code_sha256':sha(__file__),'volume_code_sha256':sha(Path(__file__).with_name('attention_volume.py'))}
    save(output/'manifest.json',manifest)
    label_data={'frozen_before_inference':False,'annotation_source':labels['annotation_source'],
        'samples':{s['id']:labels['samples'].get(s['id'],{'accepted':[],'ambiguous':False,'review_note':''}) for s in samples}}
    save(output/'labels_template.json',label_data)
    make_gallery(output,manifest)
    fresh=[s for s in samples if s['split']=='fresh']
    for i in range(0,len(fresh),3):
        canvas=Image.new('RGB',(896,480*min(3,len(fresh)-i)),'white')
        for j,s in enumerate(fresh[i:i+3]):
            with Image.open(output/s['conditions']['CROP_018']['path']) as im:canvas.paste(im,(0,j*480))
        canvas.save(output/f'fresh_annotation_{i//3+1}.jpg',quality=96)


def make_gallery(output,m):
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>时空注意表示对照</title>
<style>body{font:15px system-ui;margin:20px;background:#f2f5f7;color:#18232f}header{position:sticky;top:0;background:#f2f5f7;padding:12px}select{font:inherit;margin:10px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:14px}figure{background:white;padding:12px;margin:0}img{width:100%}p{max-width:1200px}</style>
<header><h2>热图的构造、呈现与使用</h2><select id="sample"></select><span id="meta"></span></header>
<p>旧/中/新三段分别构造停留时长密度，统一投影至当前帧。轮廓只画覆盖50%/85%质量的边界；聚光保留区域原色；热图还可由脚本用于选择局部尺度。所有条件查询位置相同。弱观测保持弱支持，不因归一化变成高置信度。RGB本身也有位置角标；不是无gaze基线。</p><p>REGRESSION为之前看过的30帧；FRESH为未参与此前优化的3交互9帧。人工像素GT缺失，类别标签是助手复核。<a href="temporal_memory.html">查看脚本组织的历史RGB记忆</a></p><main></main>
<script>const d=__DATA__;const q=document.querySelector('#sample');d.samples.forEach((s,i)=>{let o=document.createElement('option');o.value=i;o.textContent=s.id+' | '+s.split;q.append(o)});function render(){let s=d.samples[+q.value];document.querySelector('#meta').textContent=' | '+s.angular_speed_deg_s.toFixed(1)+'deg/s | support '+s.volume.bins.map(b=>b.duration.toFixed(2)+'s').join('/');let main=document.querySelector('main');main.replaceChildren();d.conditions.forEach(c=>{let f=document.createElement('figure'),title=document.createElement('figcaption'),img=document.createElement('img');title.textContent=c;img.src=s.conditions[c].path;f.append(title,img);main.append(f)})}q.onchange=render;render();</script></html>'''
    (output/'index.html').write_text(html.replace('__DATA__',json.dumps(m,ensure_ascii=False).replace('</','<\\/')))


def infer(output,split,conditions=None,max_calls=None,temporal=False):
    manifest=read(output/('temporal_manifest.json' if temporal else 'manifest.json'))
    labels_path=output/('temporal_labels.json' if temporal else 'labels.json');labels=read(labels_path)
    if not labels['frozen_before_inference']:raise ValueError('Freeze visual review before inference')
    label_hash=sha(labels_path);lock=output/('temporal_annotation_lock.json' if temporal else 'annotation_lock.json')
    if lock.exists() and read(lock)['sha256']!=label_hash:raise ValueError('Frozen labels changed')
    save(lock,{'sha256':label_hash,'source':labels['annotation_source']})
    chosen=conditions or manifest['conditions']
    if split=='fresh':
        plock=output/('temporal_fresh_lock.json' if temporal else 'fresh_lock.json')
        protocol={'manifest_sha256':sha(output/('temporal_manifest.json' if temporal else 'manifest.json')),
                  'labels_sha256':label_hash,'conditions':chosen}
        if plock.exists() and read(plock)!=protocol:raise ValueError('Fresh protocol changed')
        save(plock,protocol)
    response_dir=output/('temporal_responses' if temporal else 'responses')
    schedule=[(s,c) for s in manifest['samples'] if s['split']==split for c in chosen]
    random.Random(20261106).shuffle(schedule)
    backend=QwenBackend(output/'model_cache',model=str(ROOT.parent/'Qwen3-VL-8B-Instruct'),local_only=True,
        namespace=VERSION,trace_visual_inputs=True);calls=0
    for s,c in schedule:
        destination=response_dir/s['id']/f'{c}.json';image=s['conditions'][c];path=output/image['path']
        if sha(path)!=image['sha256']:raise ValueError('Input hash changed')
        if destination.exists():
            response=read(destination)
            if response['status']=='COMPLETE':
                req=read(response['request_path'])
                if req['prompt']!=manifest['prompt'] or req['image_sha256']!=[image['sha256']]:raise ValueError('Cache protocol mismatch')
                continue
        if max_calls is not None and calls>=max_calls:break
        annotation=labels['samples'][s['id']]
        accepted=annotation['native_accepted' if c=='NATIVE_CROP' else 'accepted']
        if not accepted or any(v not in CATEGORIES for v in accepted):raise ValueError('Incomplete annotation')
        print('INFER',s['id'],c,flush=True);start=time.monotonic()
        try:
            parsed=backend.ask(s['id'],c,manifest['prompt'],[path],max_new_tokens=120)
            req=read(backend.requests[-1]);category=parsed.get('category')
            response={'sample':s['id'],'condition':c,'split':split,'event':s['event'],'status':'COMPLETE',
                'parsed':parsed,'category':category,'correct':category in accepted,'accepted':accepted,
                'schema_valid':set(parsed)=={'category'} and category in CATEGORIES,
                'request_path':backend.requests[-1],'seconds':time.monotonic()-start,
                'processor_audit':req['processor_audit'],'vision_verified':req['visual_tensor_trace']['input_consumption_verified']}
        except Exception as exc:
            save(destination,{'sample':s['id'],'condition':c,'split':split,'status':'FAILED','error':str(exc)})
            raise
        save(destination,response);calls+=1
        print('RESULT',category,response['correct'],flush=True)
    print('COMPLETED_THIS_INVOCATION',calls,flush=True)


def prepare_temporal(output):
    if (output/'temporal_manifest.json').exists():raise ValueError('Temporal preparation never overwrites')
    m=read(output/'manifest.json');contexts=contexts_for(set(TEMPORAL_EVENTS));samples=[]
    for event in TEMPORAL_EVENTS:
        ctx=contexts[event];records=gaze_records(ctx)
        # Select before semantic outputs: maximize support, then distinct source times.
        options=[s for s in m['samples'] if s['event']==event and all(b['modes'] for b in s['volume']['bins'])]
        if not options:raise ValueError(f'No full-history query: {event}')
        selected=max(options,key=lambda s:(min(b['duration'] for b in s['volume']['bins']),s['pts']))
        g=selected['geometry'];offset=g['selected_clock_offset_seconds']
        with Image.open(selected['source_path']) as im:current=im.convert('RGB')
        maps=np.load(output/'inputs'/selected['id']/'volume.npz')['duration_maps']
        frames,stream=probe_frames(ctx.video_path);native=[]
        for b in selected['volume']['bins']:
            obs=b['modes'][0]['observed_representative'];target=obs['gaze_time_seconds']+offset
            # Closest actual PTS not later than query; preserve PTS error explicitly.
            candidate=min([f for f in frames if f['pts_seconds']<=selected['pts']],key=lambda f:abs(f['pts_seconds']-target))
            if abs(candidate['pts_seconds']-target)>.08:raise ValueError('Native source frame too far from gaze')
            native.append((b,obs,candidate))
        selected_indices=sorted({row[2]['source_frame_index'] for row in native})
        directory=output/'native_frames'/event;directory.mkdir(parents=True,exist_ok=True)
        expression='+'.join(f'eq(n\\,{index})' for index in selected_indices)
        command=['ffmpeg','-hide_banner','-loglevel','error','-threads','4','-i',ctx.video_path,
                 '-vf',f'select={expression},scale=640:640','-vsync','0','-q:v','2',str(directory/'source_%04d.jpg')]
        subprocess.run(command,check=True,capture_output=True)
        paths=sorted(directory.glob('source_*.jpg'))
        if len(paths)!=len(selected_indices):raise ValueError('Native frame extraction index mismatch')
        image_by_index=dict(zip(selected_indices,paths))
        for b,obs,frame in native:
            sid=selected['id']+f'_T{b["index_old_to_new"]}'
            source=image_by_index[frame['source_frame_index']]
            with Image.open(source) as im:rgb=im.convert('RGB')
            query=frame['pts_seconds']-offset
            p=interpolate_pose(records,query)
            record=next(r for r in records if r['source_index']==obs['source_index'])
            projection=project_gaze(record,p,*rgb.size)
            if projection['status']!='projected_frustum_only':raise ValueError('Native gaze not supported')
            current_point=obs['pixel_xy'];native_point=projection['pixel_xy']
            cc,cm=crop_at(current,current_point,.18);nc,nm=crop_at(rgb,native_point,.18)
            # Single-bin contour exposes the same region, without requiring time-color reading.
            one_maps=np.zeros_like(maps);one_maps[b['index_old_to_new']]=maps[b['index_old_to_new']]
            ct=render_volume(current,one_maps,selected['volume'],'CONTOURS')
            views={'CURRENT_CROP':(current,cc,current_point,cm['point_in_view'],selected['pts']),
                   'NATIVE_CROP':(rgb,nc,native_point,nm['point_in_view'],frame['pts_seconds']),
                   'CURRENT_CONTOUR':(current,ct,current_point,current_point,selected['pts'])}
            row={'id':sid,'event':event,'split':selected['split'],'query_sample':selected['id'],
                'time_bin':b['index_old_to_new'],'window':b['window'],'duration':b['duration'],
                'coverage':b['coverage'],'selected_mode_mass_fraction':b['modes'][0]['mass_fraction'],
                'alternatives':b['modes'][1:],'observation':obs,
                'current_point':current_point,'native_point':native_point,
                'current_source_path':selected['source_path'],'current_pts':selected['pts'],
                'native_source_path':str(source),'native_source_sha256':sha(source),'native_frame':frame,
                'native_telemetry_query':query,'gaze_to_frame_error_seconds':frame['pts_seconds']-(obs['gaze_time_seconds']+offset),
                'selected_clock_offset_seconds':offset,'native_pose':p,'native_crop':nm,'current_crop':cm,'conditions':{}}
            out=output/'temporal_inputs'/sid;out.mkdir(parents=True,exist_ok=True)
            for c,(context,view,context_point,view_point,pts) in views.items():
                card=evidence_card(context,view,context_point,view_point,frame_id=f'T{b["index_old_to_new"]}',pts=pts)
                path=out/f'{c}.png';card.save(path)
                row['conditions'][c]={'path':str(path.relative_to(output)),'sha256':sha(path),'size':list(card.size)}
            samples.append(row)
        print('TEMPORAL_PREPARED',event,selected['id'],flush=True)
    manifest={'version':VERSION,'conditions':TCONDITIONS,'prompt':PROMPT,'categories':CATEGORIES,'samples':samples,
        'gt_loaded':False,'anchors_projected':False,'scope':'Semantic readout of each scripted time-bin representative, not final referring accuracy',
        'native_frame_policy':'nearest actual decoded PTS to observed gaze under same supported clock, <= current query, <=80ms mismatch',
        'selection':'max min-bin supported duration among predetermined frames, then latest PTS; no model outputs/labels'}
    save(output/'temporal_manifest.json',manifest)
    save(output/'temporal_labels_template.json',{'frozen_before_inference':False,
        'annotation_source':'assistant visual review, provisional; not human pixel GT or event referent GT',
        'samples':{s['id']:{'accepted':[],'native_accepted':[],'ambiguous':False,'native_ambiguous':False,'review_note':''} for s in samples}})
    for i in range(0,len(samples),3):
        canvas=Image.new('RGB',(896,480*3),'white')
        for j,s in enumerate(samples[i:i+3]):
            for col,c in enumerate(['CURRENT_CROP','NATIVE_CROP']):
                with Image.open(output/s['conditions'][c]['path']) as im:panel=im.crop((448,0,896,480))
                canvas.paste(panel,(col*448,j*480));ImageDraw.Draw(canvas).text((col*448+2,j*480+2),s['id']+' '+c,fill='yellow')
        canvas.save(output/f'temporal_annotation_{i//3+1}.jpg',quality=96)
    make_memory_viewer(output,manifest)


def make_memory_viewer(output,m):
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>热图路由的时序RGB证据</title>
<style>body{font:15px system-ui;background:#f3f6f8;margin:20px}header{position:sticky;top:0;background:#f3f6f8;padding:12px}select{font:inherit;margin:8px}main{display:grid;grid-template-columns:repeat(3,minmax(300px,1fr));gap:12px}figure{background:white;padding:10px;margin:0}img{width:100%}p{max-width:1200px}</style>
<header><h2>热图构造证据；RGB解释内容</h2><select id="event"></select><select id="condition"><option>NATIVE_CROP</option><option>CURRENT_CROP</option><option>CURRENT_CONTOUR</option></select></header>
<p>脚本按旧/中/新时间段的停留密度选出实际观测代表点，模型只识别指定局部。NATIVE_CROP取观测当时的真实视频帧，减少在当前画面中解释历史对象的假设；CURRENT则把历史位置投到当前画面，遮挡和动态对应未知。保留其他热点，但不把它们合为一个对象。时长与类别由脚本绑定，模型无需理解时间颜色。均无GT投影。</p><main></main>
<script>const d=__DATA__,q=document.querySelector('#event'),c=document.querySelector('#condition');[...new Set(d.samples.map(s=>s.event))].forEach(e=>{let o=document.createElement('option');o.value=e;o.textContent=e;q.append(o)});function render(){let main=document.querySelector('main');main.replaceChildren();d.samples.filter(s=>s.event===q.value).sort((a,b)=>a.time_bin-b.time_bin).forEach(s=>{let f=document.createElement('figure'),title=document.createElement('figcaption'),img=document.createElement('img');title.textContent=['旧段','中段','新段'][s.time_bin]+' | '+s.duration.toFixed(3)+'s支持 | '+s.window.map(v=>v.toFixed(3)).join('~')+'s | source error '+(s.gaze_to_frame_error_seconds*1000).toFixed(1)+'ms';img.src=s.conditions[c.value].path;f.append(title,img);main.append(f)})}q.onchange=render;c.onchange=render;render();</script></html>'''
    (output/'temporal_memory.html').write_text(html.replace('__DATA__',json.dumps(m,ensure_ascii=False).replace('</','<\\/')))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','infer','prepare-temporal','infer-temporal'])
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--split',choices=['regression','fresh'],default='regression')
    p.add_argument('--conditions',nargs='+')
    p.add_argument('--max-calls',type=int)
    a=p.parse_args(argv);output=a.output.resolve()
    if a.action=='prepare':prepare(output)
    elif a.action=='prepare-temporal':prepare_temporal(output)
    else:infer(output,a.split,a.conditions,a.max_calls,a.action=='infer-temporal')


if __name__=='__main__':main()
