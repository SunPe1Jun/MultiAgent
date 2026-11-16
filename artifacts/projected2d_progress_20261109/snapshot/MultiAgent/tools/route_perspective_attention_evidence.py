#!/usr/bin/env python3
"""Route model-selected visual directions to exact historical RGB evidence.

No category/identity model is called. Wrong/empty model regions remain explicit;
this exporter never replaces a selected region by the script's largest heat.
"""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from PIL import Image
from MultiAgent.camera_geometry import interpolate_pose
from MultiAgent.context import load_contexts
from MultiAgent.gaze_evidence import gaze_records,project_gaze
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.local_video_visual import probe_frames
from MultiAgent.perspective_attention import observation_for_selected_face
from MultiAgent.perspective_attention_study import OUTPUT,SOURCE,read
from MultiAgent.scripted_attention import crop_at,evidence_card
from MultiAgent.spherical_attention_study import AUDIT


def export(output):
    manifest=read(output/'manifest.json');sphere=read(SOURCE/'manifest.json')
    original={e['id']:e for e in sphere['events']};selection=read(AUDIT/'selection.json')
    keys={e['event'] for e in manifest['events']}
    contexts=load_contexts({'samples':[s for s in selection['samples'] if f"{s['scene_key']}/{s['row_index']}" in keys]},load_ground_truth=False)
    contexts={f'{c.scene_key}/{c.row_index}':c for c in contexts}
    rows=[]
    for event in manifest['events']:
        ctx=contexts[event['event']];base=original[event['id']];offset=base['frames'][-1]['geometry']['selected_clock_offset_seconds']
        contract=read(AUDIT/'pose_alignment'/event['event']/'frame_pose_contract.json')
        if sha(ctx.video_path)!=contract['source_hashes']['video_sha256'] or sha(ctx.json_path)!=contract['source_hashes']['telemetry_sha256']:
            raise ValueError('Clock-fit source hashes changed')
        records=gaze_records(ctx);by_index={r['source_index']:r for r in records}
        all_frames,stream=probe_frames(ctx.video_path)
        if stream['width']!=stream['height']:raise ValueError('This pilot expects square recorded RGB')
        planes=np.load(output/'events'/event['event']/'perspective_density.npz')['density']
        for method,root in [('joint',output),('phase_grouped',output/'phase_layout_v2'),('visual_shortlist',output/'multi_region')]:
            for phase in [0,2]:
                rid=event['id']+'_TRUE_HEAT_'+('joint' if method=='joint' else f'phase_{phase}')
                response_path=root/'responses'/f'{rid}.json'
                if not response_path.exists():continue
                response=read(response_path)
                labels=(response['parsed'].get('regions',[]) if method=='visual_shortlist' else
                        [response['parsed'].get(('early' if phase==0 else 'late') if method=='joint' else 'selected')])
                for rank,label in enumerate(labels or [None]):
                    row={'event':event['event'],'event_id':event['id'],'split':event['split'],'method':method,'phase':phase,
                        'selected_label':label,'proposal_rank':rank,'model_response_path':str(response_path),
                        'model_response_sha256':sha(response_path),'semantic_model_run':False,'object_identity':None}
                    if label not in event['labels_by_region']:
                        row['status']='MODEL_ABSTAINED_OR_INVALID';rows.append(row);continue
                    region=event['labels_by_region'].index(label);face=event['selected_faces'][region]
                    observations=base['volume_stats']['bins'][phase]['observations']
                    row.update(selected_face=face,selected_region_index=region,window=base['volume_stats']['bins'][phase]['window'])
                    observation=observation_for_selected_face(observations,base['canonical_pose'],face,planes[phase,region])
                    if observation is None:
                        row['status']='SELECTED_REGION_HAS_NO_OBSERVATIONS';rows.append(row);continue
                    row['representative_observation']=observation
                    target=observation['gaze_time_seconds']+offset;end=row['window'][1]+offset
                    frame=min([f for f in all_frames if f['pts_seconds']<=end+1e-6],key=lambda f:abs(f['pts_seconds']-target))
                    row.update(source_frame=frame,gaze_frame_error_seconds=frame['pts_seconds']-target,
                        selected_clock_offset_seconds=offset,video_sha256=contract['source_hashes']['video_sha256'])
                    if abs(row['gaze_frame_error_seconds'])>.08:
                        row['status']='NO_NATIVE_FRAME_WITHIN_80MS';rows.append(row);continue
                    row['status']='PENDING_EXTRACTION';row['_record']=by_index[observation['source_index']]
                    rows.append(row)
        pending=[r for r in rows if r['event']==event['event'] and r['status']=='PENDING_EXTRACTION']
        indices=sorted({r['source_frame']['source_frame_index'] for r in pending})
        directory=output/'routed_native'/event['event'];directory.mkdir(parents=True,exist_ok=True)
        missing=[i for i in indices if not (directory/f'source_{i:06d}.jpg').exists()]
        if missing:
            with tempfile.TemporaryDirectory(dir=directory) as temporary:
                expr='+'.join(f'eq(n\\,{i})' for i in missing)
                subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-threads','4','-i',ctx.video_path,
                    '-vf',f'select={expr},scale=640:640','-vsync','0','-q:v','2',str(Path(temporary)/'frame_%04d.jpg')],capture_output=True,check=True)
                extracted=sorted(Path(temporary).glob('frame_*.jpg'))
                if len(extracted)!=len(missing):raise ValueError('Native frame index/PTS binding mismatch')
                for i,path in zip(missing,extracted):path.rename(directory/f'source_{i:06d}.jpg')
        for row in pending:
            path=directory/f"source_{row['source_frame']['source_frame_index']:06d}.jpg"
            q=row['source_frame']['pts_seconds']-offset;pose=interpolate_pose(records,q)
            with Image.open(path) as im:rgb=im.convert('RGB')
            projection=project_gaze(row.pop('_record'),pose,*rgb.size)
            row.update(native_source_path=str(path.relative_to(output)),native_source_sha256=sha(path),
                       native_query_telemetry_seconds=q,native_pose=pose,projection=projection)
            if projection['status']!='projected_frustum_only':row['status']='NATIVE_PROJECTION_UNSUPPORTED';continue
            view,meta=crop_at(rgb,projection['pixel_xy'],.18)
            image=evidence_card(rgb,view,projection['pixel_xy'],meta['point_in_view'],frame_id=f"P{row['phase']}_{row['selected_label']}",pts=row['source_frame']['pts_seconds'])
            destination=directory/f"{row['method']}_phase{row['phase']}_{row['selected_label']}.png";image.save(destination)
            row.update(status='RGB_EVIDENCE_READY',evidence_card={'path':str(destination.relative_to(output)),'sha256':sha(destination)},crop=meta)
        print('ROUTED',event['event'],Counter(r['status'] for r in rows if r['event']==event['event']),flush=True)
    result={'schema':'model_selected_visual_attention_rgb_routes_v1','gt_loaded':False,'anchors_projected':False,
        'new_model_calls':0,'selection_policy':'Follow model-selected direction; never substitute script argmax or object GT.',
        'scope':'Historical RGB evidence for later semantic/object filtering. No final referent selection or object identity established.',
        'status_counts':dict(Counter(r['status'] for r in rows)),'rows':rows,'script_sha256':sha(__file__)}
    save(output/'routed_visual_evidence.json',result)
    payload=__import__('json').dumps(result,ensure_ascii=False).replace('</','<\\/')
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>热图筛选后的原生RGB证据</title><style>body{font:15px system-ui;margin:20px;background:#f3f5f7}header{position:sticky;top:0;background:#f3f5f7;padding:12px}select{font:inherit;margin:8px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:15px}figure{background:white;padding:12px;margin:0}img{width:100%}p{max-width:1300px}</style><header><h2>先看视觉热图选方向，再取对应时段原生RGB</h2><select id="event"></select><select id="method"><option>visual_shortlist</option><option>phase_grouped</option><option>joint</option></select></header><p>方向由模型依据热图图像选择。脚本绑定时间，找到该方向内实际注视观测和观测当时的精确视频帧，再生成RGB证据。模型选错、该方向无观测、图像不可用均保留状态，不用最大热点偷偷替换。这里的角标不是最终目标标签；局部图尚未运行对象/指代模型。</p><main></main><script>const d=__DATA__,q=document.querySelector('#event'),method=document.querySelector('#method');[...new Set(d.rows.map(r=>r.event))].forEach(e=>{let o=document.createElement('option');o.value=e;o.textContent=e;q.append(o)});function render(){let main=document.querySelector('main');main.replaceChildren();d.rows.filter(r=>r.event===q.value&&r.method===method.value).forEach(r=>{let f=document.createElement('figure'),text=document.createElement('figcaption');text.textContent=(r.phase===0?'早段':'晚段')+' | 模型选择 '+r.selected_label+' | '+r.status+(r.gaze_frame_error_seconds===undefined?'':' | PTS误差 '+(r.gaze_frame_error_seconds*1000).toFixed(1)+'ms');f.append(text);if(r.evidence_card){let im=document.createElement('img');im.src=r.evidence_card.path;f.append(im)}main.append(f)})}q.onchange=render;method.onchange=render;render();</script></html>'''
    (output/'routed_visual_evidence.html').write_text(html.replace('__DATA__',payload))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    export(p.parse_args().output.resolve())
