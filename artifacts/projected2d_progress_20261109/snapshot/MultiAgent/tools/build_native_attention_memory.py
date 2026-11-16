#!/usr/bin/env python3
"""Export engineering attention segments as historical RGB cards, with no MLLM."""
import argparse
import json
from pathlib import Path
import subprocess
from PIL import Image
from MultiAgent.heatmap_forms_study import OUTPUT,AUDIT,contexts_for,read
from MultiAgent.heatmap_temporal_pilot import sha,save
from MultiAgent.attention_memory import build_memory_proposals
from MultiAgent.gaze_evidence import gaze_records,project_gaze
from MultiAgent.camera_geometry import interpolate_pose
from MultiAgent.scripted_attention import crop_at,evidence_card
from MultiAgent.local_video_visual import probe_frames


def export(output,query_ids=None,stem="native_attention_memory"):
    m=read(output/'manifest.json');tm=read(output/'temporal_manifest.json')
    queries=({s['event']:s['query_sample'] for s in tm['samples']} if not query_ids else
             {s['event']:s['id'] for s in m['samples'] if s['id'] in query_ids})
    if query_ids and set(queries.values())!=set(query_ids):raise ValueError('Unknown or repeated-event query IDs')
    contexts=contexts_for(set(queries));events={}
    for event,sid in queries.items():
        sample=next(s for s in m['samples'] if s['id']==sid);ctx=contexts[event];g=sample['geometry']
        contract=read(AUDIT/'pose_alignment'/event/'frame_pose_contract.json')
        if sha(ctx.video_path)!=contract['source_hashes']['video_sha256'] or sha(ctx.json_path)!=contract['source_hashes']['telemetry_sha256']:
            raise ValueError('Video or telemetry changed from clock fit')
        records=gaze_records(ctx)
        with Image.open(sample['source_path']) as im:size=im.size
        memory=build_memory_proposals(records,query=g['telemetry_time_seconds'],current_pose=g['reference_pose'],
            image_size=size,clock_supported=g['clock_available'])
        frames,stream=probe_frames(ctx.video_path)
        if stream['width']!=stream['height']:raise ValueError('Square native source required for this pilot')
        selected={}
        for p in memory['proposals']:
            target=p['representative']['gaze_time_seconds']+g['selected_clock_offset_seconds']
            frame=min([f for f in frames if f['pts_seconds']<=sample['pts']],key=lambda f:abs(f['pts_seconds']-target))
            p['native_frame']=frame;p['gaze_frame_error_seconds']=frame['pts_seconds']-target
            selected[frame['source_frame_index']]=frame
        directory=output/'episode_frames'/sid;directory.mkdir(parents=True,exist_ok=True)
        expression='+'.join(f'eq(n\\,{i})' for i in sorted(selected))
        if selected:
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-threads','4','-i',ctx.video_path,
                '-vf',f'select={expression},scale=640:640','-vsync','0','-q:v','2',str(directory/'frame_%04d.jpg')],check=True,capture_output=True)
        paths=sorted(directory.glob('frame_*.jpg'))
        if len(paths)!=len(selected):raise ValueError('Source frame/PTS extraction mismatch')
        by_index=dict(zip(sorted(selected),paths))
        for p in memory['proposals']:
            frame=p['native_frame'];path=by_index[frame['source_frame_index']]
            p['source_frame_path']=str(path.relative_to(output));p['source_frame_sha256']=sha(path)
            observation=p['representative'];record=next(r for r in records if r['source_index']==observation['source_index'])
            query=frame['pts_seconds']-g['selected_clock_offset_seconds'];pose=interpolate_pose(records,query)
            with Image.open(path) as im:rgb=im.convert('RGB')
            projection=project_gaze(record,pose,*rgb.size)
            p['frame_telemetry_query_seconds']=query;p['frame_projection_status']=projection['status']
            if abs(p['gaze_frame_error_seconds'])>.08 or projection['status']!='projected_frustum_only':
                p['card']=None;continue
            point=projection['pixel_xy'];crop,meta=crop_at(rgb,point,.18)
            card=evidence_card(rgb,crop,point,meta['point_in_view'],frame_id=p['proposal_id'],pts=frame['pts_seconds'])
            destination=output/'episode_inputs'/sid/f"{p['proposal_id']}.png";destination.parent.mkdir(parents=True,exist_ok=True);card.save(destination)
            p['card']={'path':str(destination.relative_to(output)),'sha256':sha(destination),'crop':meta,'model_inference_run':False}
        events[event]={'query_sample':sid,'split':sample['split'],'current_pts':sample['pts'],
            'selected_clock_offset_seconds':g['selected_clock_offset_seconds'],'video_sha256':sha(ctx.video_path),
            'telemetry_sha256':sha(ctx.json_path),'memory':memory}
        print('MEMORY',event,'segments',len(memory['proposals']),'offscreen observations',sum(p['outside_current_frustum_count'] for p in memory['proposals']),flush=True)
    result={'schema':'native_rgb_attention_memory_v1','gt_loaded':False,'anchors_projected':False,
        'model_called':False,'events':events,'script_sha256':sha(__file__),
        'memory_code_sha256':sha(Path(__file__).resolve().parents[1]/'MultiAgent/attention_memory.py')}
    save(output/f'{stem}.json',result)
    payload=json.dumps(result,ensure_ascii=False).replace('</','<\\/')
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>按注意转移组织的历史RGB</title><style>body{font:15px system-ui;background:#f3f6f8;margin:20px}header{position:sticky;top:0;background:#f3f6f8;padding:12px}select{font:inherit;margin:8px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:12px}figure{margin:0;background:white;padding:12px}img{width:100%}p{max-width:1200px}</style><header><h2>按注意转移组织历史RGB</h2><select></select></header><p>不再固定切成三个时间段：脚本按世界坐标注视方向的变化和采样缺口分段，保留原始时长、源视频时间和局部图。过去注意过、当前已在画外的观测也保留。这是工程分段，不是已验证的fixation或对象轨迹。图像尚未运行语义推理。</p><main></main><script>const d=__DATA__,q=document.querySelector('select');Object.keys(d.events).forEach(e=>{let o=document.createElement('option');o.value=e;o.textContent=e;q.append(o)});function render(){let main=document.querySelector('main');main.replaceChildren();d.events[q.value].memory.proposals.forEach(p=>{let f=document.createElement('figure'),c=document.createElement('figcaption');c.textContent=p.proposal_id+' | '+p.window_telemetry_seconds.map(t=>t.toFixed(3)).join('~')+'s | '+p.supported_duration_seconds.toFixed(3)+'s支持 | 画外历史 '+p.outside_current_frustum_count+'/'+p.observation_count+(p.low_support?' | 支持较少':'');f.append(c);if(p.card){let img=document.createElement('img');img.src=p.card.path;f.append(img)}else{let text=document.createElement('p');text.textContent='当前无可用原生帧局部';f.append(text)}main.append(f)})}q.onchange=render;render();</script></html>'''
    (output/f'{stem}.html').write_text(html.replace('__DATA__',payload))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--query-ids',nargs='+');p.add_argument('--stem',default='native_attention_memory')
    a=p.parse_args();export(a.output.resolve(),a.query_ids,a.stem)
