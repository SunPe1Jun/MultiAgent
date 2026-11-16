#!/usr/bin/env python3
"""Export aligned past-gaze bins plus image observations; never run an MLLM."""
import argparse
import json
from pathlib import Path

from PIL import Image
from MultiAgent.context import load_contexts
from MultiAgent.gaze_evidence import build_heatmap,gaze_records
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.scripted_attention import summarize_projected_history,crop_at,evidence_card


def export(root,audit_root):
    manifest=json.loads((root/'manifest.json').read_text())
    keys={s['event'] for s in manifest['samples']}
    selection=json.loads((audit_root/'selection.json').read_text())
    contexts=load_contexts({'samples':[s for s in selection['samples']
        if f"{s['scene_key']}/{s['row_index']}" in keys]},load_ground_truth=False)
    contexts={f'{c.scene_key}/{c.row_index}':c for c in contexts}
    records={key:gaze_records(c) for key,c in contexts.items()}
    observations=json.loads((root/'temporal_observations.json').read_text())
    indexed={s['sample']:s for items in observations['events'].values() for s in items}
    events={}
    for sample in manifest['samples']:
        ctx=contexts[sample['event']];geometry=sample['geometry']
        if sha(sample['source_path'])!=sample['source_sha256']:raise ValueError('Source frame changed')
        with Image.open(sample['source_path']) as im:size=im.size
        _,stats,projected=build_heatmap(ctx,records[sample['event']],
            frame_time_video=sample['pts'],frame_time_telemetry=geometry['telemetry_time_seconds'],
            event_window=[float(ctx.api_row['t_start']),float(ctx.api_row['t_end'])],
            clock_available=geometry['clock_available'],image_size=size,strategy='local_history',
            reference_pose=geometry['reference_pose'])
        history=summarize_projected_history(projected,query_time=geometry['telemetry_time_seconds'],image_size=size)
        with Image.open(sample['source_path']) as im:rgb=im.convert('RGB')
        for item in history['bins_old_to_new']:
            representative=item['representative_observed_point']
            if representative is None:
                item['review_card']=None
                continue
            point=representative['pixel_xy']
            detail,meta=crop_at(rgb,point,.18)
            card=evidence_card(rgb,detail,point,meta['point_in_view'],frame_id=sample['frame_id'],pts=sample['pts'])
            path=root/'history_inputs'/sample['id']/f'bin_{item["bin_index_old_to_new"]}.png'
            path.parent.mkdir(parents=True,exist_ok=True);card.save(path)
            item['review_card']={'path':str(path.relative_to(root)),'sha256':sha(path),'crop':meta,
                                 'model_inference_run':False}
        row={**indexed[sample['id']], 'source_sha256':sample['source_sha256'],
            'selected_clock_offset_seconds':geometry['selected_clock_offset_seconds'],
            'history_projection_stats':stats,'past_gaze_history':history}
        events.setdefault(sample['event'],[]).append(row)
    result={'schema':'scripted_temporal_evidence_v1','gt_loaded':False,'anchors_projected':False,
        'model_called_by_export':False,'manifest_sha256':sha(root/'manifest.json'),
        'script_sha256':sha(__file__),
        'history_helper_sha256':sha(Path(__file__).resolve().parents[1]/'MultiAgent/scripted_attention.py'),
        'scope':'Time bins computed by script in each current frame camera; no heatmap color judgments. '
                'Per-frame categories do not establish same object across frames. History visibility and dynamic correspondence unknown.',
        'events':{key:sorted(rows,key=lambda r:r['pts']) for key,rows in events.items()}}
    save(root/'scripted_temporal_evidence.json',result)
    write_viewer(root,result,manifest)
    print('EXPORTED',len(events),'events',sum(map(len,events.values())),'frames; no new model calls')


def write_viewer(root,evidence,manifest):
    payload=json.dumps({'evidence':evidence,'samples':{s['id']:s for s in manifest['samples']}},ensure_ascii=False).replace('</','<\\/')
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>脚本时序证据预览</title>
<style>body{font:15px system-ui;margin:20px;background:#f4f5f7;color:#18232e}header{position:sticky;top:0;background:#f4f5f7;padding:12px}select,input{font:inherit;margin:8px}#history{display:grid;grid-template-columns:repeat(3,minmax(250px,1fr));gap:12px}figure{margin:0;padding:12px;background:white}img{width:100%}#instant{max-width:1100px}p{max-width:1200px}figcaption{min-height:80px;line-height:1.5}</style>
<header><h2>时间由脚本组织；模型只识别局部内容</h2><select id="event"></select><input type="range" min="0" max="2" value="0" id="frame"><span id="status"></span></header>
<p>上方是当前帧即时局部图，下面按旧→中→新分列过去2秒的gaze。每段取一个实际投影观测点，展示该点在当前RGB中的局部；并非历史时刻的RGB。热图颜色不参与模型判断。三段局部仅供复核，尚未运行模型。</p>
<p>投影只验证视锥范围，遮挡、动态物体移动及跨帧身份尚未验证。代表点不是目标标签；分段停留占比不是概率或置信度。高运动收益尚未证实。</p>
<figure id="instant"><figcaption></figcaption><img></figure><h3>脚本计算的短时历史</h3><main id="history"></main>
<script>const d=__DATA__;const events=Object.keys(d.evidence.events),select=document.querySelector('#event'),slider=document.querySelector('#frame');events.forEach(key=>{let o=document.createElement('option');o.value=key;o.textContent=key;select.append(o)});
function render(){let frames=d.evidence.events[select.value];slider.max=frames.length-1;let row=frames[+slider.value],s=d.samples[row.sample];document.querySelector('#status').textContent=row.frame_id+' | PTS '+row.pts.toFixed(3)+' | head '+row.angular_speed_deg_s.toFixed(1)+'deg/s';let instant=document.querySelector('#instant');instant.querySelector('img').src=s.conditions.DETAIL_018.path;instant.querySelector('figcaption').textContent='即时：模型类别 '+(row.predictions.DETAIL_018?.category||'尚无')+'；telemetry '+row.telemetry_time_seconds.toFixed(3)+'s';let main=document.querySelector('#history');main.replaceChildren();row.past_gaze_history.bins_old_to_new.forEach((b,i)=>{let f=document.createElement('figure'),c=document.createElement('figcaption');c.textContent=['旧段','中段','新段'][i]+' | '+b.window_telemetry_seconds.map(x=>x.toFixed(3)).join(' ~ ')+'s | 观测 '+b.projected_sample_count+' | 视锥内加权时长 '+b.frustum_supported_duration_seconds.toFixed(3)+'s | 分散度 '+(b.spatial_rms_pixels===null?'无':b.spatial_rms_pixels.toFixed(1)+'px');f.append(c);if(b.review_card){let img=document.createElement('img');img.src=b.review_card.path;f.append(img)}else{let p=document.createElement('p');p.textContent='该时间段无受支持观测';f.append(p)}main.append(f)})}select.onchange=()=>{slider.value=0;render()};slider.oninput=render;render();</script></html>'''
    (root/'temporal_viewer.html').write_text(html.replace('__DATA__',payload))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('.local-sync/scripted_region_semantics_20261105'))
    p.add_argument('--audit-root',type=Path,default=Path('.local-sync/projection_expansion24_20261105'))
    a=p.parse_args();export(a.output.resolve(),a.audit_root.resolve())
