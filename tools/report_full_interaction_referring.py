#!/usr/bin/env python3
"""Browse complete inputs and report event-level final-set effects."""
from html import escape
import json
from pathlib import Path
import numpy as np
from MultiAgent.heatmap_temporal_pilot import save
from tools.run_full_interaction_referring import OUTPUT, read


def paired_micro_interval(rows, control, seed=20261106):
    """Paired interaction bootstrap of the aggregate micro-F1 difference."""
    arrays={}
    for condition in ('TRUE_HEAT',control):
        arrays[condition]=np.array([[r['metrics'][condition][key] for key in ('tp','fp','fn')] for r in rows],float)
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(rows),size=(5000,len(rows)))
    scores={}
    for condition,a in arrays.items():
        totals=a[indices].sum(1);denom=2*totals[:,0]+totals[:,1]+totals[:,2]
        scores[condition]=np.divide(2*totals[:,0],denom,out=np.zeros(len(totals)),where=denom>0)
    def f1(a):
        tp,fp,fn=a.sum(0);return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    return {'interactions':len(rows),'gain':f1(arrays['TRUE_HEAT'])-f1(arrays[control]),
            'interval_95':np.quantile(scores['TRUE_HEAT']-scores[control],[.025,.975]).tolist(),
            'replicates':5000,'resampling_unit':'interaction; all conditions paired; no frame/mention multiplication'}


def report(output=OUTPUT):
    manifest=read(output/'manifest.json')
    responses={p.stem:read(p) for p in (output/'responses').glob('*.json')}
    analysis=read(output/'analysis.json') if (output/'analysis.json').exists() else None
    if analysis:
        analysis['paired_micro_f1']={condition:paired_micro_interval(analysis['events'],condition)
            for condition in ('NO_ADDED_HEAT','SPATIAL_MIRROR','TIME_REVERSE')}
        save(output/'analysis.json',analysis)
    compact=[]
    for event in manifest['events']:
        compact.append({'id':event['id'],'event':event['event'],'instruction':event['instruction'],
            'candidate_count':len(event['scene_candidate_table']),
            'heat_supported_frames':event['clock_available_frames'],
            'frames':[{k:f[k] for k in ('frame_id','time_seconds','images','source_size')} for f in event['frames']],
            'temporal_groups':event['qwen_temporal_groups']})
    rr={k:{'parsed':v['parsed'],'validation':v['validation'],'seconds':v['seconds'],
           'effective_size':v['processor_audit']['processor_effective_frame_size_px'],
           'visual_tokens':v['processor_audit']['expanded_video_token_count'],
           'visual_referents':v.get('grounding_validation',{}).get('referents',[]),
           'decoder':v.get('decoder')} for k,v in responses.items()}
    review=read(output/'review_referent_projection.json') if (output/'review_referent_projection.json').exists() else None
    data=json.dumps({'events':compact,'conditions':manifest['conditions'],'responses':rr,'analysis':analysis,'review':review},ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>完整交互指代对照</title><style>body{font:16px system-ui;line-height:1.65;background:#f1f4f7;margin:18px}header{position:sticky;top:0;background:#f1f4f7;padding:10px;z-index:1}select{font:inherit;padding:4px;margin:5px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(530px,1fr));gap:14px}figure{background:white;padding:10px;margin:0}img{width:100%}pre{white-space:pre-wrap}#result{padding:12px;background:white}</style><header><h2>完整交互 → 全场景候选 → 最终指代集合</h2><select id="event"></select><select id="condition"></select><div id="instruction"></div></header><p>固定__EVENTS__条旧交互、每段12张完整画面，四条件共享RGB/相机姿态/全场景候选与预算。图内无对象裁剪、无anchor投影、无GT条件候选；GT在全部输出冻结后才评分。每帧左上为RGB，其余三个完整视野为早/中/近注意。源RGB已缩放为672方图，并非原视频全像素无损。Qwen会进一步按token预算调整尺寸，两源帧合成一时间patch。原视频绿色标记各组均保留，无新增热图不等于完全无gaze。</p><p>白色十字为模型选择（同一G组的两源帧均显示，未假定唯一帧）；橙色十字仅在全部推理后显示当前交互referent中心，未校准像素/遮挡，不是人工对象GT。十字是查看器复核层，原模型输入无这些标注。可关闭后查看原输入。</p><label><input id="overlay" type="checkbox" checked> 显示推理后复核十字</label><div id="result"></div><main></main><script>const data=__DATA__,q=document.querySelector('#event'),c=document.querySelector('#condition');data.events.forEach(e=>{let o=document.createElement('option');o.value=e.id;o.textContent=e.event;q.append(o)});data.conditions.forEach(x=>{let o=document.createElement('option');o.value=x;o.textContent=x;c.append(o)});function render(){let e=data.events.find(e=>e.id===q.value);document.querySelector('#instruction').textContent=e.instruction+' | 全场景候选 '+e.candidate_count+' | 热图支持 '+e.heat_supported_frames+'/12帧';let r=data.responses[e.id+'_'+c.value],sc=data.analysis?.events.find(x=>x.id===e.id);document.querySelector('#result').textContent=(r?'模型输出 '+JSON.stringify(r.parsed)+' | schema '+r.validation.schema_valid+' | '+r.visual_tokens+'视觉token | 实际组合帧 '+r.effective_size.join('×'):'尚未运行')+(sc?' | GT（推理后评价） '+JSON.stringify(sc.gt)+' | 集合评分 '+JSON.stringify(sc.metrics[c.value]):'');let main=document.querySelector('main');main.replaceChildren();e.frames.forEach(f=>{let fig=document.createElement('figure'),im=document.createElement('img'),cv=document.createElement('canvas'),t=document.createElement('figcaption');im.src=f.images[c.value].path;im.loading='eager';im.onload=()=>{cv.width=im.naturalWidth;cv.height=im.naturalHeight;cv.style.width='100%';let cx=cv.getContext('2d');cx.drawImage(im,0,0);if(document.querySelector('#overlay').checked){function cross(x,y,color,label){cx.strokeStyle=color;cx.fillStyle=color;cx.lineWidth=3;cx.beginPath();cx.moveTo(x-10,y);cx.lineTo(x+10,y);cx.moveTo(x,y-10);cx.lineTo(x,y+10);cx.stroke();cx.font='15px sans-serif';cx.fillText(label,x+12,y-6)}for(const point of r?.visual_referents||[]){let group=e.temporal_groups.find(g=>g.view_id===point.view_id);if(group?.frame_ids.includes(f.frame_id)){cross(point.x*f.source_size[0],28+point.y*f.source_size[1],'white',point.mention)}}for(const ref of data.review?.events?.[e.id]?.[f.frame_id]||[]){cross(ref.normalized_xy[0]*f.source_size[0],28+ref.normalized_xy[1]*f.source_size[1],'#ffad32',ref.name)}}};t.textContent=f.frame_id+' | PTS '+f.time_seconds.toFixed(6)+'s';fig.append(cv,t);main.append(fig)})}q.onchange=render;c.onchange=render;document.querySelector('#overlay').onchange=render;render();</script></html>'''
    (output/'index.html').write_text(page.replace('__DATA__',data).replace('__EVENTS__',str(len(manifest['events']))))
    print(json.dumps({'condition_results_completed':len(responses),'unique_model_requests':len({r['request_path'] for r in responses.values()}),'condition_results_planned':len(manifest['requests']),'evaluated':analysis is not None},indent=2))


if __name__=='__main__':report()
