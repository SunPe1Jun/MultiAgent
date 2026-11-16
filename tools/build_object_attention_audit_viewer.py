#!/usr/bin/env python3
"""Review referent-only center checks after all inference inputs are frozen."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.perspective_attention_study import read
from tools.run_object_attention_study import OUTPUT


def build(output=OUTPUT):
    m=read(output/'manifest.json');audit=read(output/'referent_center_audit.json');rows=[]
    for event in m['events']:
        for frame in event['frames']:
            refs=[r for r in audit['eligible_reference_frames'] if r['event']==event['event'] and r['frame_id']==frame['frame_id']]
            with Image.open(frame['source_path']) as im:rgb=im.convert('RGB')
            draw=ImageDraw.Draw(rgb);font=ImageFont.load_default
            for obj in frame['objects']:
                draw.rectangle(obj['box'],outline='white',width=2);x,y=obj['box'][:2]
                draw.text((max(0,x),max(0,y-17)),obj['id'],fill='white',stroke_width=1,stroke_fill='black',font=font(size=15))
            for i,ref in enumerate(refs):
                x,y=ref['point_xy'];draw.line((x-6,y,x+6,y),fill='#ff9300',width=3);draw.line((x,y-6,x,y+6),fill='#ff9300',width=3)
                draw.text((min(rgb.width-35,x+8),max(0,y-16)),f'R{i+1}',fill='#ff9300',stroke_width=1,stroke_fill='black',font=font(size=17))
            directory=output/'posthoc_audit'/event['event'];directory.mkdir(parents=True,exist_ok=True)
            path=directory/f"{frame['frame_id']}.png";rgb.save(path)
            rows.append({'event':event['event'],'frame_id':frame['frame_id'],'path':str(path.relative_to(output)),'sha256':sha(path),
                'references':[{'audit_id':f'R{i+1}',**r} for i,r in enumerate(refs)]})
    save(output/'posthoc_audit/manifest.json',{'rows':rows,'scope':'Only existing interaction referent projections after model inference; not model inputs, not pixel/visibility GT.'})
    data=json.dumps(rows,ensure_ascii=False).replace('</','<\\/')
    page=r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>仅referent投影的事后审计</title><style>body{font:15px system-ui;background:#f3f5f7;margin:20px}select{font:inherit}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:15px}figure{background:white;margin:0;padding:10px}img{width:100%}pre{white-space:pre-wrap}</style><h2>事后复核：检测物体框与当前交互referent中心</h2><select id="event"></select><p>白框/O号是语言条件检测提案；橙十字/R号只复用当前交互已有referent投影，不投影所有anchor。图像在全部模型推理完成后生成，不进入模型输入。中心落框只是弱几何检查：anchor中心可能不在可见表面，遮挡和像素标定未知，漏框不能直接当检测召回，落框也不能证明选对实例。</p><main></main><script>const d=__DATA__,q=document.querySelector('#event');[...new Set(d.map(r=>r.event))].forEach(e=>{let o=document.createElement('option');o.value=e;o.textContent=e;q.append(o)});function render(){let main=document.querySelector('main');main.replaceChildren();d.filter(r=>r.event===q.value).forEach(r=>{let f=document.createElement('figure'),im=document.createElement('img'),t=document.createElement('pre');im.src='../'+r.path;t.textContent=r.frame_id+'\n'+r.references.map(v=>v.audit_id+' '+v.reference_name+' | compatible boxes '+v.box_matching_ids.join(',')).join('\n');f.append(im,t);main.append(f)})}q.onchange=render;render();</script></html>'''
    (output/'posthoc_audit/index.html').write_text(page.replace('__DATA__',data))
    print('REFERENT_ONLY_AUDIT_IMAGES',len(rows))


if __name__=='__main__':build()
