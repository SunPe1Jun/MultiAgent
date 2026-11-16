#!/usr/bin/env python3
"""Interactive local review for original 2D source points and frozen routed inputs."""
from collections import defaultdict
import html
import json
import os
from pathlib import Path
import sys

from MultiAgent.projected2d_attention_study import read_csv, sha_file
from MultiAgent.projected2d_routed_rgb import GROUPS


def build(output):
    frozen = json.loads((output/'predictions_frozen.json').read_text())
    for name,digest in frozen['responses'].items():
        if sha_file(Path(name)) != digest:
            raise ValueError('Response changed after freeze')
    manifest = json.loads((output/'input_manifest.json').read_text())
    annotation = defaultdict(lambda: defaultdict(list))
    for row in read_csv(output/'rebuilt_manifest_all.csv'):
        annotation[(row['scene'],int(row['row_index']))][row['panel_id']].append({
            'name':row['referent_name'], 'valid':row['projection_valid']=='True',
            'acceptable':row['evidence_acceptable']=='True',
            'uv': [float(row['gt_u_norm']),float(row['gt_v_norm'])] if row['projection_valid']=='True' else None})
    events=[]
    for event in manifest['events']:
        panels=[]
        for panel in event['panels']:
            panels.append({'id':panel['frame_id'], 'time':panel['requested_video_time'],
                'image':os.path.relpath(panel['source_path'],output),
                'clock_supported':panel['clock_supported'],
                'inputs':{g:os.path.relpath(panel['inputs'][g]['path'],output) for g in GROUPS},
                'gt':annotation[(event['scene_key'],event['row_index'])][panel['frame_id']]})
        responses={g:json.loads((output/'responses'/f"{event['id']}_{g}.json").read_text()) for g in GROUPS}
        events.append({'id':event['id'],'instruction':event['instruction'],'panels':panels,
            'responses':{g:{'status':r['status'],'mentions':r.get('parsed',{}).get('mentions',[]),
                            'warnings':r.get('parse_warning'), 'request_path':r.get('request_path'),
                            'call_paths':r.get('call_paths')} for g,r in responses.items()}})
    payload=json.dumps({'events':events,'groups':GROUPS},ensure_ascii=False).replace('<','\\u003c')
    markup='''<!doctype html><meta charset="utf-8"><title>Projected-2D RGB review</title>
<style>body{font:16px system-ui;background:#172029;color:#f3f4f5;margin:24px}a{color:#91d7ff}select{font:inherit;max-width:100%;margin:8px;padding:5px}canvas{max-width:100%;height:auto}article{background:#26333e;border-radius:8px;padding:12px;margin:16px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}img{max-width:100%}pre{white-space:pre-wrap}.notice{border-left:5px solid #f2c66b;padding:12px;background:#35342e}</style>
<h1>Projected-2D: RGB evidence and predicted points</h1>
<p class="notice">Exploratory development results. Magenta crosses are projected anchor proxies, not independently calibrated pixel truth. Red circles are frozen model points. GT overlays are post-inference review only. Temporal acceptable panels depend on the original gaze/anchor ranking. No model point is decoded to a 3D identity.</p>
<p><a href="RESULTS.md">Results</a> · <a href="summary.json">All 2D metrics</a> · <a href="exact_set_analysis.json">2D exact / coreference accounting</a> · <a href="coordinate_audit.json">Coordinate audit</a></p>
<label>Event <select id="event"></select></label><label>Condition <select id="group"></select></label>
<label><input id="gt" type="checkbox" checked> Projected proxy GT</label>
<h2 id="title"></h2><p id="instruction"></p><p id="status"></p><div class="grid" id="panels"></div><pre id="details"></pre>
<script>const data=PAYLOAD;const es=document.querySelector('#event'),gs=document.querySelector('#group');
for(const [i,e] of data.events.entries()){const o=new Option(e.id,i);es.add(o)}
for(const g of data.groups)gs.add(new Option(g,g));
function clear(node){node.replaceChildren()}
function draw(){const event=data.events[Number(es.value)],group=gs.value,response=event.responses[group];
document.querySelector('#title').textContent=event.id;document.querySelector('#instruction').textContent=event.instruction;
document.querySelector('#status').textContent='Condition status: '+response.status;
document.querySelector('#details').textContent=JSON.stringify(response,null,2);const root=document.querySelector('#panels');clear(root);
for(const panel of event.panels){const article=document.createElement('article'),title=document.createElement('h3');title.textContent=panel.id+' | '+panel.time.toFixed(3)+' s | gaze clock '+(panel.clock_supported?'supported':'fallback');article.append(title);
const canvas=document.createElement('canvas');canvas.width=600;canvas.height=600;article.append(canvas);const context=canvas.getContext('2d');const image=new Image();image.onload=()=>{const scale=Math.min(600/image.width,600/image.height),w=image.width*scale,h=image.height*scale,ox=(600-w)/2,oy=(600-h)/2;context.drawImage(image,ox,oy,w,h);context.font='14px sans-serif';context.lineWidth=3;
for(const mention of response.mentions){for(const p of mention.predictions){if(p.frame_id!==panel.id)continue;const x=ox+p.point_uv[0]*w,y=oy+p.point_uv[1]*h;context.strokeStyle='#ff4d45';context.fillStyle='#ff4d45';context.beginPath();context.arc(x,y,6,0,Math.PI*2);context.stroke();context.fillText(mention.mention_id,x+8,y-6)}}
if(document.querySelector('#gt').checked){for(const g of panel.gt){if(!g.valid)continue;const x=ox+g.uv[0]*w,y=oy+g.uv[1]*h;context.strokeStyle='#ff55ff';context.fillStyle='#ff55ff';context.beginPath();context.moveTo(x-7,y);context.lineTo(x+7,y);context.moveTo(x,y-7);context.lineTo(x,y+7);context.stroke();context.fillText(g.name+(g.acceptable?' [time✓]':''),x+8,y+12)}}};image.src=panel.image;
const details=document.createElement('details'),summary=document.createElement('summary');summary.textContent='Actual model composite for '+group;details.append(summary);const input=document.createElement('img');input.loading='lazy';input.src=panel.inputs[group];details.append(input);article.append(details);root.append(article)}}
es.onchange=draw;gs.onchange=draw;document.querySelector('#gt').onchange=draw;draw();</script>'''.replace('PAYLOAD',payload)
    (output/'index.html').write_text(markup,encoding='utf-8')
    print(output/'index.html')


if __name__=='__main__':
    build(Path(sys.argv[1]))
