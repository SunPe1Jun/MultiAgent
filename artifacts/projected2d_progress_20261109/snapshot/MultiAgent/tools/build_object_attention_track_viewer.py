#!/usr/bin/env python3
"""Inspect native mask tracks, gaze timing and object-level visual histories."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import binary_erosion
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.perspective_attention_study import read
from tools.run_object_attention_study import OUTPUT


def build(output=OUTPUT):
    root=output/'tracks';m=read(root/'manifest.json');events=[]
    for result in m['results']:
        masks=np.load(output/result['track_masks_path'])['masks'];n,objects,h,w=masks.shape
        directory=output/Path(result['track_masks_path']).parent
        mass=np.array(result['object_mass_seconds_by_phase']);peak=float(mass.max()) if mass.size else 0.
        native=result['frames'];start=native[0]['native_source_frame']['pts_seconds'];end=native[-1]['native_source_frame']['pts_seconds']
        annotated=[]
        for j,f in enumerate(native):
            with Image.open(output/f['native_path']) as image:rgb=image.convert('RGB')
            pixels=np.asarray(rgb).copy()
            for i in range(objects):
                edge=masks[j,i]&~binary_erosion(masks[j,i]);pixels[edge]=[255,255,255]
            rgb=Image.fromarray(pixels);draw=ImageDraw.Draw(rgb)
            for i,obj in enumerate(result['seed_objects']):
                yy,xx=np.nonzero(masks[j,i])
                if len(xx):draw.text((int(xx.min()),max(0,int(yy.min())-18)),obj['id'],fill='white',stroke_fill='black',stroke_width=1,font=ImageFont.load_default(size=17))
            for o in result['gaze_observations']:
                if o['frame_index']==j and o['status']=='PROJECTED':
                    x,y=o['projection']['pixel_xy'];draw.ellipse((x-4,y-4,x+4,y+4),outline='#ff6633',width=2)
            path=directory/f'overlay_{j:03d}.jpg';rgb.save(path,quality=92)
            annotated.append({'path':str(path.relative_to(output)),'sha256':sha(path),'pts':f['native_source_frame']['pts_seconds']})
        boards=[]
        for i,obj in enumerate(result['seed_objects']):
            board=Image.new('RGB',(1008,810),'#15212b');draw=ImageDraw.Draw(board);font=ImageFont.load_default
            draw.text((16,10),f"TRACK {obj['id']} | proposal category: {obj['category']} | identity unverified",fill='white',font=font(size=19))
            sources=[]
            for phase in range(3):
                lo=start+(end-start)*phase/3;hi=start+(end-start)*(phase+1)/3
                candidates=[j for j in range(n) if lo<=annotated[j]['pts']<hi or phase==2 and annotated[j]['pts']==hi]
                available=[j for j in candidates if masks[j,i].any()]
                x=16+phase*328;draw.text((x,46),['EARLY','MIDDLE','LATE'][phase],fill='white',font=font(size=18))
                if not available:
                    draw.text((x,100),'NO PROPAGATED MASK',fill='white',font=font(size=15));sources.append(None);continue
                # Choose readability by largest mask area in each phase, not gaze or GT.
                j=max(available,key=lambda j:(int(masks[j,i].sum()),-j));sources.append(j)
                with Image.open(output/native[j]['native_path']) as im:rgb=im.convert('RGB')
                yy,xx=np.nonzero(masks[j,i]);box=[max(0,int(xx.min())-12),max(0,int(yy.min())-12),min(w,int(xx.max())+13),min(h,int(yy.max())+13)]
                context=rgb.copy();ImageDraw.Draw(context).rectangle(box,outline='white',width=3)
                board.paste(context.resize((288,288),Image.Resampling.LANCZOS),(x,78))
                crop=rgb.crop(box);scale=min(288/crop.width,224/crop.height);crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))),Image.Resampling.LANCZOS)
                tile=Image.new('RGB',(288,224),'#222a30');tile.paste(crop,((288-crop.width)//2,(224-crop.height)//2));board.paste(tile,(x,402))
                draw.text((x,374),f"source PTS {annotated[j]['pts']:.3f}s",fill='white',font=font(size=14))
                # The graphical patch represents this tracked object's accumulated
                # native-frame attention during the phase, not the chosen frame alone.
                heat=np.full((160,288,3),20.,float);strength=math.sqrt(float(mass[phase,i])/peak) if peak>0 else 0.
                silhouette=Image.fromarray(masks[j,i].astype(np.uint8)*255).crop(box).resize((288,160),Image.Resampling.NEAREST)
                sel=np.asarray(silhouette)>0;heat[sel]=np.array([20,20,20])*(1-.9*strength)+np.array([255,85,25])*.9*strength
                board.paste(Image.fromarray(heat.astype(np.uint8)),(x,650))
            path=directory/f'object_{obj["id"]}_history.png';board.save(path)
            boards.append({'object_id':obj['id'],'category':obj['category'],'path':str(path.relative_to(output)),'sha256':sha(path),
                           'source_frame_indices':sources,'mass_seconds_by_phase':mass[:,i].tolist(),'semantic_model_run':False})
        events.append({'event':result['event'],'id':result['event_id'],'frames':annotated,'objects':boards,
            'redetection':result['redetection_compatibility'],'scope':result['scope']})
        print('TRACK_VIEWER',result['event'],len(annotated),len(boards),flush=True)
    data=json.dumps(events,ensure_ascii=False).replace('</','<\\/')
    template='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>物体轨迹与原生注视记忆</title><style>body{font:15px system-ui;background:#f3f5f7;margin:20px}header{position:sticky;top:0;background:#f3f5f7;padding:10px}select{font:inherit;margin:6px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:15px}img{width:100%}figure{background:white;padding:10px;margin:0}p{max-width:1300px}</style><header><h2>按物体轨迹累计历史注视，保留每次观测的真实源帧</h2><select id="event"></select><input id="frame" type="range" min="0" value="0"><span id="time"></span><select id="object"></select></header><p>这些是SAM2从中段检测框向前、向后传播的mask轨迹，属于整段离线回看，不能当在线因果推理。白色边界及O编号是跟踪提案，橙色圆圈是按实际PTS映射到该原生帧的gaze。种子以外新出现物体尚未被发现；跟踪可能漂移、遮挡后误关联，不把编号当已验证身份。</p><p>右侧每物体历史：早/中/晚真实上下文与局部详情；下方整物体热图表达该轨迹在该阶段累计的注视支持，而非所选代表帧单独的gaze，也不是目标标签。每段来源按mask最大可见面积选取，不用gaze/GT。所有原始观测、未承接gaze、80ms外来源及失败均保留。此页未新增语义模型调用。</p><main><figure><img id="scene"></figure><figure><img id="board"></figure></main><script>const d=__DATA__,q=document.querySelector('#event'),slider=document.querySelector('#frame'),o=document.querySelector('#object');d.forEach(e=>{let v=document.createElement('option');v.value=e.id;v.textContent=e.event;q.append(v)});function render(){let e=d.find(e=>e.id===q.value),i=+slider.value;document.querySelector('#scene').src='../'+e.frames[i].path;document.querySelector('#time').textContent='PTS '+e.frames[i].pts.toFixed(3)+'s';let b=e.objects.find(b=>b.object_id===o.value);document.querySelector('#board').src=b?'../'+b.path:''}function changeEvent(){let e=d.find(e=>e.id===q.value);slider.max=e.frames.length-1;slider.value=0;o.replaceChildren();e.objects.forEach(b=>{let v=document.createElement('option');v.value=b.object_id;v.textContent=b.object_id+' '+b.category;o.append(v)});render()}q.onchange=changeEvent;slider.oninput=render;o.onchange=render;changeEvent();</script></html>'''
    (root/'index.html').write_text(template.replace('__DATA__',data));save(root/'viewer_manifest.json',{'events':events,'script_sha256':sha(__file__)})


if __name__=='__main__':build()
