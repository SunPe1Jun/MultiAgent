#!/usr/bin/env python3
"""Display head FOV moving over a fixed angular map, with RGB source lookup."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
from MultiAgent.spherical_attention import spherical_grid,angles_to_directions,_fov_mask
from MultiAgent.spherical_attention_study import OUTPUT,read


def encode_rows(owner):
    encoded=[]
    for row in owner:
        ends=np.r_[np.flatnonzero(np.diff(row)!=0)+1,len(row)]
        starts=np.r_[0,ends[:-1]]
        encoded.append([[int(end),int(row[start])] for start,end in zip(starts,ends)])
    return encoded


def build(output):
    manifest=read(output/'manifest.json');events=[]
    for entry in manifest['events']:
        directory=output/'events'/entry['event']
        volume=np.load(directory/'volume.npz');owner=volume['rgb_source_index'];h,w=owner.shape
        yg,pg=spherical_grid(w,h);directions=angles_to_directions(yg,pg)
        with Image.open(directory/'angular_heat_overlay.png') as im:base=np.array(im.convert('RGB'))
        frames=[]
        for f in entry['frames']:
            with Image.open(f['source_path']) as im:size=im.size
            footprint=_fov_mask(directions,entry['canonical_pose'],f['geometry']['reference_pose'],size)
            # Periodic yaw padding prevents a false outline at +/-180 seam.
            padded=np.pad(footprint,((1,1),(1,1)),mode='wrap')
            boundary=footprint & ~binary_erosion(padded,iterations=1)[1:-1,1:-1]
            pixels=base.copy();pixels[boundary]=[255,255,255]
            image=Image.fromarray(pixels).resize((1080,540),Image.Resampling.BILINEAR)
            draw=ImageDraw.Draw(image);draw.text((12,12),f"FIXED MAP | CURRENT FRAME {f['frame_id']} PTS {f['pts']:.3f}s | WHITE = CAMERA FOV",fill='white',stroke_width=2,stroke_fill='black')
            path=directory/f"fov_{f['frame_id']}.jpg";image.save(path,quality=95)
            frames.append({'id':f['frame_id'],'pts':f['pts'],'source_path':os.path.relpath(f['source_path'],output),
                'source_sha256':f['source_sha256'],'map_path':str(path.relative_to(output))})
        events.append({'event':entry['event'],'frames':frames,'owner_rle':encode_rows(owner),'owner_grid_size':[w,h],
            'query_pts':entry['query_pts'],'max_rotation':entry['max_head_rotation_from_reference_degrees'],
            'translation':entry['max_camera_translation_from_reference_world_units']})
    data={'scope':'Offline whole-interaction heat and RGB atlas. Moving slider changes displayed camera footprint only, not heat cutoff.',
          'events':events}
    (output/'head_motion_demo_data.json').write_text(json.dumps(data,ensure_ascii=False)+'\n')
    payload=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>转头与累计视域</title><style>body{font:16px system-ui;margin:22px;background:#f2f5f7;color:#15212d}header{position:sticky;top:0;background:#f2f5f7;padding:12px}select,input{font:inherit;margin:8px}main{display:grid;grid-template-columns:1.6fr 1fr;gap:14px}figure{margin:0;padding:12px;background:white}img{width:100%;cursor:crosshair}p{max-width:1350px}#lookup{background:white;padding:12px}#lookup img{max-width:440px}</style><header><h2>头会转，热图坐标固定</h2><select id="event"></select><input id="frame" type="range" min="0" value="0"><span id="meta"></span></header><p>左图是整段交互的固定方向图；白色边界是右侧这一帧的相机视野。拖动滑块，看白框随转头移动，历史热图留在共同坐标中。单帧视野只是累计范围的一部分，黑色区域保留为未知。</p><p>这是离线说明图：热图和RGB图集使用整段交互，滑块只移动白色视野边界，不代表当时已经获得全部历史。图集按旋转查找记录帧，平移视差、动态物体和遮挡没有解决。点击方向图可查该RGB位置来自哪张原图；热图与RGB不一定来自同一时刻。</p><main><figure><figcaption>固定方向图＋整段注视历史＋当前帧视野边界</figcaption><img id="map"></figure><figure><figcaption id="nativeTitle"></figcaption><img id="native"></figure></main><div id="lookup">点击左图查看RGB来源。</div><script>const d=__DATA__,q=document.querySelector('#event'),slider=document.querySelector('#frame');d.events.forEach((e,i)=>{let o=document.createElement('option');o.value=i;o.textContent=e.event;q.append(o)});function render(){let e=d.events[+q.value];slider.max=e.frames.length-1;if(+slider.value>=e.frames.length)slider.value=0;let f=e.frames[+slider.value];document.querySelector('#map').src=f.map_path;document.querySelector('#native').src=f.source_path;document.querySelector('#nativeTitle').textContent=f.id+' | 原始PTS '+f.pts.toFixed(3)+'s';document.querySelector('#meta').textContent=' | 相对转角最大 '+e.max_rotation.toFixed(1)+'° | 相机位移最大 '+e.translation.toFixed(3)+'世界单位';}q.onchange=()=>{slider.value=0;render()};slider.oninput=render;document.querySelector('#map').onclick=function(evt){let e=d.events[+q.value],rect=this.getBoundingClientRect();let x=Math.max(0,Math.min(e.owner_grid_size[0]-1,Math.floor((evt.clientX-rect.left)/rect.width*e.owner_grid_size[0]))),y=Math.max(0,Math.min(e.owner_grid_size[1]-1,Math.floor((evt.clientY-rect.top)/rect.height*e.owner_grid_size[1])));let row=e.owner_rle[y],owner=row.find(run=>x<run[0])[1],div=document.querySelector('#lookup');div.replaceChildren();let text=document.createElement('p');if(owner<0){text.textContent='该方向没有所保存RGB帧的覆盖，保持未知。';div.append(text);return;}let frame=e.frames[owner];text.textContent='该RGB位置来自 '+frame.id+' | PTS '+frame.pts.toFixed(3)+'s | 不是同一对象身份的证明';let img=document.createElement('img');img.src=frame.source_path;div.append(text,img)};render();</script></html>'''
    (output/'head_motion_demo.html').write_text(html.replace('__DATA__',payload))
    p=output/'index.html';page=p.read_text();needle='<p>每段交互'
    if 'head_motion_demo.html' not in page:page=page.replace(needle,'<p><a href="head_motion_demo.html">交互演示：拖动时间，查看相机视野在固定热图上移动</a> · <a href="layout_v2/index.html">调整布局后的筛选对照</a></p>'+needle)
    p.write_text(page)
    print('DEMO',len(events),'events',sum(len(e['frames']) for e in events),'FOV views; source lookup retains every atlas pixel')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    build(p.parse_args().output.resolve())
