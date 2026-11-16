#!/usr/bin/env python3
"""Point-prompted SAM2 alternatives for gaze not carried by language detections.

Preserve all three segmentation scales. No mask is called a target/object GT;
large surfaces, hands, and background remain explicit possibilities.
"""
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import binary_erosion
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.perspective_attention_study import read
from tools.run_object_attention_study import OUTPUT,MODELROOT


def build(output=OUTPUT):
    root=output/'residual_points';root.mkdir(exist_ok=True);m=read(output/'manifest.json')
    import torch
    from transformers import Sam2Model,Sam2Processor
    processor=Sam2Processor.from_pretrained(MODELROOT/'sam2.1-hiera-tiny',local_files_only=True)
    model=Sam2Model.from_pretrained(MODELROOT/'sam2.1-hiera-tiny',local_files_only=True).cuda().eval()
    rows=[]
    for event in m['events']:
        for frame in event['frames']:
            data=np.load(output/frame['npz_path']);union=data['masks'].any(axis=0);h,w=union.shape
            observations=[o for b in frame['attention_stats']['bins'] for o in b['observations']]
            # Most recent valid unassigned gaze center, solely geometry/timestamps.
            eligible=[]
            for o in observations:
                x,y=o['pixel_xy'];px=min(w-1,max(0,round(x)));py=min(h-1,max(0,round(y)))
                if not union[py,px]:eligible.append(o)
            row={'event':event['event'],'event_id':event['id'],'frame_id':frame['frame_id'],'status':'NO_UNASSIGNED_OBSERVATION'}
            rows.append(row)
            if not eligible:continue
            selected=max(eligible,key=lambda o:o['gaze_time_seconds']);row.update(observation=selected,status='PENDING')
            with Image.open(frame['source_path']) as im:rgb=im.convert('RGB')
            x=processor(images=rgb,input_points=[[[selected['pixel_xy']]]],input_labels=[[[1]]],return_tensors='pt').to('cuda')
            with torch.inference_mode():prediction=model(**x,multimask_output=True)
            masks=processor.post_process_masks(prediction.pred_masks.cpu(),x['original_sizes'].cpu())[0][0].numpy().astype(bool)
            quality=prediction.iou_scores[0,0].detach().float().cpu().tolist()
            directory=root/'events'/event['event']/frame['frame_id'];directory.mkdir(parents=True,exist_ok=True)
            npz=directory/'masks.npz';np.savez_compressed(npz,masks=masks)
            canvas=Image.new('RGB',(1008,740),'#15212b');draw=ImageDraw.Draw(canvas)
            draw.text((16,8),'UNASSIGNED GAZE -> THREE POINT-PROMPTED REGION ALTERNATIVES',fill='white',font=ImageFont.load_default(size=18))
            source=rgb.copy();sx,sy=selected['pixel_xy'];ImageDraw.Draw(source).ellipse((sx-6,sy-6,sx+6,sy+6),outline='#ff6633',width=3)
            canvas.paste(source.resize((384,384),Image.Resampling.LANCZOS),(16,48))
            for i,mask in enumerate(masks):
                pixels=np.asarray(rgb).copy();pixels[mask]=(pixels[mask]*.65+np.array([40,180,255])*.35).astype(np.uint8)
                pixels[mask&~binary_erosion(mask)]=255
                tile=Image.fromarray(pixels).resize((288,288),Image.Resampling.LANCZOS)
                canvas.paste(tile,(16+i*328,444));draw.text((16+i*328,420),f'ALT {i+1} | region, not verified object',fill='white',font=ImageFont.load_default(size=13))
            draw.text((430,70),event['event']+' '+frame['frame_id'],fill='white',font=ImageFont.load_default(size=22))
            draw.text((430,114),'All three SAM2 scales retained.',fill='white',font=ImageFont.load_default(size=18))
            draw.text((430,150),'No semantic label or target confidence inferred.',fill='white',font=ImageFont.load_default(size=16))
            path=directory/'alternatives.png';canvas.save(path)
            row.update(status='REGION_ALTERNATIVES_READY',native_source_sha256=frame['source_sha256'],
                source_pts=frame['source_pts'],mask_path=str(npz.relative_to(output)),mask_sha256=sha(npz),
                sam_predicted_quality=quality,mask_area_fractions=[float(mask.mean()) for mask in masks],
                center_inside_masks=[bool(mask[min(h-1,max(0,round(sy))),min(w-1,max(0,round(sx)))]) for mask in masks],
                preview={'path':str(path.relative_to(output)),'sha256':sha(path)},semantic_identity=None)
            print('RESIDUAL',event['event'],frame['frame_id'],row['mask_area_fractions'],flush=True)
    summary={'frames':len(rows),'with_unassigned_center':sum(r['status']=='REGION_ALTERNATIVES_READY' for r in rows),
        'mask_alternatives':sum(len(r.get('mask_area_fractions',[])) for r in rows),'new_qwen_calls':0,
        'target_or_object_accuracy_available':False}
    save(root/'manifest.json',{'parent_manifest_sha256':sha(output/'manifest.json'),'rows':rows,'summary':summary,
        'selection':'Most recent supported unassigned gaze center in existing two-second query. No referent names, GT or responses consulted.',
        'scope':'Exploratory post-audit engineering supplement, not frozen independent validation; all three masks retained, no semantic identity.',
        'script_sha256':sha(__file__)})
    data=json.dumps(rows,ensure_ascii=False).replace('</','<\\/')
    (root/'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>未承接注视的分割备选</title><style>body{font:15px system-ui;margin:20px;background:#f3f5f7}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:14px}figure{background:white;margin:0;padding:10px}img{width:100%}</style><h2>未承接注视 → 多尺度分割备选，尚不等于对象</h2><p>每帧选择最近一条未落入已有语言检测mask的受支持gaze中心，由SAM2点提示返回三种区域备选，全部保留。可能得到手、物体局部、地面或大范围表面；SAM质量分数不当目标概率，没有新增大模型语义判断。原始候选与未承接状态仍保留。这是看过主实验后的工程探索，不是独立验证。</p><main></main><script>const d=__DATA__,main=document.querySelector('main');d.forEach(r=>{let f=document.createElement('figure'),t=document.createElement('p');t.textContent=r.event+' '+r.frame_id+' | '+r.status;f.append(t);if(r.preview){let im=document.createElement('img');im.src='../'+r.preview.path;f.append(im)}main.append(f)})</script></html>'''.replace('__DATA__',data))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':build()
