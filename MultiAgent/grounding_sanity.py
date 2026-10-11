"""Single real-frame GroundingDINO sanity check; preserves VR candidate IDs."""
from __future__ import annotations
import argparse,json, re, math
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
from .baseline import module

def main():
 p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--event-json',required=True);p.add_argument('--anchor',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--model',default='/workspace/usr3/grounding-dino-base');a=p.parse_args()
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);im=Image.open(a.image).convert('RGB');labels=['forklift','truck','person','building','helicopter','cargo','van','drone','chair','lamp','curtain','cabinet','rug','window'];query='. '.join(labels)+'.'
 proc=AutoProcessor.from_pretrained(a.model,local_files_only=True);model=AutoModelForZeroShotObjectDetection.from_pretrained(a.model,local_files_only=True).to('cuda').eval();inputs=proc(images=im,text=query,return_tensors='pt').to('cuda')
 with torch.inference_mode(): result=model(**inputs)
 post=proc.post_process_grounded_object_detection(result,inputs.input_ids,threshold=.25,text_threshold=.2,target_sizes=[im.size[::-1]])[0]
 boxes=[]
 for b,s,l in zip(post['boxes'],post['scores'],post['text_labels']):boxes.append({'label':str(l),'score':float(s),'box':[float(x) for x in b]})
 # Candidate universe is the anchor table; detector labels are never IDs.
 api=module('api'); anchors=api.load_scene_anchor_table(Path(a.anchor)); event=json.loads(Path(a.event_json).read_text())
 # Existing baseline projection is used only when the event has camera pose/FOV. image_size is
 # absent in this profile, so normalized coordinates are compared against the actual frame size.
 align=[]
 for det in boxes:
  x0,y0,x1,y1=det['box']; cx,cy=(x0+x1)/2/im.width,(y0+y1)/2/im.height; typ=re.sub(r'\d+$','',det['label'].lower())
  candidates=[]
  for c in anchors:
   if re.sub(r'\d+$','',c['object_name'].lower()) != typ: continue
   u,v,ok=api.project_world_to_image(event['peak_spatial'],c)
   if ok and 0<=u<=1 and 0<=v<=1: candidates.append((math.hypot(u-cx,v-cy),c['object_name'],u,v))
  candidates.sort();align.append({'detection':det,'projected_candidate':candidates[0][1] if candidates else None,'projection_distance':candidates[0][0] if candidates else None,'candidate_projection':candidates[0][2:] if candidates else None})
 payload={'image':str(Path(a.image).resolve()),'query':query,'detector':'GroundingDINO transformers local','candidate_count':len(anchors),'detections':boxes,'candidate_alignment':'existing world->image projection + type-constrained nearest center; inspect distances before using as evidence','alignments':align}
 (out/'detections.json').write_text(json.dumps(payload,indent=2))
 canvas=im.copy();d=ImageDraw.Draw(canvas)
 for b in boxes:d.rectangle(b['box'],outline='lime',width=4);d.text((b['box'][0],b['box'][1]),f"{b['label']} {b['score']:.2f}",fill='yellow')
 canvas.save(out/'overlay.jpg');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
