#!/usr/bin/env python3
"""Rebuild actual original-wrapper tensors and verify frozen visual responses."""
import argparse
import json
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from MultiAgent.projected2d_attention_study import save_json,sha_file
from MultiAgent.visual_input_trace import tensor_record
from tools.iterate_original2d_visual import check,visual,MODEL

def verify(roots):
 v=visual();processor=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True)
 for root in roots:
  check(root);manifest=json.loads((root/'input_manifest.json').read_text());frozen=json.loads((root/'predictions_frozen.json').read_text())['responses'];calls=[]
  for e in manifest['events']:
   path=root/'responses'/(e['id']+'.json')
   if sha_file(path)!=frozen[str(path)]:raise ValueError('Frozen response changed')
   r=json.loads(path.read_text());images=[]
   for item in e['images']:
    if sha_file(Path(item['path']))!=item['sha256']:raise ValueError('Image hash mismatch')
    with Image.open(item['path']) as opened:images.append(opened.convert('RGB'))
   rebuilt=v.build_model_inputs(processor,images,e['prompt'],e['panel_meta'],'none')
   for im in images:im.close()
   trace=r['visual_tensor_trace'];assert trace['input_consumption_verified']
   for key in ('input_ids','pixel_values','image_grid_thw'):
    if tensor_record(rebuilt[key])!=trace['processor_tensors'][key]:raise ValueError(e['id']+': '+key+' mismatch')
   calls.append({'event':e['id'],'full_input_ids_and_pixels_match':True,'vision_consumption_verified':True})
  save_json(root/'processor_reconstruction_verification.json',{'calls_verified':len(calls),'calls':calls})
  print('VERIFIED_ORIGINAL_INPUTS',root,len(calls),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('roots',type=Path,nargs='+');a=p.parse_args();verify([x.resolve() for x in a.roots])
