#!/usr/bin/env python3
"""Prepare frozen confirmation or mirror-control images without changing old runs."""
import argparse
from pathlib import Path
from MultiAgent.projected2d_attention_study import save_json,sha_file
from tools import iterate_original2d_visual as study


def main():
 p=argparse.ArgumentParser();p.add_argument('--condition',choices=['HEAT_CROP','HEAT_CROP_FOV','FIXED_GAZE18','FIXED_GAZE22','FIXED_GAZE27','FIXED_GAZE30','FIXED_GAZE35'],required=True)
 p.add_argument('--scope',choices=['dev','confirm','all'],default='confirm');p.add_argument('--mirrored',action='store_true')
 p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 cache=study.ROOT/'.local-sync/2d_original_gaze_cache32_20261109';study.MAPS=cache
 fractions={'FIXED_GAZE18':.18,'FIXED_GAZE22':.22,'FIXED_GAZE27':.27,'FIXED_GAZE30':.30,'FIXED_GAZE35':.35}
 if a.condition in fractions:
  fraction=fractions[a.condition]
  study.CONFIGS[a.condition]={'kind':'paired','foveation':False,'crop':'heat','floor':1.,'dilation':12.}
  def fixed_current_gaze(meta,maps,size):
   if meta.get('gaze_projection_valid')!='True':return None
   width,height=size;side=max(32,int(min(size)*fraction));u=float(meta['gaze_u_norm']);v=float(meta['gaze_v_norm'])
   left=max(0,min(width-side,round(u*width-side/2)));top=max(0,min(height-side,round(v*height-side/2)))
   return [left,top,left+side,top+side]
  study.heat_bbox=fixed_current_gaze
 if a.mirrored:
  original=study.heat_bbox
  def mirrored(meta,maps,size):
   adjusted=dict(meta)
   if meta.get('gaze_u_norm'):adjusted['gaze_u_norm']=str(1-float(meta['gaze_u_norm']))
   return original(adjusted,maps[:,:,::-1].copy(),size)
  study.heat_bbox=mirrored
 study.prepare(a.output.resolve(),a.condition,a.scope)
 target=a.output.resolve()/a.condition
 import json
 lock=json.loads((target/'protocol_lock.json').read_text());lock['files'].update({str(Path(__file__).resolve()):sha_file(Path(__file__).resolve()),str(cache/'input_image_audit.json'):sha_file(cache/'input_image_audit.json')})
 save_json(target/'protocol_lock.json',lock)
 save_json(target/'preparation_options.json',{'scope':a.scope,'mirrored':a.mirrored,
  'fixed_gaze_crop_fraction':fractions.get(a.condition),'heatmap_source':str(cache),'rendering_uses_target_annotations':False})
if __name__=='__main__':main()
