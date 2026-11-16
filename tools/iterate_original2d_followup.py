#!/usr/bin/env python3
"""Bounded controls for image scale and foveation strength around the restored baseline."""
import argparse
import json
from pathlib import Path
import re
from MultiAgent.projected2d_attention_study import save_json,sha_file
from tools import iterate_original2d_visual as study


FIXED_FRACTION=.22

def fixed_gaze_bbox(meta,maps,size):
    # Fixed source resolution crop at current measured gaze: no heat-density adaptation.
    width,height=size
    if meta.get('gaze_projection_valid')!='True':return None
    u=float(meta['gaze_u_norm']);v=float(meta['gaze_v_norm']);side=max(32,int(min(size)*FIXED_FRACTION))
    x=max(0,min(width-side,round(u*width-side/2)))
    y=max(0,min(height-side,round(v*height-side/2)))
    return [x,y,x+side,y+side]


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','infer','evaluate'])
    p.add_argument('--condition',choices=['FIXED_GAZE18','FIXED_GAZE22','FIXED_GAZE27','FIXED_GAZE30','FIXED_GAZE35','HEAT_CROP_FOV95'],required=True)
    p.add_argument('--scope',choices=['dev','confirm','all'],default='dev');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    study.MAPS=study.ROOT/'.local-sync/2d_original_gaze_cache32_20261109'
    global FIXED_FRACTION
    fractions={'FIXED_GAZE18':.18,'FIXED_GAZE22':.22,'FIXED_GAZE27':.27,'FIXED_GAZE30':.30,'FIXED_GAZE35':.35}
    for key,value in fractions.items():
        study.CONFIGS[key]={'kind':'paired','foveation':False,'crop':'heat','floor':1.,'dilation':12.}
    study.CONFIGS['HEAT_CROP_FOV95']={'kind':'paired','foveation':True,'crop':'heat','floor':.95,'dilation':12.}
    if a.condition in fractions:
        FIXED_FRACTION=fractions[a.condition]
        study.heat_bbox=fixed_gaze_bbox
    output=a.output.resolve()
    if a.stage=='prepare':
        study.prepare(output,a.condition,a.scope)
        target=output/a.condition
        options={'condition':a.condition,'scope':a.scope,'heatmaps':str(study.MAPS),
                 'change':f'Fixed {FIXED_FRACTION:.0%} current gaze crop scale control' if a.condition in fractions else 'Heat crop plus foveation, RGB peripheral weight floor raised from .80 to .95; same 12 degree aperture'}
        save_json(target/'followup_options.json',options)
        lock=json.loads((target/'protocol_lock.json').read_text())
        for dep in [Path(__file__).resolve(),target/'followup_options.json',study.MAPS/'input_image_audit.json']:
            lock['files'][str(dep)]=sha_file(dep)
        save_json(target/'protocol_lock.json',lock)
    else:{'infer':study.infer,'evaluate':study.evaluate}[a.stage](output,a.condition)
if __name__=='__main__':main()
