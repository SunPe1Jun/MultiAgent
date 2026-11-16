#!/usr/bin/env python3
"""Explicit fixed-gaze control around immutable original-wrapper input code."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from MultiAgent.projected2d_attention_study import save_json, sha_file
from tools import iterate_original2d_visual as study


def prepare(output, scope, fraction):
    study.MAPS=study.ROOT/'.local-sync/2d_original_gaze_cache32_20261109'
    condition='FIXED_CURRENT_GAZE'
    study.CONFIGS[condition]={'kind':'paired','foveation':False,'crop':'heat','floor':1.,'dilation':12.}
    def fixed_box(meta,maps,size):
        if meta.get('gaze_projection_valid')!='True':return None
        width,height=size
        u=float(meta['gaze_u_norm']);v=float(meta['gaze_v_norm'])
        if not (0<=u<=1 and 0<=v<=1):return None
        side=max(32,int(min(size)*fraction))
        left=max(0,min(width-side,round(u*width-side/2)))
        top=max(0,min(height-side,round(v*height-side/2)))
        return [left,top,left+side,top+side]
    study.heat_bbox=fixed_box
    study.prepare(output,condition,scope)
    target=output/condition
    manifest=json.loads((target/'input_manifest.json').read_text())
    gaze_by_frame={(x['event'],x['frame_id']):x for x in json.loads((study.MAPS/'input_image_audit.json').read_text())['panels']}
    audit=[]
    for event in manifest['events']:
        for meta in event['panel_meta']:
            gaze=gaze_by_frame[(event['id'],meta['panel_id'])]
            actual=[float(x) for x in meta['crop_bbox_norm'].split(',')]
            native=json.loads((study.BASE/'responses'/f"{event['scene']}_row_{event['row_index']}.json").read_text())
            prior=next(x for x in native['panel_meta'] if x['panel_id']==meta['panel_id'])
            supported=bool(gaze['clock_supported'])
            if supported:
                from PIL import Image
                with Image.open(prior['source_frame_path']) as image:size=image.size
                box=fixed_box(prior,None,size)
                expected=[box[0]/size[0],box[1]/size[1],box[2]/size[0],box[3]/size[1]] if box else [float(x) for x in prior['crop_bbox_norm'].split(',')]
            else:expected=[float(x) for x in prior['crop_bbox_norm'].split(',')]
            assert np.max(np.abs(np.array(actual)-expected))<=.000051,(event['id'],meta['panel_id'],actual,expected)
            audit.append({'event':event['id'],'frame_id':meta['panel_id'],'fixed_fraction':fraction,
                'history_clock_supported':supported,'actual_bbox_norm':actual,'verified_bbox':True,
                'fallback_to_original_crop':not supported})
    save_json(target/'control_contract.json',{'fixed_fraction':fraction,'history_used_for_crop_shape_or_center':False,
        'same_clock_gate_as_heatmap_condition':True,'scope':scope,'all_panels_explicitly_verified':True,'panels':audit})
    lock=json.loads((target/'protocol_lock.json').read_text())
    for file in [Path(__file__).resolve(),target/'control_contract.json',study.MAPS/'input_image_audit.json']:
        lock['files'][str(file)]=sha_file(file)
    save_json(target/'protocol_lock.json',lock)
    print('EXPLICIT_FIXED_CONTROL',scope,fraction,len(audit),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','infer','evaluate'])
    p.add_argument('--scope',choices=['dev','confirm','all'],default='confirm')
    p.add_argument('--fraction',type=float,default=.22);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();output=a.output.resolve()
    if a.stage=='prepare':
        if not .1<=a.fraction<=.9:raise ValueError('fraction must be in [.1,.9]')
        prepare(output,a.scope,a.fraction)
    else:{'infer':study.infer,'evaluate':study.evaluate}[a.stage](output,'FIXED_CURRENT_GAZE')

if __name__=='__main__':main()
