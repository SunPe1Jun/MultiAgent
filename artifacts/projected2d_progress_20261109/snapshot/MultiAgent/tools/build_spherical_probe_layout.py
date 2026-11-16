#!/usr/bin/env python3
"""Reproduce the second visual layout, preserving maps, RGB and expected labels."""
import argparse
from io import BytesIO
from pathlib import Path
from PIL import Image,ImageDraw
from MultiAgent.spherical_attention_study import OUTPUT,read,gallery
from MultiAgent.heatmap_temporal_pilot import save,sha

PROMPT='''Choose which labeled RGB region should be inspected first, based on the user's visual attention.
The TOP scene and TOP RGB thumbnails define regions A, B, C in a fixed angular coordinate system.
BELOW ALL RGB are exactly three rows explicitly labeled EARLY HEAT, MIDDLE HEAT, LATE HEAT.
Each gray row shows coverage only. Added ORANGE HEAT is user attention. RGB object colors are not heat.
Choose the region A, B, or C containing the BRIGHTEST ORANGE HEAT peak in EARLY HEAT, then in LATE HEAT.
If a requested HEAT row has NO ORANGE, return none. Do not select an RGB region from appearance alone.
This is attention priority, not a target-object label. Images may contain controls.
Return only JSON with exactly two fields, A/B/C/none: {"early":"A","late":"B"}.'''


def make_card(source):
    card=Image.new('RGB',(1008,1008),'#15212b');d=ImageDraw.Draw(card)
    d.text((12,7),'RGB CONTEXT AND RGB THUMBNAILS - OBJECT COLORS ARE NOT ATTENTION',fill='white')
    card.paste(source.crop((12,28,996,268)),(12,28))
    for j in range(3):
        crop=source.crop((12+j*328,738,332+j*328,978)).resize((320,168),Image.Resampling.BILINEAR)
        card.paste(crop,(12+j*328,282));d.text((22+j*328,288),'ABC'[j]+' RGB',fill='white',stroke_width=2,stroke_fill='black')
    for i,name in enumerate(['EARLY HEAT','MIDDLE HEAT','LATE HEAT']):
        y=474+i*174
        row=source.crop((12,284+i*140,996,410+i*140)).resize((984,156),Image.Resampling.BILINEAR)
        card.paste(row,(12,y));d.rectangle((12,y,140,y+22),fill='#15212b');d.text((17,y+5),name,fill='white')
    d.text((12,992),'ONLY ORANGE IN HEAT ROWS COUNTS. NO ORANGE -> none.',fill='white')
    return card


def build(root,verify_only=False):
    target=root/'layout_v2';manifest=read(root/'manifest.json')
    if not verify_only and (target/'manifest.json').exists():raise ValueError('Never overwrite frozen layout experiment')
    if verify_only:
        prepared=read(target/'manifest.json');assert prepared['prompt']==PROMPT
        indexed={e['id']:e for e in prepared['events']}
    n=0
    for e in manifest['events']:
        for condition,file in e['conditions'].items():
            with Image.open(root/file['path']) as im:card=make_card(im.convert('RGB'))
            path=target/'inputs'/e['id']/f'{condition}.png'
            if verify_only:
                encoded=BytesIO();card.save(encoded,format='PNG')
                assert encoded.getvalue()==path.read_bytes()
                assert sha(path)==indexed[e['id']]['conditions'][condition]['sha256']
                assert file['expected_visual_priority']==indexed[e['id']]['conditions'][condition]['expected_visual_priority']
            else:
                path.parent.mkdir(parents=True,exist_ok=True);card.save(path)
                file['path']=str(path.relative_to(target));file['sha256']=sha(path)
            n+=1
    if not verify_only:
        manifest['prompt']=PROMPT;manifest['version']='spherical_visual_priority_layout_v2'
        manifest['layout_iteration']='Regression layout change after all v1 outputs; no fresh validation. Same map geometry, RGB, expected labels and model.'
        save(target/'manifest.json',manifest);gallery(target,manifest)
    print('REPRODUCED' if verify_only else 'PREPARED',n,'layout cards')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--verify-only',action='store_true')
    a=p.parse_args();build(a.output.resolve(),a.verify_only)
