#!/usr/bin/env python3
"""Group all RGB above all heat tiles; freeze after development-only failure."""
import argparse
from io import BytesIO
from pathlib import Path
from PIL import Image,ImageDraw
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.perspective_attention_study import OUTPUT,read,gallery

PROMPT='''Use only the added ORANGE in the six HEAT tiles to select an RGB region for inspection.
All six small RGB thumbnails are in the TOP ROW. These RGB colors are NOT attention evidence.
BELOW ALL RGB, the six large HEAT tiles are arranged in a 2-by-3 grid. Printed IDs A/B/C/D/E/F link each heat tile to the corresponding RGB thumbnail. Letter IDs and screen positions vary.
Gray in a HEAT tile marks a source camera footprint and is NOT heat. Black is not heat. Only added ORANGE is heat.
Select the printed ID of the HEAT tile with the BRIGHTEST ORANGE heat peak. If no HEAT tile contains orange, return none. Never choose from gray coverage or RGB appearance.
All heat tiles are for ONE interval; the script handles time. User attention is not a target-object label. Images may be controls.
Return only one JSON field A/B/C/D/E/F/none: {"selected":"A"}.'''


def make_card(source,labels,order):
    canvas=Image.new('RGB',(1008,1008),'#15212b');draw=ImageDraw.Draw(canvas)
    draw.text((12,8),'TOP ROW: RGB CONTEXT ONLY - NOT ATTENTION',fill='white')
    for col,index in enumerate(order):
        xold=12+(col%3)*328;yold=36+(col//3)*470
        rgb=source.crop((xold+80,yold+22,xold+232,yold+174))
        x=12+col*164;canvas.paste(rgb.resize((148,148)),(x,34))
        draw.rectangle((x,34,x+147,181),outline='white',width=2)
        draw.text((x+8,42),labels[index]+' RGB',fill='white',stroke_width=2,stroke_fill='black')
    draw.text((12,216),'HEAT TILES BELOW: ONLY ORANGE COUNTS. GRAY IS COVERAGE, NOT HEAT.',fill='white')
    for col,index in enumerate(order):
        xold=12+(col%3)*328;yold=36+(col//3)*470
        tile=source.crop((xold+32,yold+208,xold+288,yold+464))
        x=12+(col%3)*328+32;y=252+(col//3)*350
        canvas.paste(tile,(x,y));draw.rectangle((x,y,x+255,y+255),outline='white',width=2)
        draw.text((x+12,y+12),labels[index],fill='white',stroke_width=2,stroke_fill='black')
    draw.text((12,970),'No ORANGE in any HEAT tile -> none. Script assigns the time interval.',fill='white')
    return canvas


def build(output,verify_only=False):
    m=read(output/'manifest.json');target=output/'phase_layout_v2'
    if not verify_only and (target/'manifest.json').exists():raise ValueError('Never overwrite frozen layout')
    indexed={e['id']:e for e in m['events']};requests=[r for r in m['requests'] if r['mode']=='phase']
    if verify_only:
        prepared=read(target/'manifest.json');assert prepared['prompts']['phase']==PROMPT
        expected={r['id']:r for r in prepared['requests']}
    for r in requests:
        e=indexed[r['event_id']]
        with Image.open(output/r['path']) as im:card=make_card(im.convert('RGB'),e['labels_by_region'],e['screen_order'])
        path=target/'inputs'/e['id']/f'{r["condition"]}_{r["phase"]}.png'
        if verify_only:
            buf=BytesIO();card.save(buf,format='PNG');assert buf.getvalue()==path.read_bytes()
            assert sha(path)==expected[r['id']]['sha256']
        else:
            path.parent.mkdir(parents=True,exist_ok=True);card.save(path);r['path']=str(path.relative_to(target));r['sha256']=sha(path)
    if not verify_only:
        m['requests']=requests;m['prompts']['phase']=PROMPT;m['version']='perspective_phase_grouped_layout_v2'
        m['layout_iteration']='After initial development6 only: RGB/heat grid confused coverage with heat; group all RGB above all heat and clarify gray/none. Remaining7 not run yet.'
        m['parent_manifest_sha256']=sha(output/'manifest.json');m['layout_code_sha256']=sha(__file__)
        # RGB source views are for reviewer display only; point to their parent files.
        for e in m['events']:
            for sources in e['native_sources_by_region']:
                for s in sources:s['rendered_path']='../'+s['rendered_path']
        save(target/'manifest.json',m)
        save(target/'iteration_protocol.json',{'parent_development_analysis':read(output/'analysis.json')['stats']['development'],
            'check_has_not_started':not (output/'check_started.json').exists(),
            'changes':['group all RGB above heat','clarify gray footprint is not attention'],
            'same_heat_tile_pixels_and_RGB_context':True,'no_independent_validation_claim':True})
        gallery(target,m)
    print('REPRODUCED' if verify_only else 'PREPARED',len(requests),'phase cards')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--verify-only',action='store_true')
    a=p.parse_args();build(a.output.resolve(),a.verify_only)
