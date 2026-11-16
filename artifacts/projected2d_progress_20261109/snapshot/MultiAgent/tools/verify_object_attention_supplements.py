#!/usr/bin/env python3
"""Verify residual point alternatives and post-inference referent-only viewers."""
import json
from pathlib import Path
import sys
import numpy as np
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.perspective_attention_study import read
from tools.run_object_attention_study import OUTPUT
from tools.verify_attention_region_study import Scripts


def verify(output=OUTPUT,jscheck_path=Path('/tmp/heatmap_forms_v2_jscheck')):
    main=read(output/'manifest.json');frames={(e['event'],f['frame_id']):f for e in main['events'] for f in e['frames']}
    residual=read(output/'residual_points/manifest.json');assert residual['parent_manifest_sha256']==sha(output/'manifest.json')
    masks_checked=0;ready=0
    for row in residual['rows']:
        frame=frames[(row['event'],row['frame_id'])];masks=np.load(output/frame['npz_path'])['masks'];h,w=masks.shape[1:];union=masks.any(axis=0)
        observations=[o for b in frame['attention_stats']['bins'] for o in b['observations']]
        eligible=[]
        for obs in observations:
            x,y=obs['pixel_xy'];px=min(w-1,max(0,round(x)));py=min(h-1,max(0,round(y)))
            if not union[py,px]:eligible.append(obs)
        if not eligible:assert row['status']=='NO_UNASSIGNED_OBSERVATION';continue
        assert row['observation']==max(eligible,key=lambda o:o['gaze_time_seconds'])
        assert row['native_source_sha256']==frame['source_sha256'] and row['semantic_identity'] is None
        assert sha(output/row['mask_path'])==row['mask_sha256']
        a=np.load(output/row['mask_path'])['masks'];assert a.shape==(3,h,w)
        assert [float(m.mean()) for m in a]==row['mask_area_fractions']
        x,y=row['observation']['pixel_xy'];px=min(w-1,max(0,round(x)));py=min(h-1,max(0,round(y)))
        assert [bool(m[py,px]) for m in a]==row['center_inside_masks']
        assert sha(output/row['preview']['path'])==row['preview']['sha256'];masks_checked+=len(a);ready+=1
    audit=read(output/'posthoc_audit/manifest.json');assert len(audit['rows'])==len(frames)
    center=read(output/'referent_center_audit.json')
    for row in audit['rows']:
        assert sha(output/row['path'])==row['sha256']
        expected=[r for r in center['eligible_reference_frames'] if r['event']==row['event'] and r['frame_id']==row['frame_id']]
        assert [{k:v for k,v in r.items() if k!='audit_id'} for r in row['references']]==expected
    sys.path.insert(0,str(jscheck_path));import quickjs
    dom="class E{constructor(t='div'){this.tag=t;this.children=[];this.value=''}append(...xs){for(const x of xs){this.children.push(x);if(this.tag==='select'&&!this.value)this.value=x.value}}replaceChildren(){this.children=[]}}const roots={};const document={querySelector(s){return roots[s]||(roots[s]=new E(s==='#event'?'select':s))},createElement(t){return new E(t)}};"
    counts={}
    for name in ['residual_points','posthoc_audit']:
        root=output/name;parser=Scripts();parser.feed((root/'index.html').read_text());ctx=quickjs.Context();ctx.eval(dom)
        for script in parser.scripts:ctx.eval(script)
        if name=='posthoc_audit':n=ctx.eval("let n=0;for(const e of [...new Set(d.map(r=>r.event))]){q.value=e;render();n+=document.querySelector('main').children.length}n")
        else:n=ctx.eval("document.querySelector('main').children.length")
        assert n==39;counts[name]=n
        paths=json.loads(ctx.eval("JSON.stringify(document.querySelector('main').children.flatMap(f=>f.children).filter(c=>c.tag==='img').map(c=>c.src))"))
        for path in paths:assert (root/path).exists()
    result={'residual_ready_frames':ready,'residual_multiscale_masks_verified':masks_checked,
        'residual_selection_reproduced_without_gt':True,'no_semantic_identity_inferred':True,
        'posthoc_audit_reference_only_images':len(audit['rows']),'viewer_js_rendered':counts,'full_browser':False}
    save(output/'supplement_verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':verify()
