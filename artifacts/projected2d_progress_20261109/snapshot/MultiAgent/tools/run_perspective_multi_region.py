#!/usr/bin/env python3
"""Retain up to two visually prominent attention regions instead of hard top1.

Same frozen heat/RGB images, new prompt only. This is regression, not unseen
validation. Labels describe displayed engineering peaks, not intended referents.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.perspective_attention_study import OUTPUT,read
from MultiAgent.qwen_backend import QwenBackend

PROMPT='''Use only added ORANGE in the HEAT tiles to propose RGB regions worth inspecting.
All RGB thumbnails are at the TOP. The six large HEAT tiles are BELOW, in a 2-by-3 grid. Printed A/B/C/D/E/F IDs link heat to RGB; screen positions vary.
Gray is source camera coverage and is NOT attention. RGB object colors are NOT attention. Only added ORANGE in HEAT is attention.
Select the region with the BRIGHTEST ORANGE peak. You may also keep ONE additional region if its own orange peak is substantial, approximately at least half as bright as the brightest. Keep at most TWO IDs. A dark region is never a proposal.
If no heat tile contains orange, return an EMPTY array. Do not guess from RGB appearance. This is a shortlist of attention directions, not final target objects. The script handles chronology.
Return exactly one JSON field: {"regions":["A"]}. IDs must be unique A/B/C/D/E/F. Use {"regions":[]} for no added heat.'''


def prepare(output):
    root=output/'multi_region';path=root/'manifest.json'
    if path.exists():return read(path)
    source=read(output/'phase_layout_v2/manifest.json');requests=[]
    events={e['id']:e for e in source['events']}
    for r in source['requests']:
        e=events[r['event_id']];peaks=r['peak_values_by_time'][r['phase']];maximum=max(peaks)
        rank=sorted(range(6),key=lambda i:-peaks[i]);threshold=.5
        admissible=[e['labels_by_region'][i] for i in rank if maximum>0 and peaks[i]>=threshold*maximum]
        top1=r['expected']['selected'];imagesource=output/'phase_layout_v2'/r['path']
        requests.append({**r,'path':str(imagesource.resolve()),'file_sha256':sha(imagesource),
            'expected_top1':top1,'half_peak_admissible_labels':admissible,
            'expected_top2':admissible[:2],'scope':'relative displayed-density-peak threshold, not calibrated attention probability'})
    manifest={'version':'visual_attention_shortlist_v1','prompt':PROMPT,'requests':requests,
        'parent_manifest_sha256':sha(output/'phase_layout_v2/manifest.json'),
        'regression_only':True,'definition':'Optional second direction at least half the displayed density peak; max2.',
        'caveat':'Top2 recall is not directly comparable to top1 accuracy; report proposal count, precision and omission.',
        'new_gt_loaded':False,'code_sha256':sha(__file__)}
    save(path,manifest);return manifest


def run(output):
    root=output/'multi_region';m=prepare(output)
    if sha(output/'phase_layout_v2/manifest.json')!=m['parent_manifest_sha256']:raise ValueError('Parent changed')
    lock=root/'protocol_lock.json'
    if lock.exists() and read(lock)['manifest_sha256']!=sha(root/'manifest.json'):raise ValueError('Shortlist protocol changed')
    save(lock,{'manifest_sha256':sha(root/'manifest.json')})
    backend=QwenBackend(root/'model_cache',model='/ai/data/Qwen3-VL-8B-Instruct',local_only=True,
                        namespace='perspective_shortlist_v1',trace_visual_inputs=True)
    schedule=m['requests'][:];random.Random(20261106).shuffle(schedule)
    for r in schedule:
        dest=root/'responses'/f"{r['id']}.json";path=Path(r['path'])
        if sha(path)!=r['file_sha256']:raise ValueError('Input hash changed')
        if dest.exists() and read(dest)['status']=='COMPLETE':continue
        print('INFER',r['id'],flush=True);start=time.monotonic()
        parsed=backend.ask(r['id'],'shortlist',m['prompt'],[path],max_new_tokens=120);req=read(backend.requests[-1]);values=parsed.get('regions')
        valid=(set(parsed)=={'regions'} and isinstance(values,list) and len(values)<=2 and len(values)==len(set(values)) and all(v in list('ABCDEF') for v in values))
        result={'id':r['id'],'event_id':r['event_id'],'event':r['event'],'split':r['split'],'condition':r['condition'],'phase':r['phase'],
            'status':'COMPLETE','parsed':parsed,'schema_valid':valid,'expected_top1':r['expected_top1'],
            'admissible':r['half_peak_admissible_labels'],'expected_top2':r['expected_top2'],
            'top1_retained':valid and (r['expected_top1'] in values if r['expected_top1']!='none' else len(values)==0),
            'unsupported_proposal_count':sum(v not in r['half_peak_admissible_labels'] for v in values) if valid else None,
            'exact_expected_top2_set':valid and set(values)==set(r['expected_top2']),
            'request_path':backend.requests[-1],'seconds':time.monotonic()-start,
            'processor_audit':req['processor_audit'],'vision_verified':req['visual_tensor_trace']['input_consumption_verified']}
        save(dest,result);print('RESULT',parsed,'top1',r['expected_top1'],'admissible',r['half_peak_admissible_labels'],flush=True)
    report(output)


def report(output):
    root=output/'multi_region';m=read(root/'manifest.json');rows=[read(p) for p in (root/'responses').glob('*.json')]
    stats={}
    for split in ['development','check']:
        stats[split]={}
        for c in ['TRUE_HEAT','REVERSE_TIME','SHIFT_SPACE','NO_HEAT']:
            rr=[r for r in rows if r['split']==split and r['condition']==c]
            total=sum(len(r['parsed'].get('regions',[])) for r in rr)
            unsupported=sum(r['unsupported_proposal_count'] or 0 for r in rr)
            stats[split][c]={'n':len(rr),'top1_retained':sum(r['top1_retained'] for r in rr),
                'proposals':total,'unsupported_proposals':unsupported,
                'admissible_proposal_precision':(total-unsupported)/total if total else None,
                'exact_expected_top2_sets':sum(r['exact_expected_top2_set'] for r in rr)}
    summary={'stats':stats,'model_calls':len(rows),'all_schema_valid':all(r['schema_valid'] for r in rows),
        'tokens':sorted({r['processor_audit']['expanded_image_token_count'] for r in rows}),'all_vision_verified':all(r['vision_verified'] for r in rows),
        'regression_only':True,'scope':'visual proposal recall at max2, no final referent accuracy'}
    save(root/'analysis.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
    cards=[]
    for r in m['requests']:
        if r['condition']!='TRUE_HEAT':continue
        result=next((v for v in rows if v['id']==r['id']),None)
        cards.append({'image':__import__('os').path.relpath(r['path'],root),
            'title':r['event']+' '+('早段' if r['phase']==0 else '晚段'),
            'prediction':result['parsed'] if result else None,'top1':r['expected_top1'],'admissible':r['half_peak_admissible_labels']})
    data=json.dumps(cards,ensure_ascii=False).replace('</','<\\/')
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>多个关注方向</title><style>body{font:15px system-ui;margin:20px;background:#f3f5f7}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:14px}figure{margin:0;background:white;padding:12px}img{width:100%}</style><h2>保留最多两个视觉关注方向</h2><p>只看图中的热图，不输入数值分数。可保留第二个明显热点，后续由语言与RGB继续筛选。阈值是显示密度峰值的工程约定，不是用户真实目标标签。下面均为之前已用样本上的回归尝试；top2覆盖不能直接当成top1准确率提升。</p><main></main><script>const d=__DATA__;const main=document.querySelector('main');d.forEach(r=>{let f=document.createElement('figure'),c=document.createElement('figcaption'),im=document.createElement('img');c.textContent=r.title+' | model '+JSON.stringify(r.prediction)+' | 最高峰 '+r.top1+' | 半峰以上 '+r.admissible.join('/');im.src=r.image;f.append(c,im);main.append(f)})</script></html>'''
    (root/'index.html').write_text(html.replace('__DATA__',data))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','report'],default='run',nargs='?');p.add_argument('--output',type=Path,default=OUTPUT)
    a=p.parse_args();{'run':run,'report':report}[a.action](a.output.resolve())
