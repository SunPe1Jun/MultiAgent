#!/usr/bin/env python3
"""Analyze presentation and historical RGB use separately, without GT leakage."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from PIL import Image,ImageDraw


def read(path):return json.loads(Path(path).read_text())
def save(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def report(output):
    analysis={};all_responses=[]
    for kind,prefix,manifest_name,label_name in [('presentation','responses','manifest.json','labels.json'),
        ('temporal','temporal_responses','temporal_manifest.json','temporal_labels.json')]:
        manifest=read(output/manifest_name);labels=read(output/label_name)
        rows=[read(p) for p in (output/prefix).glob('*/*.json')];all_responses.extend(rows)
        indexed={(r['sample'],r['condition']):r for r in rows}
        def clear(r):
            return not labels['samples'][r['sample']]['native_ambiguous' if r['condition']=='NATIVE_CROP' else 'ambiguous']
        stats={};pairs={}
        for split in ['regression','fresh']:
            stats[split]={};pairs[split]={}
            for c in manifest['conditions']:
                selected=[r for r in rows if r['split']==split and r['condition']==c]
                definite=[r for r in selected if clear(r)]
                stats[split][c]={'n':len(selected),'correct':sum(r.get('correct',False) for r in selected),
                    'clear_n':len(definite),'clear_correct':sum(r.get('correct',False) for r in definite)}
                if kind=='presentation' and c!='RGB':
                    paired=[(indexed[(s['id'],'RGB')],indexed[(s['id'],c)]) for s in manifest['samples']
                        if s['split']==split and (s['id'],'RGB') in indexed and (s['id'],c) in indexed]
                    pairs[split][c]={'n':len(paired),
                        'wins':sum(not a['correct'] and b['correct'] for a,b in paired),
                        'losses':sum(a['correct'] and not b['correct'] for a,b in paired)}
        errors=[{k:r.get(k) for k in ['sample','split','condition','category','accepted']} for r in rows if not r.get('correct')]
        analysis[kind]={'stats':stats,'pairs_vs_rgb':pairs,'errors':errors}
        payload=json.dumps({'rows':rows,'labels':labels['samples']},ensure_ascii=False).replace('</','<\\/')
        page=output/('index.html' if kind=='presentation' else 'temporal_memory.html')
        html=page.read_text();marker='<!-- SEMANTIC_RESULTS -->'
        if marker in html:html=html.split(marker)[0]
        script='''<script>const results=__RESULTS__,originalRender=render;render=function(){originalRender();__HOOK__};q.onchange=render;__EXTRA__render();</script>'''
        if kind=='presentation':
            hook="let s=d.samples[+q.value];document.querySelectorAll('figure').forEach((f,i)=>{let condition=d.conditions[i];let r=results.rows.find(r=>r.sample===s.id&&r.condition===condition);if(r)f.querySelector('figcaption').textContent+=' | model '+r.category+' | reviewed '+r.accepted.join('/')+' | '+(r.correct?'✓':'×')});"
            extra=''
        else:
            hook="let samples=d.samples.filter(s=>s.event===q.value).sort((a,b)=>a.time_bin-b.time_bin);document.querySelectorAll('figure').forEach((f,i)=>{let r=results.rows.find(r=>r.sample===samples[i].id&&r.condition===c.value);if(r)f.querySelector('figcaption').textContent+=' | model '+r.category+' | reviewed '+r.accepted.join('/')+' | '+(r.correct?'✓':'×')});"
            extra='c.onchange=render;'
        html+=marker+script.replace('__RESULTS__',payload).replace('__HOOK__',hook).replace('__EXTRA__',extra)
        page.write_text(html)
    m=read(output/'temporal_manifest.json');labels=read(output/'temporal_labels.json')
    rows=[read(p) for p in (output/'temporal_responses').glob('*/*.json')]
    indexed={(r['sample'],r['condition']):r for r in rows}
    # Category evidence stays tied to the representative observation, not the entire bin.
    memories=defaultdict(list)
    for s in m['samples']:
        prediction=indexed.get((s['id'],'NATIVE_CROP'))
        memories[s['event']].append({'time_bin_old_to_new':s['time_bin'],'support_window':s['window'],
            'bin_supported_duration_seconds':s['duration'],'selected_mode_kernel_mass_fraction':s['selected_mode_mass_fraction'],
            'representative_source_index':s['observation']['source_index'],
            'representative_time_telemetry_seconds':s['observation']['gaze_time_seconds'],
            'representative_support_duration_seconds':s['observation']['weight_seconds'],
            'source_pts_seconds':s['native_frame']['pts_seconds'],'gaze_frame_error_seconds':s['gaze_to_frame_error_seconds'],
            'rgb_evidence':s['conditions']['NATIVE_CROP'],'model_category':prediction.get('category') if prediction else None,
            'object_identity':None,'representative_category_is_whole_bin_label':False})
    memory={'schema':'semantic_attention_evidence_v1','contains_ground_truth':False,'model_called_by_export':False,
        'scope':'Chronology and numerical evidence are assigned by script. Classifier labels only the selected RGB representative. '
                'A representative category cannot be assigned to the entire bin or treated as object identity.',
        'events':{key:sorted(items,key=lambda r:r['time_bin_old_to_new']) for key,items in memories.items()}}
    save(output/'semantic_attention_evidence.json',memory)
    mismatch=[]
    for s in m['samples']:
        l=labels['samples'][s['id']]
        if not (set(l['accepted'])&set(l['native_accepted'])):
            mismatch.append({'sample':s['id'],'current_accepted':l['accepted'],'native_accepted':l['native_accepted'],
                'review_note':l['review_note']})
    analysis['temporal']['different_displayed_surface_reviews']=mismatch
    analysis['audit']={'call_counts':dict(Counter(r['status'] for r in all_responses)),
        'visual_tokens':sorted({r['processor_audit']['expanded_image_token_count'] for r in all_responses if r['status']=='COMPLETE'}),
        'all_vision_verified':all(r.get('vision_verified') for r in all_responses if r['status']=='COMPLETE'),
        'all_one_category_field':all(set(r['parsed'])=={'category'} for r in all_responses if r['status']=='COMPLETE'),
        'gt_loaded':False,'anchors_projected':False,'final_referent_evaluator_run':False}
    save(output/'analysis.json',analysis)
    lines=['# 热图新构造与使用：初试结果','',
        '类别评价依据模型输出前冻结的助手视觉复核，不是人工GT或最终referent准确率。回归30帧已看过；fresh为3条此前未使用交互的9帧。','',
        '| scope | split | condition | 符合复核 | 无歧义子集 |','| --- | --- | --- | ---: | ---: |']
    for kind in ['presentation','temporal']:
        for split,conditions in analysis[kind]['stats'].items():
            for c,v in conditions.items():lines.append(f"| {kind} | {split} | {c} | {v['correct']}/{v['n']} | {v['clear_correct']}/{v['clear_n']} |")
    lines+=['','原生历史与当前重投影可能显示不同表面，各按自己的内容标签评分，不能直接解释为同一目标的准确率差。',
        f'18个历史代表点中，{len(mismatch)}个当前/原生视图的复核类别集合不相交；这是临时内容审阅，不能独立确定几何误差原因。',
        '轮廓、聚光、密度决定尺度均未证明稳定超过RGB；原生RGB也仍有手部干扰，不能宣布时序问题已解决。','',
        '注意证据按时间保存在 semantic_attention_evidence.json；不把代表点类别赋给整段停留时长。',
        f"模型审计：{analysis['audit']}。"]
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    main=read(output/'manifest.json');example=next(s for s in main['samples'] if s['id']=='scene4_room1_155_V008')
    canvas=Image.new('RGB',(448*3,480*2),'white');draw=ImageDraw.Draw(canvas)
    for index,c in enumerate(main['conditions']):
        with Image.open(output/example['conditions'][c]['path']) as im:panel=im.crop((448,32,896,480))
        x,y=index%3*448,index//3*480;canvas.paste(panel,(x,y+32));draw.text((x+10,y+9),c,fill='black')
    canvas.save(output/'presentation_overview.jpg',quality=96)
    print(json.dumps({'stats':{kind:analysis[kind]['stats'] for kind in ['presentation','temporal']},'audit':analysis['audit'],'different_surface_review_n':len(mismatch)},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('.local-sync/heatmap_forms_v2_20261105'))
    report(p.parse_args().output.resolve())
