#!/usr/bin/env python3
"""Report frozen region labels, paired presentation effects and time-indexed observations."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def report(root):
    m=read(root/'manifest.json');labels=read(root/'labels.json')
    assert hashlib.sha256((root/'labels.json').read_bytes()).hexdigest()==read(root/'annotation_lock.json')['sha256']
    samples={s['id']:s for s in m['samples']}
    rows=[read(p) for p in (root/'responses').glob('*/*.json')]
    indexed={(r['sample'],r['condition']):r for r in rows}
    def clear(row):
        key='shifted_ambiguous' if row['condition']=='SHIFTED_DETAIL' else 'ambiguous'
        return not labels['samples'][row['sample']][key]
    def stat(selected):
        definite=[r for r in selected if clear(r)]
        return {'n':len(selected),'correct':sum(bool(r.get('correct')) for r in selected),
                'clear_n':len(definite),'clear_correct':sum(bool(r.get('correct')) for r in definite),
                'failures':sum(r['status']!='COMPLETE' for r in selected)}
    counts={};pairs={};shifted={}
    for split in ['dev','validation']:
        counts[split]={c:stat([r for r in rows if r['split']==split and r['condition']==c])
                       for c in m['conditions'] if any(r['split']==split and r['condition']==c for r in rows)}
        pairs[split]={}
        for c in counts[split]:
            if c in ['RGB_REGION','SHIFTED_DETAIL']:continue
            matched=[(indexed[(s['id'],'RGB_REGION')],indexed[(s['id'],c)]) for s in m['samples']
                     if s['split']==split and (s['id'],'RGB_REGION') in indexed and (s['id'],c) in indexed]
            pairs[split][c]={'n':len(matched),'wins':sum(not a.get('correct') and b.get('correct') for a,b in matched),
                'losses':sum(a.get('correct') and not b.get('correct') for a,b in matched),
                'changed':sum(a.get('category')!=b.get('category') for a,b in matched),
                'clear_wins':sum(not a.get('correct') and b.get('correct') and clear(a) for a,b in matched),
                'clear_losses':sum(a.get('correct') and not b.get('correct') and clear(a) for a,b in matched)}
        matched=[]
        for s in m['samples']:
            if s['split']!=split:continue
            a=indexed.get((s['id'],'DETAIL_035'));b=indexed.get((s['id'],'SHIFTED_DETAIL'))
            l=labels['samples'][s['id']]
            if a and b and not (set(l['accepted']) & set(l['shifted_accepted'])):matched.append((a,b))
        shifted[split]={'disjoint_region_labels_n':len(matched),
            'prediction_changed':sum(a.get('category')!=b.get('category') for a,b in matched),
            'both_displayed_regions_correct':sum(a.get('correct') and b.get('correct') for a,b in matched)}
    by_motion={split:{c:{motion:stat([r for r in rows if r['split']==split and r['condition']==c and r.get('motion_bin')==motion])
        for motion in ['low_lt10','medium_10to45','high_ge45']} for c in counts[split]} for split in counts}
    errors=[{'sample':r['sample'],'split':r['split'],'condition':r['condition'],
             'category':r.get('category'),'accepted':r.get('accepted'),'clear':clear(r),
             'note':labels['samples'][r['sample']]['review_note']} for r in rows if not r.get('correct')]
    observations=defaultdict(list)
    for s in m['samples']:
        predictions={c:{'category':indexed[(s['id'],c)].get('category'),'image':s['conditions'][c]['path']}
            for c in m['conditions'] if c!='SHIFTED_DETAIL' and (s['id'],c) in indexed}
        observations[s['event']].append({'sample':s['id'],'frame_id':s['frame_id'],'pts':s['pts'],
            'telemetry_time_seconds':s['geometry']['telemetry_time_seconds'],
            'angular_speed_deg_s':s['angular_speed_deg_s'],'gaze_point_source_pixels':s['point'],
            'clock_supported':s['geometry']['clock_available'],'predictions':predictions})
    timeline={'schema':'time_indexed_region_observations_v1',
        'scope':'Independent frame categories only. No object identity tracking or cross-frame referent linking.',
        'contains_ground_truth':False,
        'events':{event:sorted(items,key=lambda r:r['pts']) for event,items in observations.items()}}
    audit={'call_counts':dict(Counter(r['status'] for r in rows)),
        'all_complete_vision_verified':all(r.get('vision_verified') for r in rows if r['status']=='COMPLETE'),
        'visual_token_counts':sorted({r['processor_audit']['expanded_image_token_count'] for r in rows if r['status']=='COMPLETE'}),
        'annotation_sha256':read(root/'annotation_lock.json')['sha256'],
        'annotation_source':m['annotation_source'],'gt_loaded':m['gt_loaded'],'anchors_projected':m['anchors_projected']}
    out={'audit':audit,'stats':counts,'pairs_vs_rgb':pairs,'shifted_control':shifted,'by_motion':by_motion,'errors':errors}
    write(root/'analysis.json',out);write(root/'temporal_observations.json',timeline)
    lines=['# 脚本组织区域语义：初步对照','',
        '区域内容标签来自模型推理前冻结的助手视觉复核，不是人工像素GT或交互referent GT。',
        '本试验比较同一脚本指定位置的图像呈现，RGB条件同样含位置角标；不等于无gaze基线。','',
        '| split | condition | 内容符合复核标签 | 无歧义子集 |',
        '| --- | --- | ---: | ---: |']
    for split,conditions in counts.items():
        for c,v in conditions.items():lines.append(f"| {split} | {c} | {v['correct']}/{v['n']} | {v['clear_correct']}/{v['clear_n']} |")
    lines+=['','## 与同帧 RGB 配对','', '| split | condition | 改对 | 改错 |', '| --- | --- | ---: | ---: |']
    for split,conditions in pairs.items():
        for c,v in conditions.items():lines.append(f"| {split} | {c} | {v['wins']} | {v['losses']} |")
    lines+=['','错移区域按其自己显示的内容评分，不能把识别对错移内容当模型错误。',
        '三帧按真实PTS保存在 temporal_observations.json；同一类别不表示同一对象。',
        '尚未验证跨帧身份、mention关联、头动缓解或最终集合收益。','',
        f"审计：{audit['call_counts']}；视觉token={audit['visual_token_counts']}；vision消费核验={audit['all_complete_vision_verified']}。"]
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    # Add output text to the gallery without altering any model inputs.
    html=(root/'index.html').read_text()
    marker='<!-- MODEL_RESULTS -->'
    if marker in html:html=html.split(marker)[0]
    payload=json.dumps({'rows':rows,'labels':labels['samples']},ensure_ascii=False).replace('</','<\\/')
    html+=marker+'''<script>const results=__RESULTS__;const baseRender=render;render=function(){baseRender();const s=d.samples[+select.value];document.querySelectorAll('figure').forEach(f=>{const c=f.querySelector('figcaption').textContent;const r=results.rows.find(r=>r.sample===s.id&&r.condition===c);if(!r)return;const a=results.labels[s.id];f.querySelector('figcaption').textContent=c+' | model: '+r.category+' | reviewed: '+(c==='SHIFTED_DETAIL'?a.shifted_accepted:a.accepted).join('/')+' | '+(r.correct?'✓':'×');});};select.onchange=render;render();</script>'''.replace('__RESULTS__',payload)
    (root/'index.html').write_text(html)
    print(json.dumps({'audit':audit,'stats':counts,'pairs':pairs,'shifted':shifted},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('.local-sync/scripted_region_semantics_20261105'))
    report(p.parse_args().output.resolve())
