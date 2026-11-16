#!/usr/bin/env python3
"""Summarize perspective/time/shortlist work without conflating task metrics."""
import argparse
from collections import Counter
from pathlib import Path
import json
from PIL import Image,ImageDraw
from MultiAgent.perspective_attention_study import OUTPUT,read
from MultiAgent.heatmap_temporal_pilot import save


def report(output):
    initial=read(output/'analysis.json');grouped=read(output/'phase_layout_v2/analysis.json');multi=read(output/'multi_region/analysis.json')
    sources={'joint':output,'phase_interleaved':output,'phase_grouped':output/'phase_layout_v2','shortlist':output/'multi_region'}
    manifest=read(output/'manifest.json');transitions=[e['id'] for e in manifest['events'] if e['peak_vs_mass']['TRUE_HEAT']['expected_by_time'][0]!=e['peak_vs_mass']['TRUE_HEAT']['expected_by_time'][2]]
    comparison={}
    for method,root in sources.items():
        rows=[read(p) for p in (root/'responses').glob('*.json')]
        if method=='joint':rows=[r for r in rows if r['mode']=='joint']
        if method=='phase_interleaved':rows=[r for r in rows if r['mode']=='phase']
        c={}
        for condition in ['TRUE_HEAT','REVERSE_TIME','SHIFT_SPACE','NO_HEAT']:
            rr=[r for r in rows if r['condition']==condition]
            if method=='shortlist':
                proposals=sum(len(r['parsed']['regions']) for r in rr);unsupported=sum(r['unsupported_proposal_count'] or 0 for r in rr)
                c[condition]={'fields':len(rr),'top1_retained':sum(r['top1_retained'] for r in rr),'proposal_count':proposals,
                    'unsupported_proposal_count':unsupported,'exact_expected_top2_sets':sum(r['exact_expected_top2_set'] for r in rr),
                    'expected_two_direction_intervals':sum(len(r['expected_top2'])==2 for r in rr),
                    'both_expected_directions_retained':sum(len(r['expected_top2'])==2 and set(r['expected_top2']).issubset(r['parsed']['regions']) for r in rr)}
            else:
                correct=sum(sum(r['field_correct'].values()) for r in rr);fields=sum(len(r['expected']) for r in rr)
                events={r['event_id'] for r in rr};pairs=[];transition_pairs=[]
                for event in events:
                    selected=[r for r in rr if r['event_id']==event]
                    ok=all(r['correct'] for r in selected);pairs.append(ok)
                    if event in transitions:transition_pairs.append(ok)
                c[condition]={'fields':fields,'fields_correct':correct,'event_pairs':len(pairs),'both_times_correct':sum(pairs),
                    'transition_event_pairs':len(transition_pairs),'transition_both_correct':sum(transition_pairs)}
        comparison[method]=c
    routes=read(output/'routed_visual_evidence.json')
    audit={'total_calls':initial['calls']+grouped['calls']+multi['model_calls'],
        'calls_by_stage':{'initial_joint_and_failed_phase':initial['calls'],'grouped_phase':grouped['calls'],'shortlist_regression':multi['model_calls']},
        'tokens':sorted(set(initial['tokens']+grouped['tokens']+multi['tokens'])),
        'all_vision_verified':initial['all_vision_verified'] and grouped['all_vision_verified'] and multi['all_vision_verified'],
        'routed_evidence_status':routes['status_counts'],'new_semantic_model_calls':0,'final_referent_evaluator_run':False}
    result={'comparison':comparison,'transition_event_ids':transitions,'audit':audit,
        'geometry_fix':'cube preview horizontal coordinate changed from [0,2] to [-1,1]; existing spherical/old model inputs not overwritten',
        'check_is_independent_unseen_data':False,'shortlist_is_regression':True}
    save(output/'comparison.json',result)
    lines=['# 透视方向图、脚本时间绑定与多方向候选','',
        '热图始终是图像输入；下面评价图中视觉热点方向，不是对象/referent准确率。13条旧交互按6开发/7检查预分；均在历史实验出现过。','',
        '| 方法 | 真热图正确时间段/保留最强方向 | 早晚都对 | 无热图正确处理 |',
        '| --- | ---: | ---: | ---: |']
    for method,title in [('joint','联合早晚'),('phase_interleaved','交错逐段（仅开发，已弃用）'),('phase_grouped','分开RGB/热图逐段')]:
        c=comparison[method];t=c['TRUE_HEAT'];n=c['NO_HEAT']
        lines.append(f"| {title} | {t['fields_correct']}/{t['fields']} | {t['both_times_correct']}/{t['event_pairs']} | {n['fields_correct']}/{n['fields']} |")
    c=comparison['shortlist'];t=c['TRUE_HEAT'];n=c['NO_HEAT']
    lines.append(f"| 最多两个方向（同批回归） | {t['top1_retained']}/{t['fields']} | 不与单选比较 | {n['top1_retained']}/{n['fields']} |")
    lines+=['',f"多方向真热图共输出{t['proposal_count']}个方向，{t['unsupported_proposal_count']}个不满足约半峰的工程复核定义；top2覆盖不能当top1准确率提升。",
        '真实热图中4个按半峰工程规则应保留两个方向的时间段，模型没有一次同时保留两个；候选列表能力尚不可靠。',
        '逐段方案每事件两次调用，且每段热图面积更大，不是等计算预算消融。早晚真正跨方向的只有4个交互，需单独看comparison.json。',
        '',f"全部模型审计：{audit}",
        '原生RGB已经按模型选择与原始时间导出，未调用后续语义/指代模型；不得用原生帧可用率代替筛选正确率。',
        '修复了六面预览水平网格偏移；旧模型输入为角度展开图，不受该预览修复影响。原始错误预览保留，修正版另存corrected_previews。']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    root_page=output/'index.html';html=root_page.read_text()
    links='<p><a href="phase_layout_v2/index.html">逐段版：全部RGB在上、热图在下</a> · <a href="multi_region/index.html">最多两个关注方向</a> · <a href="routed_visual_evidence.html">模型筛选后的原生RGB证据</a> · <a href="RESULTS.md">实验结果</a></p>'
    html=html.replace('<main></main>',links+'<main></main>');root_page.write_text(html)
    for name in ['phase_layout_v2/index.html','multi_region/index.html']:
        p=output/name;html=p.read_text()
        if '../routed_visual_evidence.html' not in html:html=html.replace('<main></main>','<p><a href="../routed_visual_evidence.html">查看筛选后的原生RGB</a> · <a href="../index.html">主对照</a></p><main></main>')
        p.write_text(html)
    # One concrete example links the graphical decision to historical image content.
    e=next(e for e in manifest['events'] if e['event']=='scene2/411')
    model_card=output/'phase_layout_v2'/next(r for r in read(output/'phase_layout_v2/manifest.json')['requests'] if r['event_id']==e['id'] and r['condition']=='TRUE_HEAT' and r['phase']==2)['path']
    row=next(r for r in routes['rows'] if r['event_id']==e['id'] and r['method']=='phase_grouped' and r['phase']==2)
    canvas=Image.new('RGB',(1008,1280),'white')
    with Image.open(model_card) as im:canvas.paste(im,(0,0))
    draw=ImageDraw.Draw(canvas);draw.text((12,1020),'MODEL-SELECTED DIRECTION -> EXACT HISTORICAL RGB (NOT A REFERENT LABEL)',fill='black')
    if row.get('evidence_card'):
        with Image.open(output/row['evidence_card']['path']) as im:canvas.paste(im.resize((448,240)),(280,1040))
    canvas.save(output/'routing_overview.jpg',quality=96)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    report(p.parse_args().output.resolve())
