#!/usr/bin/env python3
"""Compare graphical cue readiness, weak real proxy and duration-gated display."""
import json
from pathlib import Path
from collections import Counter
from MultiAgent.attention_reliability import cluster_paired_interval,heat_display_policy
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.perspective_attention_study import read
from tools.run_attention_reliability_study import OUTPUT,LAYOUTS,CONTROL_CONDITIONS,REAL_CONDITIONS


def summarize(m,rows,layout,support):
    indexed={(r['case_id'],r['condition'],r['permutation']):r for r in rows}
    cases={c['case_id'] for c in m['requests'] if c['family']=='injected_cue'}
    pairs=[];null=[];consistent=[]
    for case in cases:
        for p in (0,1):
            a=indexed.get((case,'FORWARD',p));b=indexed.get((case,'REVERSED',p));n=indexed.get((case,'NO_HEAT',p))
            if a and b:pairs.append(a['matches_label'] and b['matches_label'])
            if n:null.append(n['matches_label'])
        for c in CONTROL_CONDITIONS:
            a=indexed.get((case,c,0));b=indexed.get((case,c,1))
            if a and b and a['schema_valid'] and b['schema_valid']:consistent.append(a['physical_prediction']==b['physical_prediction'])
    mechanism={'forward_reverse_pairs':len(pairs),'both_directions_correct':sum(pairs),
        'null_calls':len(null),'correct_null':sum(null),'id_permutation_pairs':len(consistent),'same_physical_output':sum(consistent),
        'engineering_readiness_met':bool(pairs and null and sum(pairs)/len(pairs)>=.9 and sum(null)/len(null)>=.9)}
    real_ids={c['case_id'] for c in m['requests'] if c['family']=='real_center_proxy'};real=[];counts={}
    for condition in REAL_CONDITIONS:
        rr=[r for r in rows if r['family']=='real_center_proxy' and r['condition']==condition]
        counts[condition]={'n':len(rr),'weak_center_label_hits':sum(r['matches_label'] for r in rr),
            'unclear':sum(r['parsed'].get('selected')=='unclear' for r in rr)}
    for case in sorted(real_ids):
        scores={};both={}
        for condition in REAL_CONDITIONS:
            r=[indexed.get((case,condition,p)) for p in (0,1)]
            if all(r):scores[condition]=sum(x['matches_label'] for x in r)/2;both[condition]=all(x['matches_label'] for x in r)
        if len(scores)!=3:continue
        row={'case_id':case,'cluster':support[case]['event'],'support_bin':support[case]['support_bin'],
            'pair_supported_seconds':support[case]['pair_assigned_seconds'],**scores,
            'both_permutations_match_weak_label':both}
        chosen=heat_display_policy(row['pair_supported_seconds']);row['gated_display_condition']=chosen;row['SUPPORT_GATED']=row[chosen];real.append(row)
    contrasts={}
    for condition in ['TRUE_HEAT','SUPPORT_GATED']:
        contrasts[condition]={}
        for baseline in ['NO_HEAT','SWAP_PAIR']:
            data=[{'cluster':r['cluster'],'baseline':r[baseline],'treatment':r[condition]} for r in real]
            contrasts[condition][baseline]={**cluster_paired_interval(data),
                'helpful_cases':sum(r[condition]>r[baseline] for r in real),'harmful_cases':sum(r[condition]<r[baseline] for r in real),
                'unchanged_cases':sum(r[condition]==r[baseline] for r in real)}
    strata={}
    for group in ['below_10ms','10_to_250ms','at_least_250ms']:
        rr=[r for r in real if r['support_bin']==group]
        strata[group]={'cases':len(rr),'interactions':len({r['cluster'] for r in rr}),
            'scores_sum_over_cases':{c:sum(r[c] for r in rr) for c in REAL_CONDITIONS+['SUPPORT_GATED']}}
    real_invariance={}
    for condition in REAL_CONDITIONS:
        pairs=[(indexed.get((case,condition,0)),indexed.get((case,condition,1))) for case in sorted(real_ids)]
        pairs=[(a,b) for a,b in pairs if a and b and a['schema_valid'] and b['schema_valid']]
        real_invariance[condition]={'pairs':len(pairs),'same_physical_output':sum(a['physical_prediction']==b['physical_prediction'] for a,b in pairs)}
    interactions=[]
    for cluster in sorted({r['cluster'] for r in real}):
        within=[r for r in real if r['cluster']==cluster];outside=[r for r in real if r['cluster']!=cluster]
        interactions.append({'interaction':cluster,'cases':len(within),
            'true_minus_none_case_score_sum':sum(r['TRUE_HEAT']-r['NO_HEAT'] for r in within),
            'gain_after_leaving_out_interaction':sum(r['TRUE_HEAT']-r['NO_HEAT'] for r in outside)/len(outside) if outside else None})
    return {'mechanism':mechanism,'real_weak_proxy':{'by_condition':counts,'paired_cases':real,'support_strata':strata,'contrasts':contrasts,
        'id_permutation_invariance':real_invariance,'interaction_concentration':interactions,
        'gated_policy_scope':'Post-initial-results policy analysis: use existing TRUE_HEAT if pair support>=250ms, otherwise existing NO_HEAT prediction. Gate uses no object label, but policy is not independently validated.',
        'empirical_interval_limit':'Few old interactions, annotation-conditioned weak labels. A degenerate [0,0] interval can reflect cancellation within sampled interactions; it is not proof of population equivalence.'}}


def report(output=OUTPUT):
    m=read(output/'manifest.json');rows=[read(p) for p in (output/'responses').glob('*.json')];support={r['case_id']:r for r in read(output/'support_audit.json')['rows']}
    summaries={l:summarize(m,[r for r in rows if r['layout']==l],l,support) for l in LAYOUTS}
    local=output/'local_layout_v2';local_rows=[read(p) for p in (local/'responses').glob('*.json')];lm=read(local/'manifest.json')
    summaries['LOCAL_V2']=summarize(lm,local_rows,'LOCAL_V2',support)
    total=len(rows)+len(local_rows)
    result={'completed_model_calls':total,'base_calls':len(rows),'adaptive_local_calls':len(local_rows),
        'layouts':summaries,'support':read(output/'support_audit.json')['summary'],
        'all_schema_valid':all(r['schema_valid'] for r in rows+local_rows),'all_vision_verified':all(r['vision_verified'] for r in rows+local_rows),
        'visual_tokens':sorted({r['processor_audit']['expanded_image_token_count'] for r in rows+local_rows}),
        'independent_object_gt_available':False,'unseen_validation':False,'promote_to_default':False,
        'candidate_semantic_audit':{'reviewer':'assistant visual inspection, not human GT',
            'case_id':'real_scene5_28_V007_tank',
            'concern':'Target proposal shows yellow oil tanks; alternative shows a military tracked vehicle. Detector/canonical category tank merges two meanings. This is not a verified same-category challenge.',
            'frozen_inputs_and_labels_unchanged':True},
        'scope':'Mechanism success alone cannot establish true referent improvement. Layout2 and support policy are adaptive regression, not independent validation.'}
    # Sensitivity only: preserve the complete frozen result; no new ground truth
    # or retrospective removal from the primary study.
    suspect='real_scene5_28_V007_tank'
    for summary in summaries.values():
        subset=[r for r in summary['real_weak_proxy']['paired_cases'] if r['case_id']!=suspect]
        data=[{'cluster':r['cluster'],'baseline':r['NO_HEAT'],'treatment':r['TRUE_HEAT']} for r in subset]
        summary['real_weak_proxy']['sensitivity_without_semantically_suspect_pair']={
            **cluster_paired_interval(data),'excluded_case':suspect,'posthoc_sensitivity_only':True,
            'interactions_with_positive_net_gain':sorted(c for c in {r['cluster'] for r in subset} if sum(r['TRUE_HEAT']-r['NO_HEAT'] for r in subset if r['cluster']==c)>0)}
    save(output/'comparison.json',result)
    summaries_by_case=[]
    for case in [c for c in m['cases'] if c['family']=='real_center_proxy']:
        summaries_by_case.append({'case_id':case['id'],'event':case['event'],'frame_id':case['frame_id'],'category':case['category'],
            'weak_reference_name':case['reference_name'],'source_path':case['source_path'],'source_sha256':case['source_sha256'],
            'target_proposal':case['objects'][0]['id'],'alternative_proposal':case['objects'][1]['id'],
            'human_annotation':{'target_box_correct':None,'alternative_box_correct':None,'weak_reference_matches_visible_target':None,
                'intended_visible_target':None,'exact_mention_time':None,'annotator':None,'notes':''}})
    if not (output/'object_review_template.json').exists():
        save(output/'object_review_template.json',{'status':'UNANNOTATED','rows':summaries_by_case,
            'warning':'Weak referent-center matching remains a proposal. Never fill human fields from gaze maxima or model responses.'})
    parts=[]
    for layout,s in summaries.items():
        mech=s['mechanism'];r=s['real_weak_proxy'];a=r['by_condition']
        parts.append(f"<tr><td>{layout}</td><td>{mech['both_directions_correct']}/{mech['forward_reverse_pairs']}</td><td>{mech['correct_null']}/{mech['null_calls']}</td><td>{mech['same_physical_output']}/{mech['id_permutation_pairs']}</td><td>{a['NO_HEAT']['weak_center_label_hits']}/{a['NO_HEAT']['n']}</td><td>{a['TRUE_HEAT']['weak_center_label_hits']}/{a['TRUE_HEAT']['n']}</td><td>{a['SWAP_PAIR']['weak_center_label_hits']}/{a['SWAP_PAIR']['n']}</td></tr>")
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>热图收益可靠性报告</title><style>body{font:16px system-ui;line-height:1.7;max-width:1200px;margin:35px auto;padding:0 20px;background:#f3f5f7}table{border-collapse:collapse;background:white}th,td{padding:10px;border:1px solid #ccd5dd}li{margin:10px}</style><h1>把“能读到线索”与“指代更准确”分开检验</h1><p>全部为已使用交互；注入线索只验证图形机制。真实14物体对来自6交互，候选有标注条件选择、目标标签仅为referent中心落mask的弱代理。两种编号不是两条独立数据，按交互汇总区间。下表不是最终指代准确率。</p><table><thead><tr><th>呈现</th><th>正反方向都读对</th><th>无图拒答</th><th>换编号后实际选择一致</th><th>无图：弱标签符合</th><th>真实热图：弱标签符合</th><th>错换热图：弱标签符合</th></tr></thead><tbody>__ROWS__</tbody></table><p>第三行局部版是看过前版失败后的追加回归，改变了呈现、面积与提示，不是独立检验或纯分辨率消融。支持门控只在两个候选承接≥250ms时显示热图，减少弱支持被放大的误导；它不认证注视可靠，也不根据GT选对象。其缓存策略比较在comparison.json中，不能当新验证。</p><ul><li><a href="index.html">首轮独立/整帧RGB注册热图全部输入</a></li><li><a href="local_layout_v2/index.html">局部对象热图回归</a></li><li><a href="comparison.json">交互聚类区间、帮助/伤害和支持分层</a></li><li><a href="real_pair_review.png">真实物体对总览</a>、<a href="object_review_template.json">未标注复核表</a></li><li><a href="verification.json">首轮来源/输入核验</a>、<a href="local_layout_v2/verification.json">局部版核验</a></li></ul><p>scene5的tank配对经助手视觉复核存在油罐/军用车辆混义，完整结果保留，comparison.json另列敏感性分析。局部版3个改善案例集中在2交互；去掉此存疑配对后，改善均来自同一房间交互。可靠收益还需要：新交互、人工像素对象/可见性标签、同类干扰物、mention时间对应，固定同一RGB/对象/预算后比较真实/无图/错图；记录帮助与伤害。图形机制达标并不代替上述评价。</p></html>'''
    (output/'summary.html').write_text(page.replace('__ROWS__',''.join(parts)))
    print(json.dumps({'model_calls':total,'layouts':{l:{'mechanism':s['mechanism'],'real_conditions':s['real_weak_proxy']['by_condition'],
        'true_vs_none':s['real_weak_proxy']['contrasts']['TRUE_HEAT']['NO_HEAT'],
        'gated_vs_none':s['real_weak_proxy']['contrasts']['SUPPORT_GATED']['NO_HEAT']} for l,s in summaries.items()}},indent=2))


if __name__=='__main__':report()
