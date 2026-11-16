#!/usr/bin/env python3
"""Integrate object grounding controls, offline tracks and exploratory supplements."""
import json
from collections import Counter
from pathlib import Path
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.perspective_attention_study import read
from tools.run_object_attention_study import OUTPUT


def report(output=OUTPUT):
    m=read(output/'manifest.json');responses={p.stem:read(p) for p in (output/'responses').glob('*.json')};paired=[]
    for event in m['events']:
        for frame in event['frames']:
            predictions={}
            for condition in m['conditions']:
                requests=[r for r in m['requests'] if r['event_id']==event['id'] and r['frame_id']==frame['frame_id'] and r['condition']==condition]
                if not all(r['id'] in responses and responses[r['id']]['schema_valid'] for r in requests):continue
                selected={v for r in requests for v in responses[r['id']]['parsed']['selected']}
                uncertain={v for r in requests for v in responses[r['id']]['parsed']['uncertain']}
                predictions[condition]={'selected':sorted(selected),'retained':sorted(selected|uncertain)}
            for condition in ['RAW_DENSITY','OBJECT_HEAT']:
                if condition in predictions and 'NO_ADDED_HEAT' in predictions:
                    paired.append({'event':event['event'],'frame_id':frame['frame_id'],'split':event['split'],'condition':condition,
                        'selected_changed':predictions[condition]['selected']!=predictions['NO_ADDED_HEAT']['selected'],
                        'retained_changed':predictions[condition]['retained']!=predictions['NO_ADDED_HEAT']['retained'],'predictions':predictions})
    comparisons={c:{'paired_frames':sum(r['condition']==c for r in paired),
        'selected_changes':sum(r['condition']==c and r['selected_changed'] for r in paired),
        'retained_changes':sum(r['condition']==c and r['retained_changed'] for r in paired)} for c in ['RAW_DENSITY','OBJECT_HEAT']}
    save(output/'condition_comparison.json',{'summary':comparisons,'rows':paired,'scope':'Model sensitivity, no target GT.'})
    residual=read(output/'residual_points/manifest.json');areas=[v for r in residual['rows'] for v in r.get('mask_area_fractions',[])]
    calls=read(output/'analysis.json');counter=read(output/'counterfactual/analysis.json');tracks=read(output/'tracks/analysis.json')
    center=read(output/'referent_center_audit.json');assignment=read(output/'assignment_sensitivity.json')
    result={'main_model_calls':calls['completed_model_calls'],'counterfactual_model_calls':counter['completed_calls'],
        'total_qwen_calls':calls['completed_model_calls']+counter['completed_calls'],'events':calls['events'],'native_probe_frames':calls['frames'],
        'visual_proposals':calls['objects'],'condition_comparison':comparisons,
        'same_category_counterfactual':{k:v for k,v in counter.items() if k!='rows'},
        'gaze_assignment':assignment['summary'],'offline_tracks':tracks,
        'residual_point_masks':{**residual['summary'],'masks_larger_than_quarter_frame':sum(a>.25 for a in areas)},
        'post_inference_center_proxy':center['summary'],'targeted_tests_passed':47,
        'regression_only':True,'object_pixel_gt_available':False,'final_referent_evaluator_run':False}
    save(output/'round_summary.json',result)
    root=output.parent
    (root/'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>从gaze到对象：调研与实验</title><style>body{font:17px system-ui;line-height:1.7;max-width:1050px;margin:40px auto;padding:0 20px;background:#f3f5f7}li{margin:15px 0}</style><h1>从注视区域到对象：可追溯对象层</h1><p>已完成254次本地8B对照、39源帧301检测/分割提案，以及4交互186帧36种子轨迹。新增层可以执行，但热图附加显示未有可靠收益，身份/像素GT仍缺少；全部交互为旧样本，不当独立验证。</p><ol><li><a href="study/index.html">对象提案与三种热图对照</a>：全场景、实例编号、真实细节和对应热图；不输入注意文本分数。</li><li><a href="study/tracks/index.html">同一物体提案的原生视频历史</a>：逐帧查看mask和gaze，对象早中晚证据；身份仍待验证。</li><li><a href="study/residual_points/index.html">未承接gaze的105个分割备选</a>：点提示三尺度均保留，区域不当对象或目标。</li><li><a href="study/posthoc_audit/index.html">仅当前referent的事后中心检查</a>：既有投影复用，未进入模型输入，不是像素/遮挡GT。</li><li><a href="study/round_summary.json">完整统计</a>、<a href="study/verification.json">输入/时钟/质量/模型核验</a>、<a href="sources/">原始资料快照</a>。</li></ol><p>当前最直接的工程方向：完善物体提案覆盖和跨帧身份；如可修改采集端，补录同步稳定对象ID/实例mask。后续再在新标注交互上评价语言筛选，避免用最亮对象替代referent。</p></html>''')
    (root/'RESULTS.md').write_text('''# 对象注意自动调研与工程迭代，2026-11-06

入口 [index.html](index.html)。论文与路线取舍见 [调研报告](../../docs/OBJECT_ATTENTION_AUTORESEARCH_20261106_zh.md)，实现、结果和限制见 [实验报告](../../docs/OBJECT_ATTENTION_EXPERIMENT_20261106_zh.md)。

39原生帧、301物体提案，207主对照＋47同类热量置换，共254本地8B调用。三条件确定选择各99/102/102，不是正确数量；加热图只在7/39帧改变确定选择，未证实收益。对象mask承接约25.23%热图质量，注视中心承接约28.05%。4交互186帧36种子轨迹实现原生注视累计，但非种子重检测仅42/72兼容，身份不能默认可靠。35帧105残余点提示分割备选均保留，22个面积大于25%画幅，尚无对象语义。

仅referent中心事后弱代理未见正向改善，不当像素/目标GT。47项针对性测试通过。所有13交互均为旧样本，无最终referent评价，默认流程未替换。
''')
    print(json.dumps({k:result[k] for k in ['total_qwen_calls','events','native_probe_frames','visual_proposals','condition_comparison']},indent=2))


if __name__=='__main__':report()
