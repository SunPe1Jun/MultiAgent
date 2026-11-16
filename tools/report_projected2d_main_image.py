#!/usr/bin/env python3
"""Report direct full-frame 2D localization and render a local prediction review."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

from MultiAgent.projected2d_attention_study import read_csv, save_json, sha_file
from MultiAgent.projected2d_main_image import CONDITIONS, ROOT, prompt_for, verify_lock
from tools.report_projected2d_routed_rgb import paired_delta


def report(output):
    verify_lock(output)
    manifest = json.loads((output / 'input_manifest.json').read_text())
    summary = json.loads((output / 'summary.json').read_text())
    frozen = json.loads((output / 'predictions_frozen.json').read_text())
    for path, digest in frozen['responses'].items():
        if sha_file(Path(path)) != digest:
            raise ValueError('Frozen response changed')
    details = {c: {(r['scene'], int(r['row_index'])): r for r in read_csv(
                  output / 'evaluation' / c / '2d_eval_detail.csv')} for c in CONDITIONS}
    keys = sorted(details[CONDITIONS[0]])
    comparisons = {}
    for kind in ('point', 'joint'):
        values = {c: [tuple(int(details[c][k][f'{kind}_{part}_100']) for part in ('tp', 'fp', 'fn'))
                      for k in keys] for c in CONDITIONS}
        comparisons[kind + '_100_768_minus_512'] = paired_delta(values[CONDITIONS[1]], values[CONDITIONS[0]])
    calls_path = output / 'model_cache/projected2d_main_rgb_v1/call_audit.jsonl'
    calls = [json.loads(line) for line in calls_path.read_text().splitlines() if line]
    audit = {'events': len(keys), 'conditions': list(CONDITIONS),
        'physical_calls': sum(not c.get('cache_hit') for c in calls),
        'inference_seconds': sum(c.get('inference_seconds', 0) for c in calls if not c.get('cache_hit')),
        'visual_tokens': sum((c.get('processor_audit') or {}).get('expanded_image_token_count', 0) or 0
                             for c in calls if not c.get('cache_hit')),
        'paired_resolution_differences': comparisons, 'responses': {},
        'scope': 'Changed rendering and direct-coordinate protocol with the same frozen mention task; not an original expected_count baseline reproduction or gaze efficacy experiment.'}
    rows = ['# 单主图 RGB：保持 referent 与软时间任务', '',
        '每个时刻是一张完整 RGB，原 P1/P2/P3 按时间独立提供；模型直接在选中主图上输出目标位置。帧号和时间由相邻文字标识，不写入 RGB 像素。两组提示、mention、时间提示、输出、模型和评价一致，仅主图分辨率不同。', '',
        '| 输入 | Temporal F1 | Point@100 | Joint@100 | Joint Exact@100 |',
        '|---|---:|---:|---:|---:|']
    previous = ROOT / '.local-sync/projected2d_routed_composite_v2_20261109'
    old_summary = json.loads((previous / 'summary.json').read_text())['metrics']['FULL_RGB']
    old_exact = json.loads((previous / 'exact_set_analysis.json').read_text())['conditions']['FULL_RGB']['raw_exact']['joint_exact_100']
    rows.append(f"| 旧四宫格 FULL_RGB（历史开发比较） | {old_summary['time_f1']:.4f} | {old_summary['point_100_f1']:.4f} | {old_summary['joint_100_f1']:.4f} | {old_exact['exact']}/{old_exact['evaluable_events']} |")
    for c in CONDITIONS:
        m = summary['metrics'][c]
        exact = summary['exact'][c]['joint_exact_100']
        rows.append(f"| {c} | {m['time_f1']:.4f} | {m['point_100_f1']:.4f} | {m['joint_100_f1']:.4f} | {exact['exact']}/{exact['evaluable_events']} |")
        responses = [json.loads((output / 'responses' / f"{e['id']}_{c}.json").read_text())
                     for e in manifest['events']]
        audit['responses'][c] = {'statuses': dict(Counter(r['status'] for r in responses)),
            'vision_verified': sum(bool(r.get('vision_verified')) for r in responses),
            'image_tokens': dict(Counter(str((r.get('processor_audit') or {}).get('expanded_image_token_count'))
                                       for r in responses)),
            'panel_prediction_counts': summary['panel_prediction_counts'][c]}
    rows += ['', '相对旧四宫格的变化还包括直接主图坐标、帧号相邻文字和较少冗余视觉token，不能把差异全部归因于单一布局因素。512与768之间才是相同主图协议的分辨率配对。', '',
        '原始发布8B在同32条上的Point@100/Joint@100为0.2121/0.1970，仅作历史参考：原expected_count与当前逐mention任务不同。本轮保留用户认可的referent修改，不恢复GT数量。', '',
        '这是已评分开发样本，不是独立验证。本轮没有新增热图或脚本gaze区域输入，不能据此判断gaze有效或无效。GT只在预测冻结后进入原评价器与查看器。', '',
        f"真实调用 {audit['physical_calls']} 次，累计视觉token {audit['visual_tokens']}，模型推理 {audit['inference_seconds']:.1f} 秒。", '',
        '逐图查看：index.html。完整计数：summary.json。输入与调用核验：completion_audit.json。']
    (output / 'RESULTS.md').write_text('\n'.join(rows) + '\n')
    save_json(output / 'report.json', audit)
    return manifest, summary, audit


def viewer(output, manifest, summary):
    annotations = defaultdict(lambda: defaultdict(list))
    for row in read_csv(output / 'rebuilt_manifest_all.csv'):
        if row['projection_valid'] != 'True':
            continue
        annotations[(row['scene'], int(row['row_index']))][row['panel_id']].append(
            {'name': row['referent_name'], 'uv': [float(row['gt_u_norm']), float(row['gt_v_norm'])],
             'acceptable': row['evidence_acceptable'] == 'True'})
    events = []
    refs = []
    for event in manifest['events']:
        panels = []
        for panel in event['panels']:
            inputs = {c: os.path.relpath(panel['inputs'][c]['path'], output) for c in CONDITIONS}
            refs.extend(inputs.values())
            panels.append({'id': panel['frame_id'], 'time': panel['requested_video_time'], 'inputs': inputs,
                'gt': annotations[(event['scene_key'], event['row_index'])][panel['frame_id']]})
        responses = {}
        for c in CONDITIONS:
            r = json.loads((output / 'responses' / f"{event['id']}_{c}.json").read_text())
            responses[c] = {'status': r['status'], 'mentions': r.get('parsed', {}).get('mentions', []),
                            'warnings': r.get('parse_warning')}
        events.append({'id': event['id'], 'instruction': event['instruction'],
                       'panels': panels, 'responses': responses})
    payload = json.dumps({'events': events, 'conditions': CONDITIONS}, ensure_ascii=False).replace('<', '\\u003c')
    markup = '''<!doctype html><meta charset="utf-8"><title>完整RGB主图定位复核</title>
<style>body{font:16px system-ui;margin:24px;background:#18232c;color:#edf3f7}a{color:#9cdbff}select{font:inherit;margin:8px;padding:5px}article{margin:18px 0;padding:14px;background:#263744;border-radius:8px}canvas{display:block;max-width:100%;height:auto}pre{white-space:pre-wrap}.note{max-width:900px;line-height:1.6}h3{margin-top:0}</style>
<h1>完整 RGB 主图：目标点复核</h1>
<p class="note">每张输入只有一个完整画面，原时序面板独立提供。红点为冻结的模型预测；可选紫色标记为评价时使用的投影代理，并非人工像素真值。GT没有进入模型输入。本轮保留逐mention任务与软时间提示，未添加热图。</p>
<p><a href="RESULTS.md">结果</a> · <a href="summary.json">完整指标</a> · <a href="completion_audit.json">核验</a></p>
<label>交互<select id="event"></select></label><label>主图<select id="condition"></select></label>
<label><input id="gt" type="checkbox">显示事后GT投影</label><h2 id="title"></h2><p id="instruction"></p><div id="panels"></div><pre id="detail"></pre>
<script>const data=PAYLOAD,es=document.querySelector('#event'),cs=document.querySelector('#condition');
for(const [i,e] of data.events.entries())es.add(new Option(e.id,i));for(const c of data.conditions)cs.add(new Option(c,c));cs.value='MAIN_RGB_768';
function draw(){const e=data.events[Number(es.value)],c=cs.value,r=e.responses[c];document.querySelector('#title').textContent=e.id;document.querySelector('#instruction').textContent=e.instruction;document.querySelector('#detail').textContent=JSON.stringify(r,null,2);const root=document.querySelector('#panels');root.replaceChildren();
for(const p of e.panels){const article=document.createElement('article'),h=document.createElement('h3');h.textContent=p.id+' | '+p.time.toFixed(3)+' s';article.append(h);const canvas=document.createElement('canvas');article.append(canvas);root.append(article);const image=new Image();image.onload=()=>{canvas.width=image.width;canvas.height=image.height;const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0);ctx.font='16px sans-serif';ctx.lineWidth=3;
for(const m of r.mentions)for(const q of m.predictions){if(q.frame_id!==p.id)continue;const x=q.point_uv[0]*canvas.width,y=q.point_uv[1]*canvas.height;ctx.strokeStyle='#ff514b';ctx.fillStyle='#ff514b';ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);ctx.stroke();ctx.fillText(m.mention_id,x+9,y-8)}
if(document.querySelector('#gt').checked)for(const g of p.gt){const x=g.uv[0]*canvas.width,y=g.uv[1]*canvas.height;ctx.strokeStyle='#ff63fa';ctx.fillStyle='#ff63fa';ctx.beginPath();ctx.moveTo(x-8,y);ctx.lineTo(x+8,y);ctx.moveTo(x,y-8);ctx.lineTo(x,y+8);ctx.stroke();ctx.fillText(g.name+(g.acceptable?' [time]':''),x+9,y+17)}};image.src=p.inputs[c]}}
es.onchange=draw;cs.onchange=draw;document.querySelector('#gt').onchange=draw;draw();</script>'''.replace('PAYLOAD', payload)
    (output / 'index.html').write_text(markup)
    if not all((output / ref).is_file() for ref in refs):
        raise ValueError('Viewer contains missing RGB images')
    return {'events': len(events), 'image_references': len(refs), 'all_images_exist': True,
            'javascript_runtime_tested': False}


if __name__ == '__main__':
    out = Path(sys.argv[1]).resolve()
    m, s, a = report(out)
    print(json.dumps(viewer(out, m, s), ensure_ascii=False))
