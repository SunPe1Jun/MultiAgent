"""Summarize completed real-gaze and explicitly separate synthetic cue controls."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from MultiAgent.heatmap_temporal_pilot import CONDITIONS, LABELS, save


def main():
    root = Path(__file__).resolve().parents[1]
    output = root/'.local-sync/heatmap_temporal_pilot_20261105'
    manifest = json.loads((output/'stimulus_manifest.json').read_text())
    summary = json.loads((output/'summary.json').read_text())
    planned = len(manifest['events'])*len(CONDITIONS)
    complete = summary['model_calls_completed']
    responses = [json.loads(p.read_text()) for p in (output/'responses').glob('*/*/*.json')]
    grids = {json.dumps(r.get('processor_audit',{}).get('image_grid_thw')) for r in responses}
    tokens = sorted({r.get('processor_audit',{}).get('expanded_image_token_count') for r in responses})
    vision = sum(bool(r.get('visual_tensor_trace',{}).get('input_consumption_verified')) for r in responses)
    schema = sum(bool(r.get('schema_valid')) for r in responses)
    snapshot=json.loads((output/'source_snapshot.json').read_text())
    changed=[p for p,digest in snapshot['source_sha256'].items()
             if hashlib.sha256((root/p).read_bytes()).hexdigest()!=digest]
    checks = {'complete_requests':complete,'planned_requests':planned,'schema_valid_requests':schema,
              'vision_consumption_verified_requests':vision,'visual_token_counts':tokens,
              'all_requests_same_image_grid':len(grids)==1,'source_files_changed':changed,
              'remote_api_calls':0,'final_referent_evaluation_run':False}
    save(output/'run_validation.json',checks)
    rows=['# 热图时序与逐帧 foveation：初步读取试验结果','',
          f'真实 gaze 条件完成 **{complete}/{planned}** 请求。'
          '本轮仅测输入cue可读性，不是gaze目标定位精度，也不是最终指代效果。','',
          '固定5交互×12帧×7条件，共420张条件图。56/60帧有受支持的注意时钟；'
          '4帧在所有条件中逐字节相同地回退RGB。源帧、PTS、参考姿态、指令和图片尺寸固定。', '',
          '## 真实 gaze 结果','',
          '| 条件 | 有cue帧 | 有效中心预测 | 中心命中（≤10%对角线） | 无cue误报 | 可区分先后的帧 | 旧/新同时命中 |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for condition,label in zip(CONDITIONS,LABELS):
        s=summary['conditions'][condition]['all']
        rows.append(f'| {label} | {s["focus_eligible"]} | {s["focus_valid_predictions"]} | '
                    f'{s["focus_success_at_10pct"]}/{s["focus_eligible"]} | '
                    f'{s["no_cue_false_coordinates"]}/{s["no_cue_frames"]} | '
                    f'{s["order_eligible"]} | {s["both_endpoints_success"]}/{s["order_eligible"]} |')
    repeated = []
    for response in responses:
        answers = response.get('parsed', {}).get('frames', [])
        xy = [tuple(r['focus_xy']) for r in answers if isinstance(r,dict)
              and isinstance(r.get('focus_xy'),list)]
        if len(xy) >= 3 and len(set(xy)) == 1:
            repeated.append(f"{response['event']} / {response['condition']}: {len(xy)}帧重复{list(xy[0])}")
    if repeated:
        rows += ['', '发现固定坐标重复输出：']
        rows += ['- '+text for text in repeated]
        rows += ['', '因此粗阈值命中不能直接解释为逐帧追踪成功。']
    rows+=['','分母为0表示该项不适用。缺失/null预测与不完整schema不计成功，保留在有cue分母；'
           '旧/新同时命中只对刺激生成时就已分离的两段主峰评分，不按模型结果选择样本。', '',
           '## 头动分层','',
           '| 条件 | 低运动中心命中 | 中等运动中心命中 | 高运动中心命中 |','|---|---:|---:|---:|']
    for condition,label in zip(CONDITIONS,LABELS):
        d=summary['conditions'][condition]
        cells=[f'{d[k]["focus_success_at_10pct"]}/{d[k]["focus_eligible"]}'
               for k in ['low_lt10','medium_10to45','high_ge45']]
        rows.append('| '+label+' | '+' | '.join(cells)+' |')
    rows+=['','这是模型读取已显示cue的分层，不能当成头动下的真实注视/对象投影误差。', '',
           '## 显示强度与工程核对','',
           f'- 完整12帧schema：{schema}/{len(responses)}请求；真实vision输入追踪通过：{vision}/{len(responses)}。',
           f'- 实际视觉token集合：{tokens}；各条件图像grid一致：{len(grids)==1}。',
           f'- 关键源码/模型配置自快照后变动：{changed}。',
           '- 无远程API调用；没有加载GT、绘制anchor/referent投影或运行最终evaluator。']
    for name,key in [('即时热图','current_stats'),('累计热图','history_stats')]:
        stats=[f[key] for e in manifest['events'] for f in e['frames'] if f[key]['max_unnormalized_heat']>0]
        alphas=[.5*(1-math.exp(-r['max_unnormalized_heat']*2)) for r in stats]
        rows.append(f'- {name}主峰透明度：中位{statistics.median(alphas):.3f}，'
                    f'范围{min(alphas):.3f}–{max(alphas):.3f}，未逐帧峰值归一化。')
    control_path=output/'synthetic_color_controls/summary.json'
    rows+=['','## 独立合成颜色正对照','',
           '这两组在固定scene2/411的RGB上添加任意位置的醒目蓝/橙高斯色块（sigma=20输入像素，峰值alpha=0.95），'
           '正/反两种时间颜色，同一模型、同一12帧提示和视觉预算。它们不是gaze、不是对象标签，'
           '只用于检查颜色/坐标协议是否可读，不并入真实gaze性能。']
    if control_path.is_file():
        control=json.loads(control_path.read_text())
        rows+=['','| 条件 | schema | 焦点命中 | 旧区命中 | 新区命中 |','|---|---|---:|---:|---:|']
        for r in control['results']:
            rows.append(f'| {r["condition"]} | {r["schema_valid"]} | '
                        f'{r["focus_hits"]}/{r["frames"]} | {r["older_hits"]}/{r["frames"]} | {r["newer_hits"]}/{r["frames"]} |')
        single_path=output/'synthetic_color_controls/SINGLE_FRAME_DIAGNOSTIC.json'
        if single_path.is_file():
            single=json.loads(single_path.read_text())
            rows += ['', '附加单帧简化提示诊断（不同提示，不能与七条件公平比较）：',
                     '```json', json.dumps(single['parsed'],ensure_ascii=False,indent=2), '```']
            expected=single['expected_pixels']
            errors={}
            for field,target in [('blue_xy','expected_older_xy'),('orange_xy','expected_newer_xy')]:
                xy=single['parsed'].get(field)
                if isinstance(xy,list) and len(xy)==2:
                    errors[field]=math.hypot(xy[0]*expected['image_size'][0]/1000-expected[target][0],
                                            xy[1]*expected['image_size'][1]/1000-expected[target][1])
            rows += ['', '输入448×480，单帧颜色中心误差（px）：'+json.dumps(errors,ensure_ascii=False)+'。',
                     '这说明醒目合成颜色在单帧简化任务中可读。由于帧数和提示同时改变，'
                     '不能单独归因于多帧编码，也不能把它当成真实gaze已经有效的证据。']
    else:
        rows+=['','正对照尚未完成，不能据此归因于显示强度或颜色协议。']
    rows+=['','## 解读与后续','',
           '真实gaze条件的线索读取结果应先与独立正对照比较。若醒目标记也读不出，需要继续定位多帧观察与提示协议；'
           '若正对照能读而真实图不行，应先研究cue强度、大小、时间颜色混合和呈现方式。'
           '该试验不能证明热图已经缓解头动/时序干扰，也不能在cue尚不可读时将最终失败归因于该研究动机。', '',
           '可见物体描述没有独立人工区域标注，尚未自动评分。最旧/最新段可能多峰或互相重叠，'
           '当前主峰评分仅用于诊断。后续需要人工核对少量gaze-to-object对应，再进行同预算、'
           '带错误cue负对照的最终指代比较。', '',
           '## 文件','',
           '- `index.html`：可拖动/播放的七条件图集；JSON、图片路径及尺寸已校验，未作真实浏览器渲染测试。',
           '- `comparison.jpg`：固定V007对照总览。',
           '- `summary.json`、`per_frame_scores.json`：含全部分母的结果。',
           '- `model_cache/`、`responses/`：原始请求响应、processor输入与实际vision张量。',
           '- `synthetic_color_controls/`：独立合成正对照与图像。',
           '- `stimulus_manifest.json`、`validation.json`、`run_validation.json`：样本来源、哈希与验证。']
    (output/'RESULTS_zh.md').write_text('\n'.join(rows)+'\n')
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
