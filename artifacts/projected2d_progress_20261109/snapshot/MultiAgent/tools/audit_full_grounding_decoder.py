#!/usr/bin/env python3
"""Audit frozen full-field grounding outputs without new inference or GT reads."""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / '.local-sync/full_interaction_referring_20261106'
DEFAULT_OUTPUT = ROOT / '.local-sync/full_grounding_decoder_audit_20261108'


def read(path: Path):
    return json.loads(path.read_text())


def audit_version(run: Path, version: str):
    files = sorted((run / version / 'responses').glob('*.json'))
    reason = Counter()
    errors = Counter()
    per_event = defaultdict(Counter)
    per_condition = defaultdict(Counter)
    total_rows = total_mapped = total_unresolved = total_unaccounted = 0
    schema_valid = validation_valid = 0
    rows = []
    for path in files:
        data = read(path)
        grounding = data.get('grounding_validation', {})
        validation = data.get('validation', {})
        decoder = data.get('decoder', {})
        event_id = data.get('event_id', path.stem)
        condition = data.get('condition', 'UNKNOWN')
        referents = grounding.get('referents', [])
        total_rows += len(referents)
        total_unresolved += len(grounding.get('unresolved', []))
        total_unaccounted += len(grounding.get('unaccounted_mention_hints', []))
        schema_valid += bool(grounding.get('schema_valid'))
        validation_valid += bool(validation.get('schema_valid'))
        for error in grounding.get('errors', []):
            errors[error] += 1
        for rejected in grounding.get('rejected_referents', []):
            for error in rejected.get('reason', []):
                errors[f'rejected:{error}'] += 1
        for rejected in grounding.get('rejected_unresolved', []):
            errors[f"rejected_unresolved:{rejected.get('reason')}"] += 1
        for decision in decoder.get('decisions', []):
            decision_reason = decision.get('reason', 'missing_reason')
            reason[decision_reason] += 1
            per_event[event_id][decision_reason] += 1
            per_condition[condition][decision_reason] += 1
            if decision.get('selected_candidate'):
                total_mapped += 1
            visual = decision.get('visual_choice', {})
            rows.append({
                'version': version,
                'response_id': data.get('id'),
                'event_id': event_id,
                'condition': condition,
                'mention': visual.get('mention'),
                'category': visual.get('category'),
                'view_id': visual.get('view_id'),
                'selected_candidate': decision.get('selected_candidate'),
                'reason': decision_reason,
                'best_distance': decision.get('best_distance'),
                'runner_up_distance': decision.get('runner_up_distance'),
                'compatible_scene_candidates': decision.get('compatible_scene_candidates'),
                'supported_camera_frames': decision.get('supported_camera_frames'),
            })
        per_condition[condition]['responses'] += 1
        per_condition[condition]['accepted_rows'] += len(referents)
        per_condition[condition]['unaccounted_mentions'] += len(grounding.get('unaccounted_mention_hints', []))
    result = {
        'responses': len(files),
        'grounding_schema_valid': schema_valid,
        'decoder_validation_valid': validation_valid,
        'accepted_visual_rows': total_rows,
        'mapped_rows': total_mapped,
        'unresolved_rows': total_unresolved,
        'unaccounted_mention_hints': total_unaccounted,
        'decoder_reason_counts': dict(reason),
        'validation_error_counts': dict(errors),
        'per_condition': {key: dict(value) for key, value in sorted(per_condition.items())},
        'per_event_decoder_reasons': {key: dict(value) for key, value in sorted(per_event.items())},
    }
    analysis_path = run / version / 'analysis.json'
    if analysis_path.exists():
        analysis = read(analysis_path)
        result['posthoc_condition_metrics'] = {
            condition: data.get('set_metrics', {})
            for condition, data in analysis.get('conditions', {}).items()
        }
        result['posthoc_gt_scope'] = {
            'gt_loaded_only_after_predictions_frozen': analysis.get('gt_loaded_only_after_frozen_predictions'),
            'unseen_validation': analysis.get('unseen_validation'),
        }
    return result, rows


def write_report(summary: dict, output: Path):
    v2 = summary['grounding_v2']
    v3 = summary['grounding_v3']
    report = f'''# 完整指代 Decoder 与 mention 覆盖审计（2026-11-08）

本报告只审计 2026 年 11 月 6 日已经冻结的完整交互输出，没有新增模型调用，也没有修改输入、提示、候选、响应或预测。已有 GT 指标只作为预测冻结后的事后上下文。

## 当前发现

下一步瓶颈是 mention accounting 和视觉点到 scene ID 的解码覆盖，优先级高于继续调热图或 foveation 的呈现形式。

- V3 的复合坐标结构检查 32/32 通过，但最终只保留 {v3['accepted_visual_rows']} 条视觉行，累计 {v3['unaccounted_mention_hints']} 个 mention hint 没有被覆盖。
- V3 有 {v3['validation_error_counts'].get('rejected_unresolved:not_literal_phrase', 0)} 次把完整对象行错误放入 `unresolved`，另有 {v3['validation_error_counts'].get('rejected:unknown_category', 0)} 行类别不在场景词表中。
- V2 保留 {v2['accepted_visual_rows']} 条视觉行，其中 {v2['decoder_reason_counts'].get('anchor_center_far_from_selected_rgb_point', 0)} 条因视觉点与投影 anchor 中心距离过远被拒，{v2['decoder_reason_counts'].get('no_supported_camera_or_in_frustum_candidate', 0)} 条因没有可用相机/候选被拒，{v2['decoder_reason_counts'].get('ambiguous_same_category_anchor_centers', 0)} 条因同类 anchor 无法区分被拒。
- V3 的几何拒绝下降到远点 {v3['decoder_reason_counts'].get('anchor_center_far_from_selected_rgb_point', 0)}、无支持 {v3['decoder_reason_counts'].get('no_supported_camera_or_in_frustum_candidate', 0)}、同类歧义 {v3['decoder_reason_counts'].get('ambiguous_same_category_anchor_centers', 0)}，但 mention 覆盖成为主要损失。

## 结构计数

| 版本 | 条件响应 | 结构有效 | 有效视觉行 | 映射到 scene ID | 未覆盖 mention |
|---|---:|---:|---:|---:|---:|
| V2 | {v2['responses']} | {v2['grounding_schema_valid']} | {v2['accepted_visual_rows']} | {v2['mapped_rows']} | 不适用 |
| V3 | {v3['responses']} | {v3['grounding_schema_valid']} | {v3['accepted_visual_rows']} | {v3['mapped_rows']} | {v3['unaccounted_mention_hints']} |

V3 的“结构有效”不能单独当作任务完成率，因为模型可以返回空的 `referents` 和 `unresolved`，同时遗漏全部 parser mention。

## 对主方向的含义

主任务仍然是完整时间顺序交互指代：解析原始指令中的每个 mention，查看完整 RGB 交互和可选注意线索，让模型选择每个具体对象，最后评价 referent 集合。当前结果没有支持新的热图收益结论，只暴露出两个更早的协议损失：

1. 重复或关系 mention 只用 literal 文本表示。两个 “that car” 无法稳定区分具体发生次序。
2. `referents` 和 `unresolved` 使用不同结构；模型有时把完整对象行放进 `unresolved`，验证器只能保留原始错误并拒绝它，不能偷偷修复。
3. schema 有效不代表 mention 覆盖。V3 可以返回 `{{"referents": [], "unresolved": []}}`，但仍遗漏多个 parser hint。
4. V2/V3 的 scene-ID decoder 仍是近似的 anchor-center 代理。远点或同类中心歧义属于解码限制，不能直接当作模型视觉识别错误。

## 下一步实验设计

在再次扫热图/foveation 之前，建议在同一完整画面输入和同一候选空间上测试 V4 mention accounting 协议：

- 为每个 parser mention 分配稳定 `mention_id`，包括重复出现的 mention；
- 强制每个 mention occurrence 恰有一行输出；
- 统一行结构，使用 `status: grounded | unresolved`；
- 只有 `grounded` 才要求 `category`、`view_id` 和 RGB 点；
- `unresolved` 必须给出 literal mention 和明确原因；
- 保留原始非法行，单独报告覆盖率，不用 GT 修复；
- 继续使用同一个 GT-free scene-ID decoder，把 mention 覆盖、视觉定位和 decoder 拒绝分开报告。

这样可以先隔离输出协议和重复 mention 的损失，再进入真正未见交互的 RGB、热图和 foveation 对照，主研究方向不变。

## 产物

- 可复现工具：`tools/audit_full_grounding_decoder.py`
- 汇总：`.local-sync/full_grounding_decoder_audit_20261108/summary.json`
- 逐行决策：`.local-sync/full_grounding_decoder_audit_20261108/row_decisions.json`
- 被审计的冻结运行：`.local-sync/full_interaction_referring_20261106/grounding_v2/` 与 `grounding_v3/`
'''
    (ROOT / 'docs/FULL_GROUNDING_DECODER_AUDIT_20261108_zh.md').write_text(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    rows = []
    for version in ('grounding_v2', 'grounding_v3'):
        result, version_rows = audit_version(args.run.resolve(), version)
        summary[version] = result
        rows.extend(version_rows)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (output / 'row_decisions.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    write_report(summary, output)
    print(json.dumps({key: {
        'responses': value['responses'],
        'accepted_visual_rows': value['accepted_visual_rows'],
        'mapped_rows': value['mapped_rows'],
        'unaccounted_mention_hints': value['unaccounted_mention_hints'],
        'decoder_reason_counts': value['decoder_reason_counts'],
    } for key, value in summary.items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
