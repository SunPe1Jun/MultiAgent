"""Compare the same manifest under unified V3 and V3+GroundingDINO+SAM."""
import json
from pathlib import Path

from .evaluate import evaluate_results


def load_rows(root, manifest):
    rows, traces = {}, {}
    for item in manifest['samples']:
        key = item['event_id']
        row_path = root / 'V3' / item['scene_key'] / f"row_{item['row_index']}.json"
        trace_path = root / 'traces' / item['scene_key'] / key / 'trace.json'
        rows[key] = json.loads(row_path.read_text())
        traces[key] = json.loads(trace_path.read_text())
    return rows, traces


def main():
    project = Path('/workspace/usr3/MultiAgent')
    base_root = project / 'runs/unified_v3_20'
    sam_root = project / 'runs/unified_v3_20_sam'
    manifest = json.loads((project / 'runs/pilot20/manifest.json').read_text())
    base, base_trace = load_rows(base_root, manifest)
    sam, sam_trace = load_rows(sam_root, manifest)
    transitions = []
    for item in manifest['samples']:
        key = item['event_id']; b, s = base[key], sam[key]
        bg, sg = set(b['gt_referents_mapped']), set(s['gt_referents_mapped'])
        bt, st = base_trace[key], sam_trace[key]
        transitions.append({
            'scene': item['scene_key'], 'event_id': key,
            'base_prediction': sorted(b['selected_object_names']),
            'sam_prediction': sorted(s['selected_object_names']),
            'gt': sorted(bg), 'base_empty': not b['selected_object_names'],
            'sam_empty': not s['selected_object_names'],
            'empty_transition': ('base_empty_to_sam_nonempty' if not b['selected_object_names'] and s['selected_object_names']
                                 else 'base_nonempty_to_sam_empty' if b['selected_object_names'] and not s['selected_object_names'] else 'unchanged'),
            'base_tp': len(set(b['selected_object_names']) & bg), 'sam_tp': len(set(s['selected_object_names']) & sg),
            'base_fp': len(set(b['selected_object_names']) - bg), 'sam_fp': len(set(s['selected_object_names']) - sg),
            'base_fn': len(bg - set(b['selected_object_names'])), 'sam_fn': len(sg - set(s['selected_object_names'])),
            'sam_positive_visual_evidence': sum(e['support_score'] * e['confidence'] > 0 for e in st['visual']['evidence']),
            'sam_regions': sum(len(frame['regions']) for frame in st['visual']['frames']),
            'sam_parse_warning': st['visual'].get('parse_warning'),
            'asr_status': {m: st['temporal_plan']['mention_windows'][m]['status'] for m in st['temporal_plan']['mention_windows']},
        })
    base_metrics = evaluate_results(list(base.values()))
    sam_metrics = evaluate_results(list(sam.values()))
    transitions_by_type = {}
    for row in transitions:
        transitions_by_type[row['empty_transition']] = transitions_by_type.get(row['empty_transition'], 0) + 1
    report = {
        'manifest': str(project / 'runs/pilot20/manifest.json'), 'n': len(transitions),
        'base_metrics': base_metrics, 'sam_metrics': sam_metrics,
        'delta': {key: sam_metrics[key] - base_metrics[key] for key in ('any_hit_accuracy', 'exact_set_match', 'micro_precision', 'micro_recall', 'micro_f1', 'set_true_positive', 'set_false_positive', 'set_false_negative')},
        'empty_transition_counts': transitions_by_type, 'samples': transitions,
    }
    (sam_root / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['# V3 与 V3 + SAM 对照', '',
             '| 指标 | V3 | V3 + GroundingDINO/SAM | 变化 |', '|---|---:|---:|---:|']
    labels = [('any_hit_accuracy', 'Any-hit'), ('exact_set_match', 'Exact-set'), ('micro_precision', 'Micro Precision'),
              ('micro_recall', 'Micro Recall'), ('micro_f1', 'Micro F1')]
    for key, label in labels:
        lines.append(f"| {label} | {base_metrics[key] * 100:.2f}% | {sam_metrics[key] * 100:.2f}% | {(sam_metrics[key] - base_metrics[key]) * 100:+.2f} pp |")
    lines += ['', f"空输出变化：`{transitions_by_type}`。", '',
              '| 样本 | V3 | V3 + SAM | SAM 观察 |', '|---|---|---|---|']
    for row in transitions:
        if row['base_empty'] or row['sam_empty'] or row['base_prediction'] != row['sam_prediction']:
            lines.append(f"| {row['scene']}/{row['event_id']} | {', '.join(row['base_prediction']) or '∅'} | {', '.join(row['sam_prediction']) or '∅'} | {row['sam_positive_visual_evidence']} positive evidence, {row['sam_regions']} regions |")
    lines += ['', '## 空输出原因', '',
              '`scene2_664` 在两次运行中都为空：file mention 的 ASR 回退窗口较宽，SAM 没有生成可与 file candidate 可靠关联的 region；desk 虽有少量 region option，但 Visual 仍全部给零支持。',
              '`scene1_262` 在 SAM 运行中变为空：区域候选主要是未关联的 building/person proposals，Visual 没有把它们确认成闭集候选，因此全部为零支持；另外 Referential 已将 `side door` 映射为 `door`，但该场景的 door 候选集合为空，所以模型输出的 `door1..door3` 被 closed-set 校验拒绝。这是上游候选 schema 覆盖不足，和视觉证据为零是两个叠加问题。',
              '`scene2_123` 从空输出变为 `person1 + chair1`：SAM 区域和 Visual 产生了正支持，但 chair 实例仍选错（GT 为 chair5），说明区域辅助提升了可解析性，却没有完成实例身份对齐。', '',
              '完整逐条对照见 `comparison.json`；SAM 运行报告见 `RESULTS_zh.md`。']
    (sam_root / 'COMPARISON_zh.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'base': base_metrics, 'sam': sam_metrics, 'delta': report['delta'], 'empty_transitions': transitions_by_type}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
