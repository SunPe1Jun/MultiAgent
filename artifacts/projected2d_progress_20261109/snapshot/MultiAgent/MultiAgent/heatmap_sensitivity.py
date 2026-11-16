"""GT-free counterfactual check that model locations follow heatmap pixels."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .gaze_evidence import DEFAULTS, render_presentation
from .local_attention_probe import MODEL_DEFAULT, read_json, save, sha_file, weighted_center
from .qwen_backend import QwenBackend
from .spatial_evidence import heat_guide


def heat_geometry(heat):
    peak = float(np.max(heat))
    if peak <= 0:
        return {'center_uv': None, 'peak_uv': None, 'peak_quadrant': 'none'}
    y, x = np.unravel_index(int(np.argmax(heat)), heat.shape)
    u, v = (x+.5)/heat.shape[1], (y+.5)/heat.shape[0]
    quadrant = ('top_' if v < .5 else 'bottom_') + ('left' if u < .5 else 'right')
    return {'center_uv': weighted_center(heat), 'peak_uv': [u, v], 'peak_quadrant': quadrant}


def _score_rows(group, parsed):
    rows = parsed.get('images', []) if isinstance(parsed, dict) else parsed
    by_id = {row.get('image_id'): row for row in rows if isinstance(row, dict)}
    checked = []
    for image in group['input_order']:
        image_id = image['image_id']
        expected = group['expected_geometry_audit_only'][image_id]
        observed = by_id.get(image_id, {})
        point = observed.get('center_uv')
        with Image.open(group['paths'][image['image_index']-1]) as source_image:
            expected_size = source_image.size
        normalized, coordinate_format = None, 'missing'
        if isinstance(point, (list, tuple)) and len(point) == 2:
            try:
                u, v = map(float, point)
                if 0 <= u <= 1 and 0 <= v <= 1:
                    normalized, coordinate_format = [u, v], 'normalized_0_to_1'
                elif 0 <= u <= expected_size[0] and 0 <= v <= expected_size[1]:
                    normalized = [u/expected_size[0], v/expected_size[1]]
                    coordinate_format = 'pixel_like_posthoc_normalized'
                else:
                    coordinate_format = 'out_of_range'
            except (TypeError, ValueError):
                coordinate_format = 'invalid'
        error = (math.dist(expected['peak_uv'], normalized)
                 if expected['peak_uv'] is not None and normalized is not None else None)
        checked.append({
            'image_id': image_id, 'expected': expected, 'model_response': observed,
            'model_center_uv_normalized_for_audit': normalized,
            'center_coordinate_format': coordinate_format,
            'coordinate_schema_compliant': coordinate_format in ('normalized_0_to_1', 'missing'),
            'quadrant_correct': observed.get('quadrant') == expected['peak_quadrant'],
            'peak_center_error_uv': error,
        })
    return checked


def finalize_existing_run(output_dir, cache_dir):
    out, cache = Path(output_dir).resolve(), Path(cache_dir).resolve()
    audit = read_json(out / 'input_audit.json')
    namespace = 'heatmap_pixel_sensitivity_qwen8b_20261022_v1'
    call_dir = cache / namespace / 'heatmap_sensitivity'
    results = {}
    for kind, group in audit['groups'].items():
        if kind == 'heatmap':
            previous = read_json(out / 'heatmap_result.json')
            response_path = Path(previous['request_path'])
            saved = read_json(response_path)
            parsed = saved['parsed']
            parse_status = 'valid_json_object'
        else:
            candidates = sorted(call_dir.glob('foveation_*.failed.json'),
                                key=lambda path: path.stat().st_mtime)
            if not candidates:
                raise FileNotFoundError('Missing retained foveation parse-failure response')
            response_path = candidates[-1]
            saved = read_json(response_path)
            raw = saved.get('response', '')
            value = json.loads(raw)
            if not isinstance(value, list):
                raise ValueError('Expected the retained foveation response to be a top-level JSON array')
            parsed = {'images': value}
            parse_status = 'valid_json_array_backend_schema_mismatch_normalized_for_diagnostic'
        results[kind] = {
            'parsed': parsed, 'raw_response': saved.get('response'),
            'request_path': str(response_path), 'parse_status': parse_status,
            'checks': _score_rows(group, parsed),
            'processor_audit': saved.get('processor_audit'),
            'visual_tensor_trace': saved.get('visual_tensor_trace'),
            'elapsed_seconds': saved.get('elapsed_seconds'),
        }
        save(out / f'{kind}_result.json', results[kind])
    calls_path = cache / namespace / 'call_audit.jsonl'
    calls = [json.loads(line) for line in calls_path.read_text().splitlines() if line.strip()]
    summary = {
        'counterfactual_only': True, 'candidate_metrics': 'NOT_RUN',
        'source_sample': audit['source_sample'], 'source_frame_id': audit['source_frame_id'],
        'results': results, 'call_audit': calls,
        'heatmap_quadrant_correct': sum(row['quadrant_correct'] for row in results['heatmap']['checks']),
        'heatmap_quadrant_total': len(results['heatmap']['checks']),
        'foveation_detected_count': sum(row['model_response'].get('detected') is True
                                        for row in results['foveation']['checks']),
        'foveation_total': len(results['foveation']['checks']),
        'verified_vision_calls': all(
            result.get('visual_tensor_trace', {}).get('input_consumption_verified')
            for result in results.values()),
        'foveation_backend_parse_status': results['foveation']['parse_status'],
    }
    save(out / 'summary.json', summary)
    (out / 'AUDIT.md').write_text(
        '# 热图与 foveation 像素敏感性诊断\n\n'
        '同一真实关键帧构造三种输入：180 度翻转的真实 gaze 热图、零热图、原始真实热图。'
        '翻转仅用于反事实检查；未输入目标标签、坐标答案或 GT。\n\n'
        f"- 热图热点象限：{summary['heatmap_quadrant_correct']}/{summary['heatmap_quadrant_total']} 正确。\n"
        f"- Foveation 可见焦点：{summary['foveation_detected_count']}/{summary['foveation_total']} 张被报告。\n"
        f"- 视觉编码器实际消费张量校验：{'通过' if summary['verified_vision_calls'] else '失败'}。\n"
        f"- 热图中心坐标有 {sum(not row['coordinate_schema_compliant'] for row in results['heatmap']['checks'])} 张以像素值输出，"
        '不符合归一化坐标格式；象限正确不能替代精确定位。\n'
        f"- Foveation 响应是有效 JSON 数组，但 backend 契约要求 JSON 对象；原始内容和失败记录均保留。"
        f"诊断汇总时将数组包装为对象，解析状态为 `{summary['foveation_backend_parse_status']}`。\n\n"
        '该测试证明线索图的像素变化是否被视觉读取，不检验目标指代精度，也不代表系统级性能收益。\n',
        encoding='utf-8')
    return summary


def run(args):
    source_dir = Path(args.sample_dir).resolve()
    frozen = read_json(source_dir / 'frozen_input_audit.json')
    config = read_json(Path(args.config))
    frame_id = args.frame_id
    if frame_id not in {row['frame_id'] for row in frozen['frames']}:
        raise ValueError('Counterfactual frame is not in the fixed source-frame list')
    source = source_dir / 'inputs' / frame_id
    with Image.open(source / 'R.png') as image:
        rgb = image.convert('RGB')
    heat = np.load(source / 'attention_map.npz')['H_unnormalized']
    variants = {'original': heat, 'flipped_180': heat[::-1, ::-1].copy(),
                'zero': np.zeros_like(heat)}
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    groups = {}
    order = [('X1', 'flipped_180'), ('X2', 'zero'), ('X3', 'original')]
    for kind in ('heatmap', 'foveation'):
        images, geometry, paths = [], {}, []
        for name, variant in order:
            attention = variants[variant]
            if kind == 'heatmap':
                image = heat_guide(attention, scale=config['gaze'].get('display_scale', 2.0))
            else:
                image, _ = render_presentation(rgb, attention, 'F_local', {**DEFAULTS, **config['gaze']})
            path = out / f'{kind}_{name}.png'
            image.save(path)
            images.append({'image_id': name, 'image_index': len(paths)+1})
            paths.append(str(path))
            geometry[name] = {'variant': variant, **heat_geometry(attention),
                              'path': str(path), 'sha256': sha_file(path)}
        if kind == 'heatmap':
            protocol = ('The images are grayscale saliency maps. Find the brightest actual blob in each. '
                        'An image with no visible blob must return quadrant=none and center_uv=null.')
        else:
            protocol = ('The images show the same RGB scene, possibly with soft foveation. Find any locally '
                        'sharper retained region that differs from its blurred surroundings. A uniformly '
                        'sharp image has no foveation focus: return quadrant=none and center_uv=null. '
                        'Do not use scene object salience or image center as a substitute for a visible focus.')
        prompt = (
            'Inspect only the supplied image pixels. This is a visual encoding test, not a referring or target task. '
            + protocol +
            ' The images may differ. Do not assume the center is attended. Return JSON with an images array, '
            'exactly one row per listed image. Each row must contain image_id, detected (true/false), '
            'quadrant (top_left/top_right/bottom_left/bottom_right/none/unclear), '
            'center_uv ([u,v] at the brightest blob/sharp focus, top-left origin, or null), '
            'confidence (0..1), and a brief observation. No numerical location hints are provided. Image order: '
            + json.dumps(images))
        groups[kind] = {'input_order': images, 'expected_geometry_audit_only': geometry,
                        'prompt': prompt, 'paths': paths}
    save(out / 'input_audit.json', {
        'source_frame_id': frame_id, 'source_sample': str(source_dir),
        'transforms': 'flip the real unnormalized gaze heatmap 180 degrees; RGB unchanged; zero map is the no-gaze control',
        'counterfactual_only_not_candidate_experiment': True,
        'model_receives_transform_labels_or_expected_locations': False,
        'ground_truth_loaded': False, 'groups': groups,
    })
    backend = QwenBackend(args.cache_dir, str(MODEL_DEFAULT), local_only=True,
        namespace='heatmap_pixel_sensitivity_qwen8b_20261022_v1', trace_visual_inputs=True)
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(20261022)
    results = {}
    for kind, group in groups.items():
        print(f'{kind}: three actual-input images; original/flip/zero labels hidden', flush=True)
        parsed = backend.ask('heatmap_sensitivity', kind, group['prompt'], group['paths'], max_new_tokens=1024)
        request_path = Path(backend.requests[-1])
        saved = read_json(request_path)
        model_rows = {row.get('image_id'): row for row in parsed.get('images', []) if isinstance(row, dict)}
        checked = []
        for image in group['input_order']:
            expected = group['expected_geometry_audit_only'][image['image_id']]
            observed = model_rows.get(image['image_id'], {})
            point = observed.get('center_uv')
            distance = None
            if expected['peak_uv'] is not None and isinstance(point, list) and len(point) == 2:
                try:
                    distance = math.dist(expected['peak_uv'], [float(value) for value in point])
                except (ValueError, TypeError):
                    pass
            checked.append({'image_id': image['image_id'], 'expected': expected,
                'model_response': observed,
                'quadrant_correct': observed.get('quadrant') == expected['peak_quadrant'],
                'peak_center_error_uv': distance})
        results[kind] = {'parsed': parsed, 'raw_response': saved.get('response'),
            'request_path': str(request_path), 'checks': checked,
            'processor_audit': saved.get('processor_audit'),
            'visual_tensor_trace': saved.get('visual_tensor_trace'),
            'elapsed_seconds': saved.get('elapsed_seconds')}
        save(out / f'{kind}_result.json', results[kind])
        print(f"  quadrant_correct={sum(row['quadrant_correct'] for row in checked)}/3", flush=True)
    save(out / 'summary.json', {'counterfactual_only': True, 'candidate_metrics': 'NOT_RUN',
        'source_sample': str(source_dir), 'source_frame_id': frame_id,
        'results': results, 'call_audit': backend.call_records})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-dir', required=True)
    parser.add_argument('--frame-id', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--finalize-only', action='store_true')
    args = parser.parse_args()
    if args.finalize_only:
        finalize_existing_run(args.output_dir, args.cache_dir)
    else:
        run(args)


if __name__ == '__main__':
    main()
