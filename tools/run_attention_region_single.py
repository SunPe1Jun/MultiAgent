#!/usr/bin/env python3
"""Single-region readability regression, with visual heat retained as images.

Prespecifies three source regions per old event; all other generated regions
remain in the parent memory. This is a semantic spot check, not complete object
screening or a temporal-comprehension / referent benchmark.
"""
import argparse
from collections import Counter
import io
import hashlib
import json
from pathlib import Path
import random
import time
from PIL import Image, ImageDraw, ImageFont
from MultiAgent.heatmap_temporal_pilot import save, sha
from MultiAgent.perspective_attention_study import DEV, read
from MultiAgent.qwen_backend import QwenBackend
from tools.run_attention_region_study import OUTPUT, CONDITIONS, PHASES

PROMPT = '''Identify what is visible in the DETAIL RGB image, then judge whether that visible content could be an object or spatial reference in the instruction.
LEFT = source-frame context; RIGHT = enlarged DETAIL of a script-proposed attention region. Judge the object(s) INSIDE DETAIL. Context can clarify relations; a mentioned object elsewhere in context does not make an unrelated detail plausible.
Below are EARLY / MIDDLE / LATE heat images registered to this same detail crop and native camera direction. Orange shows recorded attention, not object identity or target labels. These may be controls. Missing heat supplies no extra attention information. Use RGB and the instruction for relevance; retain uncertainty when detail is insufficient.
Return only {"visible":"short concrete visual description", "relevance":"plausible"}. relevance must be plausible, unclear or unrelated. Describe what you actually see, even when unrelated. Do not output boxes, coordinates or lists. This is not final target identity.
INSTRUCTION: __INSTRUCTION__'''


def font(size): return ImageFont.load_default(size=size)


def single_image(parent, row, condition):
    image = Image.new('RGB', (1008, 1008), '#15212b'); draw = ImageDraw.Draw(image)
    draw.text((20, 10), f"{row['id']} | SOURCE {PHASES[row['phase']]} | PTS {row['source_frame']['pts_seconds']:.3f}s", fill='white', font=font(22))
    for j, key in enumerate(['context_image', 'crop_image']):
        draw.text((28 + j * 496, 52), ['SOURCE CONTEXT', 'DETAIL TO REVIEW'][j], fill='white', font=font(20))
        with Image.open(parent / row[key]['path']) as source:
            tile = source.convert('RGB').resize((448, 448), Image.Resampling.LANCZOS)
        image.paste(tile, (28 + j * 496, 84))
    for k, phase in enumerate(PHASES):
        draw.text((32 + k * 328, 576), phase, fill='white', font=font(20))
        if condition == 'NO_HEAT':
            heat = Image.new('RGB', (288, 288), (20, 20, 20))
        else:
            with Image.open(parent / row['heat_images'][2-k if condition == 'REVERSE_TIME' else k]['path']) as source:
                heat = source.convert('RGB').resize((288, 288), Image.Resampling.BILINEAR)
        image.paste(heat, (32 + k * 328, 614))
        draw.rectangle((32 + k * 328, 614, 319 + k * 328, 901), outline='#52616e')
    draw.text((24, 944), 'Heat = historical attention. RGB detail = evidence for object relevance.', fill='white', font=font(18))
    return image


def prepare(parent):
    output = parent / 'single_region'
    if (output / 'manifest.json').exists(): raise ValueError('Never overwrite prepared probe')
    output.mkdir(parents=True, exist_ok=True)
    m = read(parent / 'manifest.json'); requests = []; selected = []
    for event in m['events']:
        ready = sorted([r for r in event['regions'] if r['status'] == 'RGB_EVIDENCE_READY'],
                       key=lambda r: (-r['mass_seconds'], r['component'], r['phase']))
        rows = [next(r for r in ready if r['phase'] == k) for k in (0, 2)]
        rows.append(next(r for r in ready if r['id'] not in {v['id'] for v in rows}))
        split = 'development' if event['event'] in DEV else 'check'
        for row in rows:
            selected.append({'event_id': event['id'], 'event': event['event'], 'region_id': row['id'], 'phase': row['phase'], 'split': split})
            for condition in CONDITIONS:
                rid = f"{event['id']}_{row['id']}_{condition}"
                path = output / 'images' / f'{rid}.png'; path.parent.mkdir(parents=True, exist_ok=True)
                single_image(parent, row, condition).save(path)
                requests.append({'id': rid, 'event_id': event['id'], 'event': event['event'], 'region_id': row['id'],
                    'phase': row['phase'], 'condition': condition, 'split': split,
                    'image': str(path.relative_to(output)), 'sha256': sha(path),
                    'prompt': PROMPT.replace('__INSTRUCTION__', event['instruction'])})
    save(output / 'manifest.json', {'parent_manifest_sha256': sha(parent / 'manifest.json'),
        'protocol': 'Three regions/event: largest early, largest late, largest remaining region. No model outputs used for selection.',
        'scope': '39 old-region RGB relevance spot checks; 114 ready parent regions remain stored. No object or relevance GT.',
        'temporal_scope': 'Relevance anywhere in the instruction; time reversal is a perturbation, not a correctness test of action-stage alignment.',
        'regression_only': True, 'selected_regions': selected, 'requests': requests,
        'code_sha256': sha(__file__), 'conditions': CONDITIONS})
    from shutil import copy2
    copy2(__file__, output / 'source.py')
    save(output / 'protocol_lock.json', {'manifest_sha256': sha(output / 'manifest.json'), 'before_model': True})
    report(parent)


def run(parent, split, limit=None):
    output = parent / 'single_region'; m = read(output / 'manifest.json')
    assert sha(parent / 'manifest.json') == m['parent_manifest_sha256']
    assert sha(output / 'manifest.json') == read(output / 'protocol_lock.json')['manifest_sha256']
    if split == 'check' and not (output / 'development_frozen.json').exists():
        raise ValueError('Finish and freeze development before check')
    backend = QwenBackend(output / 'model_cache', model='/ai/data/Qwen3-VL-8B-Instruct', local_only=True,
        namespace='single_attention_region_screen_v1', trace_visual_inputs=True)
    schedule = [r for r in m['requests'] if r['split'] == split]; random.Random(20261106).shuffle(schedule)
    count = 0
    for request in schedule:
        destination = output / 'responses' / f"{request['id']}.json"
        if destination.exists(): continue
        path = output / request['image']; assert sha(path) == request['sha256']
        print('INFER', request['id'], flush=True); started = time.monotonic()
        parsed = backend.ask(request['id'], 'screen', request['prompt'], [path], max_new_tokens=120)
        raw = read(backend.requests[-1])
        valid = (set(parsed) == {'visible', 'relevance'} and isinstance(parsed['visible'], str)
                 and parsed['relevance'] in {'plausible', 'unclear', 'unrelated'})
        save(destination, {'id': request['id'], 'event_id': request['event_id'], 'region_id': request['region_id'],
            'condition': request['condition'], 'split': split, 'parsed': parsed, 'schema_valid': valid,
            'request_path': backend.requests[-1], 'processor_audit': raw['processor_audit'],
            'vision_verified': raw['visual_tensor_trace']['input_consumption_verified'],
            'seconds': time.monotonic() - started})
        print('RESULT', parsed, flush=True); count += 1
        if limit is not None and count >= limit: break
    report(parent)
    expected = [r for r in m['requests'] if r['split'] == 'development']
    if split == 'development' and all((output / 'responses' / f"{r['id']}.json").exists() for r in expected):
        save(output / 'development_frozen.json', {'manifest_sha256': sha(output / 'manifest.json'),
            'completed_development_calls': len(expected), 'check_not_started': not any(r['split'] == 'check'
              for r in [read(p) for p in (output / 'responses').glob('*.json')])})


def report(parent):
    output = parent / 'single_region'; m = read(output / 'manifest.json')
    responses = {p.stem: read(p) for p in (output / 'responses').glob('*.json')}
    counts = {}; changes = {}
    for condition in CONDITIONS:
        rows = [r for r in responses.values() if r['condition'] == condition]
        counts[condition] = {'n': len(rows), 'valid_schema': sum(r['schema_valid'] for r in rows),
            'relevance': dict(Counter(r['parsed'].get('relevance', 'INVALID') for r in rows)),
            'literal_unclear_descriptions': sum(r['parsed'].get('visible', '').strip().lower() == 'unclear' for r in rows)}
    for condition in ['NO_HEAT', 'REVERSE_TIME']:
        pairs = []
        for selected in m['selected_regions']:
            prefix = f"{selected['event_id']}_{selected['region_id']}_"
            true = responses.get(prefix + 'TRUE_HEAT'); control = responses.get(prefix + condition)
            if true and control and true['schema_valid'] and control['schema_valid']:
                pairs.append((true, control))
        changes[condition] = {'paired_regions': len(pairs),
            'relevance_changes': sum(a['parsed']['relevance'] != b['parsed']['relevance'] for a,b in pairs),
            'keep_for_review_set_changes': sum((a['parsed']['relevance'] != 'unrelated') !=
                (b['parsed']['relevance'] != 'unrelated') for a,b in pairs)}
    summary = {'planned_calls': len(m['requests']), 'completed_calls': len(responses),
        'selected_region_count': len(m['selected_regions']), 'by_condition': counts, 'condition_changes': changes,
        'all_vision_verified': all(r['vision_verified'] for r in responses.values()),
        'image_tokens': sorted({r['processor_audit']['expanded_image_token_count'] for r in responses.values()}),
        'gt_accuracy_available': False, 'regression_only': True,
        'caveat': 'A concrete visual description or schema success is not verified object recognition. Changed relevance is not evidence of improvement; unchanged relevance does not disprove internal heat use.'}
    save(output / 'analysis.json', summary)
    payload = json.dumps({'m': m, 'responses': responses}, ensure_ascii=False).replace('</', '<\\/')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>单关注区域审阅</title><style>body{font:15px system-ui;background:#f3f5f7;margin:20px}header{position:sticky;top:0;background:#f3f5f7;padding:10px}select{font:inherit;margin:6px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:15px}figure{background:white;padding:12px;margin:0}img{width:100%}pre{white-space:pre-wrap}</style><header><h2>放大单区域：RGB语义筛选与图形热图对照</h2><select id="event"></select><select id="condition"><option>TRUE_HEAT</option><option>NO_HEAT</option><option>REVERSE_TIME</option></select></header><p>每交互预选早段最大区域、晚段最大区域、其余时段最大区域，共39区域；不是对114个区域的完整筛选。其余原始证据仍存于上级目录。左右为源帧上下文/局部详情，下方热图均对应该源帧方向及同一裁剪；灰暗是无额外热图。未标注对象或相关性GT。模型的文字描述不是人工核验的识别结论。</p><p>本任务判断候选是否与整条指令有关，没有mention时间GT；时间反转只是扰动，输出不变不能证明没用热图，输出变化也不能证明更好。所有交互均为旧样例。</p><main></main><script>const d=__DATA__,q=document.querySelector('#event'),c=document.querySelector('#condition');[...new Set(d.m.requests.map(r=>r.event))].forEach(e=>{let o=document.createElement('option');o.value=e;o.textContent=e;q.append(o)});function render(){let main=document.querySelector('main');main.replaceChildren();d.m.requests.filter(r=>r.event===q.value&&r.condition===c.value).forEach(r=>{let f=document.createElement('figure'),im=document.createElement('img'),text=document.createElement('pre');im.src=r.image;let z=d.responses[r.id];text.textContent=r.region_id+' | '+(z?JSON.stringify(z.parsed,null,2):'尚未运行');f.append(im,text);main.append(f)})}q.onchange=render;c.onchange=render;render();</script></html>'''
    (output / 'index.html').write_text(page.replace('__DATA__', payload))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare','run','report'])
    p.add_argument('--parent', type=Path, default=OUTPUT)
    p.add_argument('--split', choices=['development','check'], default='development')
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    if a.action == 'run': run(a.parent.resolve(), a.split, a.limit)
    else: {'prepare':prepare,'report':report}[a.action](a.parent.resolve())
