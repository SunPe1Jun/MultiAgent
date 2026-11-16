#!/usr/bin/env python3
"""Browse every preserved spherical attention region and real RGB provenance."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion
from MultiAgent.attention_regions import extract_regions, region_observations
from MultiAgent.heatmap_temporal_pilot import save, sha
from MultiAgent.perspective_attention_study import SOURCE, read
from MultiAgent.spherical_attention import solid_angle_weights
from tools.run_attention_region_study import OUTPUT, PHASES, heat_tile


def build(output=OUTPUT):
    manifest = read(output / 'manifest.json')
    sphere = {e['id']: e for e in read(SOURCE / 'manifest.json')['events']}
    previews = {}; sensitivity = []
    for event in manifest['events']:
        data = np.load(output / event['regions_npz']); maps = data['duration_maps']; labels = data['labels']
        _, h, w = maps.shape; density = maps / solid_angle_weights(w, h)[None]
        image = Image.new('RGB', (1008, 660), '#15212b'); draw = ImageDraw.Draw(image)
        for phase in range(3):
            # Full 360x180 storage shown, no per-face truncation or top-k display.
            tile = heat_tile(density[phase], float(density.max())).resize((720, 180), Image.Resampling.NEAREST)
            boundary = np.zeros((h, w), bool)
            for component in np.unique(labels[phase]):
                if component:
                    mask = labels[phase] == component
                    boundary |= mask & ~binary_erosion(mask)
            pixels = np.asarray(tile).copy(); edges = np.asarray(Image.fromarray(boundary).resize(tile.size, Image.Resampling.NEAREST))
            pixels[edges] = [70, 190, 255]
            x, y = 270, 32 + phase * 208
            image.paste(Image.fromarray(pixels), (x, y))
            draw.text((14, y + 8), PHASES[phase], fill='white', font=ImageFont.load_default(size=18))
            for j, region in enumerate(r for r in event['regions'] if r['phase'] == phase):
                py, px = region['peak_yx']
                draw.text((x + px * 2 + 3, y + py - 5), region['id'], fill='white',
                          stroke_width=1, stroke_fill='black', font=ImageFont.load_default(size=13))
            draw.text((14, y + 44), f"{event['mass_audits'][phase]['components']} connected regions", fill='white', font=ImageFont.load_default(size=13))
            draw.text((14, y + 68), 'Blue outline = mass envelope', fill='#b8d5eb', font=ImageFont.load_default(size=12))
        draw.text((270, 8), 'FULL LONGITUDE -180 ... 0 ... +180 | fixed reference orientation', fill='white', font=ImageFont.load_default(size=14))
        path = output / 'events' / event['event'] / 'envelopes.png'; image.save(path)
        previews[event['id']] = {'path': str(path.relative_to(output)), 'sha256': sha(path)}
        for fraction in (.8, .9, .95):
            lab, regions, audits = extract_regions(maps, fraction)
            assigned = total = 0.
            for phase in range(3):
                observations = sphere[event['id']]['volume_stats']['bins'][phase]['observations']
                total += sum(o['duration_seconds'] for o in observations)
                for region in (r for r in regions if r['phase'] == phase):
                    assigned += sum(o['duration_seconds'] for o in region_observations(observations, event['canonical_pose'], lab[phase], region['component']))
            sensitivity.append({'event': event['event'], 'mass_fraction_parameter': fraction, 'components': len(regions),
                'retained_seconds': sum(a['retained_seconds'] for a in audits),
                'total_seconds': sum(a['total_seconds'] for a in audits),
                'observation_center_duration_in_envelope': assigned, 'observation_duration_seconds': total})
    summaries = []
    for fraction in (.8, .9, .95):
        rows = [r for r in sensitivity if r['mass_fraction_parameter'] == fraction]
        summaries.append({'mass_fraction_parameter': fraction, 'components': sum(r['components'] for r in rows),
            'retained_seconds': sum(r['retained_seconds'] for r in rows), 'total_seconds': sum(r['total_seconds'] for r in rows),
            'observation_center_duration_fraction': sum(r['observation_center_duration_in_envelope'] for r in rows) /
                sum(r['observation_duration_seconds'] for r in rows)})
    save(output / 'envelope_sensitivity.json', {'summary': summaries, 'rows': sensitivity,
        'scope': 'Post-protocol engineering sensitivity only. No parameter changed, no new model calls, no object/target scores.'})
    save(output / 'memory_previews.json', previews)
    template = []
    for event in manifest['events']:
        for r in event['regions']:
            template.append({'event': event['event'], 'region_id': r['id'], 'source_phase': r['phase'], 'status': r['status'],
                'native_source_path': r.get('native_source_path'), 'crop_image': r.get('crop_image'),
                'visible_object_labels': None, 'instruction_relevance': None, 'mention_spans': None,
                'same_object_as_regions': None, 'annotator': None, 'notes': ''})
    if not (output / 'annotation_template.json').exists():
        save(output / 'annotation_template.json', {'status': 'UNANNOTATED', 'rows': template,
            'warning': 'Null means not annotated. Never initialize GT from model descriptions or heat rankings.'})
    predictions = {}
    for path in (output / 'single_region/responses').glob('*.json'):
        r = read(path)
        if r['condition'] == 'TRUE_HEAT': predictions[(r['event_id'], r['region_id'])] = r['parsed']
    payload = {'events': manifest['events'], 'previews': previews,
        'predictions': {event + ':' + rid: value for (event, rid), value in predictions.items()}}
    data = json.dumps(payload, ensure_ascii=False).replace('</', '<\\/')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>完整关注区域记忆</title><style>body{font:15px system-ui;background:#f3f5f7;margin:20px}header{position:sticky;top:0;background:#f3f5f7;padding:10px}select{font:inherit}#overview{max-width:1100px;width:100%}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px}figure{background:white;padding:12px;margin:0}img{width:100%}pre{white-space:pre-wrap}p{max-width:1200px}</style><header><h2>全部115个关注区域均保留</h2><select id="event"></select><p id="instruction"></p></header><p>固定球面上的早、中、晚密度，蓝线圈出每段90%支持时长包络的连通区域。原始热图与剩余约10%都保留在 regions.npz；完整球面不会随头动重新居中。区域是观察方向，不是对象；同一对象可能重复出现，低注意对象也可能是目标。相机只做旋转对齐，没有解决平移视差、动态对象和可见性。</p><p>114条有真实源帧；1条画外失败保留状态。下面模型文字只对预选39条做了抽查，不能作为GT，也不用于自动删候选。源视频自带的绿色标记仍在原始RGB中，NO_HEAT仅删除额外生成的热图。<a href="single_region/index.html">三条件抽查</a> · <a href="matched_angular_coverage.json">相同预算覆盖比较</a> · <a href="envelope_sensitivity.json">包络阈值敏感性</a> · <a href="annotation_template.json">未标注复核表</a></p><img id="overview"><main></main><script>const d=__DATA__,q=document.querySelector('#event');d.events.forEach(e=>{let o=document.createElement('option');o.value=e.id;o.textContent=e.event;q.append(o)});function render(){let e=d.events.find(e=>e.id===q.value);document.querySelector('#instruction').textContent=e.instruction;document.querySelector('#overview').src=d.previews[e.id].path;let main=document.querySelector('main');main.replaceChildren();e.regions.forEach(r=>{let f=document.createElement('figure'),text=document.createElement('pre');let model=d.predictions[e.id+':'+r.id];text.textContent=r.id+' | '+['早','中','晚'][r.phase]+' | '+r.status+'\\n'+(model?JSON.stringify(model):'未运行语义模型，证据保留');f.append(text);if(r.context_image){let a=document.createElement('img'),b=document.createElement('img');a.src=r.context_image.path;b.src=r.crop_image.path;f.append(a,b)}main.append(f)})}q.onchange=render;render();</script></html>'''
    (output / 'region_memory.html').write_text(page.replace('__DATA__', data))
    print(json.dumps({'previews': len(previews), 'regions': len(template), 'sensitivity': summaries}, indent=2))


if __name__ == '__main__': build()
