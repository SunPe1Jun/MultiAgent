#!/usr/bin/env python3
"""Add actual panel IDs to RGB headers for reordered mention-focused 2D input."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from MultiAgent.gaze_rgb_routing import HEADER
from MultiAgent.projected2d_attention_study import save_json, sha_file
from MultiAgent.projected2d_focus import prepare as prepare_focus
from MultiAgent.projected2d_routed_rgb import GROUPS


def prepare(source, output):
    prepare_focus(source, output)
    manifest = json.loads((output/'input_manifest.json').read_text())
    count = 0
    for event in manifest['events']:
        for panel in event['panels']:
            width, height = panel['display_size']
            for group in GROUPS:
                original = Path(panel['inputs'][group]['path'])
                if sha_file(original) != panel['inputs'][group]['sha256']:
                    raise ValueError('Frozen input image hash mismatch')
                with Image.open(original) as opened:
                    image = opened.convert('RGB')
                before = np.asarray(image).copy()
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, width-1, HEADER-1), fill='#15212b')
                draw.text((8, 3), f"{panel['frame_id']} | {panel['requested_video_time']:.3f}s | FULL RGB",
                          fill='white', font=ImageFont.load_default(size=17))
                after = np.asarray(image)
                if not np.array_equal(before[HEADER:, :, :], after[HEADER:, :, :]):
                    raise ValueError('Label changed pixels outside the header')
                target = output/'labelled_inputs'/event['id']/panel['frame_id']/f'{group}.png'
                target.parent.mkdir(parents=True, exist_ok=True)
                image.save(target)
                panel['inputs'][group] = {'path': str(target), 'sha256': sha_file(target)}
                count += 1
    manifest.update(version='projected2d_focused_labelled_v2',
        focus_policy='Each mention has an independent call; order all original panels by soft phase center and visibly label every image with its real panel ID and video time.',
        input_header_label_contract='The only added pixels are the real P1/P2/P3 and time in the first title bar; RGB pixels and crop geometry are unchanged.',
        label_source_sha256=sha_file(Path(__file__)))
    save_json(output/'input_manifest.json', manifest)
    lock = json.loads((output/'protocol_lock.json').read_text())
    lock.update(input_manifest_sha256=sha_file(output/'input_manifest.json'),
                label_source_sha256=sha_file(Path(__file__)))
    save_json(output/'protocol_lock.json', lock)
    save_json(output/'header_audit.json', {'labelled_images': count,
        'pixels_outside_first_header_unchanged': count,
        'source_ids_and_video_times_only': True, 'gt_used_for_images': False})
    print('LABELLED_IMAGES', count, flush=True)


def infer(output):
    from MultiAgent import projected2d_focus as focus
    lock = json.loads((output/'protocol_lock.json').read_text())
    if sha_file(Path(__file__)) != lock['label_source_sha256']:
        raise ValueError('Frozen image label/prompt code changed')
    original_prompt = focus.prompt_for

    def labelled_prompt(focused, mention, hint, group):
        return ('The real frame_id and its video time are written in the top-left title bar of each image. '
                'Use those labels to associate images with frame_id after reordering.\n' +
                original_prompt(focused, mention, hint, group))

    focus.prompt_for = labelled_prompt
    try:
        focus.infer(output)
    finally:
        focus.prompt_for = original_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'infer'))
    parser.add_argument('--source', type=Path,
        default=Path('/ai/data/MultiAgent/.local-sync/projected2d_routed_composite_v2_20261109'))
    parser.add_argument('--output', type=Path,
        default=Path('/ai/data/MultiAgent/.local-sync/projected2d_mention_focus8_labelled_20261109'))
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == 'prepare':
        prepare(args.source.resolve(), output)
    else:
        infer(output)


if __name__ == '__main__':
    main()
