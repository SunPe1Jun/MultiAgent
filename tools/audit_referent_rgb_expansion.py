#!/usr/bin/env python3
"""Read-only integrity check of a completed referent-aware RGB experiment."""
from __future__ import annotations
import argparse
from html.parser import HTMLParser
from pathlib import Path

from MultiAgent.heatmap_temporal_pilot import save, sha
from tools.run_referent_rgb_expansion import OUTPUT, check_study, read, require_verified_inputs


class LocalLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        self.paths.extend(value for key, value in attrs
                          if key in ('href', 'src') and value and ':' not in value and not value.startswith('#'))


def audit(output):
    manifest = check_study(output)
    require_verified_inputs(output, manifest)
    frozen = read(output/'predictions_frozen.json')
    if frozen['manifest_sha256'] != sha(output/'manifest.json'):
        raise ValueError('Prediction freeze belongs to another manifest')
    responses = {}
    for request in manifest['requests']:
        path = output/'responses'/f"{request['id']}.json"
        if sha(path) != frozen['response_sha256'][request['id']]:
            raise ValueError('Frozen prediction changed')
        responses[request['id']] = read(path)
    aliases = 0
    for key, response in responses.items():
        canonical = responses[response['canonical_request_id']]
        for field in ['initial', 'recovery', 'stages', 'final_validation', 'selected_object_names']:
            if response[field] != canonical[field]:
                raise ValueError('Identical inputs have different outputs')
        aliases += key != response['canonical_request_id']
    analysis = read(output/'analysis.json')
    for stage, conditions in analysis['stages'].items():
        for condition, row in conditions.items():
            if row['set_metrics']['events'] != len(manifest['events']):
                raise ValueError('Selected sample dropped from denominator')
            if abs(row['set_metrics']['micro_f1']-row['original_evaluator']['micro_f1']) > 1e-12:
                raise ValueError('Original evaluator disagreement')
    calls = list((output/'calls').glob('*.json'))
    failures = []
    visual_verified = 0
    for path in calls:
        call = read(path)
        visual_verified += call['vision_verified']
        if call['failure']:
            failures.append({'path':str(path.relative_to(output)), 'failure':call['failure']})
    links = 0
    for name in ['index.html', 'exact_set_review.html']:
        parser = LocalLinks()
        parser.feed((output/name).read_text())
        for relative in parser.paths:
            if not (output/relative).exists():
                raise ValueError(f'Broken {name} link: {relative}')
            links += 1
    tensors = list((output/'model_cache').rglob('*.pt')) + list((output/'model_cache').rglob('*.safetensors'))
    if tensors:
        raise ValueError('Regenerable visual tensors remain in model cache')
    result = {'manifest_source_lock_unchanged':True,
        'responses_frozen_before_evaluation':len(responses),
        'identical_input_aliases_verified':aliases,
        'selected_interactions_retained':len(manifest['events']),
        'original_evaluator_agreement':len(manifest['stages'])*len(manifest['conditions']),
        'completed_visual_calls':len(calls), 'visual_consumption_verified_calls':visual_verified,
        'visual_runtime_failures':failures, 'local_html_links_verified':links,
        'regenerable_visual_tensors_remaining':len(tensors),
        'input_verification':read(output/'input_verification.json'),
        'scope':'Artifact and protocol integrity only; not object-point correctness or scene identity calibration.'}
    save(output/'completion_audit.json',result)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    audit(parser.parse_args().output.resolve())
