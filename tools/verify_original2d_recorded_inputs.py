#!/usr/bin/env python3
"""Rebuild frozen model requests, explicitly recording historical helper source drift."""
import argparse
import json
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from MultiAgent.projected2d_attention_study import save_json,sha_file
from MultiAgent.visual_input_trace import tensor_record
from tools.iterate_original2d_visual import visual,MODEL

# These helpers generated immutable manifests. They were edited AFTER early runs;
# preserve and disclose the mismatch rather than silently changing original locks.
HISTORICAL_HELPERS={'iterate_original2d_followup.py','prepare_original2d_confirmation.py'}


def verify(roots):
    wrapper=visual();processor=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True)
    for root in roots:
        lock=json.loads((root/'protocol_lock.json').read_text())['files'];drift=[]
        for name,expected in lock.items():
            actual=sha_file(Path(name))
            if actual!=expected:
                if Path(name).name not in HISTORICAL_HELPERS:
                    raise ValueError('Nonhistorical dependency changed: '+name)
                drift.append({'file':name,'expected_sha256':expected,'current_sha256':actual,
                              'scope':'Image preparation/CLI helper edited after the frozen run; source lock no longer completely matches. Frozen inputs are independently verified below.'})
        manifest=json.loads((root/'input_manifest.json').read_text())
        frozen=json.loads((root/'predictions_frozen.json').read_text())['responses'];calls=[]
        if len(frozen)!=len(manifest['events']):raise ValueError('Incomplete frozen predictions')
        for e in manifest['events']:
            path=root/'responses'/(e['id']+'.json')
            if sha_file(path)!=frozen[str(path)]:raise ValueError('Frozen response changed: '+str(path))
            r=json.loads(path.read_text());images=[]
            for item in e['images']:
                if sha_file(Path(item['path']))!=item['sha256']:raise ValueError('Image hash mismatch')
                with Image.open(item['path']) as opened:images.append(opened.convert('RGB'))
            rebuilt=wrapper.build_model_inputs(processor,images,e['prompt'],e['panel_meta'],'none')
            for image in images:image.close()
            trace=r['visual_tensor_trace']
            if not trace['input_consumption_verified']:raise ValueError('Actual vision consumption not verified')
            for key in ('input_ids','pixel_values','image_grid_thw'):
                if tensor_record(rebuilt[key])!=trace['processor_tensors'][key]:
                    raise ValueError(e['id']+':'+key+' differs from generated request')
            calls.append({'event':e['id'],'input_ids_pixels_and_grid_match':True,'vision_consumption_verified':True})
        save_json(root/'recorded_input_verification.json',{'calls_verified':len(calls),'source_lock_fully_matches':not drift,
            'historical_preparation_helper_drift':drift,'model_and_evaluator_sources_match':True,'calls':calls})
        print('VERIFIED_RECORDED',root.name,len(calls),'historical_helper_drifts',len(drift),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('roots',type=Path,nargs='+');a=p.parse_args()
    verify([x.resolve() for x in a.roots])
