#!/usr/bin/env python3
"""Reproducible heat-guided crop recipe on the restored original 2D protocol."""
import argparse
import json
from pathlib import Path
from MultiAgent.projected2d_attention_study import save_json,sha_file
from tools import iterate_original2d_visual as study


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','infer','evaluate'])
    parser.add_argument('--scope',choices=['dev','confirm','all'],default='all')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gaze-cache',type=Path,default=study.ROOT/'.local-sync/2d_original_gaze_cache32_20261109')
    args=parser.parse_args();output=args.output.resolve()
    if args.stage=='prepare':
        study.MAPS=args.gaze_cache.resolve()
        study.prepare(output,'HEAT_CROP',args.scope)
        target=output/'HEAT_CROP'
        manifest=json.loads((target/'input_manifest.json').read_text())
        rows=json.loads((target/'image_audit.json').read_text())['panels']
        sources={}
        for panel in rows:
            path=Path(panel['heatmap'])
            if sha_file(path)!=panel['heatmap_sha256']:raise ValueError('History map changed')
            sources[str(path)]=panel['heatmap_sha256']
        save_json(target/'heatmap_recipe.json',{
            'method':'Original source-resolution RGB plus gaze-history guided magnified RGB crop.',
            'history_seconds':2,'recent_bin_seconds':2/3,'density_mass':.90,'padding':1.35,
            'min_crop_fraction':.22,'max_crop_fraction':.45,
            'no_history_action':'Exact original 35% gaze crop fallback',
            'foveation_enabled':False,'why':'Two evaluated foveation strengths degraded the paired 2D score.',
            'reference_baseline':'Current native original Experiment 2 wrapper, same expected_count, prompt, 768 token output budget and coordinate mapping.',
            'explicit_scope':'Uses frozen 32-event source inventory; this entry does not claim evaluation on 4000 events.',
            'events':len(manifest['events']),'heatmap_files':sources})
        lock=json.loads((target/'protocol_lock.json').read_text())
        dependencies=[Path(__file__).resolve(),target/'heatmap_recipe.json',study.MAPS/'input_image_audit.json']
        for path in dependencies:lock['files'][str(path)]=sha_file(path)
        lock['files'].update(sources)
        save_json(target/'protocol_lock.json',lock)
    else:{'infer':study.infer,'evaluate':study.evaluate}[args.stage](output,'HEAT_CROP')

if __name__=='__main__':main()
