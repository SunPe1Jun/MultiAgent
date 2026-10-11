"""Seeded sample selection across all scenes; no prediction- or correctness-based selection."""
import argparse
import csv
import json
import random
import re
from pathlib import Path
from .baseline import ROOT, scene_paths, module


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=20260922)
    a=p.parse_args();rng=random.Random(a.seed); selected=[]
    ev=module('eval')
    for scene in ['scene1','scene2','scene3','scene4_room1','scene4_room2','scene4_room3','scene4_room4','scene5']:
        paths=scene_paths(scene)
        rows=list(csv.DictReader(paths['input'].open(encoding='utf-8-sig')))
        aliases,_=ev.load_anchor_alias_map(paths['anchor']); gt=ev.load_gt_rows(paths['gt'],aliases)
        n=2 if 'room' in scene else 3
        ids=list(range(len(rows)));rng.shuffle(ids)
        multi=[i for i in ids if len(gt[i]['mapped_referents'])>1]
        region=[i for i in ids if re.search(r'\b(area|region|location|point|between|behind|front|near|around)\b',rows[i]['instruction_text'],re.I)]
        chosen=[]
        for bucket in [multi,region,ids]:
            for i in bucket:
                if i not in chosen:
                    chosen.append(i);break
            if len(chosen)==n:break
        for i in chosen:
            selected.append({'scene_key':scene,'row_index':i,'event_id':rows[i]['event_id'],
                             'instruction':rows[i]['instruction_text'],
                             'selection_tags':(['multi_reference_gt'] if len(gt[i]['mapped_referents'])>1 else ['single_or_unmapped_gt']) +
                                (['region_relation_language'] if i in region else [])})
    manifest={'seed':a.seed,'split':'existing evaluation pool (no supplied train/val/test split)',
              'selection':'scene stratified seeded sampling with multi-reference / relation coverage; no accuracy-based filtering',
              'samples':selected}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(manifest,indent=2));print(out,len(selected))

if __name__=='__main__':main()
