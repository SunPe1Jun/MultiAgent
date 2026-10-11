"""Canonical workflow launcher and evaluator helpers."""
from pathlib import Path

from .baseline import module, scene_paths


def evaluate_original(results, directory):
    ev = module('eval')
    all_rows = []
    for scene in sorted({r['scene_key'] for r in results}):
        paths = scene_paths(scene)
        aliases,_ = ev.load_anchor_alias_map(paths['anchor'])
        gt = ev.load_gt_rows(paths['gt'],aliases)
        files = [directory/scene/f"row_{r['row_index']}.json" for r in results if r['scene_key']==scene]
        rows = ev.build_eval_rows(files,gt,aliases)
        ev.write_eval_csv(directory/scene/'evaluation.csv',rows)
        ev.write_summary_json(directory/scene/'evaluation.json',ev.summarize(rows,gt))
        all_rows.extend(rows)
    pilot_gt = [{'unmapped_referents':[]} for _ in all_rows]
    summary = ev.summarize(all_rows,pilot_gt)
    summary['scope']='selected pilot rows only; original evaluator functions, GT loaded per scene'
    ev.write_summary_json(directory/'evaluation.json',summary)
    return summary


def main():
    """Forward the historical module name to the canonical v3 workflow."""
    from .run_v3 import main as v3_main
    v3_main()


if __name__=='__main__':main()
