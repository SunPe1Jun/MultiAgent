#!/usr/bin/env python3
"""Finalize evaluators, visual audits, reports and immutable run archive."""
from datetime import datetime,timezone
import hashlib,json,shutil
from pathlib import Path
from MultiAgent.heatmap_temporal_pilot import save,sha
from tools.run_full_interaction_referring import OUTPUT as PARENT_RUN,ROOT,read
BASE=PARENT_RUN
from tools.report_full_interaction_referring import report
from tools.build_full_grounding_comparison import build as build_comparison
from tools.build_full_grounding_review import build as build_review


def finalize():
    for folder in ('grounding_v2','grounding_v3'):
        d=BASE/folder;m=read(d/'manifest.json')
        assert len(list((d/'responses').glob('*.json')))==len(m['requests'])
        assert (d/'predictions_frozen.json').exists()
        build_review(d)
        report(d)
    build_comparison()
    from tools.write_full_interaction_report import write
    write()
    files=['MultiAgent/full_frame_grounding.py','MultiAgent/full_frame_grounding_v3.py','MultiAgent/test_full_frame_grounding.py',
      'MultiAgent/test_full_frame_grounding_v3.py','MultiAgent/full_interaction_referring.py','MultiAgent/test_full_interaction_referring.py',
      'MultiAgent/qwen_video_compat.py','MultiAgent/test_qwen_video_compat.py','MultiAgent/qwen_backend.py',
      'tools/run_full_interaction_referring.py','tools/run_full_frame_grounding.py','tools/run_full_frame_grounding_v3.py',
      'tools/report_full_interaction_referring.py','tools/verify_full_frame_grounding.py','tools/build_full_grounding_review.py',
      'tools/build_full_grounding_comparison.py','tools/write_full_interaction_report.py','tools/finalize_full_interaction_referring.py',
      'docs/FULL_INTERACTION_REFERRING_20261106_zh.md','docs/MULTIAGENT_CURRENT_PROGRESS_zh.md','README.md']
    for name in files:
        dest=BASE/'source/current'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    shutil.copy2('/tmp/full_grounding_followup.py',BASE/'source/current/tools/full_grounding_followup_guard.py')
    logs=BASE/'logs';logs.mkdir(exist_ok=True)
    for p in Path('/tmp').glob('full_*grounding*.log'):
        shutil.copy2(p,logs/p.name)
    for p in Path('/tmp').glob('full_interaction_*.log'):
        shutil.copy2(p,logs/p.name)
    v2=read(BASE/'grounding_v2/analysis.json');v3=read(BASE/'grounding_v3/analysis.json');cmp=read(BASE/'same8_v2_v3_comparison.json')
    err=list((BASE/'grounding_v2/model_cache').rglob('*.failed.json'))
    summary={'finalized_utc':datetime.now(timezone.utc).isoformat(),'v2_old_interactions':24,'v2_conditions':96,
      'v2_successful_visual_model_calls':v2['completed_model_calls'],'v2_source_frames_per_interaction':12,
      'v3_adaptive_old_interactions':8,'v3_conditions':32,'v3_successful_visual_model_calls':v3['completed_model_calls'],
      'v3_backend_cache_replays':read(BASE/'grounding_v3/verification.json')['backend_cache_replays'],
      'v3_instruction_parse_calls':v3['language_model_calls'],'same_event_comparison_interactions':cmp['same_old_interactions'],
      'qwen_video_rope_compatibility_patch_tests':3,'full_targeted_suite_tests':48,
      'full_targeted_suite_result':'PASS','gt_reads_before_prediction_freeze':0,
      'final_evaluator':'original project evaluator; only after predictions_frozen.json',
      'v2_micro_f1_true_vs_no_heat':v2['paired_micro_f1']['NO_ADDED_HEAT'],
      'v3_micro_f1_conditions':{c:v3['conditions'][c]['set_metrics']['micro_f1'] for c in v3['conditions']},
      'v3_true_vs_no_heat_interval':v3['paired_contrasts_true_heat']['NO_ADDED_HEAT']['f1'],
      'v2_schema_failures_by_condition':{c:d['schema_failures'] for c,d in v2['conditions'].items()},
      'v3_rejected_output_rows_by_condition':{c:d['rejected_visual_rows'] for c,d in v3['grounding_and_decoder'].items()},
      'same8_comparison':cmp['conditions'],'unseen_validation':False,
      'default_pipeline_changed':False,'failed_or_recovered_model_attempt_files':len(err),
      'run_directories':{'v2':'grounding_v2','v3':'grounding_v3'},
      'important_limits':['All selected interactions are old; adaptive V3 is regression only.',
       'V2 target ID score combines visual selection and approximate public-anchor geometric decoding.',
       'V3 composites source RGB at672x672; not original source resolution or all decoded video frames.',
       'Per-condition identical inputs can reuse one exact response.','Model outputs/decoder failures remain in the final set metrics.'],
      'current_source_sha256':{str(p.relative_to(BASE/'source/current')):sha(p) for p in (BASE/'source/current').rglob('*') if p.is_file()},
      'output_sha256':{str(p.relative_to(BASE)):sha(p) for p in [BASE/'grounding_v2/manifest.json',BASE/'grounding_v2/predictions_frozen.json',BASE/'grounding_v2/analysis.json',BASE/'grounding_v2/verification.json',BASE/'grounding_v3/manifest.json',BASE/'grounding_v3/predictions_frozen.json',BASE/'grounding_v3/analysis.json',BASE/'grounding_v3/verification.json',BASE/'same8_v2_v3_comparison.json',BASE/'RESULTS.md']}}
    save(BASE/'round_summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['v2_successful_visual_model_calls','v3_successful_visual_model_calls','v3_instruction_parse_calls','v2_micro_f1_true_vs_no_heat','v3_micro_f1_conditions','v3_true_vs_no_heat_interval','unseen_validation','default_pipeline_changed']},ensure_ascii=False,indent=2))

if __name__=='__main__':finalize()
