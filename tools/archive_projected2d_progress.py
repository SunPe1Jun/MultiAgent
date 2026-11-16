#!/usr/bin/env python3
"""Create a verifiable local progress archive without model calls or changing runs."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
TRI=ROOT.parent/'TriModal-Referring'
MODEL=ROOT.parent/'Qwen3-VL-8B-Instruct'
RESULT=ROOT/'.local-sync/2d_original_visual_results32_20261109/FINAL_VERIFIED_REPORT.json'
BASE=ROOT/'.local-sync/2d_original_wrapper_recovery_20261109'
DEFAULT=ROOT/'.local-sync/archives/20261109_projected2d_visual_progress'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(path.read_text())

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def command(args):
    r=subprocess.run(args,capture_output=True,text=True)
    return r.stdout.strip() if r.returncode==0 else None


def verify_archive(archive,external=False):
    manifest=read(archive/'ARCHIVE_MANIFEST.json');verified=0
    for item in manifest['files']:
        path=archive/item['archive_path']
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
            raise ValueError('Archive content differs: '+str(path))
        verified+=1
    references=read(archive/'REFERENCED_VISUAL_INPUTS.json')['files']
    external_count=0
    if external:
        for item in references:
            path=Path(item['path'])
            if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
                raise ValueError('Referenced visual input differs: '+str(path))
            external_count+=1
    return {'archive_files_verified':verified,'external_visual_files_verified':external_count,
            'external_visual_file_count':len(references),'archive_manifest_sha256':sha(archive/'ARCHIVE_MANIFEST.json')}


def create_archive(target):
    if target.exists():raise ValueError('Archive path exists; choose a new --output')
    report=read(RESULT);result_hash=sha(RESULT)
    if report['calls_this_iteration']!=318 or report['visual_calls_reverified']!=286:
        raise ValueError('Unexpected final experimental inventory; review before archive')
    source_files=set();external={};frozen_count=0;drift=[]
    roots=[BASE]+[Path(item['root']) for item in report['roots_verified']]
    for run in roots:
        frozen=read(run/'predictions_frozen.json')['responses']
        for name,digest in frozen.items():
            path=Path(name)
            if sha(path)!=digest:raise ValueError('Frozen response changed: '+name)
        frozen_count+=len(frozen)
        for path in run.rglob('*'):
            if path.is_file() and path.suffix in {'.json','.csv','.md','.txt'}:
                source_files.add(path)
        if run==BASE:
            for path in (run/'responses').glob('*.json'):
                row=read(path)
                for image in row['model_input_images']:
                    add_external(external,Path(image),'native_wrapper_model_image')
                for p in row['panel_meta']:
                    for field in ('source_frame_path','crop_path'):
                        if p.get(field):add_external(external,Path(p[field]),'native_'+field)
        else:
            for event in read(run/'input_manifest.json')['events']:
                for image in event['images']:
                    add_external(external,Path(image['path']),'intervention_model_image',image['sha256'])
            image_audit=run/'image_audit.json'
            if image_audit.is_file():
                for panel in read(image_audit).get('panels',[]):
                    if panel.get('heatmap'):
                        add_external(external,Path(panel['heatmap']),'gaze_history_map',panel['heatmap_sha256'])
        for filename,digest in read(run/'protocol_lock.json').get('files',{}).items():
            path=Path(filename)
            if path.suffix=='.py' and path.is_file():source_files.add(path)
            current=sha(path)
            if current!=digest:
                drift.append({'run':str(run),'source':filename,'frozen_sha256':digest,'current_sha256':current})
    if frozen_count!=318:raise ValueError(f'Expected 318 preserved responses, got {frozen_count}')
    for directory,extension in [(ROOT/'docs','.md'),(ROOT/'tools','.py'),(ROOT/'MultiAgent','.py')]:
        source_files.update(p for p in directory.glob('*'+extension) if p.is_file())
    for path in ROOT.glob('requirements*.txt'):source_files.add(path)
    for name in ('README.md','pyproject.toml','setup.py','pytest.ini'):
        if (ROOT/name).is_file():source_files.add(ROOT/name)
    for name in ('config.json','generation_config.json','preprocessor_config.json','processor_config.json',
                 'tokenizer_config.json','chat_template.jinja','model.safetensors.index.json'):
        if (MODEL/name).is_file():source_files.add(MODEL/name)
    for path in RESULT.parent.rglob('*'):
        if path.is_file() and path.suffix in {'.json','.csv'}:source_files.add(path)
    old_audit=ROOT/'.local-sync/projected2d_baseline_audit_20261109'
    source_files.update(p for p in old_audit.rglob('*') if p.is_file() and p.suffix in {'.json','.csv'})
    sample_plan=ROOT/'.local-sync/referent_frame_choice_audit_20261109/input_manifest.json'
    source_files.add(sample_plan)
    gaze_inventory=ROOT/'.local-sync/2d_original_gaze_cache32_20261109/input_image_audit.json'
    source_files.add(gaze_inventory)
    target.mkdir(parents=True)
    copied=[]
    for path in sorted(source_files):
        relative=archive_name(path);destination=target/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        before=sha(path);shutil.copyfile(path,destination)
        if sha(destination)!=before or sha(path)!=before:raise ValueError('File changed during snapshot: '+str(path))
        copied.append({'archive_path':str(relative),'source_path':str(path),'sha256':before,'bytes':destination.stat().st_size})
    if sha(RESULT)!=result_hash:raise ValueError('Authoritative final report changed while archiving')
    save(target/'REFERENCED_VISUAL_INPUTS.json',{
        'policy':'Original images and gaze arrays remain at their existing locations. This archive hashes and references them; it does not duplicate all RGB/NPZ, videos or model weights.',
        'files':[dict(path=name,**info) for name,info in sorted(external.items())]})
    dev_ids={e['id'] for e in read(sample_plan)['events']}
    native_detail=read_csv(BASE/'evaluation_all/2d_eval_detail.csv')
    detail={(x['scene'],int(x['row_index'])):x for x in native_detail}
    by_event={}
    for row in read_csv(BASE/'manifest.csv'):
        key=(row['scene'],int(row['row_index']))
        e=by_event.setdefault(key,{'id':f'{key[0]}_{key[1]}','scene':key[0],'row_index':key[1],
            'instruction':row['instruction'],'event_id':row['event_id'],'panels':{},'annotated_referents':set()})
        if row.get('referent_name'):e['annotated_referents'].add(row['referent_name'])
        if row.get('panel_id'):
            e['panels'][row['panel_id']]={'frame_id':row['panel_id'],'video_time_seconds':float(row['video_frame_time']),
                                        'source_frame_path':row['frame_path']}
    inventory=[]
    for key,e in sorted(by_event.items()):
        e['annotated_physical_count']=len(e.pop('annotated_referents'))
        e['panels']=list(e['panels'].values())
        e['partition']='dev10' if e['id'] in dev_ids else 'confirm22_previously_seen'
        e['evaluable_gt_count']=int(detail[key]['gt_referent_count'])
        inventory.append(e)
    if len(inventory)!=32 or sum(len(e['panels']) for e in inventory)!=95:
        raise ValueError('Sample inventory differs from frozen original experiment')
    save(target/'SAMPLE_INVENTORY.json',{'events':inventory,'independent_holdout':False,
        'counts':{'interactions':32,'panels':95,'dev10':sum(x['partition']=='dev10' for x in inventory),
                  'confirm22':sum(x['partition']!='dev10' for x in inventory)}})
    sources=[('OLD_RELEASED',report['old_released_reference']),('NATIVE_ORIGINAL',report['native_baseline'])]
    sources.extend(report['conditions'].items())
    with (target/'RESULTS_TABLE.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['condition','events','time_f1','point100_f1','joint100_f1','joint_tp100','exact100','evaluable_events'])
        w.writeheader()
        for name,item in sources:
            m=item['metrics'];ex=item['exact']['joint_exact_100']
            w.writerow({'condition':name,'events':m['events'],'time_f1':m['time_f1'],'point100_f1':m['point_100_f1'],
                        'joint100_f1':m['joint_100_f1'],'joint_tp100':m['joint_tp_100'],
                        'exact100':ex['exact'],'evaluable_events':ex['evaluable_events']})
    packages={}
    for name in ('torch','transformers','numpy','scipy','Pillow','accelerate'):
        try:packages[name]=metadata.version(name)
        except metadata.PackageNotFoundError:packages[name]=None
    save(target/'ENVIRONMENT.json',{'python':sys.version,'executable':sys.executable,'packages':packages,
        'workspace':str(ROOT),'repository_head':command(['git','-C',str(ROOT),'rev-parse','HEAD']),
        'repository_status':'Snapshot includes tracked and untracked source files. No Git commit is created.',
        'original_repository_head':command(['git','-C',str(TRI),'rev-parse','HEAD']),
        'setup_script':'/ai/data/TriModal-Referring/.local-sync/reconstructed_pilot100/env.sh'})
    save(target/'SOURCE_LOCK_DRIFT.json',{'scope':'Current source snapshot; it cannot reconstruct already overwritten historical preparation helper versions.',
        'unchanged_frozen_outputs_verified':True,'historical_source_drift':drift,
        'interpretation':'Preserve original protocol locks and disclose differences. Model inputs and response/tensor evidence are archived independently.'})
    save(target/'RESUME_STATE.json',{
        'experiment_date':'2026-11-09','timezone':'Asia/Hong_Kong','objective':'Improve original projected-2D visual grounding using gaze heatmap and foveation while retaining a matched original baseline.',
        'best_current':'HEAT_CROP','point100_f1':report['conditions']['HEAT_CROP']['metrics']['point_100_f1'],
        'joint100_f1':report['conditions']['HEAT_CROP']['metrics']['joint_100_f1'],'joint_exact100':'6/30',
        'sample':'32 previously seen interactions; not full4000 or heldout',
        'task_contract':report['task_protocol'],
        'unresolved':['Current native wrapper does not exactly reproduce older released predictions.',
            'Independent contribution of history heatmap relative to crop scale/mirror not established.',
            'Tested heatmap+foveation combinations lose performance relative to heat-crop alone.',
            'Gaze/GT projection remains a geometric proxy without complete absolute calibration/occlusion validation.',
            'Per-mention single-main-image protocol is a different experiment; do not mix its metrics.'],
        'next_steps':['Freeze truly new interactions and matched original-wrapper/candidate/control inputs before scoring.',
            'Use HEAT_CROP as incumbent and preserve original paired full-RGB plus local details.',
            'Test foveation changes singly with source-detail and aperture evidence, instead of replacing baseline task.',
            'Use target GT only for original expected_count parity and frozen evaluation; never use GT positions to render attention.'],
        'authoritative_report_archive_path':str(archive_name(RESULT)),
        'frozen_responses_verified':frozen_count,'new_model_calls_for_archiving':0})
    (target/'README.md').write_text('''# 2026-11-09 Projected-2D 实验进度档案

入口：`RESUME_STATE.json`（恢复上下文）、`RESULTS_TABLE.csv`（最终分数）、`SAMPLE_INVENTORY.json`（32条/95帧与分组）。正式实验文档与最终机器报告都在 `snapshot/MultiAgent/` 下按原相对路径保存。

已复制当前代码、文档、各run的清单/配置/冻结响应/trace JSON和评分；源码为归档时快照，历史准备脚本漂移见 `SOURCE_LOCK_DRIFT.json`。318条冻结响应哈希逐一验证，归档没有新增模型调用。

原RGB、gaze NPZ、视频和模型权重仍在原目录；`REFERENCED_VISUAL_INPUTS.json` 保存图片/热图哈希。将本目录移到另一机器不等于已经迁移这些大文件。需要原环境和数据才能重新推理。

校验复制文件：

```bash
python tools/archive_projected2d_progress.py verify --output ARCHIVE_DIRECTORY
```

同时校验原位置图片/热图可加 `--external`。正式文件清单与SHA-256在 `ARCHIVE_MANIFEST.json`；其自身哈希在 `ARCHIVE_MANIFEST.sha256`。
''')
    # Include generated index files in the checksum inventory; avoid self-reference.
    for p in sorted(target.glob('*')):
        if p.is_file():copied.append({'archive_path':p.name,'source_path':None,'sha256':sha(p),'bytes':p.stat().st_size})
    timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    save(target/'ARCHIVE_MANIFEST.json',{'archive_version':1,'experiment_date':'2026-11-09',
        'created_at':timestamp,'timezone':'Asia/Hong_Kong','authoritative_report_source_sha256':result_hash,
        'frozen_response_count_verified':frozen_count,'new_model_calls':0,
        'files':copied,'bytes':sum(x['bytes'] for x in copied),
        'external_visual_file_count':len(external),'historical_source_lock_drift_records':len(drift)})
    (target/'ARCHIVE_MANIFEST.sha256').write_text(sha(target/'ARCHIVE_MANIFEST.json')+'  ARCHIVE_MANIFEST.json\n')
    result=verify_archive(target,external=False)
    save(target/'VERIFICATION.json',result)
    print(json.dumps({'archive':str(target),'archived_bytes':sum(x['bytes'] for x in copied),
                      'frozen_responses_verified':frozen_count,**result},ensure_ascii=False,indent=2),flush=True)


def add_external(items,path,role,expected=None):
    name=str(path)
    if name in items:
        if role not in items[name]['roles']:items[name]['roles'].append(role)
        if expected and items[name]['sha256']!=expected:raise ValueError('Conflicting input hash: '+name)
        return
    digest=sha(path)
    if expected and digest!=expected:raise ValueError('Referenced visual input changed: '+name)
    items[name]={'sha256':digest,'bytes':path.stat().st_size,'roles':[role]}


def archive_name(path):
    path=path.absolute()
    for base,prefix in [(ROOT,'snapshot/MultiAgent'),(TRI,'snapshot/TriModal-Referring'),(MODEL,'snapshot/model_config/Qwen3-VL-8B-Instruct')]:
        try:return Path(prefix)/path.relative_to(base)
        except ValueError:pass
    raise ValueError('Unexpected archive source outside known roots: '+str(path))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['create','verify'])
    p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--external',action='store_true');a=p.parse_args()
    if a.stage=='create':create_archive(a.output.resolve())
    else:print(json.dumps(verify_archive(a.output.resolve(),external=a.external),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
