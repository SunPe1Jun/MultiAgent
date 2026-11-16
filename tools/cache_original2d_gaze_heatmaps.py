#!/usr/bin/env python3
"""Cache label-free gaze history at the original Experiment 2 panel times."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from MultiAgent.attention_volume import build_volume
from MultiAgent.camera_geometry import CameraPoseTimeline, aligned_frame_geometry
from MultiAgent.context import vrtriref_sample_to_agent_context
from MultiAgent.frame_pose_alignment import source_frame_pose
from MultiAgent.gaze_evidence import gaze_records
from MultiAgent.head_motion_diagnostics import camera_records,probe_pts,verify_saved_frame_pts
from MultiAgent.projected2d_attention_study import read_csv,save_json,sha_file
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'.local-sync/2d_original_wrapper_recovery_20261109'
OLD=ROOT/'.local-sync/2d_heatmap_foveation_recovery_20261109'
OUT=ROOT/'.local-sync/2d_original_gaze_cache32_20261109'

def main():
 if (OUT/'input_image_audit.json').exists():
  raise ValueError('Refuse overwrite completed heatmap cache')
 prior=ROOT/'.local-sync/referent_rgb_expansion32_20261109_repaired'
 original=ROOT/'.local-sync/referent_rgb_expansion32_20261109'
 sources=json.loads((ROOT/'.local-sync/projected2d_main_rgb32_20261109/input_manifest.json').read_text())['events']
 existing={(x['event'],x['frame_id']):x for x in json.loads((OLD/'input_image_audit.json').read_text())['panels']}
 rows={(r['scene'],int(r['row_index']),r['panel_id']):r for r in read_csv(BASE/'manifest.csv') if r['panel_id']}
 audits=[]
 for e in sources:
  scene=e['scene_key'];idx=e['row_index']
  if all((e['id'],p['frame_id']) in existing for p in e['panels']):
   for p in e['panels']:
    row=existing[(e['id'],p['frame_id'])]
    if sha_file(Path(row['heatmap_path']))!=row['heatmap_sha256']:raise ValueError('Old heatmap changed')
    audits.append(row)
   continue
  cached=prior/'inputs'/scene/f'{idx}_row.json';contract=original/'pose_alignment'/scene/str(idx)/'frame_pose_contract.json'
  ctx=vrtriref_sample_to_agent_context(json.loads(cached.read_text())['row'],load_ground_truth=False)
  ct=json.loads(contract.read_text());assert ct['source_hashes']=={'video_sha256':sha_file(Path(ctx.video_path)),'telemetry_sha256':sha_file(Path(ctx.json_path))}
  timeline=CameraPoseTimeline(camera_records(ctx));records=gaze_records(ctx);pts,_=probe_pts(ctx.video_path,'ffprobe')
  for p in e['panels']:
   r=rows[(scene,idx,p['frame_id'])];requested=float(r['video_frame_time']);s=sha_file(Path(p['source_path']));assert s==p['source_sha256']
   verified=verify_saved_frame_pts(ctx.video_path,{'video_time_seconds':requested,'source_sha256':s},pts,'ffmpeg')
   camera=source_frame_pose(timeline,verified,ct['clock'],allow_candidate=True)
   g=aligned_frame_geometry({'frame_time':verified,'source_sha256':s,'attention_camera':camera},s,requested,float(r['video_time_offset_seconds'] or 0))
   with Image.open(p['inputs']['MAIN_RGB_768']['path']) as im:size=im.size
   maps,stats=build_volume(records,query=g['telemetry_time_seconds'],pose=g['reference_pose'],image_size=size,clock_supported=g['clock_available'],history=2.,bins=3,sigma_degrees=2.)
   folder=OUT/'maps'/e['id'];folder.mkdir(parents=True,exist_ok=True);path=folder/(p['frame_id']+'.npz');np.savez_compressed(path,maps=maps)
   audits.append({'event':e['id'],'frame_id':p['frame_id'],'source_sha256':s,'video_time':g['video_time_seconds'],
     'clock_supported':g['clock_available'],'heatmap_path':str(path),'heatmap_sha256':sha_file(path),
     'phase_support_seconds':[x['duration'] for x in stats['bins']],
     'fov_degrees':(g['reference_pose'] or {}).get('fov_vertical_degrees',60.),'history_seconds':2.})
  print('HEATMAP',e['id'],len(audits),flush=True)
 save_json(OUT/'input_image_audit.json',{'panels':audits,'gaze_only_no_target_annotations':True,'source_manifest_sha256':sha_file(BASE/'manifest.csv')})
 print('CACHED',len(audits),'panels',flush=True)
if __name__=='__main__':main()
