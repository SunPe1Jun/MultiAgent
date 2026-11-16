#!/usr/bin/env python3
"""Offline SAM2 mask propagation with gaze accumulated in native source frames.

Only seed objects are propagated. Future/middle seed information makes this an
offline memory, never an online tracker claim. Redetection checks compatibility,
not true identity. The original image/PTS/seed and all failures stay explicit.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from MultiAgent.attention_volume import support_intervals,_kernel
from MultiAgent.camera_geometry import interpolate_pose
from MultiAgent.context import load_contexts
from MultiAgent.gaze_evidence import gaze_records,project_gaze
from MultiAgent.heatmap_temporal_pilot import save,sha
from MultiAgent.local_video_visual import probe_frames
from MultiAgent.object_attention import mask_attention_links,match_masks_one_to_one
from MultiAgent.perspective_attention_study import read
from MultiAgent.spherical_attention_study import AUDIT
from tools.run_object_attention_study import OUTPUT,MODELROOT

EVENTS=['scene2/411','scene3/577','scene4_room1/72','scene5/28']


def build(output,limit=None):
    root=output/'tracks';root.mkdir(exist_ok=True);m=read(output/'manifest.json')
    selection=read(AUDIT/'selection.json');keys=set(EVENTS[:limit] if limit else EVENTS)
    contexts=load_contexts({'samples':[r for r in selection['samples'] if f"{r['scene_key']}/{r['row_index']}" in keys]},load_ground_truth=False)
    contexts={f'{c.scene_key}/{c.row_index}':c for c in contexts}
    import torch
    from transformers import Sam2VideoModel,Sam2VideoProcessor
    processor=Sam2VideoProcessor.from_pretrained(MODELROOT/'sam2.1-hiera-tiny',local_files_only=True)
    model=Sam2VideoModel.from_pretrained(MODELROOT/'sam2.1-hiera-tiny',local_files_only=True).cuda().eval()
    results=[]
    for event in [e for e in m['events'] if e['event'] in keys]:
        final=root/'events'/event['event']/'tracking.json'
        if final.exists():results.append(read(final));continue
        ctx=contexts[event['event']];records=gaze_records(ctx);offset=event['frames'][1]['geometry']['selected_clock_offset_seconds']
        seed=event['frames'][1];all_frames,stream=probe_frames(ctx.video_path)
        start=event['frames'][0]['source_pts'];end=event['frames'][-1]['source_pts']
        available=[f for f in all_frames if start-1e-6<=f['pts_seconds']<=end+1e-6]
        # Sample actual PTS; add all three detector source frames exactly. No FPS conversion.
        times=np.linspace(start,end,min(48,math.ceil((end-start)*8)+1))
        chosen={min(available,key=lambda f:abs(f['pts_seconds']-t))['source_frame_index'] for t in times}
        for f in event['frames']:chosen.add(min(available,key=lambda r:abs(r['pts_seconds']-f['source_pts']))['source_frame_index'])
        native=[r for r in available if r['source_frame_index'] in chosen]
        seed_index=min(range(len(native)),key=lambda i:abs(native[i]['pts_seconds']-seed['source_pts']))
        directory=final.parent;directory.mkdir(parents=True,exist_ok=True)
        with Image.open(seed['source_path']) as im:target_width,target_height=im.size
        existing=sorted(directory.glob('frame_*.jpg'))
        if existing:
            with Image.open(existing[0]) as im:needs_resize=im.size!=(target_width,target_height)
            if needs_resize:
                debug=directory/'unscaled_debug';debug.mkdir(exist_ok=True)
                for path in existing:path.rename(debug/path.name)
        expr='+'.join(f'eq(n\\,{f["source_frame_index"]})' for f in native)
        if not (directory/f'frame_{len(native):04d}.jpg').exists():
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-threads','4','-i',ctx.video_path,'-vf',f'select={expr},scale={target_width}:{target_height}',
                '-vsync','0','-q:v','2',str(directory/'frame_%04d.jpg')],capture_output=True,check=True)
        paths=sorted(directory.glob('frame_*.jpg'))
        assert len(paths)==len(native)
        images=[Image.open(p).convert('RGB') for p in paths];h,w=images[0].height,images[0].width
        assert (w,h)==(target_width,target_height)
        # Seed boxes were measured on the same source dimensions; fail rather than silently rescale.
        with Image.open(seed['source_path']) as im:assert im.size==(w,h)
        session=processor.init_video_session(video=images,inference_device='cuda',inference_state_device='cuda',
            processing_device='cpu',video_storage_device='cpu',dtype=torch.float32)
        ids=list(range(1,len(seed['objects'])+1));boxes=[[r['box'] for r in seed['objects']]]
        processor.add_inputs_to_inference_session(session,frame_idx=seed_index,obj_ids=ids.copy(),input_boxes=boxes)
        masks_by_frame={};frame_meta={}
        with torch.inference_mode():
            for reverse in [False,True]:
                for prediction in model.propagate_in_video_iterator(session,start_frame_idx=seed_index,reverse=reverse):
                    index=int(prediction.frame_idx)
                    restored=processor.post_process_masks([prediction.pred_masks],original_sizes=[[h,w]])[0]
                    mask=restored.detach().cpu().numpy().astype(bool)
                    if mask.ndim==4:mask=mask[:,0]
                    if mask.ndim==2:mask=mask[None]
                    returned_ids=list(prediction.object_ids)
                    assert sorted(returned_ids)==ids
                    mask=mask[[returned_ids.index(i) for i in ids]]
                    masks_by_frame[index]=mask
                    frame_meta[index]={'frame_index':index,'native_source_frame':native[index],
                        'native_path':str(paths[index].relative_to(output)),'native_sha256':sha(paths[index]),
                        'object_ids':ids.copy(),'mask_areas':[int(v.sum()) for v in mask]}
                    print('TRACKED',event['event'],index,'/',len(native),flush=True)
        indices=sorted(masks_by_frame);assert indices==list(range(len(native)))
        stack=np.stack([masks_by_frame[i] for i in indices]);np.savez_compressed(directory/'track_masks.npz',masks=stack)
        save(directory/'propagation_frames.json',{'frames':[frame_meta[i] for i in indices],
            'parent_manifest_sha256':sha(output/'manifest.json'),'track_masks_sha256':sha(directory/'track_masks.npz')})
        for im in images:im.close()
        observations=[];counts=Counter();per_frame_maps=np.zeros((len(native),3,h,w),np.float32)
        t_start=start-offset;t_end=end-offset
        for item in support_intervals(records,t_end,t_end-t_start):
            record=item['record'];duration=item['weight_seconds'] if 'weight_seconds' in item else max(0.,min(t_end,item['support_interval'][1])-max(t_start,item['support_interval'][0]))
            target=record['telemetry_time_seconds']+offset;j=min(range(len(native)),key=lambda i:abs(native[i]['pts_seconds']-target))
            error=native[j]['pts_seconds']-target
            row={'source_index':record['source_index'],'gaze_time_seconds':record['telemetry_time_seconds'],'frame_index':j,
                 'duration_seconds':duration,'pts_error_seconds':error,'support_interval':item['support_interval']}
            observations.append(row)
            if abs(error)>.08:row['status']='NO_SOURCE_WITHIN_80MS';counts[row['status']]+=1;continue
            pose=interpolate_pose(records,native[j]['pts_seconds']-offset);projection=project_gaze(record,pose,w,h);row['projection']=projection
            if projection['status']!='projected_frustum_only':row['status']=projection['status'];counts[row['status']]+=1;continue
            phase=min(2,max(0,int((record['telemetry_time_seconds']-t_start)/(t_end-t_start)*3)))
            sigma=h/(2*math.tan(math.radians(pose['fov_vertical_degrees'])/2))*math.radians(2.)
            area,kernel=_kernel((h,w),projection['pixel_xy'],sigma)
            per_frame_maps[j,phase][area]+=kernel*duration
            row['status']='PROJECTED';row['phase']=phase;counts['PROJECTED']+=1
        object_mass=np.zeros((3,len(seed['objects'])));unknown=np.zeros(3);total=np.zeros(3);ambiguity=np.zeros(3);frame_links=[]
        for j in indices:
            links,_,_=mask_attention_links(per_frame_maps[j],stack[j]);object_mass+=np.array(links['mass_seconds_by_phase_object'])
            unknown+=links['unassigned_seconds'];total+=links['total_seconds'];ambiguity+=links['ambiguous_seconds'];frame_links.append(links)
        # Compare tracked masks with independently detected+segmented masks at the
        # other original frames; no IDs/GT from the language candidate anchors.
        compatibility=[]
        for f in event['frames']:
            j=min(indices,key=lambda i:abs(native[i]['pts_seconds']-f['source_pts']))
            detection_masks=np.load(output/f['npz_path'])['masks']
            matches=match_masks_one_to_one(stack[j],detection_masks,[r['category'] for r in seed['objects']],
                [r['category'] for r in f['objects']],minimum_iou=.3)
            compatibility.append({'source_frame_id':f['frame_id'],'track_frame_index':j,'is_seed':f['frame_id']==seed['frame_id'],
                'matches':matches,'track_count':len(seed['objects']),'redetected_count':len(detection_masks)})
        result={'event':event['event'],'event_id':event['id'],'parent_manifest_sha256':sha(output/'manifest.json'),
            'seed_frame_id':seed['frame_id'],'seed_frame_index':seed_index,'seed_objects':seed['objects'],'frames':[frame_meta[i] for i in indices],
            'track_masks_path':str((directory/'track_masks.npz').relative_to(output)),'track_masks_sha256':sha(directory/'track_masks.npz'),
            'video_sha256':sha(ctx.video_path),'original_video_size':[stream['width'],stream['height']],'processed_frame_size':[w,h],'source_clock_offset_seconds':offset,'gaze_status_counts':dict(counts),'gaze_observations':observations,
            'gaze_links_by_frame':frame_links,'object_mass_seconds_by_phase':object_mass.tolist(),'unassigned_seconds_by_phase':unknown.tolist(),
            'ambiguous_seconds_by_phase':ambiguity.tolist(),'total_seconds_by_phase':total.tolist(),'redetection_compatibility':compatibility,
            'scope':'Offline bidirectional propagation from middle seed. Object IDs are tracker proposals, not verified identities. Later-entering objects are not discovered.',
            'gt_loaded':False,'anchors_projected':False,'new_qwen_calls':0,'script_sha256':sha(__file__)}
        save(final,result);results.append(result)
        del session;torch.cuda.empty_cache()
    summary={'events':len(results),'frames':sum(len(r['frames']) for r in results),'seed_tracks':sum(len(r['seed_objects']) for r in results),
        'total_seconds':sum(sum(r['total_seconds_by_phase']) for r in results),'unassigned_seconds':sum(sum(r['unassigned_seconds_by_phase']) for r in results),
        'nonseed_compatible_matches':sum(len(c['matches']) for r in results for c in r['redetection_compatibility'] if not c['is_seed']),
        'nonseed_track_checks':sum(c['track_count'] for r in results for c in r['redetection_compatibility'] if not c['is_seed']),
        'new_qwen_calls':0,'identity_accuracy_available':False}
    save(root/'analysis.json',summary);save(root/'manifest.json',{'results':results,'summary':summary})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--limit',type=int)
    a=p.parse_args();build(a.output.resolve(),a.limit)
