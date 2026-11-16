#!/usr/bin/env python3
"""Post-evaluation visual review: only current-event referents are projected."""
import argparse
from pathlib import Path
from MultiAgent.camera_geometry import project_world
from MultiAgent.heatmap_temporal_pilot import save
from tools.run_full_interaction_referring import read


def build(output):
    manifest=read(output/'manifest.json');analysis=read(output/'analysis.json')
    assert (output/'predictions_frozen.json').exists()
    labels={r['id']:r['gt'] for r in analysis['events']};events={}
    for event in manifest['events']:
        table={c['candidate_id']:c for c in event['scene_candidate_table']}
        frames={}
        for frame in event['frames']:
            refs=[]
            for name in labels[event['id']]:
                if name not in table:continue
                geometry=frame['geometry']
                if not geometry['clock_available']:continue
                p=project_world(table[name]['position_world'],geometry['reference_pose'],*frame['source_size'])
                if p['in_frustum']:
                    refs.append({'name':name,'normalized_xy':p['normalized_uv_unclipped'],
                        'visibility':'unknown','pixel_calibration':'unverified'})
            frames[frame['frame_id']]=refs
        events[event['id']]=frames
    save(output/'review_referent_projection.json',{'after_all_predictions_frozen':True,
        'scope':'Only current interaction GT referents, post-evaluation diagnostic. Never model input or object-pixel GT.',
        'events':events})
    print('Post-evaluation referent-only review',len(events),'interactions')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();build(a.output.resolve())
