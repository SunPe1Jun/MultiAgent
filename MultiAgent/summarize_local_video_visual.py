"""Summarize audited local video Visual runs without loading labels or GT."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def summarize(run_dir: Path, cache_dir: Path | None = None):
    config = read_json(run_dir / 'config.json') or {}
    manifest = read_json(run_dir / 'manifest.json') or {}
    progress = read_json(run_dir / 'progress.json') or {}
    events = []
    event_failures = []
    pending_events = []
    recovered_events = []
    semantic_failures = []
    ground_truth_context_loads = 0
    ground_truth_load_unknown_events = 0
    evidence_citations, observation_citations = Counter(), Counter()
    supported_without_positive = 0
    mentions = []
    instance_mapping_failures = []
    frame_counts, group_rates, visual_tokens = [], [], []
    for item in manifest.get('samples', []):
        sample_dir = run_dir / 'samples' / item['scene_key'] / item['event_id']
        event = read_json(sample_dir / 'visual_event.json')
        failure = read_json(sample_dir / 'event_failure.json')
        if failure is not None:
            event_failures.append(failure)
            continue
        if event is None:
            pending_events.append(f'{item["scene_key"]}/{item["event_id"]}')
            continue
        if 'ground_truth_loaded' in event:
            ground_truth_context_loads += int(bool(event['ground_truth_loaded']))
        else:
            ground_truth_load_unknown_events += 1
        for mention in event.get('referential', {}).get('mentions', []):
            if (str(mention.get('object_type', '')).casefold() in
                    {'instruction', 'utterance', 'sentence', 'command'}):
                semantic_failures.append({'scene_key': item['scene_key'], 'event_id': item['event_id'],
                                          'mention_id': mention.get('mention_id'),
                                          'error': 'whole instruction returned as object mention'})
        frame_manifest = event.get('frame_manifest', {})
        frame_counts.append(frame_manifest.get('frame_count', 0))
        recovery = event.get('referential_recovery_source')
        if recovery:
            recovered_events.append({
                'scene_key': item['scene_key'], 'event_id': item['event_id'],
                'method': recovery.get('method'),
                'cached_response': recovery.get('cached_response'),
                'cached_response_sha256': recovery.get('cached_response_sha256'),
                'referential_model_recalled': recovery.get('referential_model_recalled'),
                'invalid_previous_artifacts_preserved': recovery.get(
                    'invalid_previous_artifacts_preserved', []),
            })
        if isinstance(frame_manifest.get('actual_qwen_temporal_group_rate'), (float, int)):
            group_rates.append(frame_manifest['actual_qwen_temporal_group_rate'])
        for mention in event.get('visual_mentions', []):
            visual = mention.get('visual') or {}
            positive = [row for row in visual.get('evidence', [])
                        if float(row.get('support_score', 0)) * float(row.get('confidence', 0)) > 0]
            evidence_citations.update(row.get('view_id') or row.get('frame_id')
                                      for row in visual.get('evidence', []))
            observation_citations.update(row.get('view_id') or row.get('frame_id')
                                         for row in visual.get('visual_observations', []))
            if visual.get('reported_identity_status') == 'supported' and not positive:
                supported_without_positive += 1
            mentions.append({
                'scene_key': item['scene_key'],
                'event_id': item['event_id'],
                'mention_id': mention.get('mention', {}).get('mention_id'),
                'text': mention.get('mention', {}).get('text'),
                'candidate_scope': mention.get('candidate_scope'),
                'candidate_count': mention.get('scene_candidate_count'),
                'visual_status': mention.get('status'),
                'identity_status': visual.get('identity_status'),
                'positive_evidence_count': len(positive),
                'nonpositive_candidate_rows': sum(
                    1 for row in visual.get('evidence', [])
                    if float(row.get('support_score', 0)) * float(row.get('confidence', 0)) <= 0),
                'observation_count': len(visual.get('visual_observations', [])),
                'rejected_evidence_count': len(visual.get('rejected_evidence', [])),
                'failure': mention.get('error'),
            })
            if mention.get('candidate_scope') == 'mention_closed_set' and mention.get('candidate_ids'):
                if not positive and visual.get('identity_status') in (
                        'ambiguous_instances', 'no_positive_instance_support'):
                    instance_mapping_failures.append({
                        'scene_key': item['scene_key'], 'event_id': item['event_id'],
                        'mention_id': mention.get('mention', {}).get('mention_id'),
                        'text': mention.get('mention', {}).get('text'),
                        'candidate_ids': mention.get('candidate_ids'),
                        'raw_identity_status': visual.get('reported_identity_status',
                                                          visual.get('identity_status')),
                        'visual_observations': visual.get('visual_observations', []),
                    })
        events.append({
            'scene_key': item['scene_key'], 'event_id': item['event_id'],
            'frame_count': frame_manifest.get('frame_count'),
            'temporal_group_count': len(frame_manifest.get('qwen_temporal_groups', [])),
            'actual_temporal_group_rate': frame_manifest.get('actual_qwen_temporal_group_rate'),
            'target_group_rate_met': frame_manifest.get('target_group_rate_met'),
            'visual_model_calls': event.get('visual_model_calls', 0),
            'mention_count': len(event.get('visual_mentions', [])),
            'visual_failures': len(event.get('failures', [])),
            'elapsed_seconds': event.get('elapsed_seconds'),
            'referential_recovered_from_cache': bool(recovery),
            'frozen_input_source': event.get('frozen_input_source'),
            'visual_call_granularity': event.get('visual_call_granularity',
                                                 config.get('visual_call_granularity')),
        })

    audit = []
    cache_dir = cache_dir or (Path(config['cache_dir']) if config.get('cache_dir') else None)
    if cache_dir and cache_dir.is_dir():
        for log in cache_dir.rglob('call_audit.jsonl'):
            for line in log.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                audit.append(record)
    for record in audit:
        audit_grid = record.get('processor_audit') or {}
        if isinstance(audit_grid.get('processor_video_tokens_estimate'), int):
            visual_tokens.append(audit_grid['processor_video_tokens_estimate'])

    planned = len(manifest.get('samples', []))
    completed = len(events)
    processed = completed + len(event_failures)
    status = ('COMPLETE_WITH_EVENT_FAILURES' if processed == planned and event_failures
              else 'COMPLETE_WITH_SEMANTIC_FAILURES' if completed == planned and semantic_failures
              else 'COMPLETE_WITH_VISUAL_FAILURES' if completed == planned and any(
                  event['visual_failures'] for event in events)
              else 'COMPLETE' if completed == planned else 'IN_PROGRESS')
    return {
        'status': status,
        'run_dir': str(run_dir.resolve()),
        'version': config.get('version'),
        'model': config.get('model'),
        'manifest_sample_count': planned,
        'completed_events': completed,
        'event_failures': event_failures,
        'pending_events': pending_events,
        'referential_semantic_failures': semantic_failures,
        'referential_recovery_count': len(recovered_events),
        'referential_recoveries': recovered_events,
        'progress': progress,
        'offline_full_interaction': True,
        'fusion': 'NOT_RUN',
        'ground_truth_context_load_count': ground_truth_context_loads,
        'ground_truth_load_unknown_events': ground_truth_load_unknown_events,
        'ground_truth_used_for_prediction': False,
        'ground_truth_read_by_summary': False,
        'referent_accuracy_reported': False,
        'visual_model_calls': sum(event['visual_model_calls'] for event in events),
        'visual_failures': sum(event['visual_failures'] for event in events),
        'local_model_call_count': sum(not record.get('cache_hit', False) for record in audit),
        'cache_hit_count': sum(bool(record.get('cache_hit')) for record in audit),
        'visual_token_estimate_per_call': {
            'min': min(visual_tokens) if visual_tokens else None,
            'median': statistics.median(visual_tokens) if visual_tokens else None,
            'max': max(visual_tokens) if visual_tokens else None,
            'source': 'Qwen video_grid_thw product divided by merge_size squared',
        },
        'frame_count_per_event': {
            'min': min(frame_counts) if frame_counts else None,
            'median': statistics.median(frame_counts) if frame_counts else None,
            'max': max(frame_counts) if frame_counts else None,
        },
        'actual_temporal_groups_per_second': {
            'min': min(group_rates) if group_rates else None,
            'median': statistics.median(group_rates) if group_rates else None,
            'max': max(group_rates) if group_rates else None,
        },
        'candidate_scope_counts': dict(Counter(row['candidate_scope'] for row in mentions)),
        'visual_status_counts': dict(Counter(row['visual_status'] for row in mentions)),
        'identity_status_counts': dict(Counter(row['identity_status'] for row in mentions)),
        'mention_count': len(mentions),
        'candidate_instance_mapping_failures': instance_mapping_failures,
        'positive_candidate_evidence_mentions': sum(row['positive_evidence_count'] > 0 for row in mentions),
        'nonpositive_candidate_rows': sum(row['nonpositive_candidate_rows'] for row in mentions),
        'evidence_citation_view_counts': dict(evidence_citations),
        'observation_citation_view_counts': dict(observation_citations),
        'supported_without_positive_evidence_mentions': supported_without_positive,
        'observation_only_mentions': sum(row['observation_count'] > 0 for row in mentions),
        'mention_failures': [row for row in mentions if row['failure']],
        'events': events,
        'mentions': mentions,
        'call_audit': audit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--cache-dir')
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    result = summarize(run_dir, Path(args.cache_dir) if args.cache_dir else None)
    (run_dir / 'audit_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    summary = {key: result[key] for key in (
        'status', 'version', 'manifest_sample_count', 'completed_events',
        'event_failures', 'pending_events', 'referential_semantic_failures',
        'visual_model_calls', 'visual_failures',
        'mention_count', 'positive_candidate_evidence_mentions',
        'nonpositive_candidate_rows', 'candidate_scope_counts', 'identity_status_counts',
        'local_model_call_count', 'cache_hit_count', 'visual_token_estimate_per_call',
        'frame_count_per_event', 'actual_temporal_groups_per_second', 'offline_full_interaction',
        'fusion', 'ground_truth_context_load_count', 'ground_truth_load_unknown_events',
        'ground_truth_used_for_prediction',
        'ground_truth_read_by_summary', 'referent_accuracy_reported',
        'evidence_citation_view_counts', 'observation_citation_view_counts',
        'supported_without_positive_evidence_mentions',
        'referential_recovery_count', 'referential_recoveries')}
    (run_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in (
        'status', 'manifest_sample_count', 'completed_events', 'mention_count',
        'visual_model_calls', 'positive_candidate_evidence_mentions',
        'candidate_scope_counts', 'identity_status_counts')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
