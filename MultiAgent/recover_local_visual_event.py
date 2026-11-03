"""Recover one local-video Visual event from a cached, audited parse."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .context import load_contexts
from .local_video_visual import run_event, save
from .mention_mapping import resolve_mentions
from .qwen_backend import QwenBackend


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recover(run_dir: Path, event_id: str, cached_response: Path) -> dict:
    config = json.loads((run_dir / 'config.json').read_text())
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    ctx = next((item for item in load_contexts(manifest, load_ground_truth=False)
                if item.event_id == event_id), None)
    if ctx is None:
        raise ValueError(f'Event is not in fixed manifest: {event_id}')
    event_dir = run_dir / 'samples' / ctx.scene_key / ctx.event_id
    existing_event = event_dir / 'visual_event.json'
    if existing_event.is_file():
        previous_event = json.loads(existing_event.read_text())
        if previous_event.get('referential_recovery_source'):
            raise ValueError(f'Event already recovered: {event_id}')
    cached = json.loads(cached_response.read_text())
    raw_parse = cached.get('parsed')
    if not isinstance(raw_parse, dict) or not isinstance(raw_parse.get('mentions'), list):
        raise ValueError('Cached response does not contain a parsed mentions list')
    parse = resolve_mentions(ctx, raw_parse)
    if len(parse['mentions']) < 2:
        raise ValueError('Refusing recovery with fewer than two literal mentions')
    if any(row['text'].strip().casefold() == ctx.text.strip().casefold()
           for row in parse['mentions']):
        raise ValueError('Refusing instruction-level text as a mention')

    previous = event_dir / 'recovery' / 'invalid_attempt_20261022'
    preserved = []
    for name in ('visual_event.json', 'referential.json', 'visual_response.json',
                 'visual_m0.json'):
        source = event_dir / name
        if source.is_file():
            target = previous / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            preserved.append(str(target.relative_to(run_dir)))
            if name == 'visual_event.json':
                source.unlink()

    recovery_source = {
        'method': 'cached_qwen_referential_response_then_deterministic_scene_local_mapping',
        'cached_response': str(cached_response.resolve()),
        'cached_response_sha256': sha256(cached_response),
        'model_inference_reused': True,
        'referential_model_recalled': False,
        'invalid_previous_artifacts_preserved': preserved,
    }
    save(event_dir / 'referential_recovery.json', {
        **recovery_source,
        'raw_model_response': cached.get('response'),
        'parsed_model_mentions': raw_parse,
        'resolved_referential': parse,
    })

    backend = QwenBackend(
        config['cache_dir'], config['model'],
        namespace=config['backend_namespace'], local_only=True,
        video_max_pixels=config.get('model_video_max_pixels', 8_388_608))
    try:
        event = run_event(
            ctx, backend, run_dir,
            fps=config['fps'], min_frames=config['min_frames'], max_frames=config['max_frames'],
            referential_override=parse, referential_recovery_source=recovery_source)
    except Exception as exc:
        save(event_dir / 'event_failure.json', {
            'scene_key': ctx.scene_key, 'event_id': ctx.event_id,
            'error': f'{type(exc).__name__}: {exc}',
            'recovery_source': recovery_source,
        })
        raise
    stale_failure = event_dir / 'event_failure.json'
    if stale_failure.exists():
        stale_failure.unlink()
    if len(event['referential']['mentions']) != len(parse['mentions']):
        raise RuntimeError('Recovered event did not preserve the validated mention list')

    failures = []
    completed = 0
    for item in manifest['samples']:
        folder = run_dir / 'samples' / item['scene_key'] / item['event_id']
        event_file = folder / 'visual_event.json'
        failure_file = folder / 'event_failure.json'
        if event_file.is_file() and not failure_file.exists():
            completed += 1
        elif failure_file.is_file():
            failures.append(json.loads(failure_file.read_text()))
    audit_path = Path(config['cache_dir']) / config['backend_namespace'] / 'call_audit.jsonl'
    audit = [json.loads(line) for line in audit_path.read_text().splitlines()
             if line.strip()] if audit_path.is_file() else []
    real_requests = {row.get('request_path') for row in audit
                     if not row.get('cache_hit') and row.get('request_path')}
    progress = {
        'completed': completed, 'total': len(manifest['samples']),
        'failures': len(failures), 'event_failures': failures,
        'local_model_calls': len(real_requests),
        'cache_hits': sum(bool(row.get('cache_hit')) for row in audit),
        'last_recovered_event': f'{ctx.scene_key}/{ctx.event_id}',
        'recovery_visual_cache_hit': bool(backend.call_records and
                                          backend.call_records[-1].get('cache_hit')),
    }
    save(run_dir / 'progress.json', progress)
    return {'event_id': event_id, 'mention_count': len(parse['mentions']),
            'frame_count': event['frame_manifest']['frame_count'],
            'visual_calls': event['visual_model_calls'],
            'completed': completed, 'total': len(manifest['samples']),
            'failures': failures, 'local_model_calls': len(real_requests),
            'cache_hits': progress['cache_hits'],
            'recovery_visual_cache_hit': progress['recovery_visual_cache_hit']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--event-id', required=True)
    parser.add_argument('--cached-response', required=True)
    args = parser.parse_args()
    result = recover(Path(args.run_dir), args.event_id, Path(args.cached_response))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
