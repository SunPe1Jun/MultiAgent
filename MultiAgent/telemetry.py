"""Strict readers for the released VR-TriRef telemetry JSON variants."""
from __future__ import annotations

import json
from pathlib import Path
import re


def repair_telemetry_json(raw: str) -> str:
    text = raw.lstrip('\ufeff').strip()
    if not text:
        raise ValueError('empty telemetry file')
    # The corpus contains a JSON array missing its final bracket and a file
    # containing a comma-separated object stream without an outer array.
    if text.startswith('{'):
        text = '[' + text + ']'
    elif text.startswith('['):
        depth = 0
        quoted = escaped = False
        for char in text:
            if quoted:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                quoted = True
            elif char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
                if depth < 0:
                    raise ValueError('unbalanced telemetry array')
        if depth == 1 and text.rstrip().endswith('}'):
            text += ']'
        elif depth != 0:
            raise ValueError(f'unbalanced telemetry delimiters: depth={depth}')
    return text


def load_telemetry(path: str | Path) -> list[dict]:
    source = Path(path)
    payload = json.loads(repair_telemetry_json(source.read_text(encoding='utf-8-sig')))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f'telemetry must be a JSON list of objects: {source}')
    return payload


def timestamp_key(value) -> str:
    return re.sub(r'\s+', '', str(value or ''))
