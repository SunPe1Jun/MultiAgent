"""Local Qwen3-VL calls with input-addressed cache and complete prompt provenance."""
import hashlib
import json
import re
from pathlib import Path

from .baseline import module


class QwenBackend:
    def __init__(self, cache_dir, model='/workspace/usr3/Qwen3-VL-30B-A3B-Instruct'):
        self.cache = Path(cache_dir)
        self.model_path = model
        self.runner = None
        self.requests = []

    def load(self):
        if self.runner is None:
            local = module('local')
            self.runner = local.LocalQwen3VLRunner(self.model_path, 'bfloat16', False, 2048, True,
                                                  'image', 8, None, None, 'full')
            self.runner.load()

    def ask(self, key, stage, prompt, image_paths=(), *, max_new_tokens=2048):
        images_hash = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in image_paths]
        request = {'model': self.model_path, 'prompt': prompt,
                   'image_paths': [str(p) for p in image_paths], 'image_sha256': images_hash,
                   'max_new_tokens': max_new_tokens, 'do_sample': False}
        digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]
        path = self.cache / key / f'{stage}_{digest}.json'
        self.requests.append(str(path))
        if path.exists():
            return json.loads(path.read_text())['parsed']
        failed_path = path.with_suffix('.failed.json')
        # A prior deterministic parse failure may only be an omitted closing
        # delimiter. Reuse its raw response after the repair routine below so
        # a rerun does not spend another model call on the same input.
        if failed_path.exists():
            failed = json.loads(failed_path.read_text())
            parsed = self._parse_response(failed.get('response', ''))
            if parsed is not None:
                if parsed.get('_parse_warning') and max_new_tokens == 2048:
                    return self.ask(key, stage, prompt, image_paths, max_new_tokens=8192)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({**failed, 'parsed': parsed,
                                            'parse_repaired_from_failed': True},
                                           ensure_ascii=False, indent=2))
                return parsed
            if max_new_tokens == 2048:
                return self.ask(key, stage, prompt, image_paths, max_new_tokens=8192)
        self.load()
        r = self.runner
        content = [{'type': 'image'} for _ in image_paths] + [{'type': 'text', 'text': prompt}]
        messages = [{'role': 'system', 'content': 'Return only valid compact JSON. Base claims on the supplied evidence; mark missing evidence explicitly.'},
                    {'role': 'user', 'content': content}]
        text = r.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kwargs = {'text': [text], 'return_tensors': 'pt'}
        if image_paths:
            from PIL import Image
            kwargs['images'] = [Image.open(p).convert('RGB') for p in image_paths]
        inputs = r.processor(**kwargs)
        local = module('local')
        inputs = local.move_batch_to_device(inputs, r.device)
        with r.runtime.torch.inference_mode():
            output = r.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        response = local.decode_response(r.processor, local.trim_generated_ids(inputs, output))
        try:
            parsed = self._parse_response(response)
            if parsed is None: raise ValueError('JSON object required')
        except Exception:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.with_suffix('.failed.json').write_text(json.dumps({**request, 'response': response}, indent=2))
            if max_new_tokens == 2048:
                return self.ask(key, stage, prompt, image_paths, max_new_tokens=8192)
            raise
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**request, 'response': response, 'parsed': parsed}, ensure_ascii=False, indent=2))
        return parsed

    @staticmethod
    def _balance_json(text):
        """Close only unmatched JSON delimiters; never invent keys or values."""
        stack, quoted, escaped = [], False, False
        pairs = {'{': '}', '[': ']'}
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
            elif char in pairs:
                stack.append(pairs[char])
            elif char in (']', '}'):
                if not stack or stack[-1] != char:
                    return None
                stack.pop()
        if quoted:
            return None
        return text + ''.join(reversed(stack))

    @classmethod
    def _parse_response(cls, response):
        # Accept per-mention evidence groups, with optional omitted closing
        # braces. Each evidence array must independently be complete JSON.
        # No partial array, extra field, or trailing content is accepted.
        stripped = (response or '').strip()
        if stripped.startswith('```'):
            stripped = re.sub(r'^```(?:json)?\s*', '', stripped).rsplit('```', 1)[0].strip()
        # Hand cue prompts historically elicited a bare JSON array even though
        # the prompt requested an object. Normalize that safe shape so the
        # hand stage can validate the same cue rows without another inference.
        if stripped.startswith('[') and stripped.endswith(']'):
            try:
                rows = json.loads(stripped)
                if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
                    if rows and all({'frame_id', 'mention_ids', 'cue_type', 'confidence', 'explanation'} <= row.keys()
                                    and isinstance(row['mention_ids'], list) for row in rows):
                        return {'cues': rows}
            except json.JSONDecodeError:
                pass
        if stripped.startswith('[') and stripped.endswith(']'):
            remaining = stripped[1:-1].strip()
            rows = []
            while remaining:
                match = re.match(r'^\{\s*"evidence"\s*:\s*', remaining)
                if not match:
                    break
                try:
                    group, end = json.JSONDecoder().raw_decode(remaining[match.end():])
                except json.JSONDecodeError:
                    break
                if not isinstance(group, list) or not all(isinstance(row, dict) for row in group):
                    break
                rows.extend(group)
                remaining = remaining[match.end()+end:].strip()
                if remaining.startswith('}'):
                    remaining = remaining[1:].strip()
                if not remaining:
                    return {'evidence': rows}
                if not remaining.startswith(','):
                    break
                remaining = remaining[1:].strip()
                if not remaining:
                    break
            return None
        try:
            text = module('api').extract_json_object_text(response)
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            # Handle a fenced response whose outer object lost only one or
            # more closing delimiters at generation end.
            match = re.search(r'\{.*', response or '', flags=re.S)
            if not match:
                return None
            candidate = match.group(0).split('```', 1)[0].rstrip()
            repaired = cls._balance_json(candidate)
            if repaired is not None:
                try:
                    parsed = json.loads(repaired)
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    pass
        # Last-resort auditable recovery for a response truncated after one
        # or more complete evidence rows. Only rows accepted by the standard
        # JSON decoder are retained; the incomplete tail is discarded.
        evidence_start = re.search(r'\{\s*"evidence"\s*:\s*\[', response or '')
        if evidence_start is None:
            return None
        cursor, rows = evidence_start.end(), []
        decoder = json.JSONDecoder()
        while cursor < len(response):
            while cursor < len(response) and response[cursor] in ' \t\r\n,':
                cursor += 1
            if cursor >= len(response) or response[cursor] != '{':
                break
            try:
                row, end = decoder.raw_decode(response[cursor:])
            except json.JSONDecodeError:
                break
            if not isinstance(row, dict):
                break
            rows.append(row)
            cursor += end
        return {'evidence': rows, '_parse_warning': 'truncated_evidence_response'} if rows else None
