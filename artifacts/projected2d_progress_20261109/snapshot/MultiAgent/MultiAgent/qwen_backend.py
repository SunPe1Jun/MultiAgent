"""Local Qwen3-VL calls with input-addressed cache and complete prompt provenance."""
import base64
import hashlib
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from .baseline import WORKSPACE_ROOT, module


def _image_content_with_captions(images, captions=None):
    """Attach frame identities as text without drawing over RGB pixels."""
    if captions is None:
        return list(images)
    if len(captions) != len(images) or any(not isinstance(c, str) or not c.strip() for c in captions):
        raise ValueError('Provide one nonempty text caption per image')
    return [part for caption, image in zip(captions, images)
            for part in ({'type': 'text', 'text': caption}, image)]


class QwenBackend:
    def __init__(self, cache_dir, model=str(WORKSPACE_ROOT / 'Qwen3-VL-30B-A3B-Instruct'),
                 *, base_url=None, api_key=None, timeout=180, namespace='remote8b_20261021',
                 local_only=False, video_max_pixels=8_388_608,
                 trace_visual_inputs=False):
        self.cache = Path(cache_dir)
        self.model_path = model
        self.base_url = ('' if local_only else (base_url or os.environ.get('VRTRIREF_BASE_URL', ''))).rstrip('/')
        if self.base_url and not self.base_url.endswith('/v1'):
            self.base_url += '/v1'
        self.api_key = api_key or os.environ.get('VRTRIREF_API_KEY', 'EMPTY')
        self.timeout = timeout
        self.namespace = namespace
        self.runner = None
        self.local_video_max_frames = 32
        self.video_max_pixels = int(video_max_pixels)
        self.trace_visual_inputs = bool(trace_visual_inputs)
        self.requests = []
        self.call_records = []

    def _record_call(self, record):
        self.call_records.append(record)
        log = self.cache / self.namespace / 'call_audit.jsonl'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False)+'\n')

    def load(self):
        if self.runner is None:
            local = module('local')
            self.runner = local.LocalQwen3VLRunner(self.model_path, 'bfloat16', False, 2048, True,
                                                  'video', self.local_video_max_frames, None, None, 'full')
            self.runner.load()

    def ask_video(self, key, stage, prompt, frame_paths, frame_metadata=(), *,
                  max_new_tokens=1536, video_path=None, video_time_range=None,
                  video_max_pixels=None):
        """Run one local Transformers Qwen3-VL request over an ordered frame video.

        The existing remote/image request path remains unchanged.  This method is
        deliberately local-only and stores frame hashes plus the exact prompt so
        a visual result can be audited without exposing the model input paths to
        an HTTP service.
        """
        if self.base_url:
            raise RuntimeError('ask_video requires the local backend; remote URL is configured')
        from PIL import Image
        max_pixels = (self.video_max_pixels if video_max_pixels is None
                      else int(video_max_pixels))
        if max_pixels < 4096:
            raise ValueError('Video pixel budget must be at least 4096')

        paths = [Path(p) for p in frame_paths]
        if not paths:
            raise ValueError('ask_video requires at least one frame')
        images = []
        image_records = []
        for path in paths:
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            with Image.open(path) as image:
                width, height = image.size
            image_records.append({'path': str(path), 'sha256': digest, 'bytes': len(raw),
                                  'width': width, 'height': height})

        system = 'Return only valid compact JSON. Base claims on the supplied evidence; mark missing evidence explicitly.'
        import transformers
        config_path = Path(self.model_path) / 'config.json'
        model_config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.is_file() else None
        payload_config = {
            'backend': 'local_transformers_video',
            'model': self.model_path,
            'namespace': self.namespace,
            'max_new_tokens': max_new_tokens,
            'video_path': str(video_path) if video_path else None,
            'video_time_range': list(video_time_range) if video_time_range else None,
            'frame_metadata': list(frame_metadata),
            'images': image_records,
            'system_prompt': system,
            'prompt': prompt,
            'processor_contract': 'qwen3vl_video_v2_no_resample_timestamp_ticks',
            'video_rope_compat_contract': 'qwen3vl_timestamp_video_rope_grid_split_v1',
            'video_processor_size': {'shortest_edge': 4096,
                                     'longest_edge': max_pixels},
            'transformers_version': transformers.__version__,
            'model_config_sha256': model_config_sha256,
        }
        if self.trace_visual_inputs:
            payload_config['visual_tensor_trace_contract'] = 'processor_and_vision_forward_v1'
        digest = hashlib.sha256(json.dumps(payload_config, sort_keys=True).encode()).hexdigest()
        path = self.cache / self.namespace / key / f'{stage}_{digest[:20]}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        self.requests.append(str(path))
        if path.exists():
            saved = json.loads(path.read_text())
            self._record_call({'key': key, 'stage': stage, 'cache_hit': True,
                'backend': 'local_transformers_video', 'request_path': str(path),
                'frame_count': len(paths), 'frames': image_records,
                'frame_metadata': list(frame_metadata), 'elapsed_seconds': 0.,
                'original_elapsed_seconds': saved.get('elapsed_seconds'),
                'parse_warning': saved.get('parse_warning')})
            return saved['parsed']

        path.with_suffix('.request.json').write_text(
            json.dumps(payload_config, ensure_ascii=False, indent=2))
        started = time.monotonic()
        response, processor_audit, inference_seconds = None, None, None
        visual_trace = None
        try:
            load_started = time.monotonic()
            self.load()
            load_seconds = time.monotonic() - load_started
            runner = self.runner
            from .qwen_video_compat import ensure_timestamp_video_rope
            video_rope_compatibility = ensure_timestamp_video_rope(runner.model)
            messages = [{'role': 'system', 'content': [{'type': 'text', 'text': system}]},
                        {'role': 'user', 'content': [
                            {'type': 'video'}, {'type': 'text', 'text': prompt}]}]
            chat_text = runner.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            for frame_path in paths:
                images.append(Image.open(frame_path).convert('RGB'))
            from transformers.video_utils import VideoMetadata
            if len(frame_metadata) != len(images):
                raise ValueError('frame_metadata length must equal frame count')
            absolute_times = [float(row.get('time_seconds', 0.0)) for row in frame_metadata]
            if any(b < a for a, b in zip(absolute_times, absolute_times[1:])):
                raise ValueError('frame_metadata timestamps must be chronological')
            time_origin = absolute_times[0] if absolute_times else 0.0
            relative_times = [max(0.0, value - time_origin) for value in absolute_times]
            temporal_patch_size = int(getattr(runner.processor.video_processor, 'temporal_patch_size', 2))
            if len(images) % temporal_patch_size:
                raise ValueError(f'Frame count {len(images)} must be divisible by temporal patch size {temporal_patch_size}')
            timestamp_ticks = [int(round(value * 1_000_000)) for value in relative_times]
            sampled_fps = 1_000_000.0
            metadata = VideoMetadata(
                total_num_frames=max(timestamp_ticks[-1] + 1, len(images)), fps=sampled_fps,
                width=image_records[0]['width'], height=image_records[0]['height'],
                duration=max(relative_times[-1], 1e-6),
                frames_indices=timestamp_ticks)
            # These are already selected event frames.  Passing metadata and
            # disabling processor sampling prevents a silent second sampling
            # pass and preserves the recorded temporal sequence.
            preprocess_started = time.monotonic()
            inputs = runner.processor(
                text=[chat_text], videos=[images],
                videos_kwargs={'video_metadata': [metadata], 'do_sample_frames': False,
                               'size': {'shortest_edge': 4096,
                                        'longest_edge': max_pixels}},
                return_tensors='pt')
            preprocess_seconds = time.monotonic() - preprocess_started
            grid = inputs.get('video_grid_thw')
            grid_rows = grid.detach().cpu().tolist() if hasattr(grid, 'detach') else grid
            merge_size = int(getattr(runner.processor.video_processor, 'merge_size', 1) or 1)
            temporal_groups = []
            for start_index in range(0, len(frame_metadata), temporal_patch_size):
                members = frame_metadata[start_index:start_index + temporal_patch_size]
                member_times = [float(row['time_seconds']) for row in members]
                group_time = sum(member_times) / len(member_times)
                temporal_groups.append({
                    'group_index': len(temporal_groups),
                    'view_id': f'G{len(temporal_groups)+1:03d}',
                    'view_type': 'qwen_temporal_patch',
                    'frame_ids': [row.get('frame_id') for row in members],
                    'source_frame_ids': [row.get('source_frame_id') for row in members],
                    'source_frame_indices': [row.get('source_frame_index') for row in members],
                    'source_times_seconds': member_times,
                    'source_time_range': [member_times[0], member_times[-1]],
                    'qwen_template_timestamp_seconds': group_time,
                    'qwen_template_timestamp_label': f'{group_time - time_origin:.1f} seconds',
                })
            processor_audit = {
                'video_rope_compatibility': video_rope_compatibility,
                'input_frame_count': len(images),
                'input_frame_ids': [row.get('frame_id') for row in frame_metadata],
                'input_source_frame_ids': [row.get('source_frame_id') for row in frame_metadata],
                'input_absolute_time_seconds': absolute_times,
                'input_relative_time_seconds': relative_times,
                'metadata_fps': sampled_fps,
                'timestamp_encoding': 'relative_pts_microsecond_ticks',
                'time_origin_seconds': time_origin,
                'metadata_duration_seconds': metadata.duration,
                'transformers_version': transformers.__version__,
                'torch_version': runner.runtime.torch.__version__,
                'model_dtype': str(next(runner.model.parameters()).dtype),
                'model_device': str(runner.device),
                'attention_implementation': getattr(runner.model.config, '_attn_implementation', None),
                'model_config_sha256': model_config_sha256,
                'gpu_name': (runner.runtime.torch.cuda.get_device_name(0)
                             if runner.runtime.torch.cuda.is_available() else None),
                'processor_sampling_disabled': True,
                'processor_resampling_performed': False,
                'qwen_temporal_patch_size': temporal_patch_size,
                'qwen_temporal_groups': temporal_groups,
                'qwen_template_group_count': len(temporal_groups),
                'qwen_template_timestamps_verified': False,
                'video_grid_thw': grid_rows,
                'processor_video_tokens_estimate': (
                    int(grid[0].prod().item()) // (merge_size * merge_size)
                    if hasattr(grid, 'numel') and grid.numel() else None),
                'processor_temporal_grid': int(grid[0][0].item()) if hasattr(grid, 'numel') and grid.numel() else None,
                'processor_temporal_padding_frames': 0,
                'processor_effective_frame_size_px': (
                    [int(grid[0][1].item()) * int(getattr(runner.processor.video_processor, 'patch_size', 16)),
                     int(grid[0][2].item()) * int(getattr(runner.processor.video_processor, 'patch_size', 16))]
                    if hasattr(grid, 'numel') and grid.numel() else None),
                'input_token_count': int(inputs['input_ids'].shape[1])
                if hasattr(inputs.get('input_ids'), 'shape') else None,
            }
            if not hasattr(grid, 'numel') or not grid.numel():
                raise ValueError('Processor returned no video_grid_thw')
            expected_temporal = len(images) // temporal_patch_size
            if int(grid[0][0].item()) != expected_temporal:
                raise ValueError(f'Processor temporal grid {int(grid[0][0].item())} != expected {expected_temporal}')
            rendered_ids = runner.processor.tokenizer.decode(
                inputs['input_ids'][0], skip_special_tokens=False)
            rendered_timestamps = re.findall(r'<(\d+(?:\.\d+)?) seconds>', rendered_ids)
            expected_labels = [f"{group['qwen_template_timestamp_seconds'] - time_origin:.1f}"
                               for group in temporal_groups]
            if rendered_timestamps != expected_labels:
                raise ValueError('Qwen chat-template timestamps do not match source-PTS temporal groups')
            processor_audit['qwen_template_timestamps_verified'] = True
            processor_audit['rendered_chat_template_timestamps'] = rendered_timestamps
            expected_video_tokens = int(grid[0].prod().item()) // (merge_size * merge_size)
            budget_tokens = max(1, max_pixels // 2048)
            if expected_video_tokens > budget_tokens * 1.12:
                raise ValueError(f'Processor video tokens {expected_video_tokens} exceed configured budget {budget_tokens}')
            video_token_id = getattr(runner.processor, 'video_token_id', None)
            actual_video_tokens = int((inputs['input_ids'] == video_token_id).sum().item()) if video_token_id is not None else None
            if actual_video_tokens is not None and actual_video_tokens != expected_video_tokens:
                raise ValueError(f'Expanded video token count {actual_video_tokens} != grid count {expected_video_tokens}')
            processor_audit['expanded_video_token_count'] = actual_video_tokens
            processor_audit['load_seconds'] = load_seconds
            processor_audit['preprocess_seconds'] = preprocess_seconds
            local = module('local')
            inputs = local.move_batch_to_device(inputs, runner.device)
            if self.trace_visual_inputs:
                from .visual_input_trace import VisualInputTrace
                visual_trace = VisualInputTrace(runner, inputs, path, kind='video')
            inference_started = time.monotonic()
            with runner.runtime.torch.inference_mode():
                output = runner.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                               do_sample=False, use_cache=True)
            inference_seconds = time.monotonic() - inference_started
            if visual_trace is not None:
                processor_audit['visual_tensor_trace'] = visual_trace.finish()
                if not processor_audit['visual_tensor_trace']['input_consumption_verified']:
                    raise ValueError('Visual tensor trace failed to verify vision input consumption')
            response = local.decode_response(
                runner.processor, local.trim_generated_ids(inputs, output))
            parsed = self._parse_response(response)
            if parsed is None:
                raise ValueError('Local video response is not a JSON object')
            elapsed = time.monotonic() - started
            saved = {**payload_config, 'response': response, 'parsed': parsed,
                     'elapsed_seconds': elapsed,
                     'processor_audit': processor_audit,
                     'inference_seconds': inference_seconds,
                     'parse_warning': parsed.get('_parse_warning')}
            path.write_text(json.dumps(saved, ensure_ascii=False, indent=2))
            self._record_call({'key': key, 'stage': stage, 'cache_hit': False,
                'backend': 'local_transformers_video', 'request_path': str(path),
                'frame_count': len(paths), 'frames': image_records,
                'frame_metadata': list(frame_metadata), 'elapsed_seconds': elapsed,
                'inference_seconds': inference_seconds,
                'processor_audit': processor_audit,
                'parse_warning': saved['parse_warning']})
            return parsed
        except Exception as exc:
            failure = {**payload_config, 'error': f'{type(exc).__name__}: {exc}',
                       'response': response, 'processor_audit': processor_audit,
                       'inference_seconds': inference_seconds,
                       'elapsed_seconds': time.monotonic() - started}
            path.with_suffix('.failed.json').write_text(
                json.dumps(failure, ensure_ascii=False, indent=2))
            self._record_call({'key': key, 'stage': stage, 'cache_hit': False,
                'backend': 'local_transformers_video', 'request_path': str(path),
                'frame_count': len(paths), 'frames': image_records,
                'frame_metadata': list(frame_metadata),
                'processor_audit': processor_audit, 'inference_seconds': inference_seconds,
                'elapsed_seconds': time.monotonic() - started, 'error': str(exc)})
            raise
        finally:
            if visual_trace is not None:
                visual_trace.close()
            for image in images:
                image.close()

    def ask(self, key, stage, prompt, image_paths=(), *, max_new_tokens=2048, image_captions=None):
        if image_captions is not None:
            _image_content_with_captions([None] * len(image_paths), image_captions)
        if self.base_url:
            return self._ask_remote(key, stage, prompt, image_paths,
                                    max_new_tokens=max_new_tokens, image_captions=image_captions)
        from PIL import Image
        image_records = []
        for filename in image_paths:
            path = Path(filename)
            raw = path.read_bytes()
            with Image.open(path) as image:
                width, height = image.size
            image_records.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                                  'bytes': len(raw), 'width': width, 'height': height})
        images_hash = [row['sha256'] for row in image_records]
        request = {'model': self.model_path, 'prompt': prompt,
                   'image_paths': [str(p) for p in image_paths], 'images': image_records,
                   'image_sha256': images_hash,
                   'max_new_tokens': max_new_tokens, 'do_sample': False}
        if image_captions is not None:
            request['image_captions'] = list(image_captions)
        if self.trace_visual_inputs:
            request['visual_tensor_trace_contract'] = 'processor_and_vision_forward_v1'
        digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]
        path = self.cache / self.namespace / key / f'{stage}_{digest}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        self.requests.append(str(path))
        if path.exists():
            saved = json.loads(path.read_text())
            self._record_call({'key': key, 'stage': stage, 'cache_hit': True,
                'backend': 'local_transformers_image', 'request_path': str(path),
                'image_count': len(image_paths), 'image_sha256': images_hash,
                'elapsed_seconds': 0., 'original_elapsed_seconds': saved.get('elapsed_seconds')})
            return saved['parsed']
        failed_path = path.with_suffix('.failed.json')
        # A prior deterministic parse failure may only be an omitted closing
        # delimiter. Reuse its raw response after the repair routine below so
        # a rerun does not spend another model call on the same input.
        if failed_path.exists():
            failed = json.loads(failed_path.read_text())
            parsed = self._parse_response(failed.get('response', ''))
            if parsed is not None:
                if parsed.get('_parse_warning') and max_new_tokens == 2048:
                    return self.ask(key, stage, prompt, image_paths, max_new_tokens=8192, image_captions=image_captions)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({**failed, 'parsed': parsed,
                                            'parse_repaired_from_failed': True},
                                           ensure_ascii=False, indent=2))
                return parsed
            if max_new_tokens == 2048:
                return self.ask(key, stage, prompt, image_paths, max_new_tokens=8192, image_captions=image_captions)
        started = time.monotonic()
        path.with_suffix('.request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2))
        self.load()
        r = self.runner
        content = _image_content_with_captions(
            [{'type': 'image'} for _ in image_paths], image_captions) + [{'type': 'text', 'text': prompt}]
        messages = [{'role': 'system', 'content': 'Return only valid compact JSON. Base claims on the supplied evidence; mark missing evidence explicitly.'},
                    {'role': 'user', 'content': content}]
        text = r.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kwargs = {'text': [text], 'return_tensors': 'pt'}
        opened_images = []
        if image_paths:
            kwargs['images'] = []
            for filename in image_paths:
                with Image.open(filename) as image:
                    converted = image.convert('RGB')
                    kwargs['images'].append(converted)
                    opened_images.append(converted)
        try:
            inputs = r.processor(**kwargs)
        finally:
            for image in opened_images:
                image.close()
        image_grid = inputs.get('image_grid_thw')
        image_grid_rows = image_grid.detach().cpu().tolist() if hasattr(image_grid, 'detach') else image_grid
        merge_size = int(getattr(r.processor.image_processor, 'merge_size', 1) or 1)
        expected_image_tokens = (sum(int(row[0]) * int(row[1]) * int(row[2])
                                     for row in image_grid_rows) // (merge_size * merge_size)
                                 if image_grid_rows else None)
        image_token_id = r.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        actual_image_tokens = (int((inputs['input_ids'] == image_token_id).sum().item())
                               if image_token_id is not None else None)
        processor_audit = {
            'image_count': len(image_paths),
            'image_grid_thw': image_grid_rows,
            'image_merge_size': merge_size,
            'processor_image_tokens_estimate': expected_image_tokens,
            'expanded_image_token_count': actual_image_tokens,
            'input_token_count': int(inputs['input_ids'].shape[1]),
            'transformers_version': __import__('transformers').__version__,
            'model_config_sha256': hashlib.sha256(
                (Path(self.model_path) / 'config.json').read_bytes()).hexdigest()
                if (Path(self.model_path) / 'config.json').is_file() else None,
        }
        if expected_image_tokens is not None and actual_image_tokens is not None \
                and expected_image_tokens != actual_image_tokens:
            raise ValueError(f'Expanded image tokens {actual_image_tokens} != grid estimate {expected_image_tokens}')
        local = module('local')
        inputs = local.move_batch_to_device(inputs, r.device)
        visual_trace = None
        trace_data = None
        try:
            if self.trace_visual_inputs and image_paths:
                from .visual_input_trace import VisualInputTrace
                visual_trace = VisualInputTrace(r, inputs, path, kind='image')
            inference_started = time.monotonic()
            with r.runtime.torch.inference_mode():
                output = r.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            inference_seconds = time.monotonic() - inference_started
        finally:
            if visual_trace is not None:
                trace_data = visual_trace.close()
        if trace_data is not None and not trace_data['input_consumption_verified']:
            raise ValueError('Visual tensor trace failed to verify vision input consumption')
        response = local.decode_response(r.processor, local.trim_generated_ids(inputs, output))
        try:
            parsed = self._parse_response(response)
            if parsed is None: raise ValueError('JSON object required')
        except Exception:
            path.parent.mkdir(parents=True, exist_ok=True)
            elapsed = time.monotonic() - started
            path.with_suffix('.failed.json').write_text(json.dumps(
                {**request, 'response': response, 'processor_audit': processor_audit,
                 'visual_tensor_trace': trace_data, 'inference_seconds': inference_seconds,
                 'elapsed_seconds': elapsed}, indent=2))
            self._record_call({'key': key, 'stage': stage, 'cache_hit': False,
                'backend': 'local_transformers_image', 'request_path': str(path),
                'image_count': len(image_paths), 'image_sha256': images_hash,
                'elapsed_seconds': elapsed, 'error': 'JSON parse failure'})
            if max_new_tokens == 2048:
                return self.ask(key, stage, prompt, image_paths, max_new_tokens=8192, image_captions=image_captions)
            raise
        path.parent.mkdir(parents=True, exist_ok=True)
        elapsed = time.monotonic() - started
        path.write_text(json.dumps({**request, 'response': response, 'parsed': parsed,
                                    'visual_tensor_trace': trace_data,
                                    'processor_audit': processor_audit,
                                    'inference_seconds': inference_seconds,
                                    'elapsed_seconds': elapsed}, ensure_ascii=False, indent=2))
        self._record_call({'key': key, 'stage': stage, 'cache_hit': False,
            'backend': 'local_transformers_image', 'request_path': str(path),
            'image_count': len(image_paths), 'image_sha256': images_hash,
            'processor_audit': processor_audit, 'inference_seconds': inference_seconds,
            'elapsed_seconds': elapsed})
        return parsed

    def check_models(self):
        request = urllib.request.Request(self.base_url + '/models',
            headers={'Authorization': 'Bearer ' + self.api_key})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        if self.model_path not in {row['id'] for row in payload.get('data', [])}:
            raise ValueError('Requested model absent from remote /models')
        return payload

    def _ask_remote(self, key, stage, prompt, image_paths, *, max_new_tokens, image_captions=None):
        from PIL import Image
        images, image_records = [], []
        for filename in image_paths:
            path = Path(filename)
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            mime = mimetypes.guess_type(path.name)[0] or 'image/jpeg'
            images.append({'type': 'image_url', 'image_url': {
                'url': 'data:' + mime + ';base64,' + base64.b64encode(raw).decode('ascii')}})
            with Image.open(path) as image:
                width, height = image.size
            image_records.append({'path': str(path), 'sha256': digest, 'bytes': len(raw),
                                  'width': width, 'height': height,
                                  'visual_tokens_estimate': ((width+31)//32)*((height+31)//32),
                                  'estimate_basis': '32px merged grid; remote resize unverified'})
        system = 'Return only valid compact JSON. Base claims on the supplied evidence; mark missing evidence explicitly.'
        payload = {'model': self.model_path, 'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': _image_content_with_captions(images, image_captions) + [{'type': 'text', 'text': prompt}]}],
            'temperature': 0, 'max_tokens': max_new_tokens, 'seed': 20261021}
        config = {'backend': 'openai_compatible_http', 'base_url': self.base_url,
                  'model': self.model_path, 'namespace': self.namespace,
                  'temperature': 0, 'seed': 20261021, 'max_tokens': max_new_tokens,
                  'system_prompt': system, 'prompt': prompt, 'images': image_records}
        if image_captions is not None:
            config['image_captions'] = list(image_captions)
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        path = self.cache / self.namespace / key / f'{stage}_{digest[:20]}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        self.requests.append(str(path))
        started = time.monotonic()
        if path.exists():
            saved = json.loads(path.read_text())
            self._record_call({'key': key, 'stage': stage, 'cache_hit': True,
                'request_path': str(path), 'image_count': len(images), 'elapsed_seconds': 0,
                'original_elapsed_seconds': saved.get('elapsed_seconds'),
                'usage': saved.get('usage'), 'images': image_records})
            return saved['parsed']
        # Store the literal wire payload (image bytes included); no auth headers or keys.
        path.with_suffix('.request.json').write_text(json.dumps(payload, ensure_ascii=False))
        error = None
        for attempt in range(2):
            attempt_start = time.monotonic()
            raw_response = ''
            try:
                request = urllib.request.Request(self.base_url + '/chat/completions',
                    data=json.dumps(payload).encode(), headers={
                        'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.api_key},
                    method='POST')
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw_response = response.read().decode('utf-8')
                envelope = json.loads(raw_response)
                content = envelope['choices'][0]['message']['content']
                parsed = self._parse_response(content)
                if parsed is None:
                    raise ValueError('Remote response is not a JSON object')
                saved = {**config, 'request_sha256': digest, 'response': content,
                    'raw_http_response': raw_response, 'parsed': parsed,
                    'usage': envelope.get('usage'), 'response_model': envelope.get('model'),
                    'elapsed_seconds': time.monotonic()-started,
                    'attempt_elapsed_seconds': time.monotonic()-attempt_start,
                    'attempt_count': attempt+1, 'parse_warning': parsed.get('_parse_warning')}
                path.write_text(json.dumps(saved, ensure_ascii=False, indent=2))
                self._record_call({'key': key, 'stage': stage, 'cache_hit': False,
                    'request_path': str(path), 'image_count': len(images), 'images': image_records,
                    'elapsed_seconds': saved['elapsed_seconds'], 'usage': saved['usage'],
                    'parse_warning': saved['parse_warning'], 'attempt_count': attempt+1})
                return parsed
            except Exception as exc:
                error = exc
                if isinstance(exc, urllib.error.HTTPError):
                    raw_response = exc.read().decode('utf-8', errors='replace')
                failure = {**config, 'error': f'{type(exc).__name__}: {exc}',
                           'raw_http_response': raw_response,
                           'elapsed_seconds': time.monotonic()-attempt_start, 'attempt': attempt+1}
                path.with_suffix(f'.attempt{attempt+1}.failed.json').write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2))
        self._record_call({'key': key, 'stage': stage, 'cache_hit': False,
            'request_path': str(path), 'image_count': len(images), 'images': image_records,
            'elapsed_seconds': time.monotonic()-started, 'error': str(error)})
        raise RuntimeError(f'Remote model call failed: {key}/{stage}: {error}') from error

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
        # Qwen3-VL sometimes emits the requested mention objects as a top-level
        # JSON array even when the prompt asks for {"mentions": [...]}. This
        # is a schema adaptation only: preserve every object and let the
        # downstream mention validator enforce IDs, candidate scope and frames.
        if stripped.startswith('['):
            try:
                rows = json.loads(stripped)
                if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
                    if any('mention_id' in row for row in rows):
                        return {'mentions': rows, '_parse_warning': 'top_level_mention_array_normalized'}
            except json.JSONDecodeError:
                pass
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
