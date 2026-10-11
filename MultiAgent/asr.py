"""Offline ASR and mention-to-audio time alignment for real VR-TriRef clips."""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")

# Small, auditable ASR confusions observed in the pilot recordings.  These are
# only used for the object-head check; the original instruction remains the
# alignment reference and the result keeps its lower confidence.
HEAD_CONFUSIONS = {
    'file': {'fire'},
    'cargo': {'car', 'cart'},
    'box': {'boxes'},
    'building': {'buildings', 'warehouse'},
}


def _tokens(text: str) -> list[str]:
    return [x.lower() for x in TOKEN_RE.findall(str(text or ""))]


def _parse_dt(value: Any):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        text = value.replace("Z", "+00:00")
        # Python 3.10 accepts at most six fractional-second digits while the
        # telemetry exporter writes seven.
        text = re.sub(r"\.(\d{6})\d+(?=[+-]\d{2}:\d{2}$)", r".\1", text)
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _video_datetime(video_path: Path, reference):
    m = re.search(r"ScreenRecord_(\d{4})-(\d{2})-(\d{2})-(\d{2})_(\d{2})_(\d{2})", video_path.name)
    if not m:
        return None
    vals = [int(x) for x in m.groups()]
    return datetime(*vals, tzinfo=reference.tzinfo if reference is not None else None)


def locate_audio(video_path: str | Path) -> Path | None:
    """The released recordings keep voice_*.wav beside ScreenRecord_*.mp4."""
    path = Path(video_path)
    files = sorted(path.parent.glob("voice_*.wav"))
    return files[0] if len(files) == 1 else None


def clock_alignment(ctx) -> dict[str, Any]:
    """Return offsets from audio and telemetry clocks to the video clock.

    Audio metadata has sub-second precision; the screen-recording filename only has
    whole seconds. The result records its source and is never silently treated as
    ground truth when metadata is missing.
    """
    audio = locate_audio(ctx.video_path)
    metadata = None
    if audio is not None:
        candidates = list(audio.parent.glob(audio.name.replace('voice_', 'metadata_').replace('.wav', '.json')))
        if candidates:
            try:
                metadata = _parse_dt(json.loads(candidates[0].read_text()).get("timestamp"))
            except (OSError, ValueError, TypeError):
                metadata = None
    first = None
    for _, sample in ctx.timed_samples:
        first = _parse_dt(sample.get("timestamp"))
        if first is not None:
            break
    video_dt = _video_datetime(Path(ctx.video_path), metadata or first)
    wall_audio_to_video = (metadata - video_dt).total_seconds() if metadata and video_dt else None
    wall_audio_to_telemetry = (metadata - first).total_seconds() if metadata and first else None
    # Event/sample times are relative to the first telemetry sample, while the
    # audio metadata timestamp is the recording origin used by the released
    # frame manifests.  Therefore an event time t maps to video time
    # t + (first_telemetry - audio_metadata).  The previous implementation
    # computed this delta but discarded it, shifting telemetry-driven frames.
    audio_to_video = 0.0
    audio_to_telemetry = wall_audio_to_telemetry if wall_audio_to_telemetry is not None else 0.0
    telemetry_to_video = (-wall_audio_to_telemetry
                          if wall_audio_to_telemetry is not None else 0.0)
    calibrated = bool(metadata and first)
    return {
        "audio_path": str(audio) if audio else None,
        "audio_to_video": audio_to_video,
        "audio_to_telemetry": audio_to_telemetry,
        "telemetry_to_video": telemetry_to_video,
        "wall_audio_to_video": wall_audio_to_video,
        "wall_audio_to_telemetry": wall_audio_to_telemetry,
        "clock_available": calibrated,
        "video_clock_uncertainty_seconds": 1.0,
        "calibration_status": "metadata_origin_estimate_not_pixel_calibrated" if calibrated else "unavailable_zero_offset_fallback",
        "metadata_timestamp": metadata.isoformat() if metadata else None,
        "first_telemetry_timestamp": first.isoformat() if first else None,
        "video_filename_timestamp": video_dt.isoformat() if video_dt else None,
        "source": "voice_metadata+video_filename+first_telemetry" if metadata and video_dt and first else "fallback_unaligned",
    }


def guard_clock(clock, duration, max_offset_ratio=0.30):
    """Reject excessive audio-derived offsets; preserve raw values for audit.

    This is a reliability gate, not a new clock calibration. Rejected clocks
    must not be used to match video frames to telemetry or word timestamps.
    """
    import math
    if clock.get('audio_timing_rejected'):
        return dict(clock)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Video duration must be positive and finite')
    fields = ('telemetry_to_video', 'audio_to_video', 'wall_audio_to_video')
    offsets = {key: clock[key] for key in fields
               if isinstance(clock.get(key), (int, float))}
    bad = {key: value for key, value in offsets.items()
           if not math.isfinite(value) or abs(value) > duration * max_offset_ratio}
    out = {**clock, 'offset_guard_duration': duration,
           'max_offset_ratio': max_offset_ratio, 'audio_timing_rejected': bool(bad)}
    if bad:
        out.update(raw_clock=dict(clock), rejected_offsets=bad,
                   audio_to_video=None, audio_to_telemetry=None,
                   telemetry_to_video=None, clock_available=False,
                   calibration_status='rejected_offset_over_duration_threshold',
                   fallback_reason='audio_derived_offset_exceeds_30_percent_of_video',
                   fallback_policy='full_video_rgb_without_audio_or_telemetry_timing')
    return out


class OfflineASR:
    def __init__(self, model_path: str | Path, cache_dir: str | Path | None = None,
                 device: str = "cuda", compute_type: str = "float16", align_model=None):
        self.model_path = str(model_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device, self.compute_type = device, compute_type
        self._model = None
        self.align_model = str(align_model) if align_model else None
        self._aligner = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_path, device=self.device,
                                       compute_type=self.compute_type)
        return self._model

    def transcribe(self, audio_path: str | Path) -> dict[str, Any]:
        audio_path = Path(audio_path)
        audio_digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
        settings = {'version': 2, 'model': self.model_path, 'align_model': self.align_model,
                    'device': self.device, 'compute_type': self.compute_type, 'audio_sha256': audio_digest,
                    'beam_size': 5, 'language': 'en', 'vad_filter': True}
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
        cache = self.cache_dir / f"{audio_path.stem}_{digest}.json" if self.cache_dir else None
        if cache and cache.exists():
            return json.loads(cache.read_text())
        model = self._load()
        segments, info = model.transcribe(str(audio_path), language="en", beam_size=5,
                                          word_timestamps=True, vad_filter=True,
                                          condition_on_previous_text=False)
        result = {"audio_path": str(audio_path.resolve()), "model": self.model_path, 'settings': settings,
                  "language": getattr(info, "language", "en"), "words": [], "segments": []}
        for seg in segments:
            words = []
            for word in (seg.words or []):
                item = {"word": word.word, "start": float(word.start), "end": float(word.end),
                        "probability": float(getattr(word, "probability", 0.0) or 0.0)}
                words.append(item); result["words"].append(item)
            result["segments"].append({"start": float(seg.start), "end": float(seg.end),
                                       "text": seg.text, "words": words})
        if self.align_model and result['words']:
            if self._aligner is None:
                self._aligner = WordAligner(self.align_model, self.device)
            result = self._aligner.refine(audio_path, result)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result


class WordAligner:
    """Use TorchAudio's CTC forced alignment on the recognized transcript.

    Aligning the reference instruction could fabricate timestamps for words that
    were never spoken, so only the actual ASR transcript is aligned.
    """
    def __init__(self, model_path, device):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.torch = torch
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_path, local_files_only=True)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path, local_files_only=True).to(device).eval()

    def refine(self, audio_path, result):
        import numpy as np
        import subprocess
        import torchaudio.functional as audio_ops
        torch = self.torch
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(audio_path), '-ac', '1',
                              '-ar', '16000', '-f', 'f32le', 'pipe:1'], capture_output=True, check=True).stdout
        waveform = np.frombuffer(raw, dtype=np.float32).copy()
        text = ' '.join(''.join(_tokens(w['word'])).upper() for w in result['words'])
        ids = self.processor.tokenizer(text, add_special_tokens=False).input_ids
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            emissions = self.model(**inputs).logits.log_softmax(-1).cpu()
        paths, scores = audio_ops.forced_align(emissions, torch.tensor([ids], dtype=torch.int64),
                                               blank=self.processor.tokenizer.pad_token_id)
        spans = audio_ops.merge_tokens(paths[0], scores[0].exp(), blank=self.processor.tokenizer.pad_token_id)
        scale = len(waveform) / 16000 / emissions.shape[1]
        cursor = 0
        refined = []
        for word in result['words']:
            n = len(self.processor.tokenizer(''.join(_tokens(word['word'])).upper(), add_special_tokens=False).input_ids)
            pieces = spans[cursor:cursor+n]
            if len(pieces) != n or not pieces:
                raise ValueError('Forced alignment token count mismatch')
            refined.append({**word, 'asr_start': word['start'], 'asr_end': word['end'],
                            'start': float(pieces[0].start * scale), 'end': float(pieces[-1].end * scale),
                            'alignment_score': float(sum(float(s.score) for s in pieces)/n)})
            cursor += n + 1  # The tokenizer's pipe token separates words.
        return {**result, 'words': refined, 'timestamp_source': 'wav2vec2_ctc_forced_alignment',
                'align_model': str(self.model.config._name_or_path)}


def match_mention(mention_text: str, words: list[dict[str, Any]], instruction='',
                  object_type='', occurrence=0) -> dict[str, Any]:
    target = _tokens(mention_text)
    observed = [_tokens(w.get("word", "")) for w in words]
    observed = [x[0] if x else "" for x in observed]
    if not target or not observed:
        return {"matched": False, "text": mention_text, "confidence": 0.0}
    reference = _tokens(instruction) if instruction else target
    spans = [i for i in range(len(reference)-len(target)+1) if reference[i:i+len(target)] == target]
    if not spans or occurrence >= len(spans):
        return {'matched': False, 'text': mention_text, 'confidence': 0., 'reason': 'literal_span_not_found'}
    start = spans[occurrence]
    mapping = {}
    for block in difflib.SequenceMatcher(None, reference, observed, autojunk=False).get_matching_blocks():
        mapping.update({block.a+k: block.b+k for k in range(block.size)})
    positions = [mapping[p] for p in range(start,start+len(target)) if p in mapping]
    content = [x for x in target if x not in {'the','a','an','this','that','these','those','of','on','near','to'}]
    heads = _tokens(object_type) or content[-1:]
    def same_word(a, b):
        return (a == b or a.rstrip('s') == b.rstrip('s') or
                (a == 'people' and b == 'person') or
                b in HEAD_CONFUSIONS.get(a, set()) or
                a in HEAD_CONFUSIONS.get(b, set()))
    head_present = any(same_word(observed[p], head) for p in positions for head in heads)
    score = len(positions)/len(target)
    # If the transcript changed modifiers or a determiner but preserved the
    # noun head, align around that head instead of discarding the mention. This
    # handles e.g. ``front building`` → ``this building`` and file/fire. A
    # low-coverage match is explicitly marked and is never treated as full
    # confidence.
    recovered_head = False
    if (not head_present) and observed:
        head_positions = [p for p, token in enumerate(observed)
                          if any(same_word(token, head) for head in heads)]
        if len(head_positions) == 1:
            positions = head_positions
            head_present = True
            score = max(score, 1.0 / max(1, len(target)))
            recovered_head = True
    if not positions or not head_present:
        return {'matched': False, 'text': mention_text, 'confidence': round(score,3),
                'reason': 'missing_object_head_or_low_span_coverage'}
    i, j = min(positions), max(positions)+1
    low_coverage = score < .65 or recovered_head
    return {"matched": True, "text": mention_text, "word_start": i, "word_end": j,
            "audio_start": float(words[i]["start"]), "audio_end": float(words[j-1]["end"]),
            "confidence": round(min(score, .6) if low_coverage else score, 3),
            "alignment_quality": 'head_only' if low_coverage else 'phrase',
            "matched_words": [w.get("word", "") for w in words[i:j]]}


def align_mentions(ctx, referential: dict[str, Any], asr: OfflineASR,
                   cache_dir: str | Path | None = None) -> dict[str, Any]:
    alignment = clock_alignment(ctx)
    audio_path = alignment["audio_path"]
    if not audio_path:
        return {"status": "no_audio", "clock": alignment, "mentions": {}}
    transcript = asr.transcribe(audio_path)
    result = {"status": "ok", "clock": alignment, "transcript": transcript,
              "mentions": {}, "instruction_coverage": 0.0,
              "policy": "supplement_only"}
    occurrences = {}
    for mention in referential.get("mentions", []):
        text = mention.get('text', '')
        occurrence = occurrences.get(text.lower(),0)
        occurrences[text.lower()] = occurrence+1
        match = match_mention(text, transcript['words'], ctx.text,
                              mention.get('spoken_head', mention.get('object_type','')), occurrence)
        if match.get("matched"):
            pad_before, pad_after = 0.5, 0.75
            a0, a1 = max(0.0, match["audio_start"] - pad_before), match["audio_end"] + pad_after
            match.update({"audio_clip": [a0, a1],
                          "video_clip": [a0 + alignment["audio_to_video"],
                                         a1 + alignment["audio_to_video"]],
                          "telemetry_clip": [a0 + alignment["audio_to_telemetry"],
                                              a1 + alignment["audio_to_telemetry"]],
                          'clock_available': alignment['clock_available']})
            if not alignment['clock_available']:
                match['matched'] = False
                match['reason'] = 'clock_alignment_unavailable'
        result["mentions"][mention["mention_id"]] = match
    matched_phrase = [m for m in result['mentions'].values()
                      if m.get('matched') and m.get('alignment_quality') == 'phrase'
                      and m.get('confidence', 0) >= .75]
    result['instruction_coverage'] = (len(matched_phrase) / max(1, len(referential.get('mentions', []))))
    for mention in referential.get('mentions', []):
        match = result['mentions'][mention['mention_id']]
        anchor = mention.get('anchor_text','')
        if anchor and match.get('matched'):
            anchor_match = match_mention(anchor, transcript['words'], ctx.text)
            if anchor_match.get('matched'):
                a0 = max(0., min(match['audio_start'],anchor_match['audio_start'])-.5)
                a1 = max(match['audio_end'],anchor_match['audio_end'])+.75
                match['relation_audio_clip'] = [a0,a1]
                match['video_clip'] = [a0+alignment['audio_to_video'],a1+alignment['audio_to_video']]
                match['telemetry_clip'] = [a0+alignment['audio_to_telemetry'],a1+alignment['audio_to_telemetry']]
    if cache_dir:
        path = Path(cache_dir) / "mention_alignment" / ctx.scene_key / f"{ctx.event_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result
