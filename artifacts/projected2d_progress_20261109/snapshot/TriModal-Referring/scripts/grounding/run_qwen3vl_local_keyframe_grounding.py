#!/usr/bin/env python3
"""Run local Qwen3-VL grounding for multimodal non-entity spatial reference events."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

DEFAULT_INPUT_COLUMNS = ("event_id", "keyframe_path", "gaze_summary", "hand_summary")
DEFAULT_OUTPUT_COLUMNS = (
    "event_id",
    "prompt_text",
    "model_raw_output",
    "parsed_json",
    "referent_type",
    "primary_source",
    "prior_usage",
    "u_norm",
    "v_norm",
    "x_world",
    "y_world",
    "z_world",
    "referent_text",
    "reasoning_summary",
    "confidence",
    "parse_ok",
    "error_message",
    "event_json_path",
    "spatial_context_text",
    "spatial_context_json",
    "spatial_prior_u_norm",
    "spatial_prior_v_norm",
    "spatial_prior_source",
)
DEFAULT_INSTRUCTION_TEXT = (
    "Ground the user's intended event-level referent using the visual input together with language, gaze, hand, and JSON spatial context. "
    "A valid referent may be an entity, a spatial region, or not reliably inferable."
)
BASE_SYSTEM_PROMPT = (
    "You are a precise multimodal grounding model for egocentric VR interaction. "
    "Your task is to localize the user's intended referent in the current frame as a single normalized 2D point, "
    "using language, image evidence, and structured geometric signals from the event JSON. "
    "This is a geometric grounding task, not free-form guessing and not generic image captioning. "

    "The referent may be one of three types: "
    "entity = a visible object or object part; "
    "spatial = a point, placement area, contact location, operating spot, destination, or local region in space; "
    "none = the evidence is insufficient for a reliable referent. "

    "You must follow a strict decision order: "
    "1) decide whether a reliable referent exists; "
    "2) classify it as entity, spatial, or none; "
    "3) determine which evidence source is the primary localization source; "
    "4) decide whether the final point should equal the projected prior, stay very near it, or be offset from it; "
    "5) output the final normalized point only after the above decisions. "

    "You must understand the geometric fields as follows. "
    "gazePoint is a 3D world point estimated from the user's eye gaze and is usually the strongest cue for intended visual attention when valid. "
    "gazeVector is the eye gaze direction in 3D and indicates where the user is looking. "
    "gazeOrigin is the 3D world origin of the eye gaze ray. "
    "cameraGazeOrigin and cameraGazeDirection define the camera-centered viewing ray. "
    "cameraHitPoint is the 3D world point hit by the camera-centered ray and is a fallback visual center cue, not the default target. "
    "rightIndexFingerRayHitPoint is the 3D world hit point of the right index finger pointing ray and is a manipulation or action cue, not necessarily the final referent. "
    "cameraPosition is the 3D world position of the camera. "
    "cameraRotation is the camera orientation in 3D world coordinates. "
    "cameraFOV is the camera field of view and affects projection from 3D to 2D. "

    "You must treat projected 2D priors as follows. "
    "If spatial_prior_source is gazePoint and the instruction semantics are consistent with looking at or selecting a target, "
    "the final 2D point should usually stay at or very near the gaze prior. "
    "Do not move away from a valid gaze prior unless image evidence or action semantics clearly require an offset. "
    "If spatial_prior_source is cameraHitPoint, use it only as a fallback center-view cue. "
    "If spatial_prior_source is a hand-related source, use it mainly for manipulation targets, contact points, or placement regions. "

    "Important anti-failure rules: "
    "Do not default to the image center. "
    "Do not invent a point when referent_type should be none. "
    "Do not ignore the provided geometric fields. "
    "Do not describe what might be somewhere in the scene and then guess a coordinate. "
    "Do not choose a point far from the selected primary source unless you explicitly justify the offset. "

    "Return strict JSON only. No markdown, no code fences, no commentary outside the JSON object."
)
DEBUG_USER_PROMPT_TEMPLATE = """Event ID: {event_id}

Visual input mode:
{visual_input_mode}

Task:
Localize the intended referent for this event in the current frame as one final normalized 2D image point.

You must perform geometric grounding, not free-form guessing.

====================
INPUT FIELD MEANINGS
====================

Language fields:
- instruction_text: highest-level task instruction. This defines the task semantics.
- utterance_text: spoken command or natural language expression of the user's intent.
- target_description: short text describing the intended target or referent.

Summary fields:
- gaze_summary: summarized gaze behavior over the event or selected window.
- hand_summary: summarized hand behavior over the event or selected window.

Structured geometry fields:
- spatial_context_text: structured natural-language summary derived from event JSON.
- spatial_context_json: structured JSON-derived geometry evidence. Use it as the most important non-visual evidence when available.

Projected 2D prior fields:
- spatial_prior_source: the source used to build the projected 2D prior.
- spatial_prior_u_norm: normalized horizontal coordinate of the projected prior in [0, 1].
- spatial_prior_v_norm: normalized vertical coordinate of the projected prior in [0, 1].

Normalized coordinates:
- u_norm = horizontal image coordinate in [0,1], where 0 is the left border and 1 is the right border.
- v_norm = vertical image coordinate in [0,1], where 0 is the top border and 1 is the bottom border.

=========================
JSON GEOMETRY DEFINITIONS
=========================

Interpret the JSON geometry fields in this exact way:

1. eyeGaze.gazePoint
- 3D world point estimated from eye gaze intersection.
- This is usually the strongest cue for intended attention when valid and stable.
- If a projected 2D prior comes from gazePoint and the task is consistent with looking/selecting/indicating a target, the final point should usually remain equal or very close to that prior.

2. eyeGaze.gazeVector
- 3D direction of eye gaze.
- Use this as directional evidence, especially if gazePoint is noisy or partially missing.

3. eyeGaze.gazeOrigin
- 3D origin of the eye gaze ray.

4. eyeGaze.cameraGazeOrigin
- 3D origin of the camera-centered viewing ray.

5. eyeGaze.cameraGazeDirection
- 3D direction of the camera-centered viewing ray.
- This is a viewing-direction fallback cue, not the default target cue.

6. eyeGaze.cameraHitPoint
- 3D world point hit by the camera-centered viewing ray.
- This is a fallback center-view cue.
- Do not prefer cameraHitPoint over gazePoint unless gazePoint is missing, invalid, unstable, or semantically inconsistent.

7. handData.rightIndexFingerRayHitPoint
- 3D world point hit by the right index finger pointing ray.
- Use this for pointing, contact, operating, manipulation, or placement semantics.
- Do not treat it as the final referent if the user is merely moving the hand while attention is elsewhere.

8. cameraPosition
- 3D world position of the egocentric camera.

9. cameraRotation
- 3D orientation of the camera in world coordinates.

10. cameraFOV
- Camera field of view.
- Affects the mapping from 3D world geometry to 2D image position.

=========================
PRIMARY SOURCE PRIORITY
=========================

You must explicitly choose one primary source for the final point:

Allowed primary_source values:
- gazePoint
- cameraHitPoint
- rightIndexFingerRayHitPoint
- fused
- visual_only
- none

Use this priority unless strong evidence requires otherwise:

Priority rule A:
- If spatial_prior_source is gazePoint and the task semantics match visual attention, selection, pointing-to-visible-target, or identification, prefer gazePoint.

Priority rule B:
- If the task is about contact, touching, placing, operating, or interacting with a location and the hand cue is valid and more action-relevant than gaze, you may prefer rightIndexFingerRayHitPoint or fused.

Priority rule C:
- Only prefer cameraHitPoint when gazePoint is missing/invalid/unreliable and the intended target is likely near the visual center.

Priority rule D:
- Use fused only when two sources agree semantically and geometrically.
- Example: gazePoint indicates the area, hand ray indicates the exact contact spot.

Priority rule E:
- Use visual_only only when the projected prior is missing or clearly wrong and the target is visually unambiguous in the image.
- Use none when reliable grounding is not possible.

=========================
OFFSET DECISION STANDARD
=========================

You must decide one prior_usage mode:

Allowed prior_usage values:
- exact_prior
- near_prior
- offset_from_prior
- no_prior

Interpretation:
- exact_prior: final point should be numerically equal to the projected prior or effectively identical after rounding.
- near_prior: final point should remain very close to the projected prior; only a small correction is allowed.
- offset_from_prior: final point may move away from the prior, but only when there is a clear semantic or visual reason.
- no_prior: there is no usable projected prior.

Strict rule:
- If primary_source is gazePoint and the gaze prior is valid, do not use offset_from_prior unless the image clearly shows that the true referent is adjacent to the gaze target rather than at the gaze target itself.
- If you offset from a gaze prior, the reasoning_summary must clearly say why.

=========================
VALIDITY CHECKS
=========================

Before predicting, check these failure conditions:
- Missing or contradictory language
- Missing or unreliable priors
- Prior source inconsistent with the action semantics
- Target not visible or not inferable in the current frame
- Strong disagreement between image and geometry with no reliable resolution

If reliable grounding is not possible:
- set referent_type = none
- set primary_source = none
- set prior_usage = no_prior
- set u_norm = null
- set v_norm = null

=========================
DECISION PROCEDURE
=========================

Step 1. Decide whether a reliable referent exists.
- entity / spatial / none

Step 2. Identify the true referent, not just a landmark.
- A named object may be the referent itself, or it may only define a nearby target location.

Step 3. Choose the primary_source.
- gazePoint / cameraHitPoint / rightIndexFingerRayHitPoint / fused / visual_only / none

Step 4. Choose prior_usage.
- exact_prior / near_prior / offset_from_prior / no_prior

Step 5. Produce the final point.
- If exact_prior: final point should match the prior.
- If near_prior: final point should stay very close to the prior.
- If offset_from_prior: final point must reflect a justified semantic or visual offset.
- If no_prior and referent_type is not none: use visual and language evidence conservatively.

Step 6. Estimate world coordinates only if reasonably inferable.
- Otherwise use null.

=========================
INPUTS
=========================

instruction_text:
{instruction_text}

utterance_text:
{utterance_text}

target_description:
{target_description}

gaze_summary:
{gaze_summary}

hand_summary:
{hand_summary}

structured spatial context from event JSON:
{spatial_context_text}

structured spatial context JSON:
{spatial_context_json}

projected 2D prior from spatial JSON:
- source: {spatial_prior_source}
- u_norm_prior: {spatial_prior_u_norm}
- v_norm_prior: {spatial_prior_v_norm}

=========================
OUTPUT RULES
=========================

Return strict JSON only.

Required JSON keys:
- referent_type
- primary_source
- prior_usage
- u_norm
- v_norm
- x_world
- y_world
- z_world
- referent_text
- reasoning_summary
- confidence

Allowed referent_type:
- entity
- spatial
- none

Allowed primary_source:
- gazePoint
- cameraHitPoint
- rightIndexFingerRayHitPoint
- fused
- visual_only
- none

Allowed prior_usage:
- exact_prior
- near_prior
- offset_from_prior
- no_prior

Confidence:
- confidence must be in [0,1]

Output schema:
{{
  "referent_type": "spatial",
  "primary_source": "gazePoint",
  "prior_usage": "exact_prior",
  "u_norm": 0.0,
  "v_norm": 0.0,
  "x_world": null,
  "y_world": null,
  "z_world": null,
  "referent_text": "short referent phrase",
  "reasoning_summary": "brief concrete justification mentioning source choice and whether the point equals, stays near, or offsets from the prior",
  "confidence": 0.0
}}
"""
MINIMAL_USER_PROMPT_TEMPLATE = """Event ID: {event_id}

Visual input mode:
{visual_input_mode}

Task:
Predict the intended referent as one final normalized 2D image point using language, image evidence, and structured geometric cues.

This is a geometric grounding task.

Field meanings:
- instruction_text: task semantics
- utterance_text: user intent
- target_description: short description of the referent
- gaze_summary: summarized gaze cue
- hand_summary: summarized hand cue
- spatial_context_text: structured geometry summary from event JSON
- spatial_context_json: structured geometry evidence
- spatial_prior_source: source of the 2D projected prior
- spatial_prior_u_norm, spatial_prior_v_norm: normalized 2D prior coordinates

Coordinate meanings:
- u_norm in [0,1]: left to right
- v_norm in [0,1]: top to bottom

Important geometry rules:
- gazePoint usually represents the strongest intended attention cue when valid.
- cameraHitPoint is a fallback center-view cue.
- rightIndexFingerRayHitPoint is mainly a manipulation/contact/placement cue.
- If spatial_prior_source is gazePoint and the task semantics agree, the final point should usually equal or stay very near that prior.
- Do not move away from a valid gaze prior unless the image or action semantics clearly require an offset.
- Do not default to the image center.

You must decide:
1. referent_type = entity / spatial / none
2. primary_source = gazePoint / cameraHitPoint / rightIndexFingerRayHitPoint / fused / visual_only / none
3. prior_usage = exact_prior / near_prior / offset_from_prior / no_prior
4. final u_norm, v_norm

If reliable grounding is impossible:
- referent_type = none
- primary_source = none
- prior_usage = no_prior
- u_norm = null
- v_norm = null

Inputs:

instruction_text:
{instruction_text}

utterance_text:
{utterance_text}

target_description:
{target_description}

gaze_summary:
{gaze_summary}

hand_summary:
{hand_summary}

structured spatial context from event JSON:
{spatial_context_text}

structured spatial context JSON:
{spatial_context_json}

projected 2D prior:
- source: {spatial_prior_source}
- u_norm_prior: {spatial_prior_u_norm}
- v_norm_prior: {spatial_prior_v_norm}

Return strict JSON only.

Output schema:
{{
  "referent_type": "spatial",
  "primary_source": "gazePoint",
  "prior_usage": "exact_prior",
  "u_norm": 0.0,
  "v_norm": 0.0,
  "confidence": 0.0
}}
"""
MINIMAL_USER_PROMPT_TEMPLATE = """Event ID: {event_id}

Visual input mode:
{visual_input_mode}

Task objective:
Determine the intended event-level referent and output the most reliable grounding result.
This frame may be one of several plausible grounding peaks within the same event. 
Ground the referent that is most strongly supported by the current frame together with the language and multimodal cues. 

Possible referent types:
- entity
- spatial
- none

Core rule:
This is not unconstrained coordinate guessing.
You must:
1. decide whether a valid referent exists
2. classify it as entity or spatial if it exists
3. determine its coarse image region
4. decide whether the final point is near the projected prior or offset from it
5. output the final point only after the above decisions

Field roles and priority:
- instruction_text: main task constraint
- utterance_text: action semantics and intent
- target_description: strongest short description of the intended referent
- gaze_summary: attention and aiming cue
- hand_summary: action-direction or contact cue
- spatial_context_text: structured geometric evidence from JSON
- spatial priors: candidate 2D point from JSON, useful only when consistent with the scene and action

Coordinate interpretation:
- left third: u_norm < 0.33
- center third: 0.33 <= u_norm <= 0.67
- right third: u_norm > 0.67
- top third: v_norm < 0.33
- middle third: 0.33 <= v_norm <= 0.67
- bottom third: v_norm > 0.67

Allowed coarse regions:
- top_left
- top_center
- top_right
- middle_left
- middle_center
- middle_right
- bottom_left
- bottom_center
- bottom_right

Allowed prior relations:
- near_prior
- left_of_prior
- right_of_prior
- above_prior
- below_prior
- far_from_prior
- no_prior

Required decision order:
1. Decide whether a valid referent exists.
2. If it exists, classify it as entity or spatial.
3. Distinguish landmark objects from the true referent.
4. Determine the coarse image region.
5. Determine the relation to the projected prior.
6. Output the final point.

Failure-avoidance rules:
- Do not force a point when evidence is insufficient.
- Do not simply choose the named object.
- Do not simply choose the most salient visible object.
- Do not blindly copy the spatial prior.
- Do not default to the image center.

Input fields:
instruction_text:
{instruction_text}

utterance_text:
{utterance_text}

target_description:
{target_description}

gaze_summary:
{gaze_summary}

hand_summary:
{hand_summary}

structured spatial context from event JSON:
{spatial_context_text}

projected 2D prior from spatial JSON:
- source: {spatial_prior_source}
- u_norm_prior: {spatial_prior_u_norm}
- v_norm_prior: {spatial_prior_v_norm}

Output requirements:
- Output strict JSON only.
- referent_type must be entity, spatial, or none.
- coarse_region must be one of the allowed regions, or null when referent_type is none.
- prior_relation must be one of the allowed prior relations.
- If referent_type is none, set u_norm and v_norm to null.

Required JSON schema:
{{
  "referent_type": "entity",
  "coarse_region": "middle_right",
  "prior_relation": "right_of_prior",
  "u_norm": 0.0,
  "v_norm": 0.0,
  "confidence": 0.0
}}
"""


class GroundingRunError(Exception):
    pass


@dataclass
class RuntimeDeps:
    torch: Any
    image_module: Any
    image_draw_module: Any
    image_font_module: Any
    auto_processor_cls: Any
    auto_config_cls: Any
    auto_model_for_image_text_to_text: Any
    auto_model_for_vision2seq: Any
    auto_model_for_causal_lm: Any
    qwen3_vl_for_conditional_generation: Any
    qwen3_vl_moe_for_conditional_generation: Any


@dataclass
class Sample:
    event_id: str
    keyframe_path: Path
    video_path: Optional[Path]
    t_start: str
    t_end: str
    gaze_summary: str
    hand_summary: str
    instruction_text: str
    utterance_text: str
    target_description: str
    event_json_path: Optional[Path]
    spatial_context_text: str
    spatial_context_json: str
    spatial_prior_u_norm: str
    spatial_prior_v_norm: str
    spatial_prior_source: str


@dataclass
class SpatialContext:
    prompt_text: str
    structured_json: str
    prior_u_norm: str
    prior_v_norm: str
    prior_source: str


@dataclass
class PredictionResult:
    event_id: str
    prompt_text: str
    model_raw_output: str
    parsed_json: str
    referent_type: str
    primary_source: str
    prior_usage: str
    u_norm: str
    v_norm: str
    x_world: str
    y_world: str
    z_world: str
    referent_text: str
    reasoning_summary: str
    confidence: str
    parse_ok: str
    error_message: str
    event_json_path: str
    spatial_context_text: str
    spatial_context_json: str
    spatial_prior_u_norm: str
    spatial_prior_v_norm: str
    spatial_prior_source: str


def ensure_runtime_dependencies() -> RuntimeDeps:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise GroundingRunError("Missing dependency: torch. Please install PyTorch before running inference.") from exc
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError as exc:
        raise GroundingRunError("Missing dependency: Pillow. Please install pillow before running inference.") from exc
    try:
        from transformers import AutoProcessor, AutoConfig  # type: ignore
    except ImportError as exc:
        raise GroundingRunError("Missing dependency: transformers. Please install transformers before running inference.") from exc
    try:
        from transformers import AutoModelForImageTextToText  # type: ignore
    except ImportError:
        AutoModelForImageTextToText = None
    try:
        from transformers import AutoModelForVision2Seq  # type: ignore
    except ImportError:
        AutoModelForVision2Seq = None
    try:
        from transformers import AutoModelForCausalLM  # type: ignore
    except ImportError:
        AutoModelForCausalLM = None
    try:
        from transformers import Qwen3VLForConditionalGeneration  # type: ignore
    except ImportError:
        Qwen3VLForConditionalGeneration = None
    try:
        from transformers import Qwen3VLMoeForConditionalGeneration  # type: ignore
    except ImportError:
        Qwen3VLMoeForConditionalGeneration = None
    return RuntimeDeps(
        torch,
        Image,
        ImageDraw,
        ImageFont,
        AutoProcessor,
        AutoConfig,
        AutoModelForImageTextToText,
        AutoModelForVision2Seq,
        AutoModelForCausalLM,
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration,
    )


class LocalQwen3VLRunner:
    def __init__(
        self,
        model_name: str,
        dtype_name: str,
        use_flash_attn: bool,
        max_new_tokens: int,
        local_files_only: bool,
        input_mode: str,
        max_video_frames: int,
        ffmpeg_path: Optional[str],
        ffprobe_path: Optional[str],
        prompt_variant: str,
        offload_folder: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.dtype_name = dtype_name
        self.use_flash_attn = use_flash_attn
        self.max_new_tokens = max_new_tokens
        self.local_files_only = local_files_only
        self.input_mode = input_mode
        self.max_video_frames = max_video_frames
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.prompt_variant = prompt_variant
        self.offload_folder = offload_folder
        self.runtime: Optional[RuntimeDeps] = None
        self.processor = None
        self.model = None
        self.device = None

    def load(self) -> None:
        self.runtime = ensure_runtime_dependencies()
        model_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "dtype": resolve_dtype(self.dtype_name, self.runtime.torch),
            "trust_remote_code": True,
            "local_files_only": self.local_files_only,
        }
        if self.offload_folder:
            offload_path = Path(self.offload_folder).expanduser().resolve()
            offload_path.mkdir(parents=True, exist_ok=True)
            model_kwargs["offload_folder"] = str(offload_path)
        self.processor = self.runtime.auto_processor_cls.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        self.model = load_model_with_fallbacks(
            self.model_name,
            model_kwargs,
            self.runtime,
            self.use_flash_attn,
        )
        self.device = resolve_model_device(self.model)

    def predict(self, sample: Sample, spatial_context: SpatialContext) -> PredictionResult:
        if self.runtime is None or self.processor is None or self.model is None:
            raise GroundingRunError("Model runner has not been loaded.")
        system_prompt, prompt_text = build_prompts(sample, spatial_context, self.prompt_variant, self.input_mode)
        media_mode = resolve_sample_input_mode(sample, self.input_mode)
        model_inputs = build_model_inputs(
            processor=self.processor,
            runtime=self.runtime,
            sample=sample,
            system_prompt=system_prompt,
            prompt_text=prompt_text,
            media_mode=media_mode,
            ffmpeg_path=self.ffmpeg_path,
            ffprobe_path=self.ffprobe_path,
            max_video_frames=self.max_video_frames,
        )
        model_inputs = move_batch_to_device(model_inputs, self.device)
        with self.runtime.torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        generated_ids = trim_generated_ids(model_inputs, generated)
        raw_response = decode_response(self.processor, generated_ids).strip()
        parsed = parse_prediction_json(raw_response, self.prompt_variant)
        if parsed is None:
            return PredictionResult(
                event_id=sample.event_id,
                prompt_text=prompt_text,
                model_raw_output=raw_response,
                parsed_json="",
                referent_type="",
                primary_source="",
                prior_usage="",
                u_norm="",
                v_norm="",
                x_world="",
                y_world="",
                z_world="",
                referent_text="",
                reasoning_summary="",
                confidence="",
                parse_ok="false",
                error_message="Failed to parse model output into the required JSON schema.",
                event_json_path=str(sample.event_json_path) if sample.event_json_path else "",
                spatial_context_text=spatial_context.prompt_text,
                spatial_context_json=spatial_context.structured_json,
                spatial_prior_u_norm=spatial_context.prior_u_norm,
                spatial_prior_v_norm=spatial_context.prior_v_norm,
                spatial_prior_source=spatial_context.prior_source,
            )
        return PredictionResult(
            event_id=sample.event_id,
            prompt_text=prompt_text,
            model_raw_output=raw_response,
            parsed_json=json.dumps(parsed, ensure_ascii=False),
            referent_type=str(parsed.get("referent_type", "")),
            primary_source=str(parsed.get("primary_source", "")),
            prior_usage=str(parsed.get("prior_usage", "")),
            u_norm=format_optional_float(parsed.get("u_norm")),
            v_norm=format_optional_float(parsed.get("v_norm")),
            x_world=format_optional_float(parsed.get("x_world")),
            y_world=format_optional_float(parsed.get("y_world")),
            z_world=format_optional_float(parsed.get("z_world")),
            referent_text=str(parsed.get("referent_text", "")),
            reasoning_summary=str(parsed.get("reasoning_summary", "")),
            confidence=format_float(parsed["confidence"]),
            parse_ok="true",
            error_message="",
            event_json_path=str(sample.event_json_path) if sample.event_json_path else "",
            spatial_context_text=spatial_context.prompt_text,
            spatial_context_json=spatial_context.structured_json,
            spatial_prior_u_norm=spatial_context.prior_u_norm,
            spatial_prior_v_norm=spatial_context.prior_v_norm,
            spatial_prior_source=spatial_context.prior_source,
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local Qwen3-VL grounding inference for multimodal non-entity spatial reference events."
    )
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV. Legacy and extended columns are both supported.")
    parser.add_argument("--output_csv", required=True, help="Path to the output CSV for predictions.")
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct", help="Local model directory or Hugging Face model name.")
    parser.add_argument("--dtype", default="auto", help="Torch dtype: auto, float16, bfloat16, float32.")
    parser.add_argument("--use_flash_attn", action="store_true", help="Try flash_attention_2 first, then fall back automatically.")
    parser.add_argument("--local_files_only", action="store_true", help="Load model and processor from local files only.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens for model generation.")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue processing later rows when one sample fails.")
    parser.add_argument("--vis_dir", help="Optional directory for saving keyframe overlays with predicted 2D points.")
    parser.add_argument("--input_mode", choices=("auto", "image", "video"), default="auto", help="Use keyframe image, event video, or automatically choose per row.")
    parser.add_argument("--max_video_frames", type=int, default=16, help="Maximum number of video frames to sample when input_mode uses video.")
    parser.add_argument("--ffmpeg_path", help="Optional path to ffmpeg executable for video frame extraction.")
    parser.add_argument("--ffprobe_path", help="Optional path to ffprobe executable for video duration probing.")
    parser.add_argument("--prompt_variant", choices=("debug", "minimal"), default="debug", help="Prompt/output style: debug keeps referent_text and reasoning_summary; minimal requests only coordinates and confidence.")
    parser.add_argument("--offload_folder", help="Optional directory for Accelerate disk offload when loading very large models or MoE checkpoints.")
    parser.add_argument("--input_columns", default=",".join(DEFAULT_INPUT_COLUMNS), help="Comma-separated required legacy input CSV columns.")
    parser.add_argument("--output_columns", default=",".join(DEFAULT_OUTPUT_COLUMNS), help="Comma-separated output CSV columns.")
    return parser.parse_args()


def split_columns(raw_columns: str, minimum_count: int, label: str) -> List[str]:
    columns = [item.strip() for item in raw_columns.split(",") if item.strip()]
    if len(columns) < minimum_count:
        raise GroundingRunError(f"{label} must contain at least {minimum_count} column names, but received {len(columns)}.")
    return columns


def validate_columns(column_names: Sequence[str], required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in column_names]
    if missing:
        raise GroundingRunError(f"{label} is missing required names: {', '.join(missing)}")


def resolve_dtype(dtype_name: str, torch: Any) -> Any:
    normalized = dtype_name.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise GroundingRunError(f"Unsupported dtype: {dtype_name}. Use one of auto, float16, bfloat16, float32.")
    return mapping[normalized]


def load_model_with_fallbacks(model_name: str, model_kwargs: Mapping[str, Any], runtime: RuntimeDeps, use_flash_attn: bool) -> Any:
    last_error: Optional[Exception] = None
    attempted_errors: List[str] = []
    config = None
    model_type = ""
    try:
        config = runtime.auto_config_cls.from_pretrained(model_name, trust_remote_code=True)
        model_type = str(getattr(config, "model_type", "") or "")
    except Exception as exc:
        attempted_errors.append(f"AutoConfig: {exc}")

    preferred_model_classes: List[Any] = []
    if model_type == "qwen3_vl_moe":
        preferred_model_classes.extend([
            runtime.qwen3_vl_moe_for_conditional_generation,
            runtime.auto_model_for_image_text_to_text,
            runtime.auto_model_for_vision2seq,
        ])
    elif model_type == "qwen3_vl":
        preferred_model_classes.extend([
            runtime.qwen3_vl_for_conditional_generation,
            runtime.auto_model_for_image_text_to_text,
            runtime.auto_model_for_vision2seq,
        ])
    preferred_model_classes.extend([
        runtime.auto_model_for_image_text_to_text,
        runtime.auto_model_for_vision2seq,
        runtime.auto_model_for_causal_lm,
    ])

    seen_classes: List[Any] = []
    model_classes: List[Any] = []
    for model_class in preferred_model_classes:
        if model_class is None or model_class in seen_classes:
            continue
        seen_classes.append(model_class)
        model_classes.append(model_class)

    attempts: List[Dict[str, Any]] = []
    if use_flash_attn:
        kwargs = dict(model_kwargs)
        kwargs["attn_implementation"] = "flash_attention_2"
        attempts.append(kwargs)
    attempts.append(dict(model_kwargs))
    for kwargs in attempts:
        for model_class in model_classes:
            try:
                return model_class.from_pretrained(model_name, **kwargs)
            except Exception as exc:
                last_error = exc
                attempted_errors.append(f"{getattr(model_class, '__name__', str(model_class))}: {exc}")
    error_tail = attempted_errors[-3:] if attempted_errors else []
    joined_errors = ' || '.join(attempted_errors)
    if model_type == "qwen3_vl_moe" and "offload_folder" in joined_errors:
        raise GroundingRunError(
            f"Failed to load model {model_name}. model_type=qwen3_vl_moe. "
            "This MoE checkpoint needs an explicit disk offload directory when Accelerate decides to offload weights. "
            "Re-run with --offload_folder /path/to/offload_dir (and keep using local_files_only if desired). "
            f"Recent errors: {' || '.join(error_tail or attempted_errors[-5:])}"
        )
    if error_tail:
        raise GroundingRunError(f"Failed to load model {model_name}. model_type={model_type or 'unknown'}. Recent errors: {' || '.join(error_tail)}")
    raise GroundingRunError(f"Failed to load model {model_name}. Last error: {last_error}")


def resolve_model_device(model: Any) -> Optional[Any]:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        for target in device_map.values():
            if isinstance(target, int):
                return f"cuda:{target}"
            if isinstance(target, str):
                lowered = target.lower()
                if lowered.startswith("cuda"):
                    return target
        for target in device_map.values():
            if isinstance(target, str) and target.lower() not in {"cpu", "disk", "meta"}:
                return target
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def load_image(image_path: Path, image_module: Any) -> Any:
    if not image_path.exists():
        raise GroundingRunError(f"Missing keyframe file: {image_path}")
    if not image_path.is_file():
        raise GroundingRunError(f"Expected keyframe_path to be a file, but found: {image_path}")
    try:
        return image_module.open(image_path).convert("RGB")
    except Exception as exc:
        raise GroundingRunError(f"Failed to load image {image_path}: {exc}") from exc


def build_messages(system_prompt: str, prompt_text: str, media_mode: str) -> List[Dict[str, Any]]:
    media_content = {"type": "video"} if media_mode == "video" else {"type": "image"}
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [media_content, {"type": "text", "text": prompt_text}]},
    ]


def resolve_binary_path(explicit_path: Optional[str], binary_name: str) -> str:
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    which_path = shutil.which(binary_name)
    if which_path:
        candidates.append(Path(which_path))
    candidates.append(Path(r"E:\Anaconda_1\envs\yolo\Library\bin") / f"{binary_name}.exe")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    raise GroundingRunError(f"{binary_name} was not found. Install it or pass --{binary_name}_path.")


def parse_time_value(raw_value: str, label: str) -> Optional[float]:
    text = raw_value.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise GroundingRunError(f"Invalid {label} value: {raw_value}") from exc
    if value < 0.0:
        raise GroundingRunError(f"{label} must be non-negative, but got {raw_value}.")
    return value


def probe_video_duration(video_path: Path, ffprobe_path: Optional[str]) -> Optional[float]:
    binary = resolve_binary_path(ffprobe_path, "ffprobe")
    command = [
        binary,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return None
    text = completed.stdout.strip()
    if not text:
        return None
    try:
        duration = float(text)
    except ValueError:
        return None
    return duration if math.isfinite(duration) and duration > 0.0 else None


def load_video_frames(
    video_path: Path,
    runtime: RuntimeDeps,
    ffmpeg_path: Optional[str],
    ffprobe_path: Optional[str],
    max_video_frames: int,
    t_start: Optional[float],
    t_end: Optional[float],
) -> List[Any]:
    if not video_path.exists():
        raise GroundingRunError(f"Missing video file: {video_path}")
    if max_video_frames <= 0:
        raise GroundingRunError(f"max_video_frames must be positive, but got {max_video_frames}.")
    duration = probe_video_duration(video_path, ffprobe_path)
    clip_start = t_start if t_start is not None else 0.0
    clip_end = t_end
    if clip_end is not None and clip_end <= clip_start:
        raise GroundingRunError(f"t_end must be greater than t_start for video input: {video_path}")
    if clip_end is None and duration is not None:
        clip_end = duration
    clip_duration = None if clip_end is None else max(clip_end - clip_start, 1e-3)
    fps = min(8.0, max(1.0, max_video_frames / clip_duration)) if clip_duration is not None else 1.0
    ffmpeg_binary = resolve_binary_path(ffmpeg_path, "ffmpeg")
    with tempfile.TemporaryDirectory(prefix="qwen3vl_video_") as temp_dir:
        frame_pattern = str(Path(temp_dir) / "frame_%04d.jpg")
        command = [ffmpeg_binary, "-y"]
        if clip_start > 0.0:
            command.extend(["-ss", f"{clip_start:.3f}"])
        command.extend(["-i", str(video_path)])
        if clip_duration is not None:
            command.extend(["-t", f"{clip_duration:.3f}"])
        command.extend(["-vf", f"fps={fps:.6f}", "-frames:v", str(max_video_frames), frame_pattern])
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise GroundingRunError(
                f"ffmpeg failed while extracting frames from {video_path}: {completed.stderr.strip() or completed.stdout.strip()}"
            )
        frame_paths = sorted(Path(temp_dir).glob("frame_*.jpg"))
        if not frame_paths:
            raise GroundingRunError(f"No video frames were extracted from {video_path}.")
        return [runtime.image_module.open(frame_path).convert("RGB") for frame_path in frame_paths]


def resolve_sample_input_mode(sample: Sample, requested_mode: str) -> str:
    if requested_mode == "image":
        return "image"
    if requested_mode == "video":
        if sample.video_path is None:
            raise GroundingRunError(f"input_mode=video requires video_path, but event_id={sample.event_id} has none.")
        return "video"
    return "video" if sample.video_path is not None else "image"


def build_model_inputs(
    processor: Any,
    runtime: RuntimeDeps,
    sample: Sample,
    system_prompt: str,
    prompt_text: str,
    media_mode: str,
    ffmpeg_path: Optional[str],
    ffprobe_path: Optional[str],
    max_video_frames: int,
) -> Mapping[str, Any]:
    messages = build_messages(system_prompt, prompt_text, media_mode)
    if hasattr(processor, "apply_chat_template"):
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if media_mode == "video":
            frames = load_video_frames(
                video_path=sample.video_path or Path(),
                runtime=runtime,
                ffmpeg_path=ffmpeg_path,
                ffprobe_path=ffprobe_path,
                max_video_frames=max_video_frames,
                t_start=parse_time_value(sample.t_start, "t_start"),
                t_end=parse_time_value(sample.t_end, "t_end"),
            )
            return processor(text=[chat_text], videos=[frames], return_tensors="pt")
        image = load_image(sample.keyframe_path, runtime.image_module)
        return processor(text=[chat_text], images=[image], return_tensors="pt")
    if media_mode == "video":
        frames = load_video_frames(
            video_path=sample.video_path or Path(),
            runtime=runtime,
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
            max_video_frames=max_video_frames,
            t_start=parse_time_value(sample.t_start, "t_start"),
            t_end=parse_time_value(sample.t_end, "t_end"),
        )
        return processor(text=[system_prompt + "\n\n" + prompt_text], videos=[frames], return_tensors="pt")
    image = load_image(sample.keyframe_path, runtime.image_module)
    return processor(text=[system_prompt + "\n\n" + prompt_text], images=[image], return_tensors="pt")


def move_batch_to_device(batch: Mapping[str, Any], device: Optional[Any]) -> Mapping[str, Any]:
    if device is None:
        return batch
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def trim_generated_ids(model_inputs: Mapping[str, Any], generated_ids: Any) -> Any:
    input_ids = model_inputs.get("input_ids")
    if input_ids is None:
        return generated_ids
    prompt_length = input_ids.shape[1]
    if getattr(generated_ids, "ndim", None) == 2 and generated_ids.shape[1] >= prompt_length:
        return generated_ids[:, prompt_length:]
    return generated_ids


def decode_response(processor: Any, token_ids: Any) -> str:
    if hasattr(processor, "batch_decode"):
        decoded = processor.batch_decode(token_ids, skip_special_tokens=True)
        if decoded:
            return decoded[0]
    raise GroundingRunError("Processor does not support batch_decode.")


def collect_json_candidates(raw_response: str) -> List[str]:
    candidates: List[str] = []
    stripped = raw_response.strip()
    if stripped:
        candidates.append(stripped)
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw_response, flags=re.DOTALL))
    candidates.extend(re.findall(r"\{.*?\}", raw_response, flags=re.DOTALL))
    unique: List[str] = []
    seen = set()
    for candidate in candidates:
        text = candidate.strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def parse_prediction_json(raw_response: str, prompt_variant: str) -> Optional[Dict[str, Any]]:
    for candidate in collect_json_candidates(raw_response):
        parsed = try_parse_candidate(candidate, prompt_variant)
        if parsed is not None:
            return parsed
    return None


def try_parse_candidate(candidate: str, prompt_variant: str) -> Optional[Dict[str, Any]]:
    cleaned = re.sub(r"^```(?:json)?", "", candidate.strip()).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        referent_type = normalize_referent_type(payload.get("referent_type"))
        primary_source = normalize_primary_source(payload.get("primary_source"))
        prior_usage = normalize_prior_usage(payload.get("prior_usage"))
        u_norm = parse_optional_unit_float(payload.get("u_norm"))
        v_norm = parse_optional_unit_float(payload.get("v_norm"))
        if referent_type == "none":
            if primary_source != "none" or prior_usage != "no_prior":
                return None
            u_norm = None
            v_norm = None
        elif u_norm is None or v_norm is None:
            return None
        parsed = {
            "referent_type": referent_type,
            "primary_source": primary_source,
            "prior_usage": prior_usage,
            "u_norm": u_norm,
            "v_norm": v_norm,
            "x_world": parse_optional_float(payload.get("x_world")),
            "y_world": parse_optional_float(payload.get("y_world")),
            "z_world": parse_optional_float(payload.get("z_world")),
            "referent_text": str(payload.get("referent_text", "")).strip(),
            "reasoning_summary": str(payload.get("reasoning_summary", "")).strip(),
            "confidence": clamp_unit_float(payload["confidence"]),
        }
    except (KeyError, TypeError, ValueError):
        return None
    if not validate_source_consistency(parsed):
        return None
    if prompt_variant == "debug":
        if not parsed["reasoning_summary"]:
            return None
        if parsed["referent_type"] != "none" and not parsed["referent_text"]:
            return None
    return parsed


def normalize_referent_type(value: Any) -> str:
    text = str(value).strip().lower()
    if text not in {"entity", "spatial", "none"}:
        raise ValueError(f"Invalid referent_type: {value}")
    return text


def normalize_primary_source(value: Any) -> str:
    text = str(value).strip()
    allowed = {
        "gazePoint",
        "cameraHitPoint",
        "rightIndexFingerRayHitPoint",
        "fused",
        "visual_only",
        "none",
    }
    if text not in allowed:
        raise ValueError(f"Invalid primary_source: {value}")
    return text


def normalize_prior_usage(value: Any) -> str:
    text = str(value).strip()
    allowed = {
        "exact_prior",
        "near_prior",
        "offset_from_prior",
        "no_prior",
    }
    if text not in allowed:
        raise ValueError(f"Invalid prior_usage: {value}")
    return text


def validate_source_consistency(parsed: Dict[str, Any]) -> bool:
    referent_type = parsed.get("referent_type")
    primary_source = parsed.get("primary_source")
    prior_usage = parsed.get("prior_usage")

    if referent_type == "none":
        return primary_source == "none" and prior_usage == "no_prior"

    if primary_source == "none":
        return False

    return True

def clamp_unit_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite value")
    return max(0.0, min(1.0, number))


def parse_optional_float(value: Any) -> Optional[float]:
    if value in (None, "", "null", "None"):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def parse_optional_unit_float(value: Any) -> Optional[float]:
    if value in (None, "", "null", "None"):
        return None
    return clamp_unit_float(value)


def format_float(value: float) -> str:
    return f"{value:.6f}"


def format_optional_float(value: Any) -> str:
    if value in (None, "", "null", "None"):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number:.6f}" if math.isfinite(number) else ""


def normalize_text(value: Any, fallback: str = "") -> str:
    text = str(value).strip() if value not in (None, "") else ""
    return text if text else fallback


def normalize_iso_timestamp(raw_value: str) -> str:
    text = raw_value.strip()
    if not text or "T" not in text or "." not in text:
        return text
    time_start = text.find("T")
    fraction_start = text.find(".", time_start)
    timezone_start = len(text)
    for marker in ("+", "-", "Z"):
        marker_index = text.find(marker, time_start + 1)
        if marker_index != -1:
            timezone_start = min(timezone_start, marker_index)
    fraction = text[fraction_start + 1 : timezone_start]
    if not fraction.isdigit() or len(fraction) <= 6:
        return text
    return text[: fraction_start + 1] + fraction[:6] + text[timezone_start:]


def repair_json_array_if_needed(raw_text: str) -> str:
    stripped = raw_text.strip()
    if not stripped:
        raise GroundingRunError("Input event JSON file is empty.")
    if stripped.startswith("[") and stripped.count("[") == stripped.count("]") + 1 and stripped.endswith("}"):
        return stripped + "\n]"
    # Released scene5/408 omits the outer array; keep strict JSON validation.
    if stripped.startswith("{") and stripped.endswith("}"):
        return "[" + stripped + "]"
    return stripped


def load_event_json(json_path: Path) -> Any:
    if not json_path.exists():
        raise GroundingRunError(f"Missing event JSON file: {json_path}")
    raw_text = json_path.read_text(encoding="utf-8-sig")
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return json.loads(repair_json_array_if_needed(raw_text))


def parse_sample_time(timestamp_value: Any) -> Optional[float]:
    if not isinstance(timestamp_value, str) or not timestamp_value.strip():
        return None
    normalized = normalize_iso_timestamp(timestamp_value)
    match = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?", normalized)
    if match is None:
        return None
    hour = int(match.group(1)[11:13])
    minute = int(match.group(2))
    second = int(match.group(3))
    fraction = match.group(4) or "0"
    return hour * 3600.0 + minute * 60.0 + second + float(f"0.{fraction}")

def format_xyz(point: Mapping[str, Any]) -> str:
    try:
        return f"({float(point['x']):.3f}, {float(point['y']):.3f}, {float(point['z']):.3f})"
    except Exception:
        return "unknown"


def sample_representative_indices(length: int, max_points: int) -> List[int]:
    if length <= 0:
        return []
    if length <= max_points:
        return list(range(length))
    if max_points <= 1:
        return [length // 2]
    step = (length - 1) / float(max_points - 1)
    return sorted({int(round(step * idx)) for idx in range(max_points)})[:max_points]


def summarize_event_json(payload: Any) -> SpatialContext:
    if payload is None:
        return SpatialContext("No structured event JSON was provided.", json.dumps({}, ensure_ascii=False), "", "", "none")
    if isinstance(payload, Mapping) and "spatial_prior" in payload:
        prior = payload.get("spatial_prior") if isinstance(payload.get("spatial_prior"), Mapping) else {}
        peak = payload.get("peak_spatial") if isinstance(payload.get("peak_spatial"), Mapping) else {}
        lines = [
            f"window_sample_count={payload.get('window_sample_count', 'unknown')}",
            f"time_window={payload.get('time_window', {})}",
            f"image_size={payload.get('image_size', {})}",
            f"spatial_prior_source={prior.get('source', 'none')}",
            f"spatial_prior_u_norm={format_optional_float(prior.get('u_norm'))}",
            f"spatial_prior_v_norm={format_optional_float(prior.get('v_norm'))}",
            f"peak_camera_hit_point={format_xyz(peak.get('camera_hit_point')) if isinstance(peak.get('camera_hit_point'), Mapping) else 'unknown'}",
            f"peak_gaze_point={format_xyz(peak.get('gaze_point')) if isinstance(peak.get('gaze_point'), Mapping) else 'unknown'}",
            f"peak_right_index_ray_hit_point={format_xyz(peak.get('right_index_ray_hit_point')) if isinstance(peak.get('right_index_ray_hit_point'), Mapping) else 'unknown'}",
            f"peak_camera_position={format_xyz(peak.get('camera_position')) if isinstance(peak.get('camera_position'), Mapping) else 'unknown'}",
            f"peak_camera_fov={peak.get('camera_fov', 'unknown')}",
        ]
        return SpatialContext(
            "\n".join(lines),
            json.dumps(payload, ensure_ascii=False),
            format_optional_float(prior.get("u_norm")),
            format_optional_float(prior.get("v_norm")),
            str(prior.get("source", "none")),
        )
    if isinstance(payload, list):
        valid = [sample for sample in payload if isinstance(sample, Mapping)]
        if not valid:
            return SpatialContext("Event JSON list was provided but no valid sample objects were found.", json.dumps({"sample_count": 0}, ensure_ascii=False), "", "", "none")
        eye_open_count = 0
        left_tracked = 0
        right_tracked = 0
        hit_points: List[Mapping[str, Any]] = []
        gaze_points: List[Mapping[str, Any]] = []
        right_hits: List[Mapping[str, Any]] = []
        cam_positions: List[Mapping[str, Any]] = []
        times: List[float] = []
        for sample in valid:
            t = parse_sample_time(sample.get("timestamp"))
            if t is not None:
                times.append(t)
            eye_gaze = sample.get("eyeGaze")
            if isinstance(eye_gaze, Mapping):
                if eye_gaze.get("isEyeOpen") is True:
                    eye_open_count += 1
                if isinstance(eye_gaze.get("cameraHitPoint"), Mapping):
                    hit_points.append(eye_gaze["cameraHitPoint"])
                if isinstance(eye_gaze.get("gazePoint"), Mapping):
                    gaze_points.append(eye_gaze["gazePoint"])
            hand_data = sample.get("handData")
            if isinstance(hand_data, Mapping):
                if hand_data.get("isLeftHandTracked") is True:
                    left_tracked += 1
                if hand_data.get("isRightHandTracked") is True:
                    right_tracked += 1
                if isinstance(hand_data.get("rightIndexFingerRayHitPoint"), Mapping):
                    right_hits.append(hand_data["rightIndexFingerRayHitPoint"])
            if isinstance(sample.get("cameraPosition"), Mapping):
                cam_positions.append(sample["cameraPosition"])
        rep_hit = [format_xyz(hit_points[i]) for i in sample_representative_indices(len(hit_points), 3)]
        rep_gaze = [format_xyz(gaze_points[i]) for i in sample_representative_indices(len(gaze_points), 3)]
        rep_hand = [
            format_xyz(right_hits[i])
            for i in sample_representative_indices(len(right_hits), 2)
            if any(abs(float(right_hits[i].get(axis, 0.0))) > 1e-6 for axis in ("x", "y", "z"))
        ]
        valid_positions = [item for item in cam_positions if all(axis in item for axis in ("x", "y", "z"))]
        avg_pos = None
        if valid_positions:
            avg_pos = {axis: mean(float(item[axis]) for item in valid_positions) for axis in ("x", "y", "z")}
        duration = max(times) - min(times) if len(times) >= 2 else None
        structured = {
            "sample_count": len(valid),
            "duration_seconds": duration,
            "eye_open_count": eye_open_count,
            "left_hand_tracked_frames": left_tracked,
            "right_hand_tracked_frames": right_tracked,
            "camera_hit_points": rep_hit,
            "gaze_points": rep_gaze,
            "right_index_ray_hits": rep_hand,
            "avg_camera_position": avg_pos,
        }
        lines = [
            f"sample_count={structured['sample_count']}",
            f"duration_seconds={duration:.3f}" if duration is not None else "duration_seconds=unknown",
            f"eye_open_count={eye_open_count}",
            f"left_hand_tracked_frames={left_tracked}",
            f"right_hand_tracked_frames={right_tracked}",
            f"representative_camera_hit_points={rep_hit or ['none']}",
            f"representative_gaze_points={rep_gaze or ['none']}",
            f"representative_right_index_ray_hits={rep_hand or ['none']}",
            f"avg_camera_position={format_xyz(avg_pos) if isinstance(avg_pos, Mapping) else 'unknown'}",
        ]
        return SpatialContext("\n".join(lines), json.dumps(structured, ensure_ascii=False), "", "", "none")
    if isinstance(payload, Mapping):
        extracted = {
            key: payload[key]
            for key in payload.keys()
            if key in {
                "gaze_hit_point",
                "gaze_point",
                "hand_ray_origin",
                "hand_ray_direction",
                "camera_pose",
                "camera_position",
                "camera_rotation",
                "world_point",
                "hit_object",
                "referent_text",
                "utterance_text",
            }
        }
        if not extracted:
            extracted = {"available_top_level_keys": list(payload.keys())[:20]}
        lines = [
            f"{key}={format_xyz(value) if isinstance(value, Mapping) and all(axis in value for axis in ('x', 'y', 'z')) else value}"
            for key, value in extracted.items()
        ]
        return SpatialContext("\n".join(lines), json.dumps(extracted, ensure_ascii=False), "", "", "none")
    return SpatialContext(f"Unsupported event JSON type: {type(payload).__name__}", json.dumps({"json_type": type(payload).__name__}, ensure_ascii=False), "", "", "none")


def resolve_language_inputs(sample: Sample) -> Tuple[str, str, str]:
    instruction_text = normalize_text(sample.instruction_text)
    utterance_text = normalize_text(sample.utterance_text)
    target_description = normalize_text(sample.target_description)
    if not instruction_text:
        instruction_text = DEFAULT_INSTRUCTION_TEXT
    if not target_description:
        if utterance_text:
            target_description = utterance_text
        elif instruction_text and instruction_text != DEFAULT_INSTRUCTION_TEXT:
            target_description = instruction_text
    if not utterance_text:
        utterance_text = "No utterance text was provided."
    if not target_description:
        target_description = "No target description was provided."
    return instruction_text, utterance_text, target_description


def build_prompts(sample: Sample, spatial_context: SpatialContext, prompt_variant: str, input_mode: str) -> Tuple[str, str]:
    media_mode = resolve_sample_input_mode(sample, input_mode)
    visual_input_mode = "event video clip" if media_mode == "video" else "keyframe image"
    template = DEBUG_USER_PROMPT_TEMPLATE if prompt_variant == "debug" else MINIMAL_USER_PROMPT_TEMPLATE
    instruction_text, utterance_text, target_description = resolve_language_inputs(sample)
    prompt_text = template.format(
        event_id=sample.event_id,
        visual_input_mode=visual_input_mode,
        instruction_text=instruction_text,
        utterance_text=utterance_text,
        target_description=target_description,
        gaze_summary=normalize_text(sample.gaze_summary, "No gaze summary was provided."),
        hand_summary=normalize_text(sample.hand_summary, "No hand summary was provided."),
        spatial_context_text=normalize_text(spatial_context.prompt_text or sample.spatial_context_text, "No structured event JSON was provided."),
        spatial_context_json=normalize_text(spatial_context.structured_json or sample.spatial_context_json, "No spatial_context_json was provided."),
        spatial_prior_source=normalize_text(spatial_context.prior_source, "none"),
        spatial_prior_u_norm=normalize_text(spatial_context.prior_u_norm, "unknown"),
        spatial_prior_v_norm=normalize_text(spatial_context.prior_v_norm, "unknown"),
    )
    return BASE_SYSTEM_PROMPT, prompt_text


def read_samples(input_csv: Path, input_columns: Sequence[str]) -> List[Sample]:
    if not input_csv.exists() or not input_csv.is_file():
        raise GroundingRunError(f"Input CSV does not exist or is not a file: {input_csv}")
    samples: List[Sample] = []
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise GroundingRunError(f"Input CSV has no header row: {input_csv}")
        missing = [column for column in input_columns if column not in reader.fieldnames]
        if missing:
            raise GroundingRunError("Input CSV is missing required legacy columns: " + ", ".join(missing))
        for row_index, row in enumerate(reader, start=1):
            if not any(row.values()):
                continue
            event_id = normalize_text(row.get("event_id"))
            keyframe_path_raw = normalize_text(row.get("keyframe_path"))
            video_path_raw = normalize_text(row.get("video_path"))
            if not event_id:
                raise GroundingRunError(f"Row {row_index} is missing event_id.")
            if not keyframe_path_raw and not video_path_raw:
                raise GroundingRunError(f"Row {row_index} must provide keyframe_path or video_path.")
            event_json_path_raw = normalize_text(row.get("event_json_path"))
            keyframe_path = Path(keyframe_path_raw).expanduser().resolve() if keyframe_path_raw else Path()
            samples.append(
                Sample(
                    event_id=event_id,
                    keyframe_path=keyframe_path,
                    video_path=Path(video_path_raw).expanduser().resolve() if video_path_raw else None,
                    t_start=normalize_text(row.get("t_start")),
                    t_end=normalize_text(row.get("t_end")),
                    gaze_summary=normalize_text(row.get("gaze_summary")),
                    hand_summary=normalize_text(row.get("hand_summary")),
                    instruction_text=normalize_text(row.get("instruction_text"), DEFAULT_INSTRUCTION_TEXT),
                    utterance_text=normalize_text(row.get("utterance_text")),
                    target_description=normalize_text(row.get("target_description")),
                    event_json_path=Path(event_json_path_raw).expanduser().resolve() if event_json_path_raw else None,
                    spatial_context_text=normalize_text(row.get("spatial_context_text")),
                    spatial_context_json=normalize_text(row.get("spatial_context_json")),
                    spatial_prior_u_norm=normalize_text(row.get("spatial_prior_u_norm")),
                    spatial_prior_v_norm=normalize_text(row.get("spatial_prior_v_norm")),
                    spatial_prior_source=normalize_text(row.get("spatial_prior_source"), "none"),
                )
            )
    if not samples:
        raise GroundingRunError(f"Input CSV contains no valid rows: {input_csv}")
    return samples


def write_results(output_csv: Path, results: Iterable[PredictionResult], output_columns: Sequence[str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_columns))
        writer.writeheader()
        for result in results:
            writer.writerow({column: getattr(result, column, "") for column in output_columns})


def sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized or "event"


def clamp_pixel(value: float, limit: int) -> int:
    return max(0, min(limit - 1, int(round(value))))


def render_overlay(runtime: RuntimeDeps, sample: Sample, result: PredictionResult, vis_dir: Path) -> None:
    if not sample.keyframe_path or not sample.keyframe_path.exists() or not sample.keyframe_path.is_file():
        return
    image = load_image(sample.keyframe_path, runtime.image_module)
    draw = runtime.image_draw_module.Draw(image)
    width, height = image.size
    font = runtime.image_font_module.load_default()

    if result.spatial_prior_u_norm and result.spatial_prior_v_norm:
        prior_x = clamp_pixel(float(result.spatial_prior_u_norm) * width, width)
        prior_y = clamp_pixel(float(result.spatial_prior_v_norm) * height, height)
        prior_radius = max(5, int(min(width, height) * 0.012))
        draw.ellipse(
            (prior_x - prior_radius, prior_y - prior_radius, prior_x + prior_radius, prior_y + prior_radius),
            outline=(0, 255, 255),
            width=3,
        )
        draw.line((prior_x - prior_radius - 3, prior_y, prior_x + prior_radius + 3, prior_y), fill=(0, 255, 255), width=2)
        draw.line((prior_x, prior_y - prior_radius - 3, prior_x, prior_y + prior_radius + 3), fill=(0, 255, 255), width=2)

    if result.parse_ok == "true" and result.u_norm and result.v_norm:
        x_pixel = clamp_pixel(float(result.u_norm) * width, width)
        y_pixel = clamp_pixel(float(result.v_norm) * height, height)
        radius = max(6, int(min(width, height) * 0.015))
        draw.ellipse(
            (x_pixel - radius, y_pixel - radius, x_pixel + radius, y_pixel + radius),
            outline=(255, 0, 0),
            width=3,
            fill=(255, 220, 0),
        )
        text_x = min(max(8, x_pixel + radius + 4), max(8, width - 320))
        text_y = min(max(8, y_pixel - radius - 34), max(8, height - 90))
    else:
        text_x = 8
        text_y = 8

    text = "\n".join(
        [
            f"event_id: {sample.event_id}",
            f"referent_type: {result.referent_type or 'unknown'}",
            f"referent: {result.referent_text or 'unknown'}",
            f"confidence: {result.confidence or 'unknown'}",
            f"prior: {result.spatial_prior_source or 'none'} ({result.spatial_prior_u_norm or '?'} , {result.spatial_prior_v_norm or '?'})",
        ]
    )
    draw.multiline_text(
        (text_x, text_y),
        text,
        fill=(255, 255, 255),
        font=font,
        stroke_width=2,
        stroke_fill=(0, 0, 0),
        spacing=4,
    )
    vis_dir.mkdir(parents=True, exist_ok=True)
    image.save(vis_dir / f"{sanitize_filename(sample.event_id)}.png")


def build_failed_result(sample: Sample, spatial_context: SpatialContext, prompt_text: str, message: str) -> PredictionResult:
    return PredictionResult(
        event_id=sample.event_id,
        prompt_text=prompt_text,
        model_raw_output="",
        parsed_json="",
        referent_type="",
        primary_source="",
        prior_usage="",
        u_norm="",
        v_norm="",
        x_world="",
        y_world="",
        z_world="",
        referent_text="",
        reasoning_summary="",
        confidence="",
        parse_ok="false",
        error_message=message,
        event_json_path=str(sample.event_json_path) if sample.event_json_path else "",
        spatial_context_text=spatial_context.prompt_text,
        spatial_context_json=spatial_context.structured_json,
        spatial_prior_u_norm=spatial_context.prior_u_norm,
        spatial_prior_v_norm=spatial_context.prior_v_norm,
        spatial_prior_source=spatial_context.prior_source,
    )


def main() -> int:
    args = parse_args()
    try:
        input_columns = split_columns(args.input_columns, len(DEFAULT_INPUT_COLUMNS), "input_columns")
        output_columns = split_columns(args.output_columns, len(DEFAULT_OUTPUT_COLUMNS), "output_columns")
        validate_columns(input_columns, DEFAULT_INPUT_COLUMNS, "input_columns")
        validate_columns(output_columns, DEFAULT_OUTPUT_COLUMNS, "output_columns")

        input_csv = Path(args.input_csv).resolve()
        output_csv = Path(args.output_csv).resolve()
        vis_dir = Path(args.vis_dir).resolve() if args.vis_dir else None
        model_name_or_path = (
            str(Path(args.model_name).expanduser().resolve())
            if Path(args.model_name).expanduser().exists()
            else args.model_name
        )
        samples = read_samples(input_csv, input_columns)
        runner = LocalQwen3VLRunner(
            model_name_or_path,
            args.dtype,
            args.use_flash_attn,
            args.max_new_tokens,
            args.local_files_only,
            args.input_mode,
            args.max_video_frames,
            args.ffmpeg_path,
            args.ffprobe_path,
            args.prompt_variant,
            args.offload_folder,
        )
        runner.load()
        results: List[PredictionResult] = []
        for index, sample in enumerate(samples, start=1):
            if sample.event_json_path:
                spatial_context = summarize_event_json(load_event_json(sample.event_json_path))
            else:
                spatial_context = SpatialContext(
                    sample.spatial_context_text or "No structured event JSON was provided.",
                    sample.spatial_context_json or json.dumps({}, ensure_ascii=False),
                    sample.spatial_prior_u_norm,
                    sample.spatial_prior_v_norm,
                    sample.spatial_prior_source,
                )
            if not spatial_context.prior_u_norm:
                spatial_context.prior_u_norm = sample.spatial_prior_u_norm
            if not spatial_context.prior_v_norm:
                spatial_context.prior_v_norm = sample.spatial_prior_v_norm
            if not spatial_context.prior_source or spatial_context.prior_source == "none":
                spatial_context.prior_source = sample.spatial_prior_source or "none"
            _, prompt_text = build_prompts(sample, spatial_context, args.prompt_variant, args.input_mode)
            try:
                result = runner.predict(sample, spatial_context)
                if vis_dir is not None and runner.runtime is not None:
                    render_overlay(runner.runtime, sample, result, vis_dir)
            except Exception as exc:
                if not args.continue_on_error:
                    raise GroundingRunError(f"Failed on event_id={sample.event_id}: {exc}") from exc
                result = build_failed_result(sample, spatial_context, prompt_text, str(exc))
            results.append(result)
            print(f"Processed {index}/{len(samples)} | event_id={sample.event_id} | parse_ok={result.parse_ok}", flush=True)
        write_results(output_csv, results, output_columns)
        print(f"Saved predictions to: {output_csv}")
        if vis_dir is not None:
            print(f"Saved overlays to: {vis_dir}")
        return 0
    except GroundingRunError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
