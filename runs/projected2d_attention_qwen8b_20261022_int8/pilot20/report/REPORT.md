# Projected-2D pilot20

Model: Qwen3-VL-8B; quantization=int8; no comparison to historical model results.

**PROVISIONAL_GEOMETRY_NOT_CALIBRATED**: RGB/telemetry alignment is unverified. The 2026-10-23 GT-free motion audit covers the fixed 20 events and supports clock-alignment concerns; candidate residuals are lower in 16 events and higher in 4. These saved scores cannot establish heatmap/foveation efficacy. See diagnostics/REVIEW_20261023_zh.md at the experiment root.

|Condition|Successful calls / events|Temporal F1|Point@100 F1|Joint@100 F1|Joint@200 F1|Delta Joint@100|
|---|---:|---:|---:|---:|---:|---:|
|RGB|16/20|0.4835|0.1538|0.1099|0.2198|+0.0000|
|CURRENT_HEATMAP|16/20|0.5217|0.1087|0.0870|0.2609|-0.0229|
|HISTORY_HEATMAP|16/20|0.5714|0.1538|0.0879|0.2418|-0.0220|
|HISTORY_FOVEATION|16/20|0.5000|0.1087|0.0652|0.2391|-0.0447|

All fixed events, including upstream failures, remain in the original evaluator denominator.
The original projected-anchor proxy and matching rules are unchanged. No mention-level accuracy is asserted.
Inputs reuse multi_image + paired_crop; prompts use instruction and frozen extracted mentions, without annotation-derived expected_count.
These are given-expression extension results, not an exact reproduction of the historical GT-count-conditioned wrapper.

Saved examples draw projected GT only after inference. They are never inference inputs.

## Actual inputs and outputs

All conditions use the same instruction, frozen Referential mentions, source frames, paired-crop bounding boxes, media form, decoder settings, and expanded image-token budget.
The original protocol selects three evidence panels per event. This diagnostic is not the earlier 34–100-frame video experiment.
Each condition is one multi-image Visual call per event with all frozen mentions; outputs retain per-mention status, frame_id, point_uv and evidence. No Coordinator/fusion call is included in this Visual diagnostic.

|Condition|Actual visual input|Output|
|---|---|---|
|RGB|Original source pixels; original gaze-centred paired crop|Per-mention normalized source-frame points|
|CURRENT_HEATMAP|Current gaze reprojected onto the reference camera, overlay before identical paired crop|Same schema|
|HISTORY_HEATMAP|Past 2 seconds of observed gazePoint reprojected using each reference camera; mild overlay|Same schema|
|HISTORY_FOVEATION|Same numerical history heatmap; smooth peripheral weakening with retained RGB detail|Same schema|

RGB already contains the original gaze-centred crop. The comparisons measure adding overlays/foveation to that existing input, not introducing gaze to a gaze-free baseline.
Original evaluation is event/referent level, using projected 3D anchor points and acceptable panels. It is not independent hand-labelled per-mention pixel GT, and no mention accuracy is reported.
The adapter flattens all per-mention point predictions into the original referents schema. It does not use GT to merge repeated mentions or remove false positives; the original one-referent matching may penalize repeated predictions of one physical target.

## Full original metrics

|Condition|Point@50|Point@100|Point@150|Point@200|Joint@50|Joint@100|Joint@150|Joint@200|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|RGB|0.0879|0.1538|0.1758|0.2857|0.0440|0.1099|0.1099|0.2198|
|CURRENT_HEATMAP|0.0217|0.1087|0.1957|0.3913|0.0217|0.0870|0.1304|0.2609|
|HISTORY_HEATMAP|0.0220|0.1538|0.1978|0.3516|0.0000|0.0879|0.1099|0.2418|
|HISTORY_FOVEATION|0.0870|0.1087|0.1957|0.2826|0.0435|0.0652|0.1522|0.2391|

## Visual budget and recorded cost

Each completed event uses three source frames and three paired full-frame/crop images. Crops do not add source frames.
Image tokens below are counted from the actual expanded image placeholders, not total prompt tokens.
Inference seconds sum the saved original calls, including exact-match replayed calls; they are not the wall time of the last resume.

|Condition|Visual calls|Images|Image tokens / call|Inference seconds|Parse-warning events|
|---|---:|---:|---|---:|---:|
|RGB|16|48|[3528]|2384.0|1|
|CURRENT_HEATMAP|16|48|[3528]|2402.1|0|
|HISTORY_HEATMAP|16|48|[3528]|2363.7|0|
|HISTORY_FOVEATION|16|48|[3528]|2211.1|0|

## Effective attention strength

Statistics use all fixed source panels, including panels from upstream-failure events.

|Map|Panels|Zero-map panels|Median maximum alpha|Median pixel fraction alpha > 0.01|
|---|---:|---:|---:|---:|
|current|60|0|0.0558|0.0021|
|history|60|0|0.2349|0.0111|

## Paired output changes

A changed prediction is evidence of a response to the input, not evidence of a correct response.

|Condition|Completed pairs with changed localization|Joint@100 TP increased / decreased / unchanged|
|---|---:|---|
|CURRENT_HEATMAP|15/16|1 / 2 / 13|
|HISTORY_HEATMAP|14/16|1 / 2 / 13|
|HISTORY_FOVEATION|15/16|1 / 3 / 12|

## Delivery verification

Captured full input IDs and real vision-forward tensor hashes for 64 successful calls.
Fully paired events: 16; identical actual input IDs: 16; identical image grids: 16.
Decoded system/user prompts, processor-restored paired images, pixel deltas and file hashes are indexed under actual_model_inputs/provenance.json.
Delivery and prompt instructions are verified. Whether the model benefits from the cue is evaluated by the paired metrics, not inferred from the tensor audit.
The stored alpha=0.28 is a cap, not the actual tint strength. attention_strength.csv records effective alpha after duration weighting and rendering for every source panel.
Source P1/P2/P3 IDs identify exact saved JPEGs by path and SHA-256; video_time is the original extractor's requested seek time. Exact decoded video PTS/frame number is not captured by the original extractor.
Current gaze uses the nearest valid observation within its configured tolerance and can fall slightly after the panel time; history uses observations through the reference time. The overall experiment is offline.
Finite gazePoint intersections are world coordinates; gazeVector fallback is a direction. Historical points are reprojected to the reference camera, but occlusion and dynamic-object correspondence cannot be verified without depth/mesh.
Original source hashes and package versions are saved in environment_and_sources.json. Coordinate consistency over the fixed 20 events is saved at the experiment root in coordinate_audit.json; this does not validate video clock alignment.

## Inherited annotation limitation

original_annotation_parity.json compares all rebuilt GT pixel coordinates, projection validity and status to the original released manifest. The prepared subset has 153 rows and 0 differences.
Posthoc image inspection reveals visible offsets in some projected-anchor labels: for example, scene4_room2/145 P1 places the person proxy near the wall rather than the visible person at the left. This is an inherited proxy/alignment concern, not a new manual pixel annotation or a verified clock calibration.
The cause (clock alignment, anchor placement or camera projection) has not been isolated. The evaluator and labels remain unchanged; the reported pilot metrics must not be interpreted as validated human-labelled pixel localization accuracy.
Equal image tokens and shared GPU load mean the elapsed differences do not establish inference acceleration from foveation.

## Failures

- scene2/592 RGB: NOT_RUN; ValueError: parent_text 'laptop' must be a literal span inside mention 'battery'
- scene4_room3/151 RGB: NOT_RUN; ValueError: parent_text 'the chair' must be a literal span inside mention 'its front legs'
- scene4_room4/163 RGB: NOT_RUN; ValueError: parent_text 'the curtain fabric' must be a literal span inside mention 'its left fold'
- scene2/416 RGB: NOT_RUN; ValueError: parent_text 'that file' must be a literal span inside mention 'its label'
- scene2/592 CURRENT_HEATMAP: NOT_RUN; ValueError: parent_text 'laptop' must be a literal span inside mention 'battery'
- scene4_room3/151 CURRENT_HEATMAP: NOT_RUN; ValueError: parent_text 'the chair' must be a literal span inside mention 'its front legs'
- scene4_room4/163 CURRENT_HEATMAP: NOT_RUN; ValueError: parent_text 'the curtain fabric' must be a literal span inside mention 'its left fold'
- scene2/416 CURRENT_HEATMAP: NOT_RUN; ValueError: parent_text 'that file' must be a literal span inside mention 'its label'
- scene2/592 HISTORY_HEATMAP: NOT_RUN; ValueError: parent_text 'laptop' must be a literal span inside mention 'battery'
- scene4_room3/151 HISTORY_HEATMAP: NOT_RUN; ValueError: parent_text 'the chair' must be a literal span inside mention 'its front legs'
- scene4_room4/163 HISTORY_HEATMAP: NOT_RUN; ValueError: parent_text 'the curtain fabric' must be a literal span inside mention 'its left fold'
- scene2/416 HISTORY_HEATMAP: NOT_RUN; ValueError: parent_text 'that file' must be a literal span inside mention 'its label'
- scene2/592 HISTORY_FOVEATION: NOT_RUN; ValueError: parent_text 'laptop' must be a literal span inside mention 'battery'
- scene4_room3/151 HISTORY_FOVEATION: NOT_RUN; ValueError: parent_text 'the chair' must be a literal span inside mention 'its front legs'
- scene4_room4/163 HISTORY_FOVEATION: NOT_RUN; ValueError: parent_text 'the curtain fabric' must be a literal span inside mention 'its left fold'
- scene2/416 HISTORY_FOVEATION: NOT_RUN; ValueError: parent_text 'that file' must be a literal span inside mention 'its label'
