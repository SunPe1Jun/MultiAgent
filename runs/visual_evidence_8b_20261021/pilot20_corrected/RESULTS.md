# VR-TriRef 8B Visual-only paired pilot

Offline diagnostic pilot, not a complete-dataset result. No training or official development split.

|Group|N|Any-hit|Exact-set|Micro-P|Micro-R|Micro-F1|Failed|
|---|---:|---:|---:|---:|---:|---:|---:|
|R|20|0.750|0.200|0.677|0.438|0.532|0|
|H_inst|20|0.750|0.200|0.724|0.438|0.545|0|
|H_local|20|0.800|0.200|0.759|0.458|0.571|0|
|F_inst|20|0.750|0.200|0.808|0.438|0.568|0|
|F_local|20|0.650|0.200|0.731|0.396|0.514|0|

All main groups replay the same real 8B Referential, scene candidate mapping, direct event plan and raw-RGB Hand observations. Visual calls use the same granularity. Coordinator prompt/rules/images remain fixed; it recomputes against each group's new evidence.

Gaze points are measured world intersections, not target IDs. Historical reprojection is frustum-only and has no occlusion/dynamic-object correspondence guarantee. Camera or clock failures retain RGB and stay in the denominator.

The existing coordinator still receives original gaze numeric/text cues. These comparisons measure additional Visual encoding, not total gaze contribution. Instant and local share their respective numerical H across heatmap/foveation; all images keep geometry and resolution.

Pilot outcome: H_local has the highest Micro-F1 (0.571 vs R 0.532) and Any-hit (0.800 vs 0.750); Exact-set is 0.200 in every group. F_inst Micro-F1 is 0.568 and F_local is 0.514. All four gaze variants changed Visual input pixels in 7/20 events; the other 13/20 were full-RGB fallbacks after clock rejection. These are small-pilot observations, not evidence of a stable or significant gain.

See geometry_audit.json, failures.json, failure_analysis.json, budget.json, call_audit.json and per-sample source/heatmap/projection records.
