# VR-TriRef 8B Visual-only paired pilot

Offline diagnostic pilot, not a complete-dataset result. No training or official development split.

|Group|N|Any-hit|Exact-set|Micro-P|Micro-R|Micro-F1|Failed|
|---|---:|---:|---:|---:|---:|---:|---:|
|R|100|0.670|0.220|0.608|0.424|0.500|2|
|H_inst|100|0.670|0.210|0.607|0.429|0.502|2|
|H_local|100|0.670|0.210|0.615|0.437|0.511|2|
|F_inst|100|0.650|0.220|0.651|0.408|0.501|2|
|F_local|100|0.630|0.210|0.625|0.399|0.487|2|

All main groups replay the same real 8B Referential, scene candidate mapping, direct event plan and raw-RGB Hand observations. Visual calls use the same granularity. Coordinator prompt/rules/images remain fixed; it recomputes against each group's new evidence.

Gaze points are measured world intersections, not target IDs. Historical reprojection is frustum-only and has no occlusion/dynamic-object correspondence guarantee. Camera or clock failures retain RGB and stay in the denominator.

The existing coordinator still receives original gaze numeric/text cues. These comparisons measure additional Visual encoding, not total gaze contribution. Instant and local share their respective numerical H across heatmap/foveation; all images keep geometry and resolution.

Metrics above are generated from this run only. Clock coverage, explicit RGB fallbacks and per-event changes are reported in AUDIT.md and paired_comparison.json; pilot outcomes are not complete-dataset conclusions.

See geometry_audit.json, failures.json, failure_analysis.json, budget.json, call_audit.json and per-sample source/heatmap/projection records.
