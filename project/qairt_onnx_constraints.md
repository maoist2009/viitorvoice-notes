---
name: QAIRT 2.50 / qai-hub ONNX compile constraints
description: dtype + graph-shape requirements learned compiling w2v-bert fp16 ONNX through qai-hub to QNN DLC — float16 yes, bfloat16 hard-no, PostLayout addTensor bug is quantization-triggered, QDQ bypass via qai-hub quantize service
type: project
---

Learned 2026-10-12..14 while compiling the ViiTorVoice w2v-bert graph (1.12 GB fp16 ONNX) through `qai_hub.submit_compile_job` to a QNN DLC (device "Samsung Galaxy S25 (Family)", options `--target_runtime qnn_dlc --truncate_64bit_io`).

**Dtypes — QAIRT converter accepts only: float16, float32, int32, int8, uint8, int16, uint16, int64, bool. `bfloat16` is hard-rejected** with `Unsupported tensor dtype=bfloat16`. This kills the "bf16 to fit under the 2 GiB protobuf cap" plan. Use **float16** — same byte size as bf16, accepted everywhere, and onnxruntime-CPU can actually run it end-to-end (w2v verified at 2.6e-3 max rel diff vs torch-fp16), so a real numeric verification is possible.

**qai-hub runs a server-side ONNX shape-inference prepass before conversion.** It rejects some dtypes per-op (Conv with bf16 input: `[ShapeInferenceError] X typestr: T, has unsupported type: tensor(bfloat16)`), so a model can pass `onnx.checker` locally and still fail on upload. `onnx.checker.check_model(m, full_check=True)` locally reproduces it **faithfully** — it is the exact server prepass, not a stricter false gate. (I once removed a local full_check believing the server was more lenient; a subsequent job failed server-side with exactly the error full_check had reported. The earlier "passing" job had simply died earlier in the pipeline.) **Always iterate locally against full_check before submitting.**

## addTensor bug is QUANTIZATION-TRIGGERED, not structural

PostLayout `addTensor: Tensor transpose_10 already exists!` (ir_to_dlc.py:648 → graph_optimizer.py:925, QAIRT 2.50). Experiment matrix on the byte-identical fp16 ONNX:

| options | job | result |
|---|---|---|
| no quantization | jp3owwk3p | ✅ SUCCESS, 1121.3 MB fp16 DLC |
| int8 + `--quantize_io` | jpy877w8g | ❌ addTensor transpose_10 |
| int8 only | jpy8770lg | ❌ addTensor transpose_10 |
| int16 + `--quantize_io` | jp3oww6nnp | ❌ addTensor transpose_10 |

Triggered by something in the attention pattern `MatMul(softmax, v_t) -> Transpose(0,2,1,3) -> Reshape`. Verified NOT fixed by: prefixing all tensor names, renaming nodes to `n_<idx>`, stripping `allowzero` from all 93 Reshapes, inserting Identity between every Transpose->Reshape. Minimal 3-4 node repros all compile FINE — the trigger needs more graph context. **Don't bother with renames/allowzero/Identity; the pass itself is the problem.**

## QDQ bypass route (WORKING — prepass passes as of 2026-10-14)

qai-hub's quantize service is separate from the compile-time PTQ pass:
`qai_hub.submit_quantize_job(model, calibration_data, weights_dtype=INT8, activations_dtype=INT8)` emits a **QDQ ONNX** that bypasses the buggy pass entirely. Compile that artifact with NO quantize options. (job j5m9d8z7g; artifact delivered as a zip of `model.onnx` + external `model.data` under a `job_<id>_qdq_onnx/` subdirectory, ~785 MB zipped / 1.13 GB inlined.)

**qai-hub compile only ships the graph, not external data** — "Data of TensorProto ... should be stored in model.data, but it is not regular file". Inline first: `onnx.load(path, load_external_data=True)` then `onnx.save`. Watch the 2 GiB protobuf cap.

**The QDQ graph's interfaces were fp16, which the prepass rejects** (`[ShapeInferenceError] QuantizeLinear ... has unsupported type: tensor(float16)`). Fix with Cast-node surgery (`~/tmp/dl_vv/runner_scripts/fix_qdq_fp16.py`), which is free in the file — the converter materializes Casts at load (vs duplicating initializers, +293 MB on a 1.12 GB model).

opset-18 QDQ type rules, load-bearing:
- `QuantizeLinear (x: T1, x_scale: T1, x_zero_point: T2) -> y: T2`
- `DequantizeLinear (x: T2, x_scale: T1, x_zero_point: T2) -> y: T1`
- T1 must be fp32 (fp16 rejected). T2 MUST stay integer — casting zero_point to fp32 is an error.

**Derive each DQ output's dtype from its SCALE, never from the output annotation.** The quantize service annotates scales (initializers, always typed) but leaves DQ outputs UNTYPED. On the mixed fp16/fp32 w2v graph: 2099 DQs at fp16 scales (conformer), 46 at **fp32** scales (MelFrontend boundary). Defaulting untyped outputs to fp16 corrupts the fp32 region and full_check rejects every downstream op (`Inferred elem type differs from existing elem type: (10) vs (1)`). Also cache one Cast per source tensor — per-tensor quantization means several QDQ nodes share one scale, and a naive per-node Cast breaks SSA (`has been used as output names multiple times`).

## Folding the weight-side Q (the actual fix, 2026-10-14) — PROVEN, compiles clean

The deeper problem: qai-hub's quantize service stored the weights as **fp16 initializers** and expressed their int8 quantization as `fp16 w -> Q -> int8 -> DQ -> fp16 -> Conv` pairs meant to be constant-folded by the compiler. Every Q's x is therefore fp16 — exactly what the prepass rejects — and the "quantized" artifact is still 1.13 GB (int8 initializers total only 0.4 MB). Converting those weights to fp32 reaches 2.24 GB, past the 2 GiB protobuf cap.

**Fix: fold the weight Q ourselves in numpy.** For each Q whose x is a constant fp16 initializer, compute `y = clip(round(x/scale) + zp, lo, hi)` (round-half-to-even), store the result as an int8 initializer, delete the fp16 weight + the Q node, and point the DQ at the int8 data. The result is the canonical `int8 w -> DQ(fp32 scale) -> Conv` weight pattern, the model SHRINKS 1130 -> 571 MB, and zero fp16 op inputs remain. Notes: per-channel scales broadcast along the weight axis whose length equals `scale.shape[0]` (dim 0 for conv/linear); Q and DQ scales are initializers, never value_info; also drop the stale value_info wholesale and re-run `shape_inference` (the service inserts QDQ without updating the export's annotations), and cut any `Cast(fp32->fp16)` so the conformer no longer runs in fp16.

**RESULT (job jgk68x8n5): SUCCESS.** Compiled to `--target_runtime qnn_dlc
--truncate_64bit_io` on Samsung Galaxy S25 (Family), ~30 min in
OPTIMIZING_MODEL (all failed attempts died within 5-6 min — a long optimize
stage is a good signal the graph is actually being processed). Output DLC
577.5 MB vs the unquantized fp16 DLC's 1121 MB. **This is now the default
route for any graph that hits the addTensor bug under PTQ.**

Do NOT try to keep weights fp16 by inserting Casts around the ops instead — ONNX's opset-18 type constraints make every op require all fp inputs to share a dtype (`MatMul: B has inconsistent type tensor(float)`, `Conv: W has inconsistent type tensor(float)`), so a fp32 DQ output cannot feed an op whose other input is fp16. Either everything at an op is fp32 or everything is fp16. A Cast between DQ and Conv additionally breaks the converter's weight recognition (it then reads the per-channel weight scale as an activation scale: "Cannot apply per channel quantization on activation").

**Why:** w2v was the last ViiTorVoice graph needing a DLC; llm/encoder/decoder/unet DLCs were already done.

**How to apply:** for any qai-hub compile of a >2 GiB fp32 model, go straight to fp16 (never bf16); run `onnx.checker full_check=True` before uploading; on the addTensor bug, skip the QAIRT PTQ pass — either drop quantization options or pre-quantize via `submit_quantize_job` and do Cast surgery on the QDQ interfaces.
