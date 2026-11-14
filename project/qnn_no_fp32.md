---
name: QNN target does not support fp32
description: Hard constraint for DreamLite QNN NPU export — compiled model must be fully w8a16, never fp32
type: project
---

QNN (the DreamLite Qualcomm NPU target) does NOT support fp32 compute. The model compiled for QAI Hub → QNN must be fully **w8a16 (INT8 weights + INT16 activations)** with no fp32 execution paths.

**Why:** User explicit directive ("记住不能fp32，qnn不支持" / QNN does not support fp32). A model with raw fp32 activation/compute tensors will fail QNN compilation or produce a broken binary.

**How to apply:** Never upload/compile an fp32-activation intermediate. If weights are pre-quantized to int8 locally (to get the model under the 2GiB C++ protobuf limit), you MUST still run activation quantization (int16) — e.g. ORT quantize_static with activation_type=QInt16 — so the final compiled artifact is w8a16. Intermediate models with DequantizeLinear (fp32 logical dtype) are fine ONLY as a transient local step; QNN fuses the QDQ pattern, but a model left with un-quantized fp32 activations must not reach QAI Hub compile.

**Scope clarification (2026-10-24, learned on the decoder split):** the
constraint is about *compute* paths, not IO or precision modes.
- **fp32 graph IO is fine** and is what the decoder segments use — it is the
  only safe boundary when chaining graphs, because `--quantize_io` uses a
  fixed heuristic range that ignores calibration data and saturates.
- **fp16 graphs also work** (the decoder's p1 is fp16 and measures 48.1 dB
  SNR). `bf16` remains hard-rejected; use fp16.
- For small models prefer w8a16 (a16) over w8a8: on the decoder it was
  SNR 11.9 → 19.9 dB for 2× the runtime.
