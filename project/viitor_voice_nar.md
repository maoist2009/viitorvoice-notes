---
name: ViiTorVoice-NAR on-device TTS candidate
description: Research findings on ViiTorVoice-NAR (viitor-ai) as a phone-NPU TTS candidate — architecture, sizes, license, QNN feasibility; corrects the "1B main model" claim
type: project
---

ViiTorVoice-NAR (github.com/viitor-ai/viitor-voice-nar, HF ZzWater/ViiTorVoice-NAR) evaluated 2026-07-14 as a candidate on-device TTS for the sm8750 NPU project.

**Why:** User flagged it as "currently the most advanced on-device TTS, main model ~1B" and asked for research (no model clone). Subagent cloned the CODE repo only to `vv_nar_src/`; model weights NOT downloaded.

**Key correction — "1B main model" is wrong for the released artifact.** The published checkpoint uses **Qwen3-0.6B (~0.6B params)** as the NAR LLM backbone. The only "1B-class" string is `Qwen3-1.7B` inside `train_config.json`, which is a *training template*, not a shipped model. Full voice-cloning inference ≈ **1.2B total** (0.6B LLM + ~0.6B W2V-BERT 2.0 encoder) + tiny DualCodec.

**Architecture:** non-autoregressive **masked discrete diffusion** LLM (MaskGit-style iterative unmasking, 32 steps default, consistency-distilled to 8), inspired by OmniVoice + DualCodec. Pipeline: text tokenizer (char-level zh/ja/ko, subword en) → DualCodec encoder = **W2V-BERT 2.0** Conformer (semantic ref-audio features) → NAR LLM generates all 12 DualCodec codebook layers at once → DualCodec decoder → waveform. 4 langs: zh/en/ja/ko. Zero-shot **+ no-ref-text** voice cloning (~1s ref wav). Local speech editing via separate Qwen3-ForcedAligner-0.6B. Rich emotion/paralinguistic tags + CFG.

**License:** Apache-2.0 (repo + Qwen3 + DualCodec/OmniVoice) + **MIT** (W2V-BERT 2.0, Meta Seamless — NOT Google USM). Commercial-friendly, no known blocker.

**ONNX / QNN path (STATUS UPDATED 2026-10-14): ALL FIVE GRAPHS COMPILED TO DLC.**
The port is done at the model layer — every graph is a `.dlc` on
`--target_runtime qnn_dlc --truncate_64bit_io` for Samsung Galaxy S25 (Family):

| graph | artifact | size | status |
|---|---|---|---|
| w2v-bert-2.0 | w2v_int8.dlc | 577 MB | job jgk68x8n5 — int8 weights via the weight-Q fold (see `qairt_onnx_constraints.md`); fp32 IO, `input_values[1,16000]` -> `hidden_states[1,48,1024]` |
| Qwen3-0.6B backbone | llm.dlc | 442 MB | on phone |
| unet (diffusion) | unet.dlc | 393 MB | on phone |
| dualcodec decoder | decoder.dlc | 67 MB | on phone |
| dualcodec encoder | encoder.dlc | 50 MB | on phone |

The app-side loader exists: `qnn_load_dlc` in `app/src/main/cpp/qnn_executor.cpp`
(`QnnSystemDlc_createFromFile` + `composeGraphs` + `graphRetrieve`), exposed as
`RvcJni.loadGraphDlc`. See `qnn_dlc_loading.md`. Remaining work is the C++
pipeline chaining w2v -> llm -> unet -> decoder.

**STATUS 2026-10-24: all graphs now exist as executable context binaries, and
the decoder is solved.** Measured exec-only on a real SM8750 with QNN 2.51.40
(context already loaded, zero-filled inputs):

| graph | bin | exec/inf | note |
|---|---|---|---|
| w2v | w2v_250.bin 570 MB | 39 ms | fp32 IO, `input_values[1,16000]` (1 s @16 k) |
| llm | llm_245.bin 426 MB | 75 ms | 256-token context, non-autoregressive |
| encoder | encoder_245.bin 72 MB | **5801 ms** | `semantic_repr[1,1024,600]` + `audio[1,1,240000]`; once per prompt |
| unet | unet_245.bin 422 MB | 784 ms | `sample[1,4,128,128]`, 4 inputs |
| decoder | 5 segments, 132 MB | 6660 ms | per **8 s** of audio → 833 ms/s |

The decoder (the hardest part — it would not link as one graph) is
p1 fp16 + p2..p5 w8a16, SNR 19.89 dB / log-mel MAE 0.154, **1.20× realtime**.
Full recipe, cut points, calibration rules and traps: `decoder_split_a16.md`.
So the model layer is DONE; only the C++/JNI orchestration is left.

**Legacy export history (2026-10-11, runner-side):** w2v exported via
- `local_models/dualcodec/w2v-bert-2.0/w2v_bert_2_0.onnx` (3.7 MB graph + **2.24 GB `.onnx.data` external**) — EXPORT SUCCEEDED via `torch.export.export` → `torch.onnx.export(dynamo=True)` (TorchScript path crashes on torch 2.11 Termux: `'torch._C.Node' has no attribute 'b'`). ir 10, opset 20, `input_values[1,16000]` → `hidden_states[1,48,1024]` (penultimate block, mel frontend baked in). Script: `tools/export_w2v_onnx_dynamo.py`. Weights `model.safetensors` 2.3 GB + `conformer_shaw.pt` 1.07 GB both local.
- `local_models/dualcodec/dualcodec_ckpts/dualcodec_encode_core_30s.onnx` (197 MB) + `dualcodec_decoder.onnx` (268 MB) — present, from upstream.
- `local_models/llm/0p6_emotion/model.safetensors` (2.2 GB) — present.
- **`llm/0p6_emotion/cache/onnx_backbone_fp32/llm_backbone_dynamic.onnx` (651 MB) is CORRUPT** — `onnx.load` → `DecodeError: Truncated message` (export was interrupted 07-17 03:41, before the weights finished downloading 06:58). This was superseded by the clean runner-side `.pt2` re-export (llm_backbone.pt2, 1985 MB) that became llm.dlc.
- NO QNN/DLC/.bin conversion of any viitor graph exists yet. ← OBSOLETE as of 2026-10-14; see the table above.

**Feasibility verdict:** plausible on sm8750 — 0.6B LLM fits NPU trivially at w8a16/w4a16; W2V-BERT (~0.6B Conformer) is the second quantization target. 8 distilled steps → real-time-ish. No mobile runtime/benchmark exists.

**"Most advanced" claim assessment:** overstated but genuinely strong. Strengths: true NAR (~60ms first frame), no-ref-text cloning, local editing (unique), Apache/MIT, ONNX-ready, small footprint. Weaknesses: only 4 langs (CosyVoice2/F5/MegaTTS3 broader), no published MOS/RTF, self-published 2026-06, heavy W2V-BERT dependency, server/TensorRT-oriented. Competitive-to-leading on NAR latency + cloning + editing; not clearly "most advanced" overall.

**How to apply:** If user pursues on-device TTS on this NPU, ViiTorVoice-NAR is a strong zh/en/ja/ko candidate — model the QNN port after DreamLite (quantize fp32 ONNX → QNN context binaries; host-side diffusion loop). Don't take the "1B" number at face value.
