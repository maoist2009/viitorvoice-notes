---
name: ViiTorVoice decoder split into 5 context binaries + w8a16 recipe
description: The dualcodec decoder cannot be linked as one graph (host preparer OOM) — the shipped config is p1 fp16 + p2..p5 w8a16, 5 contexts, halo 48; device-measured SNR 19.89 dB, 6.66 s per 8 s audio
type: project
---

**FINAL SHIPPED CONFIG (2026-10-24) — decoder = 5 context binaries, 1 graph each.**
Artifacts in `salvage/dl_vv/dec_split/` (p1) + `dec_split/a16/` (p2..p5).
Device-measured on SM8750 with QNN 2.51.40, 8 s of audio (191996 samples @24 kHz):

| part | precision | bin | exec/inf (net-run) | exec/inf (in-app) | graph output |
|---|---|---|---|---|---|
| p1 | **fp16** | 84 MB | 321 ms | **53 ms** | `convolution` [1,768,1600] |
| p2 | w8a16 | 19 MB | 433 ms | **72 ms** | `convolution_1` [1,384,9600] |
| p3 | w8a16 | 6 MB | 673 ms | **140 ms** | `convolution_2` [1,192,47999] |
| p4 | w8a16 | 16 MB | 2613 ms | **532 ms** | `convolution_3` [1,96,191996] |
| p5chunk48 | w8a16 | 6 MB | 654 ms × 4 | **162 ms × 4** | `audio` [1,1,47999] per chunk |
| total | | **132 MB** | 6.66 s | **1.45 s (5.2× realtime)** | |

**The qnn-net-run column overstates the cost by ~4.5×** — it writes every output
tensor to disk on every inference (p4 alone dumps 74 MB per run), and that I/O
dominates its timing loop. The in-app column is the shipped app's
`qnn_timings_json()` (pure `graphExecute`). Plan with the right-hand numbers.

Quality vs fp32 reference: **SNR 19.89 dB, corr 0.9949, log-mel MAE 0.154**
(<0.1 ≈ transparent; same-SNR white noise would be MAE 2.07). Band SNR:
0–500 Hz 22.3 / 500–1k 19.4 / 1–2k 14.7 / 2–4k 11.4 / 4–8k 11.2 / 8–12k 13.2 dB.
The earlier w8a8 tail gave only SNR 11.93 / MAE 1.06 — **a16 is a different
class, not an incremental gain**; HTP int16 activations run at half int8
throughput, which is exactly the 3.27 s → 6.66 s cost.

## Why split at all

The whole decoder graph (268 MB ONNX, 200 frames) **cannot be linked**: the
host-side preparer eats 346 GB and gets OOMKilled on 2.45, 2.50 **and** 2.51.
Shrinking to 100 frames does not help. Splitting is a *build-time* necessity,
not a runtime choice — measured runtime cost of the split is <1% (p5: one
48095-sample call 654 ms vs 4 calls 2616 ms, i.e. per-call overhead ≈ 0).

## Cut points

Cut right after each of the 4 `ConvTranspose` nodes, giving
`convolution` / `convolution_1` / `convolution_2` / `convolution_3` / `audio`.
Verified: whole graph vs 5-part ORT chaining = **max abs diff 0.0**.

## Rules that cost real time to discover

- **Do NOT pass `--quantize_io` on the segments.** qai-hub's IO quantization
  uses a fixed heuristic range (p1 emitted [-2.0, 2.3], p2 wanted [0,1]) and
  ignores calibration data, so chaining saturates. Keep segment boundaries
  **fp32** and quantize inside only.
- **Feed real calibration data** (2 samples taken from the ORT boundary
  tensors); p1's `codes` must be **int64**.
- **p1 must be fp16, not int8.** An int8 graph with an fp32 output link stalls
  in `LINKING_MODELS` for >1 h (`jp09ze76p`, `jg92lknlp` both ended FAILED);
  the same graph as fp16 links in 30 s and is numerically transparent
  (SNR 48.1 dB, corr 0.99999).
- **p5 must be time-chunked**: its `InputSlicePad` needs 12.3 MB > 8 MB VTCM.
  Chunk = 47999 samples, **halo = 48** (measured receptive-field radius is 42,
  not 12 — halo 12 gives max err 0.24, halo 48 gives rms 2.4e-4 from float
  tail only). Zero-pad the first/last chunk, drop 48 samples on each side.

## Zero-upload recompiles from the phone (no runner needed)

`qai_hub.CompileJob` exposes `.model` and `.calibration_dataset`, so an a16
variant is a pure re-submit with **nothing uploaded**:

```python
qai_hub.submit_compile_job(model=src.model, calibration_data=src.calibration_dataset,
                           options="--target_runtime qnn_context_binary ... w8a16 ...")
```

Driver: `salvage/dl_vv/a16_from_phone.py` (`submit|poll|link|poll-link|fetch`),
job ids in `salvage/dl_vv/a16_jobs.json` (compile `jgl8vjzmp`, `j56mykjyp`,
`jgzy319xp`, `j5wzejv45`). w8a8 source jobs: p2 `jgjyvky7g`, p3 `jpexd4x7g`,
p4 `jgzy3vyzp`, p5chunk48 `jg92l82qp`, p1 fp16 link `jp37jxvlg`.
**A qai-hub *link* job does produce a context binary** — see the correction in
`qnn_dlc_loading.md`.

## Runtime traps

- **`qnn-net-run` parses input files as float32 by default.** Feeding int32
  `codes` without `--use_native_input_files` silently turns them all into 0 and
  reports success. Always pass it for this model.
- Input list format: one line per inference, just the file paths, no tensor
  names/dims. Outputs land in `<out_dir>/Result_0/<tensor>.raw`.
- On arm64 outputs are written in native dtype (uint8 = raw bytes); the x86
  backend dequantizes quantized outputs to float32.
- QNN dequant: `real = scale * (q + offset)` (offset -120 → zero point at q=120).
- Memory the app must budget: weights 131 MB; per-context peak RSS p1 98 /
  p2 33 / p3 64 / **p4 120** / p5 32 MB. Largest single buffer is p4's fp32
  output, 73.7 MB; p5 needs only an 18.5 MB halo buffer on top of it (slice the
  p4 output, do not copy it).

## Verifying on device

`salvage/dl_vv/verify_chain.py <bins_p1..p4 dir> <work> <out> <tail_bins_dir>`
runs the full chain with `qnn-net-run` and writes `got.npy`;
`metrics.py <got.npy> <ref_audio.npy>` prints SNR / band SNR / log-mel MAE;
`timing.py` and `mem.py` separate context load from exec and sample peak RSS.
Reference tensors (`ref_audio.npy`, `ref_sem.npy`, `ref_aco.npy`,
`ort_p1_ref.f32`) live in `dec_split/`.

**How to apply:** any future decoder work starts from this config. If more
speed is needed the lever is finer time-chunking of p3/p4 (their 7–18 MB
activations far exceed the 8 MB VTCM and stream through DDR), *not* merging
segments — p4+p5 cannot be merged at all because of the VTCM-bound
`InputSlicePad`. User declined that optimisation on 2026-10-24.
