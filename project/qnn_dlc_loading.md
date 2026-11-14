---
name: Loading a .dlc at runtime with QnnSystemDlc (vs a prebuilt .bin context binary)
description: A qai-hub qnn_dlc is UNPREPARED and CANNOT be executed on-device — graphFinalize is mandatory and OOMs the phone above ~200 MB; use a qai-hub compile/link job targeting qnn_context_binary (or qairt-dlc-prepare on a big-RAM host)
type: project
---

**CONFIRMED 2026-10-14 by direct device test — the "app consumes .dlc directly"
design does NOT work for qai-hub DLCs. Read this before any DLC-loading work.**

## The blocker, with the exact error

A qai-hub `--target_runtime qnn_dlc` artifact is **unprepared**: unzip the .dlc
and it contains only `dlc.metadata*`, `model`, `model.params`, `model.params.bin`
— **no `QNN_SYSTEM_DLC_RECORD_PREFIX_HTP_CACHE_RECORD`**. It is QNN IR topology +
weights, nothing compiled for Hexagon. (Verified by unzipping all 6 ViiTorVoice
DLCs.)

An unprepared composed graph is NOT executable. The required sequence is
**compose → finalize → execute**:

1. `QnnSystemDlc_createFromFile` → `contextCreate` (empty) →
   `QnnSystemDlc_composeGraphs` → `graphRetrieve(name)`. All of this works fine
   (577 MB w2v composes + binds in ~2.4 s — the weights stay mmap'd, so loading
   is cheap).
2. **`graphFinalize` is mandatory** before `graphExecute`. Without it, execute
   fails `0x1787` = 6023 = `QNN_GRAPH_ERROR_GRAPH_NOT_FINALIZED`. `qnn-net-run`
   itself follows compose → "Finalizing Graphs" → execute, so the call IS part of
   the real flow even though `QnnSystemDlc.h` never mentions it. (Earlier note
   claiming graphFinalize was "not part of the documented flow" was wrong — the
   header just omits it.)
3. On-device `graphFinalize` of the 577 MB w2v graph fails after **16 s** with
   **`0x3ea` = 1002 = `QNN_COMMON_ERROR_MEM_ALLOC`**. This is exactly the OOM the
   user predicted ("不要跑了。必定oom") — now confirmed with the error code. It is
   a genuine allocation failure, not a fixable config mistake.

## Why the .bin route works and the .dlc route does not

A `.bin` is a *prebuilt context binary*: the HTP prepare already ran (off-device
or locally) and the serialized context holds the DSP-compiled graph, so
`contextCreateFromBinary` + `graphRetrieve` is immediately executable. The
prepare step is what `qairt-dlc-prepare` performs; that is the step that OOMs.

Phone-side prepare ceiling: **HuBERT at 200 MB succeeded**; **w2v at 577 MB
OOMs**. The ViiTorVoice set is encoder 50 / decoder 67 / unet 393 / llm 442 /
w2v 577 MB — the small two are under the ceiling, the big three are not.

**CORRECTED 2026-10-24: qai-hub DOES produce prepared context binaries.**
The earlier claim here ("no `qnn_context_binary` target") was wrong. A
`submit_compile_job` whose `options` end in `--target_runtime
qnn_context_binary` is accepted, and a **link job**
(`submit_link_job`) combines several compiled models into a single context
binary. All 5 ViiTorVoice graphs plus the 5 decoder segments were produced
this way — see `decoder_split_a16.md`. `qnn_dlc` really is unprepared (that
part stands), but the fix is a qai-hub compile/link job, not a phone-side
`qairt-dlc-prepare`.

## The tflite target is ALSO unprepared (verified 2026-10-14)

Pulled the farm runtime log (`download_job_logs`, 318 KB) for inference job
`jgk68qey5` on a real SM8750 (SM-S931U1) and read it line by line. Two findings
that correct earlier assumptions:

1. **A qai-hub tflite is not precompiled either.** The farm app ("Tungsten",
   `ai.tetra.tungsten`) loads it with plain LiteRT 1.4.2 / TFLite 2.19 and applies
   the **QNN TFLite delegate API 0.24.0** (same API as our
   `QnnTFLiteDelegate.h`) — the HTP prepare runs **on-device at first load** in
   `libQnnHtpPrepare.so`. `No delegates specified; using compute unit=all.`
2. **A qai-hub inference job SUCCESS does NOT mean it ran on the NPU.** The HTP
   delegate prepare failed on the farm *identically* to my local run — same
   `UnPack` INT32 rejection, same `requires 0xa6b000 bytes of TCM > 0x800000`,
   same `qnn_graph_finalize failed. Error 1002` — then Tungsten printed
   `Delegate QNN/HTP failed to prepare. Retrying without this delegate.` and fell
   back: **GPUV2/OpenCL took 316/437 nodes (80 s OpenCL compile), XNNPACK took
   91/122 of the remainder, `Model is fully delegated=false`**, and it ran 100
   iterations in 58.9 s = **0.589 s/inference on Adreno GPU + CPU, never on HTP.**

**How to apply:** never cite a green qai-hub inference job as proof a model runs
on NPU — read `download_job_logs` and grep for
`Applied .* delegates` / `failed to prepare. Retrying without` first. The
int8-quantized decoder graph has an intrinsic >8 MB VTCM op on SM8750; that is a
property of the compiled graph, not a local-config mistake.

## Consequence

The prepare must run where RAM is plentiful (the cloud runner), producing a
`.bin` that the app loads via the proven `qnn_load` path. Do NOT attempt
on-device prepare for graphs above ~200 MB — it is both forbidden by the user and
empirically an OOM.

## API details still correct

- **`composeGraphs` does NOT return a graph handle.**
  `QnnSystemContext_GraphInfoV1_t` has `graphName` + tensor descriptors only;
  call `graphRetrieve` by name afterwards. Compose only REGISTERS graphs on the
  context. Getting this wrong is a compile error (`no member named 'graphHandle'`).
- **Free the CONTEXT before the DLC handle** — the DLC owns the weight memory the
  composed graphs reference. `free_graph` does `graphFinalize` → `contextFree` →
  `QnnSystemDlc_free`.
- Both entry points live in `QnnSystemInterface_t`'s version-1 union (System API
  1.9 → member `v1_9`), resolved via `QnnSystemInterface_getProviders` from
  `libQnnSystem.so`, NOT the HTP `QnnInterface_getProviders`.
- **`libQnnHtpPrepare.so` must ship in `app/src/main/jniLibs/arm64-v8a/`** (88 MB)
  or composeGraphs aborts with `HTP Prepare backend loading failed`.
- The GraphInfoV1 tensor descriptors are the same struct the `.bin` path uses, so
  `describe()` / `bind_graph()` are shared verbatim.
- **A C++ `goto` fail-path cannot jump past initialized locals** — declare every
  pointer the fail label frees (and any timing locals) BEFORE the first `goto`.
- qnn error decode: `QNN_MIN_ERROR_COMMON 1000` (1002 = MEM_ALLOC),
  `QNN_MIN_ERROR_GRAPH 6000` (6021 OPTIMIZATION_FAILED, 6022 FINALIZE_FAILED,
  6023 GRAPH_NOT_FINALIZED), `QNN_MIN_ERROR_MEM 8000`.

**How to apply:** the app executes `.bin` (context binary) only. Any new qai-hub
DLC needs a `qairt-dlc-prepare` step on the runner before it is shippable.
