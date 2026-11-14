---
name: ViiTorVoice Android demo app (independent GUI client)
description: Standalone Compose app at ~/projects/viitorvoice-demo (com.viitorvoice.demo) that now runs the WHOLE chain on device — ORT encoder (w2v frontend + dualcodec core), NAR masked-diffusion with CFG local edit, 5-context decoder — encode 10 s + edit 5 s + decode 1.5 s for a 3.2 s clip
type: project
---

**Location:** `~/projects/viitorvoice-demo` — separate from dreamlite (user
decision 2026-11-03: "独立app"), reusing dreamlite's build skeleton
(AGP 9.0.1, Kotlin 2.2.10, Compose BOM 2026.04.01, offline gradle cache).
Package `com.viitorvoice.demo`, debug installs as `com.viitorvoice.demo.tmp`
(uid 10381).

## State 2026-11-05 — full pipeline in-app

| stage | where | how long (3.2 s clip, 80 frames) |
|---|---|---|
| 24k→16k resample + w2v frontend + dualcodec encode_core | ONNX Runtime CPU (C++), `vv_encode.cpp` | **10.3 s** |
| NAR masked-diffusion, 32 steps, CFG 2.0 (2 llm + 2 head per step) | HTP, `vv_nar.cpp` | **4.8–5.1 s** |
| decoder chain (5 contexts, fixed 200-frame window) | HTP, `vv_pipeline.cpp` | 1.45 s |

Total ~27 s for the first edit of a clip (encoder load dominates); repeat edits
of the same clip skip encoding.

* Editing semantics follow `_generate_edit_iterative_single`: cond branch =
  prompt(style+text) + window, uncond branch = window only at positions
  0..T-1, `guided = logp_c + cfg*(logp_c - logp_u)` renormalised, score mask
  only lets *editable* + still-masked cells update, schedule built from the
  editable cell count.
* Verified on device: for mask frames [23,37) of an 80-frame clip the result is
  **exactly 66/80 frames unchanged, frames 23–36 regenerated** — identical to
  the python NPU reference `nar_edit_proto.py`.
* Encoder verified by dumping the device's frontend output (`feat_dev.bin`) and
  re-running the core under desktop ORT: **100 % identical codes**. The device
  frontend also matches the python one at cos 0.99997 (only the resampler
  differs, −60 dB).
* UI (Compose): record (AudioRecord 24 k, WAV) / import (MediaExtractor+
  MediaCodec → resample) → envelope waveform with drag selection → editable
  transcript → "局部重生成" → play. Char diff between the baseline transcript
  and the edited one auto-selects the frames to regenerate.

## 2026-11-06 — 音色克隆 / 情感 CFG 接入，两个真 bug

`vv_nar.cpp` 现在只有一个通用 CFG job（`vv_nar_prepare_cfg` / `cfg_step`），
编辑、克隆、情感/NVV 辅助分支、纯 TTS 都走它；`prepare_edit/edit_step` 是薄封装。

**Bug 1（致命）：CFG job 从不回灌已生成 token。**
`build_branch()` 只在 prepare 时写一次 target 窗口到 `BR_IDS[b]`，而
`apply_topk()` 把新 token 写进 `CFG_TOK`。于是每一步 `gather_branch()` 读到的
还是**全 mask 的初始窗口**，等于 32 步都在重复预测同一状态：

* 现象：token 大幅重复（语义行 4853×24、9508×27），解码出来是**高频噪声**
  （0–1 kHz 能量占比只有 0.08–0.21，正常中文 TTS 是 0.47–0.68）。
* 锚定编辑（14/80 帧）看起来"正常"，因为 66 帧真 token 撑住了解码，掩盖了它。
* 修复：`sync_branch_targets()`，每步把 `CFG_TOK` 拷回**每个分支**的
  `BR_IDS[b] + t0`（tstart 各自不同）。修复后同样 80 帧全生成：

| 用例 | rms | 0–1k 占比 | voiced |
|---|---|---|---|
| 修复前全生成 cfg=2 | 0.022 | 0.21 | 8/80 |
| 修复后全生成 cfg=2 | 0.047 | 0.38 | 41/80 |
| 修复后全生成 cfg=0 | 0.224 | 0.47 | 53/80 |
| 修复后克隆 cfg=2 | 0.049 | 0.48 | 49/88 |

编辑 [23,37) 仍然是 66/80 帧不变，音色/频段与输入一致（0.65 vs 0.68）。

**Bug 2：Kotlin 分词器给中文段乱加 metaspace `▁`(10558)。**
`tokenizePlain` 无条件给每段加 `▁`。HF `tokenizer.json` 的规则是只有"词首字符"
才加：HAN/HIRAGANA/KATAKANA 和 `。、` 不加，其余（拉丁、数字、全角标点、谚文）加。
证据：`t.encode("你好…")` = `[10932, …]`，我们的 prompt 里却是 `52667 10558 10932 …`。
修复：`needsMetaspace(ch)`。修完后 prompt 与 HF 逐 token 一致
（`<|text_start|>` 后不再多一个空格）；这一项单独并没有解决噪声，但比较 prompt
时它是实实在在的偏移。

**辅助分支实测**：`<|emotion-happy|>今天天气真不错，我们出去走走吧。` → prompt 带
52694，N_BR=3，NAR 7615 ms（2 分支 5081 ms），输出 4.08 s、0–1k 0.64，说明
aux 分支（同 prompt 剥标签 + `-aux*logp_aux`）确实跑起来了。

**Bug 3（2026-11-07）：长参考音把目标帧数挤没了 → 语速快得要死。**
256 token 窗口装的是 `prompt + 参考帧 + 1(SEP) + 目标帧`。参考音一长（比如录了 6 s），
留给目标的帧数就不够，而旧代码是 `est.coerceIn(25, room)` —— **静默**把目标砍到几十帧，
一句 24 字的话塞进 1.7 s（≈14 字/秒）。另外 `vv_encode` 的窗口固定 192900 样本
(8.04 s)，更长的参考音是**静默截断**的。
修复：`Prompt.planClone()` 从尾部逐 5 帧裁参考音（最短 1 s），直到目标的估计帧数放得下；
连 1 s 参考都放不下就明确报错。`Engine.ENC_MAX_SAMPLES` 在 Kotlin 侧先截 8 s 并打日志。
实测 9.6 s 参考 + 32 字文本：参考 200→35 帧、目标 163 帧（6.52 s，4.9 字/秒），不再压速。
**预算**：参考建议 1–5 s；参考 + 目标合计别超 ~10 s。

**解码耗时是固定成本**，不是 bug：5 个 context 的 decoder 编译成**固定 200 帧窗口**，
所以 88 帧音频和 163 帧音频都花 1451 / 1464 ms（`nativeNarFinish` 计的就是纯推理）。
想减半只能按更短窗口重编译 decoder（要云 runner，目前 SSH 已死）。

**调试入口（debug 源集）**：`EditService` 支持 `--es mode synth|clone`；
`--ez diag true` 让 `cfg_step` 在 step 0 把每个分支的 top1/top2 logit 落到
`files/diag.txt`；`--ef wc/--ef wu` 覆盖引导权重。`nativeNarEditTokens()` 现在
按模式返回（原来永远返回 `CFG_TOK`，synth 模式 dump 出来是全 0）。

**内存**：一个进程里连续跑多次会 OOM（LMK 在 `rss 4.3 GB` 时杀进程，swap 只剩
220 MB）。无头测试每次先 `am force-stop`，再起服务。

## Layout

```
app/src/main/cpp/  vv_encode.cpp/h   ORT: w2v_front8s.onnx + dualcodec_encode_core_30s.onnx
                   vv_nar.cpp/h      embeddings gather + diffusion (+ CFG edit) + decode
                   vv_pipeline.cpp/h decoder chain (200-frame window)
                   vv_jni.cpp        JNI surface
                   onnxruntime/include, ../jniLibs/arm64-v8a/libonnxruntime.so (1.27, from the old ve app)
app/src/main/java/.../audio/AudioIo.kt   record / MediaCodec import / wav write
                       ui/HomeScreen.kt  waveform + edit UI
                       ui/HomeViewModel.kt
app/src/debug/java/.../EditService.kt    headless RUN_EDIT (foreground service)
```

Models in `filesDir/models/`: `llm.bin` (885 MB fp16, the rope_theta=1e6 one),
`head.bin`, `text_emb/sem_emb/aco_emb.bin`, `dec_p1..p5.bin`,
`w2v_front8s.onnx(+.data, 2.26 GB)`, `dualcodec_encode_core_30s.onnx` (197 MB).
≈3.6 GB total.

## Traps hit for real

1. **`am` must be `/system/bin/am` inside `sudo sh -c`** — bare `sudo am …`
   resolves to the Termux shim and silently does nothing.
2. **`files/` must be chowned to the app uid** after pushing models with sudo,
   else the first `writeText` throws EACCES and the process gets "crashed too
   many times".
3. A multi-minute broadcast handler dies on the ANR window → run the job in a
   foreground service (`am start-foreground-service` from root works; a
   receiver's `startForegroundService` is blocked on 12+ background starts).
4. **Token matrix stride**: `nativeEncode` fills a fixed 12×200 matrix; the edit
   API takes 12×frames. Repack in Kotlin (`System.arraycopy(out, c*200, packed,
   c*n, n)`) or every acoustic row silently points at the wrong frame.
5. **Release the ORT sessions before the NPU loop** — 2.45 GB of frontend/core
   weights + 1.1 GB QNN contexts otherwise sit in the same process.
6. Kotlin `MutableList<Int>.addAll(IntArray)` does not exist (no primitive
   overload) — `.toList()` it.

## Debug driving (no UI)

```sh
sudo sh -c '/system/bin/am start-foreground-service \
  -n com.viitorvoice.demo.tmp/com.viitorvoice.demo.EditService \
  --es text "改后的整句" --ei a 23 --ei b 37 --ef cfg 2.0'
sudo cat /data/data/com.viitorvoice.demo.tmp/files/edit.txt
```
`edit_in.wav` must be in `filesDir` (24 kHz mono). The service also dumps
`edit_tokens.txt` (12 rows) for numeric comparison.

```sh
# 克隆（无参考文本 / 有参考文本），结果在 clone_out.wav + clone.txt
sudo sh -c '/system/bin/am start-foreground-service \
  -n com.viitorvoice.demo.tmp/com.viitorvoice.demo.EditService \
  --es mode clone --es ref /data/data/com.viitorvoice.demo.tmp/files/edit_in.wav \
  --es text "<|emotion-happy|>今天天气真不错。" --es lang zh \
  --ef cfg 2.0 --ef emo 1.0 --ef nvv 0.5'
# 纯文本（走老的单分支 diffusion），结果在 synth_out.wav + synth.txt
sudo sh -c '/system/bin/am start-foreground-service \
  -n com.viitorvoice.demo.tmp/com.viitorvoice.demo.EditService \
  --es mode synth --es text "你好" --ei frames 80'
```
跑之前先 `am force-stop com.viitorvoice.demo.tmp`，否则连续多跑会撞 LMK。

Alternatively `am start ... --es load <wav> --es text <…> --ez auto true`
drives the real UI path (fresh task only — the activity ignores
`onNewIntent`).

## Still missing

ASR (the transcript must be typed by the user) and a model-pack downloader.
音色克隆 / 情感 / NVV / ja-ko 已在 2026-11-06 打通（见上），但**音质的最终判断
要人耳**——本地只有频段/基频/浊音率这类数值代理。

## Root cause note (2026-11-05)

Everything that looked like "the DLC conversion is broken" was the RoPE base:
the export built its config from `config.json['llm_config']`, which has
`rope_parameters.rope_theta = 1e6`, but transformers 4.57's
`_compute_default_rope_parameters()` reads the legacy `config.rope_theta`
(default 10000). Forcing `cfg.rope_theta = 1e6` before export fixed the whole
chain (NPU vs torch cos 0.999999).

## 双窗口（256 + 512）落地 2026-11-07

目标：参考音 8 s 也能配满 8 s 目标（旧 256 窗口里 8 s 参考只剩 ~35 帧目标）。

**配方（全部在 qai-hub，本机不 prepare）**
- 导出：`torch.onnx.export(..., dynamo=False)`，opset 18，**静态形状**，
  fp16，`cfg.rope_theta=1e6`（同时写 `rope_parameters`），
  `inputs_embeds (1,S,1024)` + `attention_mask (1,1,S,S)` → `(1,S,1024)`。
  S=256 与 S=512 各一份，onnx 大小 881,625,573 / 881,626,597 B。
- 编译（llm）：`--target_runtime qnn_dlc --truncate_64bit_io
  --quantize_full_type float16 --qairt_version 2.51`
- 链接（出 .bin）：`submit_link_job(models=[tm256, tm512], options='--qairt_version 2.51')`
  → 893,554,688 B（**权重去重**，单份权重 + 两张图）。
- **`qnn_context_binary` 作为 compile 的 target_runtime 会被 0.57 客户端拒绝**
  （只接受 onnx/precompiled_qnn_onnx/qnn_dlc/tflite）；出 .bin 只能走 link job。
- `link_options` 不认 `--truncate_64bit_io`，只给 `--qairt_version 2.51`。
- head 也要两份：`semantic_head.weight (16387,1024)` / `acoustic_head.weight
  (11297,1024)`，纯 MatMul fp16，ONNX 图名必须不同（`head_256` / `head_512`），
  否则 link 撞名；链接后 57,413,632 B。
- 脚本在服务器 `/home/runner/vv/`：`export_llm.py`、`mkhead.py`、state.json。

**端侧加载**：`qnn_load_multi()`（qnn_executor.cpp）一次 contextCreateFromBinary
+ 多个 graphRetrieve，所有图共享一个 ctx/mmap（refcount 释放），tag = 图名。
vv_nar 里 `SEQ` 变成运行期变量（缓冲按 512 分配），`vv_nar_set_window(256|512)`，
Kotlin `Engine.useWindow()` 按 `cond+ref+sep+frames` 自动选窗口；
单图旧 .bin 仍走 `qnn_load("llm"/"head")` 回退（n<=1 时先 free 再重载）。

**两个坑（2026-11-07 实测踩到）**
1. **fp16 导出的 ONNX，图 IO 也是 fp16**（`QNN_DATATYPE_FLOAT_16 = 0x216`）。
   旧 fp32 ONNX + `--quantize_full_type float16` 的图 IO 仍是 fp32，所以
   `dtype_bytes()` 只认 32 位的写法一直没暴露问题；新图直接报
   `tensor 'inputs_embeds' dtype 534`。修法：`dtype_bytes` 加 FLOAT_16（2 B），
   `qnn_exec` 输入端 float→half（`f2h`）、输出端 half→float（`h2f`），
   宿主侧 API 仍全是 fp32，调用方无感。
2. **qai-hub 只有 llm 的图名能被 `--graph_names` 改**；head 编译出来的图名是
   `graph_naxscpm0` / `graph_4bropqko`（model id 派生），link 也不改名。
   所以**不能按图名选窗口**（曾报 `no such graph 'head_256'`），改为按输入形状
   认图：`qnn_input_dim(tag,"inputs_embeds",1)` / `("hidden",1)` == 256|512，
   存进 `TAG_LLM[2]`/`TAG_HEAD[2]`，运行期按 `SEQ` 取 tag。

**实测（SM8750，`EditService` 无头跑）**
- init 958 ms / 570 ms（两张图一起 load，权重共享，比单图没明显代价）。
- 256 窗口：文本 21 token，nar 3145 ms，decode 1552 ms，200 帧 8.00 s。
- 512 窗口（8 s 参考 + 4.32 s 目标）：ref 200→200，target 108，
  nar 9658 ms，总 24.7 s。
- 512 窗口满载（8 s 参考 + 8.0 s 目标，prompt 91 + ref 200 + sep + target 200
  = 492 ≤ 512）：nar 10770 ms，总 25.9 s，输出 8.00 s、末尾 7.72 s 仍有人声，
  `dumpsys meminfo` 中段 PSS 393 MB / RSS 442 MB / Native Heap 296 MB，无 LMK。
- 耗时与窗口近似 O(S²)：512 约是 256 的 3.4–4×。
- 产物：`/sdcard/Download/vv/v6_synth256.wav`、`v6_clone512.wav`、
  `v6_clone512_full.wav`（待用户人耳判定）。

## 无参考纯合成（synth）音质差 —— 是用法问题，不是转换问题（2026-11-07 定论）

**现象**：clone/edit 声音干净，synth（无参考）有"杂"声；帧数给得紧（54 帧）时
变成嗡嗡的破音。

**根因（token 层实测）**：没有音频上下文时，声学码本 cb1–cb11 会塌缩成长串同值。

| 用例 | 参考音 | 帧数 | cb1 最长连续同值 | 0–1 kHz 占比 |
|---|---|---|---|---|
| synth 固定 200 帧 | 无 | 200 | —（塌缩在中后段） | 0.71，flat 0.25（-67 dBFS 嘶声；clone 是 -84） |
| synth 54 帧 | 无 | 54 | **47 帧 = 87%** | **0.43**（发毛） |
| synth 80 帧（去掉逗号） | 无 | 80 | 11 帧 | 0.70，连续出声 |
| clone 512 | 8 s | 108/200 | 5–6 帧 | 0.77 |

历史最极端：2026-11-06 的 `tok_synth80.txt` 12 个码本 80 帧**全部只有一个值**。

**为什么不是转换缺陷**（三条证据）：
1. 同一套 llm/head/decoder 产物在 clone/edit 下正常 → 转换链路没坏。
2. 上游仓库只有 `/v1/voice-clone` 与 `/v1/text-local-edit` 两个端点，且
   `_audio_input()` 在没给参考音时直接 400；**上游根本没有"无参考纯合成"入口**。
   `ref_audio_tokens=None` 的分支只服务编辑（编辑时目标音频本身就是上下文）。
3. 采样超参与上游默认完全一致：`t_shift 0.1`、`layer_penalty 5.0`、
   `position_temperature 5.0`、`class_temperature 0.0`（`LLMGenerationConfig`）。

**结论**：synth 是训练分布外的用法，声学码本塌缩是必然；UI 里保留但要注明。
要真做无参考 TTS，可借上游的静音锚点机制：`assets/dualcodec_silence_2s.pt`
（12×49 的静音 token，6.5 KB）+ `no_ref_text_silence_prefix_seconds=1.0`，
把 N 帧静音当作 target 前缀/参考帧塞进音频槽（`_prepare_inference_inputs` 的
`target_prefix_tokens`），上游只在"有 ref 音频但没 ref 文本"时用它。

**另一个真 bug（应用层，与模型无关）**：`Prompt.framesFor()` 对无参考合成永远
返回 `DEFAULT_FRAMES=200`，不看文本长度估计，短句被摊成 8 s（1 s 说话 +
4.5 s 静音 + 1 s 说话）。`Duration.frames("你好，这是双窗口测试。")` = 54 帧
（2.16 s）才对。应改成"估计 + 余量 + 尾部静音裁剪"。
另注：上游 `ref_text_mask_len` 默认 10 帧，我们的 `Prompt.maskLen()` 用
4 token/s（200 帧 → 32），两者不一致，待核对。
