---
name: qai-hub 共用权重 + 伪动态上下文（静态多图 + link 去重）
description: 用 N 个静态 shape 各编译一份、再一个 link job 合成单个 context binary，权重只存一份；宿主按输入形状选图。配方、实测数字、四个坑，以及解码器 200/375 帧的套用方案。
type: project
---

QNN/QAIRT 只吃静态形状，而 NAR 的序列长度是变的（`prompt + 参考帧 + SEP + 目标帧`，几十到 512）。
"伪动态上下文" = **按需求 N 个静态 shape 各编译一份，再用一个 link job 合成一个 .bin：权重存一份，图存 N 份，运行时按输入形状选图**。
宿主侧看起来像"动态上下文"，本质是权重共享的静态多图。

## 五步配方（已在 llm / head 上跑通）

1. **导出**：每个 shape 单独导出，静态形状、fp16、`torch.onnx.export(..., dynamo=False)`、opset 18。
   ⚠️ `rope_theta` 必须显式写进 config（1e6），transforms 4.57 的默认 10000 会让整条链崩（2026-11-05 的根因）。
2. **编译**：每图单独 `submit_compile_job`，
   `--target_runtime qnn_dlc --truncate_64bit_io --quantize_full_type float16 --qairt_version 2.51`。
   `qnn_context_binary` 不能当 compile 的 target（客户端只认 onnx / precompiled_qnn_onnx / qnn_dlc / tflite）——
   **出 .bin 只能走 link job**；link 也不认 `--truncate_64bit_io`，只给 `--qairt_version`。
3. **链接**：`submit_link_job(models=[tm256, tm512], options='--qairt_version 2.51')`。
4. **加载**：`qnn_load_multi()` = 一次 `contextCreateFromBinary` + N 次 `graphRetrieve`，
   所有图共享同一个 ctx/mmap（引用计数释放），每图一个 tag。
5. **选图**：**按输入形状选，不按图名选**（见坑 2）。宿主缓冲区按最大 shape 分配，运行时切 `SEQ`。

## 实测（SM8750，QNN 2.51.40）

| 产物 | 单图 | link 后 | 说明 |
|---|---|---|---|
| llm 256+512 | 881,625,573 B | **893,554,688 B** | ≈1.01× 单图，第二张图只多 ~2 kB |
| head 256+512 | 权重 33.6 + 23.1 MB | **57,413,632 B** | ≈1.01×，权重被去重 |

- 两张图一起 init **≈0.6–1.0 s**，和单图加载没有明显代价（权重共享生效）。
- 512 窗口的 NAR 耗时是 256 的 **3.4–4×**（attention O(S²)）→ 窗口必须自动选，默认 256。
- 512 的用法：`prompt(≤91) + ref(≤200) + SEP + target(≤200) = 492 ≤ 512`。

## 四个坑

1. **图名撞车**：多个 ONNX 图节点/initializer 同名时 link 会失败或串味。给不同 shape 的图取不同图名
   （`head_256` / `head_512`）。
2. **`--graph_names` 只对部分图生效**：llm 能改名，head 编译出来的图名是 model id 派生的
   （`graph_naxscpm0` / `graph_4bropqko`），link 也不改名 → **不能按名字选图**。
   改为按输入形状认：`qnn_input_dim(tag, "inputs_embeds", 1)` / `("hidden", 1) == 256|512`，
   存进 `TAG_LLM[2]` / `TAG_HEAD[2]`。
3. **fp16 导出的 ONNX，图 IO 也是 fp16**（`QNN_DATATYPE_FLOAT_16 = 0x216`）。
   `dtype_bytes()` 要补 2 字节分支，输入端 `f2h`、输出端 `h2f`，对外仍全用 fp32。
   旧 fp32 ONNX + `--quantize_full_type float16` 的图 IO 仍是 fp32，所以这个坑平时不暴露。
4. **两图 IO 只允许差选图那一维**，dtype 必须一致，否则调用方要写两套接口。

## 可继续套用的地方

- **解码器 200 → 375 帧**（现 P0）：`dec_p1..p5` 各按 200 / 375 编译一份，然后**每段各自 link 成一对**。
  代价 ≈ 段数 × 单段权重 1.01 倍，换来短音频走 200 窗（现在固定 200 帧，45 帧音频也要付 1.68 s）、
  长音频走 375 窗。
- **FE 的 8 s 窗口**要支持更长参考音（现在固定 802 帧 mel）时同理：
  802 / 1602 两份图 link 成一份 `enc_fe.bin`。

**How to apply**：任何"长度必须变、但 QAIRT 只吃静态形状"的图，别去试动态 shape，
直接多 shape 编译 + link 去重；选图按输入维度认，不按图名。
