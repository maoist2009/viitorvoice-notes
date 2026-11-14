# notes — 项目记忆快照

**权威副本**在 `~/projects/rvc/rvc-android/.opencode/memory/`（AI 会话加载的那份）。
这里是 2026-11-14 的快照，方便本树自包含；改记忆请改权威副本再同步过来，别反向覆盖。

| 文件 | 主题 |
|---|---|
| `project/viitorvoice_app.md` | 端侧 demo 的结构与各阶段实测耗时 |
| `project/viitor_voice_nar.md` | NAR 模型 / 权重 / 上游仓库 |
| `project/clone_rate_chunking.md` | 语速来源结论 + 分段合成实现 |
| `project/decoder_split_a16.md` | 解码器 5 段切点与精度分配 |
| `project/qaihub_shared_weight_multi_graph.md` | 多 shape 共享权重编译 + link 去重 |
| `project/qnn_dlc_loading.md` | `.dlc` vs `.bin`，为什么要 prepare |
| `project/qairt_onnx_constraints.md` | fp16/bf16、2 GiB protobuf 上限 |
| `project/qnn_no_fp32.md` | QNN 不接受 fp32 执行路径 |
| `project/dreamlite_s3_network.md` | qai-hub S3 直连下载 |
| `project/no_git_repo.md` | 历史限制（本树已于 2026-11-14 建独立仓库） |
