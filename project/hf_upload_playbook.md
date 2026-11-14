---
name: HF 大文件上传操作手册（xet + 白名单直连，2026-11-14 实跑）
description: 本机把 GB 级模型传到 HuggingFace 的可照抄配方——xet 修好、控制面走梯子/数据面直连、手写 UDP DNS、失败重试循环、事后 sha256 校验；实测 2.85 GB 全程隧道流量仅 0.1 MB
type: project
---

**Why:** 用户 2026-11-14 明确要求「大文件上传不要走梯子，梯子有流量限制」。
原理与域名归属见 `hf_network_xet.md`；这份是**执行手册**——照抄就能上传，含每条踩过的坑。

## 0. 成绩单（实测）

| 指标 | 数值 |
|---|---|
| 数据量 | 2.85 GB / 20 文件（xet） |
| 耗时 | 约 15 分钟（含一次 attempt 里的完整链路） |
| 速度 | 1.4–3.6 MB/s（峰值 3.6，稳定 2.0–2.6） |
| 直连上行 | 3.55 GB |
| **隧道（梯子）上行** | **0.1 MB**（3 次 API 握手） |
| 重试 | 0 次（第 1 轮 attempt 就成功） |

## 1. 前置条件

```sh
hf_xet 1.8.0            # hf download/upload 自动用
SSL_CERT_FILE=/data/data/com.termux/files/usr/etc/tls/cert.pem   # 必须，否则 rustls panic
HF_XET_HIGH_PERFORMANCE=1
root                    # SO_BINDTODEVICE 需要
```

没有 `SSL_CERT_FILE` 时的报错是 `panicked: "Expect rustls-platform-verifier to be initialized"`
（rustls-platform-verifier 在 Android 走 JNI，Termux 没 JVM）。

## 2. 分流代理（核心）

脚本在 `models/net/pinproxy.py`（就是 `~/tmp/pinproxy.py` 的固化版）。
**必须 root 跑**，监听 `127.0.0.1:8912`，普通 CONNECT 代理：

| 目标 | 走法 |
|---|---|
| `huggingface.co` / `*.huggingface.co` / 裸 `hf.co` | 隧道（默认路由 tun0），几 KB |
| **其余全部**（`*.hf.co`、`*.xethub.hf.co`、`*.amazonaws.com`…） | `SO_BINDTODEVICE wlan0` + 直连 |

```sh
cd ~/projects/viitorvoice/models
sudo /data/data/com.termux/files/usr/bin/python3 net/pinproxy.py > ~/tmp/pinproxy.log 2>&1 &
```

三条设计要点，缺一条就会翻车：

1. **白名单，不是黑名单。** 先写成「`.hf.co` 直连、其余走隧道」，结果 xet 的大流量
   实际打到了 `s3.us-east-1.amazonaws.com`，照样吃配额。改成「只有 `huggingface.co` 走隧道」才彻底。
2. **直连域名的 DNS 必须自己解。** 系统 DNS 是 fake-ip（`198.18.x.x`），
   拿它去直连必然失败；DoH（cloudflare-dns.com）本身又走隧道、而且常超时。
   做法：手写 UDP DNS 查询打 `223.5.5.5:53`，**socket 也 `SO_BINDTODEVICE wlan0`**，结果按 host 缓存。
3. **xet 的数据面会漂。** 两次实测分别是 `s3.us-east-1.amazonaws.com`（8–9 MB/次 PUT）
   和 `cas-server.xethub.hf.co`（EC2 us-east-1）。所以规则不能只写死某一个域名。

启动后先验三条：

```sh
curl -s -o /dev/null -w "api %{http_code} %{time_total}s\n" -x http://127.0.0.1:8912 https://huggingface.co/api/models?limit=1
curl -s -o /dev/null -w "s3  %{http_code} %{time_total}s\n" -x http://127.0.0.1:8912 https://s3.us-east-1.amazonaws.com/
curl -s -o /dev/null -w "cas %{http_code} %{time_total}s\n" -x http://127.0.0.1:8912 https://cas-server.xethub.hf.co/
```

日志里应当出现 `tunnel` / `direct-wlan0:IP` 两种标签，一眼可判。

## 3. 上传（必须带重试循环）

`hf upload` **没有内建重试**，网络抖一下就以
`SSLError(SSLEOFError) ... on /api/repos/create` 退出。可用的循环：

```sh
export HTTPS_PROXY=http://127.0.0.1:8912 HTTP_PROXY=http://127.0.0.1:8912
export SSL_CERT_FILE=/data/data/com.termux/files/usr/etc/tls/cert.pem
export HF_XET_HIGH_PERFORMANCE=1
n=0
while :; do
  n=$((n+1)); echo "=== attempt $n ==="
  hf upload mistmlm/viitorvoice-nar-sm8750 . . \
    --include "artifacts/*" "MANIFEST.md" "README.md" "install.sh" "verify.sh" "push_hf.sh" \
    --exclude ".git/*" --commit-message "Add SM8750 QNN model set (xet)" && { echo DONE; break; }
  sleep 20
done
```

（实测脚本：`models/net/upload_loop.sh`）

## 4. 事后校验（必做）

```sh
TOK=$(cat ~/.cache/huggingface/token)
curl -s -H "Authorization: Bearer $TOK" \
  "https://huggingface.co/api/models/<repo>/tree/main/artifacts" -o tree.json
```

然后逐项比：文件数、每个文件的 `size`、以及 `lfs.oid`（= sha256）与本地 `artifacts/` 对照。
本次结果：20/20 文件、体积全一致、19 个 `.bin` 的 sha256 全一致；
`nar_steps`（3 B）是普通 git 文件，API 不给 `lfs.oid`，属正常。

## 5. 踩过的坑（每条都真实发生过）

| 坑 | 现象 | 处理 |
|---|---|---|
| 只按 `.hf.co` 分流 | 大流量走 `s3.us-east-1.amazonaws.com`，吃梯子配额 | 改白名单：只有 `huggingface.co` 走隧道 |
| 直连用 DoH 解析 | `urlopen error handshake timed out`，且 DoH 自己走隧道 | 手写 UDP DNS + 绑 wlan0 + 缓存 |
| `hf upload` 单轮跑 | `SSLEOFError on /api/repos/create`，无重试 | 外层 while 循环，20 s 间隔 |
| 重试以为会重传 | 实际 xet 内容寻址，已传 xorb 跳过，日志总量会从 2.85 → 2.63 GB 缩水 | 正常，不要慌 |
| 上传中删 `/usr/tmp/tmp*` | `resource_tracker: leaked semaphore` / `sem_unlink FileNotFoundError` | 无害但别在上传时删临时目录 |
| 清缓存清过头 | `~/.cache/huggingface/token` 被删就丢登录态 | 只删 `hub/ xet/ assets/`，保留 `token`、`stored_tokens` |
| 用 `pkill -f "hf upload"` 停任务 | 模式匹配到自己所在 shell，整条命令被杀 | 用 `ps` 取 PID 再 `kill` |
| 中途查仓库 | 显示 4 个旧文件 / 0.03 GB，像是没传 | xet **最后统一 commit**，传完才出现；期间"空"是正常的 |
| 日志 `Removing N file(s) from commit that have not changed` | 看着像删除文件 | 正常，是 xet 去重提示 |
| 用隧道跑 cas-server "更稳" | 几十 MB × N 的配额瞬间没 | 宁可直连 + 重试 |
| 上传前没验 MANIFEST | 传完才发现本地文件本身不全 | 先跑 `models/verify.sh`（locally sha256） |

## 6. 复用 checklist

1. `models/verify.sh` 本地校验通过
2. root 起 `net/pinproxy.py`，三条 curl 验通（api=tunnel，s3/cas=direct）
3. `net/upload_loop.sh` 后台跑，`tail -f` 看进度
4. 收尾用 API tree + `lfs.oid` 与 `MANIFEST.md` 逐项比对
5. 确认隧道侧累计上行 ≈ 0 MB（`grep tunnel ~/tmp/pinproxy.log` 汇总 `up=`）
6. 最后才删 `~/.cache/huggingface/{hub,xet,assets}`

## 附：分流日志的统计口径

每行 `OK <host> <route> up=<n> down=<n>`，汇总上行：

```sh
python3 - <<'EOF'
import re, collections
agg = collections.Counter()
for line in open('/data/data/com.termux/files/home/tmp/pinproxy.log', errors='ignore'):
    m = re.search(r'\b(OK|FAIL)\s+(\S+)\s+(\S+)\s+up=\s*(\d+)', line)
    if m: agg[m.group(3).split(':')[0]] += int(m.group(4))
for k, v in agg.most_common(): print(f"{k:14s} {v/1e6:9.1f} MB")
EOF
```

注意连接关闭时才落盘，**正在传输的连接不计入**——上传中看到的小数字是正常的。
