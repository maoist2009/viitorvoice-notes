---
name: DreamLite qai_hub S3 upload needs accelerate + IP pin
description: From this 电信 network the qai_hub S3 data plane historically needed the accelerate endpoint pinned to CDN edge IPs. RE-VERIFIED 2026-10-14: the plain host is now reachable and no pin is needed for downloads.
type: project
---

**RE-VERIFIED 2026-10-14 (~20:00 UTC): the plain (non-accelerate) host
`tetrahub-qprod-userdata.s3.amazonaws.com` is now REACHABLE from this 电信
link** — `curl https://.../` returns 403 in ~2 s, same as all three
accelerate edge IPs. No IP pin was needed for the 3.1 MB/s S3 direct
download (see `dreamlite_status.md` "S3 direct download"). Treat the
"must pin" rule below as a fallback for when a transfer stalls, not a
default; re-probe before touching code.

---

The DreamLite qai_hub export uploads (clip.bin, unet_tile.bin) fail from this
电信 (China Telecom) link UNLESS the S3 connection uses the *accelerate*
endpoint pinned to a reachable Amazon CDN edge IP.

**Why:** The non-accelerate host `tetrahub-qprod-userdata.s3.amazonaws.com`
cannot be served by the reachable CDN edges (TLS handshake failure / cert
mismatch). The accelerate host `tetrahub-qprod-userdata.s3-accelerate.amazonaws.com`
completes TLS and serves traffic on 52.84.223.17 / 18.172.34.154 / 143.204.148.99
(verified ~640-700 kB/s, 137-174 ms). The control plane (Qualcomm AI Hub API)
is fine on this link.

**How to apply:** `qai_hub_s3_patch.py` forces `use_acceleration=True` and
monkeypatches `socket.getaddrinfo` to pin the accelerate host to those 3 IPs
(curl `--resolve` equivalent: SNI/hostname preserved, no /etc/hosts change).
`resumable_upload.py` requests accelerate URLs. If an upload stalls on
"Could not connect", first re-verify the 3 edge IPs are still reachable
(`curl --resolve <host>:443:<ip> https://<host>/` → expect 403, not handshake
failure) BEFORE touching the code. Do NOT switch back to non-accelerate — the
non-accel host is unreachable from this network.
