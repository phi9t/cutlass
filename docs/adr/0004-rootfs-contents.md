# ADR 0004 — Rootfs contents: CUDA 12.8 devel image, no Python, host driver bound in

- Status: Accepted
- Date: 2026-08-13
- Depends on: ADR 0002 (rootfs is the boundary), ADR 0003 (one pinned CUDA in the rootfs).

## Context

Given the rootfs supplies the pinned CUDA 12.8 (ADR 0003), what base image and what
bind-ins? monarch's rootfs starts from a PyTorch-CUDA *runtime* base and therefore needs
the synthetic-CUDA-from-wheels trick plus a `.venv` — because monarch is a PyO3/Python
world. The cutlass fork trainer is C++/CUTLASS with no Python runtime dependency, so it
does not need Python, torch, or uv in the rootfs.

Host facts (verified): driver **580.105.08** (open kernel module); full `/dev/nvidia*`
set (nvidia0..7, nvidiactl, nvidia-uvm, nvidia-uvm-tools, nvidia-modeset, nvidia-caps);
`libcuda.so*`, the `libnvidia-*.so*` family, and `nvidia-smi` all present on the host. A
580.x driver runs a CUDA 12.8 toolkit under the driver-forward-compat guarantee.

## Decision

- **Base image**: `nvidia/cuda:12.8.x-devel-ubuntu22.04`, pinned by digest. A *devel*
  image ships a real `nvcc` and a C++17 host toolchain directly — no synthetic-from-wheels
  step. **No Python / torch / uv** in the rootfs.
- **Driver bind-in** (monarch pattern, read-only): host `libcuda.so*`, `libnvidia-*.so*`,
  `nvidia-smi`, and all `/dev/nvidia*` device nodes, so the in-rootfs 12.8 runtime links
  the matching host 580.105.08 driver userspace. Also bind host `/etc/resolv.conf` +
  `/etc/hosts` (docker export leaves resolv.conf empty).
- **Re-exec marker**: `CUTLASS_IN_ROOTFS=1`.
- Reuse monarch's **idempotent, atomic-stage** `build_rootfs.sh` (mktemp stage → validate
  → `mv` into place) and **loud-preflight** `enter_rootfs.sh` structure.

## Consequences

- Far lighter than monarch's rootfs (no Python layer); the only job is a pinned nvcc +
  C++17 toolchain + matching driver userspace.
- Reproducible by image digest; the exact `12.8.x` patch tag/digest is pinned at build
  time during implementation.
- If the fork trainer ever grows a Python dependency, the rootfs would need a venv layer
  (revisit then) — deliberately out of scope now.
