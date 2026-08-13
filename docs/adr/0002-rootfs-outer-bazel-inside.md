# ADR 0002 — bwrap rootfs is the outer isolation boundary; Bazel runs inside it

- Status: Accepted
- Date: 2026-08-13
- Reference models: monarch (bwrap rootfs, no Bazel) vs nccl (hermetic Bazel, no rootfs).

## Context

The two reference repos take opposite approaches to isolation:

- **nccl** gets isolation purely from Bazel `--spawn_strategy=sandboxed`, with a truly
  hermetic CUDA redistributable and **no rootfs**.
- **monarch** uses a full **bwrap rootfs** to supply a consistent nvcc + host-driver
  userspace + `/dev/nvidia*`, sidestepping per-host toolchain mismatch, and has **no
  Bazel build**.

We want "both" — a Bazel build *and* a rootfs. The load-bearing fact: **this host has no
`nvcc` on PATH at all.** Something must supply the CUDA compiler and the driver-userspace
consistency before any GPU build or run can happen. The scope is only the fork's `src/`
trainer; cutlass-proper stays header-only CMake.

Layering options considered:
- (a) Rootfs outer, Bazel inside it.
- (b) Bazel hermetic build on the host; rootfs only for the GPU run/verifier.
- (c) Bazel-only, no rootfs (nccl model).

## Decision

**The bwrap rootfs is the outer isolation boundary, and Bazel runs *inside* it.** The
rootfs (monarch-style) supplies nvcc, the C++17 host toolchain, matching host-driver
userspace bound in read-only, and `/dev/nvidia*` device nodes. Bazel's own sandbox still
applies within the rootfs.

## Consequences

- **One environment for build and run** — no "compiles under Bazel, breaks at runtime on
  a driver mismatch" class of bug, because both happen under the same rootfs.
- The hermetic-CUDA question becomes *what goes in the rootfs* (a rootfs-contents
  decision), not a competing mechanism to the rootfs.
- The `run_local_*` verifier re-execs into the rootfs with a **repo-relative** script
  path (monarch pattern) so reruns are stable regardless of host path layout.
- More setup weight than nccl's Bazel-only model (rejected: it assumes a host nvcc we
  don't have) and than option (b) (rejected: splitting build-host from run-rootfs
  reintroduces the mismatch the rootfs exists to prevent).
