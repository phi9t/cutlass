# ADR 0003 — CUDA provenance: rootfs devel image supplies nvcc; Bazel discovers it

- Status: Accepted
- Date: 2026-08-13
- Depends on: ADR 0002 (rootfs is the outer boundary).

## Context

`rules_cuda 0.3.0` offers a hermetic CUDA redistributable (nccl's model). But ADR 0002
already makes the bwrap rootfs the single source of environment truth, and the host has
no `nvcc`. Provenance options:

- (a) Rootfs supplies CUDA (from a CUDA **devel** image or synthetic-from-wheels); Bazel
  uses the scaffold's `local_cuda` discovery pointed at the rootfs toolchain.
- (b) Bazel hermetic `rules_cuda` + `redist_json`; rootfs supplies only host C++ + driver
  userspace. Two pinned CUDA sources risk drift.
- (c) Both pinned in lockstep.

CUDA version: nccl uses 13.0.2; cutlass's README recommends 12.8. B200 (`sm_100`) needs
CUDA 12.8+.

## Decision

**One pinned CUDA, living in the rootfs.** Use a CUDA **devel** image (real nvcc, no
synthetic-from-wheels step) pinned at **CUDA 12.8**. Inside the rootfs:

- `rules_cuda 0.3.0` is the **compiler toolchain provider** (`cuda_library`, the nvcc
  driver, and the `@rules_cuda//cuda:archs` flag). It is pointed at the **rootfs** CUDA
  as a *local* toolkit (nccl's `--config=local_cuda` style), **not** its hermetic
  `redist_json` download — so there is still exactly one CUDA.
- The scaffold's `local_cuda` (`tools/bazel/cuda_ext.bzl`) is kept only to expose the
  CUDA **libraries** (`cudart`/`cublas`/`cublasLt` + headers) discovered from the same
  rootfs `CUDA_HOME`.

Discovery note: the scaffold's `local_cuda` alone cannot compile `.cu` files (it exposes
libraries only, defines no `cuda_archs` flag, and provides no nvcc toolchain); its
`.bazelrc` even referenced a non-existent `@local_cuda//:cuda_archs`. `rules_cuda` fills
that gap, which is why it is added as a `bazel_dep`.

## Consequences

- **No redist/rootfs drift** — there is exactly one CUDA, pinned by the rootfs image
  digest. 12.8 matches cutlass's own recommendation and supports Blackwell `sm_100`.
- Reuses `cuda_ext.bzl` as-is (least new machinery); "the system" it discovers is the
  pinned rootfs, so `local_cuda`'s system-discovery is no longer non-hermetic in practice.
- A devel image is heavier than a runtime image but avoids monarch's synthetic-CUDA-from-
  wheels complexity, which existed only because monarch started from a runtime base.
- Rejected (b)/(c): two CUDA sources to keep in lockstep, for no gain once the rootfs is
  already the pinned boundary.
