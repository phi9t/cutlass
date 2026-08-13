# ADR 0005 — Default CPU build target set is the header-only subset

- Status: Accepted
- Date: 2026-08-13
- Depends on: ADR 0002 (rootfs is where real builds happen).

## Context

The fork is a CUDA trainer. A dependency survey (`bazel cquery somepath`) showed that
nearly every target transitively needs the CUDA/NCCL **libraries**: `core` → cudart,
`ops:linear` → cublasLt, `dist` → nccl, and everything layered above them (`model`,
`train`, `attention`). Only header-only targets (`//src/tensor`, `//include:cutlass`)
build on a bare CPU host with no CUDA/NCCL installed.

`.bazelrc` gates GPU work with `--build/test_tag_filters=-cuda,-gpu` by default (nccl
pattern), and all `.cu`-compiling targets are now `cuda_library` tagged `cuda`, and GPU
tests are tagged `gpu`. But `.cc` targets that merely *link* CUDA/NCCL libs (e.g. `core`,
`dist`) are not `.cu` and would otherwise be built by a default `bazel build //...`,
failing on a host where the libs are absent (a dangling `local_nccl` symlink, missing
cudart).

Options: (a) tag every CUDA/NCCL-linking target `cuda` (nearly the whole tree; tags must
track dep changes); (b) make `local_cuda`/`local_nccl` emit empty stubs when libs are
absent (hides genuinely-missing libs until later); (c) accept that the default CPU target
set is only the header-only subset.

## Decision

**(c) The default CPU-host build target set is the header-only subset.** CPU-only CI and
local CPU checks build `//src/tensor`, `//include:cutlass`, `//third_party/cutlass` (and
run `bazel build --nobuild //...` for analysis-only verification of the whole graph). The
**real trainer build always runs `--config=cuda` inside the rootfs** (ADR 0002), where
CUDA and NCCL exist. `bazel build //...` is deliberately *not* the CPU-host build command.

## Consequences

- No sprawling `cuda` tags on `.cc` link-only targets, and no lib-hiding stubs.
- CPU-host verification is two things: (1) build the header-only subset, (2)
  `--nobuild //...` to prove the entire graph still *analyzes* (catches BUILD/label/glob
  regressions without needing a GPU). Both are fast and hardware-free.
- A reader who runs `bazel build //...` on a CPU host will see link failures on
  CUDA/NCCL targets — this is expected; the verifier and rootfs docs point them at
  `--config=cuda` inside the rootfs instead.
