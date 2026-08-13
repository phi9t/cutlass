# CONTEXT — cutlass fork infra glossary

> Glossary only. No implementation details, no specs. Terms are defined as they
> crystallize during design (`/grill-with-docs`). Decisions with trade-offs live
> in `docs/adr/`.

## Terms

- **Fork trainer** — the fork-specific code under `src/` (a single-node, multi-GPU
  GPT trainer built on CUTLASS/CuTe). Disjoint from cutlass-proper's own tree so it
  rebases cleanly onto upstream releases. The *only* code we build with Bazel.

- **cutlass-proper** — the upstream NVIDIA CUTLASS/CuTe library at repo-root
  `include/{cute,cutlass}`. Header-only; never *built* by us, only *exposed* to the
  fork trainer as a header `cc_library`. Stays on its own CMake build untouched.

- **Rootfs** — a bwrap-run flattened Ubuntu userland from a pinned CUDA 12.8 *devel*
  image (real nvcc + C++17 host toolchain, **no Python**). Supplies the toolchain the
  host lacks plus the matching host-driver userspace (`libcuda.so*`, `libnvidia-*.so*`,
  `nvidia-smi`) and `/dev/nvidia*` device nodes bound in read-only. The **outer isolation
  boundary**: Bazel runs *inside* it, so build and GPU-run share one environment.
  Re-exec marker `CUTLASS_IN_ROOTFS=1`. See ADR 0002, 0004. (Contrast: nccl has no
  rootfs.)

- **Local run** — single-host execution that exercises the real GPU runtime path
  (build → on-device smoke → tests) inside the rootfs, driven by a loud-preflight
  `run_local_*` verifier. Mirrors monarch's local-run verifier.

- **Verifier** — the `run_local_*` script that runs the local-run ladder with a loud
  preflight, repo-relative re-exec into the rootfs, and distinct exit codes per
  failure stage.

- **Hermetic CUDA** — CUDA toolchain provenance is pinned and reproducible rather than
  discovered from whatever is installed on the host. Here: one CUDA (12.8, `sm_100`)
  living in the rootfs from a pinned CUDA *devel* image; Bazel discovers it via
  `local_cuda` pointed at the rootfs. See ADR 0003.

- **Smoke rung** — the single-GPU minimal CUTLASS GEMM the verifier compiles and runs on
  `sm_100` to prove nvcc + cutlass headers + driver link end-to-end, before any
  multi-GPU test. The 8-GPU capacity run is a later rung, not the gate.

- **Rootfs guard** — the build-level enforcement that makes "CUDA work runs inside the
  Rootfs" a *build* invariant, not a convention. A `//tools/bazel:rootfs_guard` action,
  pulled in transitively by `//third_party/cuda:cudart` only when CUDA is enabled,
  hard-fails any `--config=cuda` build that runs without `CUTLASS_IN_ROOTFS=1` and prints
  how to enter the Rootfs. It cannot be skipped by targeting a specific label. Escape
  hatch: `CUTLASS_ALLOW_HOST_CUDA=1` (deliberate, visible host override). See ADR 0006.

- **Land on mainline** — commit to the local `main` branch only; never push to any
  remote (see `AGENTS.md`).

- **Header-only subset** — the only fork targets that build on a bare CPU host
  (`//src/tensor`, `//include:cutlass`, `//third_party/cutlass`). Everything else is a
  CUDA trainer target that links CUDA/NCCL and builds via `--config=cuda` inside the
  rootfs. CPU-host verification = build this subset + `bazel build --nobuild //...`
  (analysis only). See ADR 0005.

- **CUDA-off gating** — the default `.bazelrc` state: `--build/test_tag_filters=-cuda,-gpu`
  skips `cuda`-tagged (`.cu`-compiling) and `gpu`-tagged (device-running test) targets.
  `--config=cuda` flips the filters open, enables `rules_cuda` + nvcc, and defaults to
  Blackwell `sm_100`.
