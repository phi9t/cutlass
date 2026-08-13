# Production Infra Research: bwrap rootfs, Bazel, hermetic CUDA, local-run scripts

> Location note: this file was placed at `docs/agents/infra-research.md`. No prior
> agent-notes convention existed in this repo (`docs/agents/` did not exist and was
> created for this note); the neighbouring `nccl` repo uses `docs/agents/` for the
> analogous role, so this path was chosen for consistency.

## 1. Purpose & scope

This is research feeding a later `/grill-with-docs` session about standing up
production-ready build/runtime infra for `~/workspace/cutlass`: a bwrap rootfs, a
Bazel build, a hermetic CUDA toolchain, and local-run scripts. Everything below is
read from the actual files in `~/workspace/monarch`, `~/workspace/nccl`, and
`~/workspace/cutlass` (primary sources, cited `path:line`). The two reference repos
sit at opposite ends of a spectrum: **monarch** is the full bwrap-rootfs + strict
local-runner model (Python/PyO3 world), while **nccl** is a Bazel-wraps-Make model
with truly hermetic CUDA redistributables and **no rootfs at all**. Cutlass's own
build is header-only CUDA C++ CMake, and there is already a partial Bazel scaffold
under `src/` that must be extended/hardened, not replaced.

---

## 2. Monarch infra patterns

### 2.1 bwrap rootfs build (`scripts/rootfs/build_rootfs.sh`)

- Builds a flattened Ubuntu userland from a pinned PyTorch-CUDA base image and
  `docker export`s it into a plain directory that bwrap runs under. Rationale is
  documented inline: some hosts (Nix) ship a `cc`/loader that mismatches the loader
  `rustc` runs under, breaking proc-macro/`.so` loading; a consistent rootfs sidesteps
  per-host toolchain patching (`monarch/scripts/rootfs/build_rootfs.sh:8-22`).
- **Pinned inputs** (all by digest/version): base image
  `ghcr.io/pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime@sha256:...`, `uv` image,
  nextest version, and CUDA nvcc/cccl wheel versions
  (`build_rootfs.sh:41-49`).
- **Synthetic CUDA_HOME from pip wheels**: the runtime base image has CUDA libs but
  no compiler; the script `pip install`s `nvidia-cuda-nvcc` + `nvidia-cuda-cccl`,
  verifies `bin/nvcc` and `include/cuda_runtime.h`, symlinks `lib64 -> lib`, and
  points `CUDA_HOME`/`CUDA_PATH` at `/opt/cuda-synth`
  (`build_rootfs.sh:148-169`).
- **Idempotent + atomic**: reuses the docker image tag unless `--rebuild`
  (`build_rootfs.sh:85-88`); exports into a `mktemp` stage dir under
  `scripts/rootfs/` then `mv`s into place so an interrupted run never leaves a
  half-populated rootfs (`build_rootfs.sh:173-199`); validates the stage produced a
  usable `bin/bash` before swap (`build_rootfs.sh:192`). Refuses `--dest` paths that
  are not managed `rootfs`/`rootfs-*` dirs (`build_rootfs.sh:69-76`).
- Fails loudly if `docker` is absent (`build_rootfs.sh:78`) and parses the pinned
  Rust channel from `rust-toolchain`, dying if unparseable (`build_rootfs.sh:80-83`).

### 2.2 Entering the rootfs (`scripts/rootfs/enter_rootfs.sh`)

- Auto-builds the rootfs on first use so a first run is a single command
  (`enter_rootfs.sh:50-58`); dies if `bwrap` is missing (`enter_rootfs.sh:48`).
- **bwrap args**: `--bind $ROOTFS /`, fresh `--proc/--tmpfs/--dev`, repo bind-mounted
  rw at `/workspace/monarch`, `--unshare-all --share-net --die-with-parent`
  (`enter_rootfs.sh:60-72`).
- **DNS**: docker export leaves `/etc/resolv.conf` empty, so host `/etc/resolv.conf`
  and `/etc/hosts` are read-only bound in for PyPI access
  (`enter_rootfs.sh:74-78`).
- **GPU access**: binds every `/dev/nvidia*` device node (`enter_rootfs.sh:80-85`)
  and read-only binds host `libcuda.so*` / `libnvidia-*.so*` + `nvidia-smi` so the
  in-rootfs CUDA runtime links the *matching host driver userspace*
  (`enter_rootfs.sh:87-96`).
- **Env**: sets `PATH`/`CUDA_HOME`/`CUDA_PATH`/`RUSTUP_HOME`/`CARGO_HOME`/
  `LD_LIBRARY_PATH`, plus `MONARCH_IN_ROOTFS=1` as the re-exec marker
  (`enter_rootfs.sh:98-109`). Honors caller `CUDA_VISIBLE_DEVICES` including the
  meaningful empty-string case (hide all GPUs) so negative preflight works
  (`enter_rootfs.sh:111-116`).

### 2.3 In-rootfs driver (`scripts/rootfs/run_in_rootfs.sh`)

- Hard-fails if `MONARCH_IN_ROOTFS != 1`, telling the caller to enter via
  `enter_rootfs.sh` (`run_in_rootfs.sh:29-30`); requires `uv`
  (`run_in_rootfs.sh:32`).
- Creates/reuses a `.venv-rootfs` inside the repo mount with
  `--system-site-packages` so the pinned rootfs torch + nvidia wheels are visible
  without re-download (`run_in_rootfs.sh:38-44`), then syncs deps and delegates to
  the suite runner (`run_in_rootfs.sh:55-58`).

### 2.4 Canonical local-run verifier (`scripts/run_local_8gpu_capacity.sh`)

- **Loud preflight then re-exec**: validates `CUDA_VISIBLE_DEVICES` via a Python
  helper (`run_local_8gpu_capacity.sh:28-43`); if not already inside the rootfs it
  `exec`s through `enter_rootfs.sh -- scripts/run_local_8gpu_capacity.sh` using a
  **repo-relative** script path (`run_local_8gpu_capacity.sh:45-49`).
- Ladder: rootfs entry -> `.venv-rootfs` -> editable build -> 8-rank tensor-engine
  smoke (`has_tensor_engine()`, `torch.cuda.device_count()==8`, shard-rank check,
  distinct exit codes 21/22/23) (`run_local_8gpu_capacity.sh:74-125`) -> full
  control-plane suites (`run_local_8gpu_capacity.sh:127-146`).
- **Failure classification**: on suite failure it parses JUnit XML via a Python
  helper and, if Rust passed, re-runs each failed Python test *in isolation*; if all
  pass alone it treats the full-run failure as suite-ordering fragility and exits 0
  (`run_local_8gpu_capacity.sh:148-203`). Complex parsing lives in
  `scripts/local_8gpu_capacity.py` (referenced at lines 33, 150, 161), keeping the
  shell entrypoint thin.

### 2.5 Bazel / CUDA / Python (PyO3)

- Monarch has **no Bazel build**; it is a PyO3 Rust extension built via `uv pip
  install -e .` inside the rootfs (`run_in_rootfs.sh:46-56`). CUDA hermeticity is
  achieved through the rootfs + pip nvidia wheels, not a Bazel toolchain.
- Documented workflow anchors (per memory, not re-read this session): `AGENTS.md`,
  `CONTEXT.md`, `docs/adr/0001-bwrap-rootfs-for-local-runs.md`, and
  `docs/adr/0002-python-isolation-reruns-for-local-run-classification.md`.

---

## 3. NCCL infra patterns

### 3.1 Build system: Bazel that wraps the existing Makefile

- Bazel pinned to **9.2.0** (`nccl/.bazelversion:1`).
- `MODULE.bazel` declares `rules_cc 0.2.22` and **`rules_cuda 0.3.0`**, and uses the
  `rules_cuda` toolchain extension with a **hermetic CUDA redistributable**:
  `cuda.redist_json(name="cuda_13_0_2_redist", version="13.0.2")` +
  `cuda.toolkit(name="cuda")` (`nccl/MODULE.bazel:1-15`). This is genuinely hermetic
  CUDA (downloaded redist), unlike cutlass's system-discovery approach.
- The Bazel target does **not** re-implement the build; `nccl_make_shared` stages the
  vendored `makefiles/**` + `src/**`, symlinks the hermetic CUDA redist components
  into a synthetic `cuda_toolkit/{bin,include,lib64}`, and shells out to
  `make -C src ... lib` with `NVCC_GENCODE=-gencode=arch=compute_100,code=sm_100`
  (`nccl/bazel/nccl_make_build.bzl:15-115`). Outputs `libnccl.so.2` + `nccl.h` via
  output groups (`nccl_make_build.bzl:117-123`).
- `BUILD.bazel` wires that rule up, exposes `nccl_headers` and a `cc_import`ed
  `nccl_shared`, and tags every CUDA target `["cuda"]` (`nccl/BUILD.bazel:1-60`).

### 3.2 `.bazelrc` — the hardened CUDA opt-in pattern (directly transferable)

- bzlmod on, sandboxed spawn strategy for Genrule/CppCompile/CppLink
  (`nccl/.bazelrc:1-9`).
- Flag aliases `cuda_archs`/`cuda_compiler`/`cuda_enable` -> `@rules_cuda//cuda:...`
  (`nccl/.bazelrc:11-13`).
- **CUDA off by default**, and `--build_tag_filters=-cuda` / `--test_tag_filters=-cuda`
  so a CPU-only host/CI skips CUDA targets (`nccl/.bazelrc:15-17`).
- `--config=cuda` flips filters open, enables rules_cuda, sets `compiler=nvcc`,
  `runtime=@cuda//:cuda_runtime`, and `archs=compute_100:sm_100`
  (`nccl/.bazelrc:19-24`). `--config=local_cuda` is an escape hatch that enables CUDA
  without forcing the hermetic redist (`nccl/.bazelrc:26-28`).

### 3.3 Rootfs / sandbox / local-run scripts

- **There is no bwrap rootfs and no `run_local_*` script in nccl.** Searches for
  `*rootfs*`/`*bwrap*` returned nothing; the only shell script under the repo top
  levels is `maint/run-clang-format.sh` (verified by directory listing).
  Isolation is Bazel's own `--spawn_strategy=sandboxed` (`nccl/.bazelrc:6-9`).
- Docs conventions: `AGENTS.md` points to `.scratch/` Markdown issues,
  `docs/agents/{issue-tracker,triage-labels,domain}.md`, and a single-context
  `CONTEXT.md` + `docs/adr/` (`nccl/AGENTS.md:1-13`). **Note:** `AGENTS.md` references
  `docs/adr/` but that directory does not currently exist (only `docs/agents/` is
  present) — an inconsistency in the reference repo, not a pattern to copy blindly.

---

## 4. Existing cutlass `src/` scaffold

Layout: `src/{MODULE.bazel,.bazelrc}` plus per-module `BUILD.bazel` under
`core, tensor, ops, kernels, attention, model, data, dist, checkpoint, train`
(directory listing). Additional Bazel packages already exist **at the cutlass repo
root**: `third_party/{cuda,cutlass,nccl}/BUILD.bazel`, `tests/BUILD.bazel`,
`tests/{integration,perf}/BUILD.bazel`, and `tools/bazel/{BUILD.bazel,cuda_ext.bzl}`.

### 4.1 Module + deps (`src/MODULE.bazel`)

- Root module `gpt_cuda_trainer` 0.1.0, a "single-node, multi-GPU GPT trainer built
  on CUTLASS/CuTe" (`src/MODULE.bazel:1-10`).
- bzlmod deps: `bazel_skylib 1.7.1`, `googletest 1.15.2`
  (`src/MODULE.bazel:16-17`). CUTLASS itself is **not** a bazel_dep — the comment
  says switch to `bazel_dep(name="cutlass", version="3.9.0")` if a native module
  appears, else vendor/local-extension it (`src/MODULE.bazel:19-21`).
- CUDA/NCCL via the local extension `//tools/bazel:cuda_ext.bzl`, calling
  `cuda.local_cuda`/`cuda.local_nccl` with paths auto-detected from
  `CUDA_HOME`/`NCCL_HOME` (`src/MODULE.bazel:27-36`).

### 4.2 `cuda_ext.bzl` — system-local (NOT hermetic) CUDA discovery

- `local_cuda` repo rule symlinks `$CUDA_HOME/{include,lib64}` (falling back to
  `/usr/local/cuda`) into the repo and generates a BUILD exposing `cuda_headers`,
  `cudart`, `cublas`, `cublaslt` (`tools/bazel/cuda_ext.bzl:8-70`).
- `local_nccl` symlinks `$NCCL_HOME/{include,lib}` (fallback `/usr`) and exposes
  `nccl_headers`, `nccl` (`tools/bazel/cuda_ext.bzl:19-99`).
- This is the **opposite of nccl's hermetic redist**: it trusts whatever CUDA/NCCL is
  installed on the host. `local=True` repo rules, keyed off env vars
  (`cuda_ext.bzl:66-70, 95-99`).

### 4.3 Third-party wrappers + arch config

- `//third_party/cuda` aliases the `@local_cuda` targets
  (`third_party/cuda/BUILD.bazel:9-27`); `//third_party/nccl` aliases `@local_nccl`
  (`third_party/nccl/BUILD.bazel:5-13`).
- `//third_party/cutlass` is a header-only `cc_library` globbing
  `include/**/*.{h,hpp,cuh}` and depending on `//third_party/cuda:cuda_headers`;
  comment says point `hdrs`/`includes` at vendored headers or alias a module target
  (`third_party/cutlass/BUILD.bazel:1-23`). **The include tree is not yet wired to
  cutlass's real `include/` — this is a TODO in the scaffold.**
- `src/.bazelrc`: **C++20** (`build --cxxopt=-std=c++20`,
  `src/.bazelrc:8-9`) — note this diverges from cutlass's own C++17. Warnings,
  dbg/opt/asan/tsan/profile configs, and **arch configs** `cuda_ampere=80,86`,
  `cuda_hopper=90,90a`, `cuda_ada=89` bound to `@local_cuda//:cuda_archs`
  (`src/.bazelrc:44-46`). **No Blackwell (100/100a/120) config** and no
  `--build_tag_filters=-cuda` default-off gating like nccl has.
- Module BUILD files: layered comments (Layer 1 core ... Layer 6 dist). `kernels`
  gemm depends on `//third_party/cutlass` and `//third_party/cuda:cublaslt`
  (`src/kernels/BUILD.bazel:8-22`); GPU tests tagged `["gpu"]`
  (`src/kernels/BUILD.bazel:70`). `dist` depends on `//third_party/nccl`
  (`src/dist/BUILD.bazel:5-13`). Tests use `//tests:test_main` + `//tests:test_utils`
  + `@googletest//:gtest` (`src/kernels/BUILD.bazel:59-71`, `tests/BUILD.bazel:11-26`).

### 4.4 Scaffold gaps found (load-bearing for grilling)

- **Root vs `src/` package-root mismatch**: BUILD files use root-relative labels
  (`//src/core`, `//third_party/cuda`, `//tests`, `//tools/bazel:cuda_ext.bzl` in
  `src/MODULE.bazel:27`), but `MODULE.bazel`/`.bazelrc` currently live in `src/`,
  while `third_party/`, `tests/`, `tools/bazel/` live at the **repo root**. There is
  **no `MODULE.bazel`, `.bazelrc`, `.bazelversion`, or `WORKSPACE` at the cutlass
  repo root** (verified). For labels like `//third_party/cuda` to resolve, the module
  root must be the repo root, so these files likely need to move up (or the labels
  are currently unbuildable as-is).
- No `.bazelversion` anywhere in the cutlass scaffold (nccl pins 9.2.0).
- `//third_party/cutlass` globs a non-existent `include/` dir under
  `third_party/cutlass/` rather than cutlass's real root `include/`.
- No rootfs, no `run_local_*` script, no CUDA-off tag gating in cutlass yet.

---

## 5. Cutlass build constraints (what a Bazel+rootfs wrapper must accommodate)

- Cutlass proper is **header-only** and "does not need to be built to be used by
  other projects" (`README.md:251`, `third_party/cutlass/BUILD.bazel:2-4`). A Bazel
  target only needs to expose the include tree, not compile it.
- **C++17 host compiler required** (`README.md:134,137`; `CMakeLists.txt:104-105`
  set `CMAKE_CXX_STANDARD 17` / `STANDARD_REQUIRED ON`). The public CMake interface
  advertises only `cxx_std_11` (`CMakeLists.txt:744`). The `src/` scaffold currently
  compiles at **C++20** (`src/.bazelrc:8`) — a deliberate choice for the trainer code
  that a wrapper must reconcile with cutlass's C++17 headers (usually fine, but a
  grilling point).
- Built primarily with **CMake** (`cmake_minimum_required 3.19`,
  `CMakeLists.txt:29`); recommends **CUDA 12.8 toolkit**, compatible with 11.4–12.x
  (`README.md:138-139`).
- **Arch selection is `CUTLASS_NVCC_ARCHS`** (`README.md:191-204`,
  `CMakeLists.txt:210-211`), with supported archs including `90a` (Hopper),
  `100/100a/120/120a` (Blackwell), etc. (`CMakeLists.txt:174-210`). "Architecture-
  accelerated features" (the `a` suffix) matter for Hopper/Blackwell perf
  (`README.md:183-204`). A Bazel/rootfs wrapper must let the operator pick the arch
  (e.g. `90a`, `100a`) the same way CMake's `-DCUTLASS_NVCC_ARCHS` does.
- Canonical CMake flow: `export CUDACXX=$CUDA/bin/nvcc; cmake .. -DCUTLASS_NVCC_ARCHS=90a`
  (`README.md:261-273`), and kernel filtering via `-DCUTLASS_LIBRARY_KERNELS=`
  (`README.md:393-409`). Any wrapper should preserve these knobs.

---

## 6. Synthesis: candidate infra shape for cutlass

Reconciling the two reference models with the existing `src/` scaffold:

- **Keep CMake for cutlass proper; Bazel is for the fork's `src/` trainer.** Cutlass
  itself is header-only CMake (§5). The fork code under `src/` is what benefits from
  Bazel. The existing scaffold already assumes this split (cutlass exposed as a
  header-only `cc_library`, `third_party/cutlass/BUILD.bazel:1-8`).
- **Adopt nccl's `.bazelrc` CUDA-gating pattern** (§3.2): default CUDA off with
  `--build_tag_filters=-cuda` and a `--config=cuda`, so CPU-only hosts/CI can build
  the non-GPU targets (`core`, `tensor` CPU pieces, tests). The GPU tests are already
  tagged (`src/kernels/BUILD.bazel:70`) so the tag filter would work immediately.
- **Two possible CUDA-hermeticity levels**: (a) keep the current `local_cuda`
  system-discovery (`cuda_ext.bzl`, simplest, matches "use the host's CUDA"), or (b)
  move to nccl's hermetic `rules_cuda 0.3.0` + `redist_json` (`nccl/MODULE.bazel:6-15`)
  for reproducibility. These are mutually exclusive top-level choices (grilling point).
- **Add a monarch-style bwrap rootfs + `run_local_*` runner** for on-device GPU
  validation (§2). The cutlass rootfs would be simpler than monarch's: no PyO3/uv/Rust
  toolchain, no `.venv`; it mainly needs a pinned nvcc + C++17 host toolchain (from a
  CUDA devel image, or the synthetic-CUDA-from-wheels trick,
  `build_rootfs.sh:148-169`) plus host-driver bind-in (`enter_rootfs.sh:87-96`) and
  `/dev/nvidia*` binds (`enter_rootfs.sh:80-85`). Reuse the atomic-stage/idempotent
  build (`build_rootfs.sh:173-199`) and the loud-preflight + repo-relative re-exec
  pattern (`run_local_8gpu_capacity.sh:28-49`).
- **First fix the scaffold's package-root mismatch** (§4.4) before layering infra on
  top — decide whether `MODULE.bazel`/`.bazelrc`/`.bazelversion` live at the repo root
  (matching the `//third_party`, `//tests` labels) or whether the scaffold is a
  self-contained `src/`-rooted module.
- **Verifier ladder analogous to monarch's**: preflight (GPU visibility, arch match)
  -> build -> a small GEMM/attention smoke test on-device -> the `test/unit` or
  `tests/` gtest suites, with distinct exit codes and loud failures
  (model on `run_local_8gpu_capacity.sh:74-125`).

---

## 7. Open decisions for grilling

- **Bazel vs keep CMake — scope.** Do we Bazel-ify only the fork `src/` trainer (and
  keep cutlass-proper on CMake), or attempt a Bazel wrapper over cutlass's CMake build
  too? The scaffold assumes the former; is that the intent?
- **CUDA hermeticity model.** Keep `local_cuda` system-discovery
  (`cuda_ext.bzl:30-70`) or switch to nccl-style hermetic `rules_cuda 0.3.0` +
  `redist_json` (`nccl/MODULE.bazel:6-15`)? If hermetic, which CUDA version — 13.0.2
  like nccl, or 12.8 as cutlass recommends (`README.md:138`)?
- **`rules_cuda` version + Bazel version.** Adopt nccl's `rules_cuda 0.3.0` / Bazel
  9.2.0 (`nccl/.bazelversion:1`)? The cutlass scaffold pins neither.
- **Package root.** Move `MODULE.bazel`/`.bazelrc` to the repo root (so
  `//third_party`, `//tests`, `//tools/bazel` labels resolve), or restructure the
  scaffold to be `src/`-rooted? (§4.4 — currently unbuildable as laid out.)
- **Target arch.** The `src/.bazelrc` only defines Ampere/Hopper/Ada
  (`src/.bazelrc:44-46`); the local host is Blackwell (B200, per memory). Add a
  `cuda_blackwell = 100,100a` config and make it the default? nccl already targets
  `sm_100` (`nccl/.bazelrc:24`).
- **C++ standard.** `src/.bazelrc` uses C++20 (`src/.bazelrc:8`) while cutlass is
  C++17 (`CMakeLists.txt:104`). Keep C++20 for the trainer, or align to C++17 to
  minimize surprise when including cutlass headers?
- **Rootfs contents + base image.** Reuse monarch's PyTorch-CUDA base
  (`build_rootfs.sh:41`), use a CUDA *devel* image (real nvcc, no synthetic trick),
  or the synthetic-CUDA-from-wheels approach? Does the cutlass rootfs need Python at
  all, or just a C++17 + nvcc toolchain?
- **How does matmul / GEMM feature work slot in?** The fork's GEMM already depends on
  `//third_party/cutlass` + cublasLt (`src/kernels/BUILD.bazel:8-22`), and
  `third_party/cutlass` isn't yet wired to the real `include/` tree
  (`third_party/cutlass/BUILD.bazel:14-19`). Wire that first? Which archs must the
  GEMM kernels compile for in the verifier smoke test?
- **Hermetic vs sandboxed isolation.** nccl relies purely on Bazel
  `--spawn_strategy=sandboxed` with no rootfs (`nccl/.bazelrc:6-9`); monarch uses a
  full bwrap rootfs. For cutlass GPU tests, is Bazel sandboxing enough, or is a bwrap
  rootfs (for driver/toolchain consistency) warranted?
- **Docs/ADR convention.** monarch uses `docs/adr/000N-*.md`; nccl's `AGENTS.md`
  points at `docs/adr/` but that dir is missing (`nccl/AGENTS.md:9-13`). Which
  convention should cutlass adopt for recording these infra decisions?
