# Repository Guidelines

## Project Structure & Module Organization
- `include/` is the header-only CUTLASS and CuTe libraries; most core APIs live under `include/cutlass/` and `include/cute/`.
- `tools/` contains the CUTLASS instance library, profiler, and utilities.
- `examples/` holds SDK examples and tutorials.
- `test/unit/` contains Google Test-based unit tests.
- `media/docs/` is the main documentation set, including coding guidelines and architecture notes.

## Build, Test, and Development Commands
Use an out-of-tree build directory and set GPU architecture via `CUTLASS_NVCC_ARCHS`.
```bash
export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=90a
```
Build tools or tests from `build/`:
```bash
make cutlass_profiler -j
make test_unit -j
```
To speed up builds when you only need the profiler:
```bash
cmake .. -DCUTLASS_NVCC_ARCHS=90a -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON
```

## Coding Style & Naming Conventions
- Follow the C++ Core Guidelines and Google C++ Style Guide, with CUTLASS-specific exceptions.
- Indentation: 2 spaces, spaces only (no tabs), and keep lines <= 100 chars.
- Do not use automatic formatters like `clang-format`.
- Keep formatting-only changes separate from functional edits.

## Testing Guidelines
- Unit tests are under `test/unit/` and use Google Test.
- Build and run via `make test_unit -j` from the `build/` directory.
- No explicit coverage target is documented; add tests alongside new kernels or utilities.

## Git Workflow & Landing
- This checkout is a **fork**. Remotes: `origin` -> `https://github.com/phi9t/cutlass.git` (the fork), `upstream` -> `https://github.com/NVIDIA/cutlass.git` (NVIDIA).
- **"Land on mainline" means commit to the local `main` branch only.** Do NOT push to `origin` (or any remote) unless the user explicitly asks to push.
- Fork-specific work lives under `src/` (the trainer scaffold) plus a small set of additive root-level infra: `MODULE.bazel`/`.bazelrc`/`.bazelversion` (Bazel 9.2.0), `scripts/rootfs/` + `scripts/run_local_gpu_smoke.sh` (hermetic bwrap rootfs + loud GPU verifier), `tests/smoke/` (CUTLASS GEMM toolchain proof), `include/BUILD.bazel` + `third_party/cutlass/` (expose upstream CUTLASS headers to Bazel), and `CONTEXT.md`/`docs/adr/`. These are disjoint from CUTLASS's own CMake build, so they rebase cleanly onto upstream releases.
- Hermetic local GPU check: `scripts/run_local_gpu_smoke.sh` re-execs into the rootfs and walks a loud ladder (21 preflight / 22 build / 23 GEMM smoke / 24 optional gtest). First run auto-builds the rootfs via `scripts/rootfs/build_rootfs.sh`.
- **CUDA builds MUST run inside the rootfs.** A build-level guard (`//tools/bazel:rootfs_guard`, ADR 0006) hard-fails any `--config=cuda` build without `CUTLASS_IN_ROOTFS=1`, pointing you to the verifier. Never work around it by building CUDA on the host; enter the rootfs (`scripts/rootfs/enter_rootfs.sh -- bash`) or run the verifier. The escape hatch `CUTLASS_ALLOW_HOST_CUDA=1` exists only for deliberate expert toolchain debugging.
- To sync with a new upstream release: `git fetch upstream --tags`, then rebase local `main` onto the latest release tag (e.g., `git rebase 4.7.0`), replaying the fork commits on top.

## Commit & Pull Request Guidelines
- Recent history favors short, imperative subjects, sometimes with a scope tag (e.g., `[SM90] ...`) or `fix:` prefix.
- Include a PR or issue reference when applicable (e.g., `(#2219)`).
- PRs should describe the change, call out architecture flags used, and include test evidence (`make test_unit` or profiler runs) when relevant.

## Configuration Tips
- Use `-DCUTLASS_NVCC_ARCHS` to match your GPU (e.g., `80`, `90a`, `100a`).
- You can filter kernel builds with `-DCUTLASS_LIBRARY_KERNELS=...` to reduce compile times.

<!-- ultron-agentic-workflow:start -->
## Agentic engineering workflow

**Mandatory:** Read and follow `CONSTITUTION.md` before acting. Before planning,
building, fixing, or changing code, read and follow
`docs/agents/agentic-engineering.md`. Direct user instructions and more specific
repository guidance take precedence.
<!-- ultron-agentic-workflow:end -->
