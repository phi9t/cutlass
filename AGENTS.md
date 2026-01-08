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

## Commit & Pull Request Guidelines
- Recent history favors short, imperative subjects, sometimes with a scope tag (e.g., `[SM90] ...`) or `fix:` prefix.
- Include a PR or issue reference when applicable (e.g., `(#2219)`).
- PRs should describe the change, call out architecture flags used, and include test evidence (`make test_unit` or profiler runs) when relevant.

## Configuration Tips
- Use `-DCUTLASS_NVCC_ARCHS` to match your GPU (e.g., `80`, `90a`, `100a`).
- You can filter kernel builds with `-DCUTLASS_LIBRARY_KERNELS=...` to reduce compile times.
