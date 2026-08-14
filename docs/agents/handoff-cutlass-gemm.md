# Handoff — CUTLASS fp32 GEMM completion and next scaffold frontier

## Status

Completed on local `main`:
- `a753d47a` — wired CUTLASS fp32 `gemm_f32` / `batched_gemm_f32` and linear
  bias add.
- `f3b9ac2e` — completed the causal attention forward path by making
  `split_heads_f32` honor strided Q/K/V views and applying the existing fused
  scale+causal-mask kernel.

Verification, inside the hermetic rootfs on the B200:
- `scripts/run_local_gpu_smoke.sh --with-gtest` -> **PASS**
- preflight PASS, CUTLASS GEMM smoke PASS, gpu-tagged `//src/...` gtest rung
  **15 / 15 PASS**

This file is now a completion record plus context for the next scaffold work.

## Additional completed slice

After the 15/15 forward-path milestone, the next small correctness hole was
closed test-first:
- `src/ops/linear_test.cc` now asserts `d_bias` in both backward GPU cases.
- `src/kernels/reductions.{h,cu}` adds `col_sum_f32`.
- `src/ops/linear.cc::linear_backward` computes `db = col_sum(dY)` when a
  `d_bias` buffer is present.
- `src/model/gpt_block.cc::block_forward` now performs both residual adds via
  `kernels::vec_add_f32`, and `src/model/gpt_block_test.cc` compares the GPU
  block output against a CPU reference instead of only checking finiteness.
- `src/model/gpt_model.cc::gpt_forward` now adds positional embeddings via a
  model-local CUDA helper, and `src/model/gpt_model_test.cc` has a zero-layer
  GPU-vs-CPU reference test for token + position embedding, final LN, and LM
  head.

## Next recommended scaffold frontier

Full attention backward and GPT backward remain larger follow-on work:
- `src/attention/attention.cc::attention_backward` returns `kNotImplemented`.
- `src/model/gpt_block.cc::block_backward` returns `kNotImplemented`.
- `src/model/gpt_model.cc::gpt_backward` returns `kNotImplemented`.

Keep landing local-only unless explicitly told to push.

---

Historical handoff below documents the original GEMM task and the layout
pitfalls that were resolved.

## Context / where we are

The cutlass **fork** infra is done and landed on local `main`:
- bwrap rootfs + hermetic Bazel + hermetic CUDA (sm_100 / B200) + NCCL + a loud
  local-run verifier. See `CONTEXT.md`, `docs/adr/0001..0006`.
- The `src/` GPT-trainer scaffold was made **buildable under C++17** (commit
  `77b87ba4`): C++20 `requires` removed, missing includes added, const-view
  conversion added, `init_gpt_params/grads` defined.

Original pre-implementation state, verified inside the rootfs on the B200:
- `bazel build --config=cuda //src/...` → **succeeds**.
- gtest rung: **12 / 15** gpu-tagged targets pass.
- The **3 failing** targets — `//src/ops:linear_test`,
  `//src/attention:attention_forward_test`, `//src/model:gpt_block_test` — all
  fail for **one root cause**: `kernels::gemm_f32` is a `kNotImplemented` stub.

## The task (this is the scoped-in feature work now)

Replace the `gemm_f32` stub in `src/kernels/gemm_cutlass.cu` with a real CUTLASS
fp32 GEMM so `linear_test` (and its dependents) pass. **Landing rule: commit to
local `main` only — never push** (AGENTS.md; origin=phi9t fork, upstream=NVIDIA).
Commit trailer required: `Co-authored-by: TRAE CLI <noreply@bytedance.com>`.

## Red bar (drive this test-first, /tdd)

Run: `scripts/run_local_gpu_smoke.sh --with-gtest` (re-execs into rootfs; rung 24
is the gtest rung). Or, faster inner loop, inside the rootfs:
`scripts/rootfs/enter_rootfs.sh -- bazel test --config=cuda //src/ops:linear_test`

`linear_test.cc` asserts `linear_forward(...).ok()` then compares against a CPU
reference with `test::vectors_near(..., 1e-4f, 1e-3f)`. Cases: no-bias, with-bias,
identity-weight, and backward (dX, dW).

## What gemm_f32 must compute — and the layout subtlety

`kernels::gemm_f32(A, B, C, alpha, beta, stream, backend)` computes
`C = alpha*A@B + beta*C`, where A,B,C are `Tensor2D<float>` (row-major
`{rows,cols}` shape, `{stride_row, stride_col}` stride).

**Critical:** `src/ops/linear.cc` does NOT pass plain row-major matrices. It
builds *transposed-stride views* to express W^T without copying:
- forward `Y = X @ W^T`: passes `W_T` with `shape={D_in,D_out}`,
  `stride={1, D_in}` (i.e. column-major over the original `[D_out,D_in]` W).
- backward `dW = dY^T @ X`: passes `dY_T` with `shape={D_out,N}`,
  `stride={1, D_out}`.
So the GEMM MUST honor arbitrary row/col strides on A and B (not assume
contiguous row-major). The cleanest CUTLASS mapping: a matrix with
`stride={s_row, s_col}` is row-major with leading dim `s_row` when `s_col==1`,
and column-major with leading dim `s_col` when `s_row==1`. Detect per-operand
and pick the CUTLASS layout + leading dimension accordingly, or normalize by
mapping (row-major MxN, lda) ↔ (column-major NxM, lda) identities.

## Working reference to model from (already passing on sm_100)

`tests/smoke/cutlass_gemm_smoke.cu` calls CUTLASS successfully:
```cpp
using ColumnMajor = cutlass::layout::ColumnMajor;
using Gemm = cutlass::gemm::device::Gemm<float, ColumnMajor, float,
                                         ColumnMajor, float, ColumnMajor>;
Gemm op;
Gemm::Arguments args({M,N,K}, {A,lda}, {B,ldb}, {C,ldc}, {C,ldc}, {alpha,beta});
if (op(args) != cutlass::Status::kSuccess) return cudaErrorUnknown;
```
Header: `#include "cutlass/gemm/device/gemm.h"`. The `gemm` Bazel target already
deps `//third_party/cutlass` (alias → `//include:cutlass`) and
`//third_party/cuda:cublaslt`, so both a CUTLASS and a cuBLASLt backend are
available without BUILD changes. `cutlass::gemm::device::Gemm` supports
Row/ColumnMajor as template args — you likely need to either template over the
detected layouts or standardize to one orientation via the transpose identity
`(A@B)^T = B^T@A^T`.

Tip: `cutlass::gemm::device::Gemm` picks a default kernel the library
specializes for the compiled arch (worked for sm_100 in the smoke). Keep the
epilogue as linear alpha/beta. Add a `cudaGetLastError()`/status check and
return `StatusCode::kCudaError` on launch failure so tests fail loudly, not
silently.

## Known follow-on issues (triage after gemm_f32 works)

1. **Bias not applied.** `linear_forward` has `// TODO: Add bias`. Test
   `LinearTest.ForwardWithBiasMatchesCPU` WILL still fail after GEMM alone —
   you must add the bias broadcast-add (`[D_out]` over `[N,D_out]`). There may
   be a suitable kernel in `src/kernels/elementwise.*` or `src/ops/*`; check
   before writing a new one. `linear_backward` also has `// TODO db` (col-sum of
   dY) — `linear_test` backward cases check dX and dW; confirm whether db is
   asserted before deciding scope.
2. **attention_forward_test / gpt_block_test** depend on linear; they may pass
   once linear works, or may need `batched_gemm_f32` (also a stub in
   `gemm_cutlass.cu`) and/or the attention score/softmax path. Re-run and triage.
   Batched GEMM feeds attention `S=Q@K^T` and `C=P@V`.

## Files

- `src/kernels/gemm_cutlass.cu` — the stubs to implement (`gemm_f32`, then maybe
  `batched_gemm_f32`).
- `src/kernels/gemm.h` — signatures (already includes `<cuda_bf16.h>`).
- `src/ops/linear.cc` — caller; transposed-stride views + the bias TODO.
- `src/ops/linear_test.cc` — the red bar.
- `tests/smoke/cutlass_gemm_smoke.cu` — working CUTLASS example.
- `src/kernels/gemm_cublaslt.cc` — alternative backend if CUTLASS layout mapping
  gets painful (cuBLASLt handles arbitrary strides/transpose naturally).

## Definition of done

- `scripts/run_local_gpu_smoke.sh --with-gtest` → **15/15** gtest targets pass
  (or a documented, justified scope cut for anything genuinely beyond fp32 GEMM,
  e.g. bf16 kernels).
- Land on local `main` with a descriptive commit (+ required trailer). Consider
  an ADR note if the layout-mapping decision is non-obvious.

## Task tracker

Open tasks in the shared list: **#24** (wire CUTLASS fp32 GEMM) and **#25** (get
15/15 green + land). Mark in_progress before starting.
