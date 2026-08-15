# Handoff — CUTLASS fp32 GEMM completion and next scaffold frontier

## Status

Completed on local `main`:
- `a753d47a` — wired CUTLASS fp32 `gemm_f32` / `batched_gemm_f32` and linear
  bias add.
- `f3b9ac2e` — completed the causal attention forward path by making
  `split_heads_f32` honor strided Q/K/V views and applying the existing fused
  scale+causal-mask kernel.
- `7109ebf9`, `13d3dc28` — wired CUTLASS bf16 GEMM and batched bf16 GEMM.
- `2706a499`, `c55001bf`, `0f0b468d`, `d81880e5` — wired the cuBLASLt fp32,
  bf16, batched fp32, and batched bf16 GEMM backends.
- `97140b30` — made contiguous repack honor non-contiguous strided 2D input.
- `bb9b1570` — wired the single-rank NCCL lifecycle and identity all-reduce.
- `c60b69e8` — loaded checkpoint metadata from `meta.json`.
- attention backward slice — wired `src/attention/attention.cc::attention_backward`
  and a tiny GPU-vs-CPU backward test for `dX`, QKV projection grads, and output
  projection grads.

Verification, inside the hermetic rootfs on the B200:
- `scripts/run_local_gpu_smoke.sh --with-gtest` -> **PASS**
- preflight PASS, CUTLASS GEMM smoke PASS, gpu-tagged `//src/...` gtest rung
  **16 / 16 PASS**

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
- `src/kernels/gemm_cutlass.cu` now covers fp32, batched fp32, bf16, batched
  bf16, and dispatch to cuBLASLt variants.
- `src/kernels/gemm_cublaslt.cc` now covers fp32, bf16, batched fp32, and
  batched bf16 through cuBLASLt.
- `src/dist/nccl_context.cc` now supports `world_size == 1`; multi-rank still
  returns `kNotImplemented` because there is no rendezvous or shared unique-ID
  API in `NcclConfig`.
- `src/checkpoint/checkpoint.cc` now parses `meta.json` and fills
  `CheckpointMetadata::{step,param_count,dataset_cursor,config_json}`.
- `src/attention/attention.cc::attention_backward` now reverses the forward path
  using existing linear, softmax, and batched GEMM primitives plus small
  attention-local gradient layout kernels.
- `src/model/gpt_block.cc::block_backward` now reverses the transformer block
  with residual-gradient fanout, MLP backward, LN backward, and attention
  backward.
- `src/model/gpt_model.cc::gpt_backward` now supports zero-layer models:
  LM-head backward, final LayerNorm backward, token embedding backward, and
  position embedding gradients.
- `src/model/gpt_model.cc::gpt_backward` now supports one or more transformer
  layers by saving each block input in `GPTForwardState::block_inputs` and
  walking `block_backward` in reverse.
- `src/train/trainer.cc::train_step` now owns GPT forward-state allocation,
  computes cross-entropy gradients, runs `gpt_backward`, and feeds real
  gradients into AdamW. `src/train/trainer_test.cc` covers two public
  `Trainer::train_step` calls on a tiny GPU model.
- `tests/integration/gpt_checkpoint_roundtrip_test.cc` now verifies that
  checkpoint save/load preserves zero-layer GPT forward logits after rebuilding
  typed parameter views from a loaded flat parameter buffer.

## Next recommended scaffold frontier

The next larger follow-on work is the integration/runtime layer: wiring
`src/train/main.cc` to real data batches, adding resume-equivalence / overfit
coverage, and defining the DDP/NCCL multi-rank rendezvous policy.

The DDP averaging TODO in `src/dist/ddp.cc` should wait until multi-rank NCCL is
wired. The only initialized `NcclContext` today is `world_size == 1`, where
dividing by `world_size` is a no-op and does not produce a useful red bar.

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

## The original task (complete)

Replace the `gemm_f32` stub in `src/kernels/gemm_cutlass.cu` with a real CUTLASS
fp32 GEMM so `linear_test` (and its dependents) pass. **Landing rule: commit to
local `main` only — never push** (AGENTS.md; origin=phi9t fork, upstream=NVIDIA).
Commit trailer required: `Co-authored-by: TRAE CLI <noreply@bytedance.com>`.

## Original red bar

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

## Resolved follow-on issues from the original GEMM task

1. **Bias and db.** `linear_forward` applies bias through
   `kernels::bias_add_f32`; `linear_backward` computes `db` through
   `kernels::col_sum_f32`; `//src/ops:linear_test` covers both.
2. **Attention/block forward.** `batched_gemm_f32`, strided Q/K/V split, causal
   masking, residual adds, and positional embedding are wired. The gpu-tagged
   `//src/...` suite is green.

## Original files touched by the GEMM task

- `src/kernels/gemm_cutlass.cu` — CUTLASS GEMM backends and dispatch.
- `src/kernels/gemm.h` — signatures (already includes `<cuda_bf16.h>`).
- `src/ops/linear.cc` — caller; transposed-stride views, bias add, and `db`.
- `src/ops/linear_test.cc` — original red bar.
- `tests/smoke/cutlass_gemm_smoke.cu` — working CUTLASS example.
- `src/kernels/gemm_cublaslt.cc` — alternative backend if CUTLASS layout mapping
  gets painful (now also implemented).

## Definition of done

- `scripts/run_local_gpu_smoke.sh --with-gtest` → **15/15** gtest targets pass
  (or a documented, justified scope cut for anything genuinely beyond fp32 GEMM,
  e.g. bf16 kernels).
- Land on local `main` with a descriptive commit (+ required trailer). Consider
  an ADR note if the layout-mapping decision is non-obvious.

## Task tracker

Open tasks in the shared list: **#24** (wire CUTLASS fp32 GEMM) and **#25** (get
15/15 green + land). Mark in_progress before starting.
