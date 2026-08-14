# ADR 0007 — fp32 GEMM maps arbitrary strides via the transpose identity

- Status: Accepted
- Date: 2026-08-14
- Scope: `src/kernels/gemm_cutlass.cu` (`gemm_f32`, `batched_gemm_f32`),
  consumed by `src/ops/linear.cc` and `src/attention/attention.cc`.

## Context

`kernels::gemm_f32(A, B, C, alpha, beta, ...)` computes `C = alpha*A@B + beta*C`
where `A`, `B`, `C` are `Tensor2D<float>`: element `(i, j)` lives at
`i*stride_row + j*stride_col`. The callers do **not** pass plain row-major
matrices. `src/ops/linear.cc` builds *transposed-stride views* to express `W^T`
without copying:

- forward `Y = X @ W^T`: `W_T` has `shape={D_in, D_out}`, `stride={1, D_in}`
  (column-major over the original `[D_out, D_in]` `W`);
- backward `dW = dY^T @ X`: `dY_T` has `shape={D_out, N}`, `stride={1, D_out}`.

`src/attention/attention.cc` does the same for the batched `S = Q@K^T` step
(`K` passed with `stride={..., 1, Dh}`). So the GEMM MUST honor an arbitrary
unit-stride axis on each operand — it cannot assume contiguous row-major.

The working smoke (`tests/smoke/cutlass_gemm_smoke.cu`) proved only the
column-major NN path on the compiled arch.

## Decision

**Detect each operand's orientation from its strides, then evaluate the
transpose `C^T = B^T @ A^T` in column-major.**

- A 2-D operand contributes exactly one unit-stride axis. If `stride_col == 1`
  it is **row-major** (leading dim `stride_row`); if `stride_row == 1` it is
  **column-major** (leading dim `stride_col`); anything else is rejected with
  `kInvalidArgument` (loud, not silent).
- CUTLASS `device::Gemm` computes a column-major product. Our output `C` is
  always row-major `M x N`, which is bit-identical in memory to a column-major
  `N x M` matrix (`C^T`). Using the identity `(A@B)^T = B^T @ A^T`, we call
  CUTLASS with operand order **(B, A)** and supply each operand's *transposed*
  layout (row-major in its own frame becomes column-major once transposed, and
  vice versa). The leading dimension is the physical stride, unchanged by
  transposition. This writes `C^T` column-major, i.e. `C` row-major, in place.
- Because CUTLASS templates the layout of each operand, we dispatch over the
  four `(RowMajor|ColMajor) x (RowMajor|ColMajor)` instantiations for `(B, A)`.
  Output layout is fixed column-major.
- `batched_gemm_f32` is the strided-batched analogue via
  `cutlass::gemm::device::GemmBatched`: the same per-batch 2-D classification
  plus a per-operand batch stride (`stride[0]`).
- Both paths check the CUTLASS `Status` and `cudaGetLastError()` and return
  `kCudaError` on failure so tests fail loudly rather than silently.

Bias for `linear_forward` is a `[D_out]` vector broadcast over `[N, D_out]`,
which is not a linear alpha/beta epilogue. It is applied by a small
`kernels::bias_add_f32` broadcast kernel added to the elementwise family
(reusing that layer's idiom) rather than fusing into the GEMM epilogue.

## Consequences

- `//src/ops:linear_test` passes all 7 cases (no-bias, with-bias, identity,
  shape-error, backward dX and dW). `//src/model:gpt_block_test` passes.
  The first GEMM slice moved the gtest ladder from **12/15 -> 14/15**.
- Rejected alternative: normalizing every operand to contiguous row-major with
  explicit transpose copies. It would work but adds device copies and scratch
  for what is purely a metadata reinterpretation; the transpose identity is
  zero-copy.
- Rejected alternative: a cuBLASLt backend (which handles arbitrary
  strides/transpose natively). Kept as the documented fallback in
  `gemm_cublaslt.cc`; the CUTLASS layout mapping turned out clean enough not to
  need it.

## Follow-on to reach 15/15

The remaining `//src/attention:attention_forward_test` failure turned out to
have two attention-layer causes exposed only after fp32 batched GEMM worked:

- `split_heads_f32` ignored `Tensor3D` input strides, but attention passes
  strided Q/K/V views into packed `[B, T, 3*D]` QKV storage. It now reads
  `input.stride[]` and writes contiguous `[B, H, T, Dh]` buffers.
- The GPU path ran causal attention with an unmasked softmax, while the CPU
  reference applies the upper-triangle causal mask. The existing
  attention-forward CUDA scale/mask kernel is now exposed through a package
  internal helper and applied after score GEMM for `config.causal`.

With those follow-ons, `scripts/run_local_gpu_smoke.sh --with-gtest` passes:
preflight, CUTLASS GEMM smoke, and all 15 gpu-tagged `//src` tests.
