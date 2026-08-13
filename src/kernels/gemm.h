#pragma once

// Layer 2 — Primitive kernels: GEMM / batched GEMM.
//
// Public API for dense matrix multiplication.  Two backends:
//   1. CUTLASS/CuTe  (primary, tunable)
//   2. cuBLASLt       (fallback, early bring-up)
//
// All functions take TensorView arguments and a CudaStream.

#include <cstdint>

#include <cuda_bf16.h>

#include "src/core/status.h"
#include "src/core/stream.h"
#include "src/tensor/tensor_view.h"

namespace gpt {
namespace kernels {

// ---------------------------------------------------------------------------
// Backend selector.
// ---------------------------------------------------------------------------

enum class GemmBackend : uint8_t {
  kCutlass = 0,
  kCublasLt,
};

// ---------------------------------------------------------------------------
// GEMM:  C = alpha * A @ B + beta * C
//
//   A: [M, K]
//   B: [K, N]
//   C: [M, N]
//
// Supports transposed inputs via stride conventions.
// ---------------------------------------------------------------------------

Status gemm_f32(Tensor2D<float> A, Tensor2D<float> B, Tensor2D<float> C,
                float alpha, float beta, const CudaStream& stream,
                GemmBackend backend = GemmBackend::kCutlass);

Status gemm_bf16(Tensor2D<__nv_bfloat16> A, Tensor2D<__nv_bfloat16> B,
                 Tensor2D<__nv_bfloat16> C, float alpha, float beta,
                 const CudaStream& stream,
                 GemmBackend backend = GemmBackend::kCutlass);

// ---------------------------------------------------------------------------
// Batched GEMM:  C[b] = alpha * A[b] @ B[b] + beta * C[b]
//
//   A: [B, M, K]
//   B: [B, K, N]
//   C: [B, M, N]
// ---------------------------------------------------------------------------

Status batched_gemm_f32(Tensor3D<float> A, Tensor3D<float> B,
                        Tensor3D<float> C, float alpha, float beta,
                        const CudaStream& stream,
                        GemmBackend backend = GemmBackend::kCutlass);

Status batched_gemm_bf16(Tensor3D<__nv_bfloat16> A,
                         Tensor3D<__nv_bfloat16> B,
                         Tensor3D<__nv_bfloat16> C, float alpha, float beta,
                         const CudaStream& stream,
                         GemmBackend backend = GemmBackend::kCutlass);

}  // namespace kernels
}  // namespace gpt
