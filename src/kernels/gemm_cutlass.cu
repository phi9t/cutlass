// Layer 2 — Primitive kernels: CUTLASS GEMM backend.
//
// This file contains the CUTLASS/CuTe-based GEMM implementations.
// For v1 correctness bring-up, we use CUTLASS 3.x collective mainloop
// with sensible default tile shapes.

#include "src/kernels/gemm.h"

#include <cuda_bf16.h>

// TODO(m1): Wire CUTLASS 3.x collective GEMM with CuTe layout.
//   - Select tile shape based on problem size.
//   - Support both [M,K] x [K,N] and transposed strides.
//   - fp32 accumulation for bf16 inputs.

namespace gpt {
namespace kernels {

Status gemm_f32(Tensor2D<float> A, Tensor2D<float> B, Tensor2D<float> C,
                float alpha, float beta, const CudaStream& stream,
                GemmBackend backend) {
  if (backend != GemmBackend::kCutlass) {
    return Status(StatusCode::kNotImplemented,
                  "Only CUTLASS backend implemented here");
  }
  // Validate shapes.
  if (A.shape[1] != B.shape[0]) {
    return Status(StatusCode::kInvalidArgument, "GEMM K-dim mismatch");
  }
  if (A.shape[0] != C.shape[0] || B.shape[1] != C.shape[1]) {
    return Status(StatusCode::kInvalidArgument, "GEMM output shape mismatch");
  }

  // TODO: Launch CUTLASS GEMM kernel.
  return Status(StatusCode::kNotImplemented,
                "CUTLASS f32 GEMM kernel not yet wired");
}

Status gemm_bf16(Tensor2D<__nv_bfloat16> A, Tensor2D<__nv_bfloat16> B,
                 Tensor2D<__nv_bfloat16> C, float alpha, float beta,
                 const CudaStream& stream, GemmBackend backend) {
  if (backend != GemmBackend::kCutlass) {
    return Status(StatusCode::kNotImplemented,
                  "Only CUTLASS backend implemented here");
  }
  if (A.shape[1] != B.shape[0]) {
    return Status(StatusCode::kInvalidArgument, "GEMM K-dim mismatch");
  }
  if (A.shape[0] != C.shape[0] || B.shape[1] != C.shape[1]) {
    return Status(StatusCode::kInvalidArgument, "GEMM output shape mismatch");
  }

  // TODO: Launch CUTLASS bf16 GEMM kernel with fp32 accumulation.
  return Status(StatusCode::kNotImplemented,
                "CUTLASS bf16 GEMM kernel not yet wired");
}

Status batched_gemm_f32(Tensor3D<float> A, Tensor3D<float> B,
                        Tensor3D<float> C, float alpha, float beta,
                        const CudaStream& stream, GemmBackend backend) {
  if (backend != GemmBackend::kCutlass) {
    return Status(StatusCode::kNotImplemented,
                  "Only CUTLASS backend implemented here");
  }
  if (A.shape[0] != B.shape[0] || A.shape[0] != C.shape[0]) {
    return Status(StatusCode::kInvalidArgument,
                  "Batched GEMM batch-dim mismatch");
  }
  if (A.shape[2] != B.shape[1]) {
    return Status(StatusCode::kInvalidArgument,
                  "Batched GEMM K-dim mismatch");
  }

  // TODO: Launch CUTLASS batched GEMM kernel.
  return Status(StatusCode::kNotImplemented,
                "CUTLASS batched f32 GEMM kernel not yet wired");
}

Status batched_gemm_bf16(Tensor3D<__nv_bfloat16> A,
                         Tensor3D<__nv_bfloat16> B,
                         Tensor3D<__nv_bfloat16> C, float alpha, float beta,
                         const CudaStream& stream, GemmBackend backend) {
  if (backend != GemmBackend::kCutlass) {
    return Status(StatusCode::kNotImplemented,
                  "Only CUTLASS backend implemented here");
  }
  if (A.shape[0] != B.shape[0] || A.shape[0] != C.shape[0]) {
    return Status(StatusCode::kInvalidArgument,
                  "Batched GEMM batch-dim mismatch");
  }
  if (A.shape[2] != B.shape[1]) {
    return Status(StatusCode::kInvalidArgument,
                  "Batched GEMM K-dim mismatch");
  }

  // TODO: Launch CUTLASS batched bf16 GEMM kernel with fp32 accumulation.
  return Status(StatusCode::kNotImplemented,
                "CUTLASS batched bf16 GEMM kernel not yet wired");
}

}  // namespace kernels
}  // namespace gpt
