#pragma once

// Package-internal cuBLASLt backend entrypoints. Public callers should use
// gemm.h and select GemmBackend::kCublasLt.

#include "src/kernels/gemm.h"

namespace gpt {
namespace kernels {

Status gemm_f32_cublaslt(Tensor2D<float> A, Tensor2D<float> B,
                         Tensor2D<float> C, float alpha, float beta,
                         const CudaStream& stream);

Status gemm_bf16_cublaslt(Tensor2D<__nv_bfloat16> A,
                          Tensor2D<__nv_bfloat16> B,
                          Tensor2D<__nv_bfloat16> C, float alpha, float beta,
                          const CudaStream& stream);

Status batched_gemm_f32_cublaslt(Tensor3D<float> A, Tensor3D<float> B,
                                 Tensor3D<float> C, float alpha, float beta,
                                 const CudaStream& stream);

Status batched_gemm_bf16_cublaslt(Tensor3D<__nv_bfloat16> A,
                                  Tensor3D<__nv_bfloat16> B,
                                  Tensor3D<__nv_bfloat16> C, float alpha,
                                  float beta, const CudaStream& stream);

}  // namespace kernels
}  // namespace gpt
