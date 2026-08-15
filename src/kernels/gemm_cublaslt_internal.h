#pragma once

// Package-internal cuBLASLt backend entrypoints. Public callers should use
// gemm.h and select GemmBackend::kCublasLt.

#include "src/kernels/gemm.h"

namespace gpt {
namespace kernels {

Status gemm_f32_cublaslt(Tensor2D<float> A, Tensor2D<float> B,
                         Tensor2D<float> C, float alpha, float beta,
                         const CudaStream& stream);

}  // namespace kernels
}  // namespace gpt
