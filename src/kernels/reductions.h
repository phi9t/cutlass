#pragma once

// Layer 2 — Primitive kernels: reduction operations.
//
// Row-wise reductions over the innermost dimension.
// All reductions operate on contiguous [rows, cols] input.

#include "src/core/status.h"
#include "src/core/stream.h"
#include "src/tensor/tensor_view.h"

namespace gpt {
namespace kernels {

// ---------------------------------------------------------------------------
// Row-wise max:  out[i] = max_j(input[i, j])
//   input:  [rows, cols]
//   output: [rows]
// ---------------------------------------------------------------------------

Status row_max_f32(Tensor2D<const float> input, Tensor1D<float> output,
                   const CudaStream& stream);

// ---------------------------------------------------------------------------
// Row-wise sum:  out[i] = sum_j(input[i, j])
// ---------------------------------------------------------------------------

Status row_sum_f32(Tensor2D<const float> input, Tensor1D<float> output,
                   const CudaStream& stream);

// ---------------------------------------------------------------------------
// Row-wise mean:  out[i] = mean_j(input[i, j])
// ---------------------------------------------------------------------------

Status row_mean_f32(Tensor2D<const float> input, Tensor1D<float> output,
                    const CudaStream& stream);

// ---------------------------------------------------------------------------
// Row-wise variance:  out[i] = var_j(input[i, j])
//   Optionally also returns mean.
// ---------------------------------------------------------------------------

Status row_variance_f32(Tensor2D<const float> input,
                        Tensor1D<float> variance,
                        Tensor1D<float> mean,  // may be null-data if unwanted
                        const CudaStream& stream);

// ---------------------------------------------------------------------------
// Row-wise inverse standard deviation:
//   out[i] = 1 / sqrt(var[i] + eps)
//
//   Fused: computes mean, variance, and inv-std in one pass.
//   Saves mean into `mean` and inv-std into `inv_std`.
// ---------------------------------------------------------------------------

Status row_mean_inv_std_f32(Tensor2D<const float> input,
                            Tensor1D<float> mean,
                            Tensor1D<float> inv_std,
                            float eps,
                            const CudaStream& stream);

}  // namespace kernels
}  // namespace gpt
