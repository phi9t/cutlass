#pragma once

// Layer 3 — Composite neural ops: Layer Normalization.
//
// Forward:
//   mean = row_mean(x)
//   var  = row_variance(x)
//   x_hat = (x - mean) / sqrt(var + eps)
//   y = gamma * x_hat + beta
//
// Backward requires saved mean and inv_std from forward.

#include "src/core/status.h"
#include "src/core/stream.h"
#include "src/tensor/tensor_view.h"

namespace gpt {
namespace ops {

struct LayerNormParams {
  Tensor1D<float> gamma;  // [D] scale
  Tensor1D<float> beta;   // [D] shift
  float eps = 1e-5f;
};

// Saved statistics from forward, needed for backward.
struct LayerNormState {
  Tensor1D<float> mean;     // [rows]
  Tensor1D<float> inv_std;  // [rows]
};

// Forward: y = gamma * ((x - mean) * inv_std) + beta
//   x:      [rows, D]
//   y:      [rows, D]
Status layernorm_forward(Tensor2D<const float> x,
                         const LayerNormParams& params,
                         Tensor2D<float> y,
                         LayerNormState& state,
                         const CudaStream& stream);

// Backward: compute dx, dgamma, dbeta.
//   dy:     [rows, D]
//   x:      [rows, D]  (saved input)
//   dx:     [rows, D]
//   dgamma: [D]
//   dbeta:  [D]
Status layernorm_backward(Tensor2D<const float> dy,
                          Tensor2D<const float> x,
                          const LayerNormParams& params,
                          const LayerNormState& state,
                          Tensor2D<float> dx,
                          Tensor1D<float> dgamma,
                          Tensor1D<float> dbeta,
                          const CudaStream& stream);

}  // namespace ops
}  // namespace gpt
