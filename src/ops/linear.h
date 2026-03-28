#pragma once

// Layer 3 — Composite neural ops: linear projection.
//
// Forward:   Y = X @ W^T + b        (optional bias)
// Backward:  dX = dY @ W
//            dW = dY^T @ X
//            db = row_sum(dY)        (if bias)
//
// X:  [B*T, D_in]
// W:  [D_out, D_in]
// b:  [D_out]
// Y:  [B*T, D_out]

#include "src/core/status.h"
#include "src/core/stream.h"
#include "src/tensor/tensor_view.h"

namespace gpt {
namespace ops {

struct LinearParams {
  Tensor2D<float> weight;    // [D_out, D_in]
  Tensor1D<float> bias;      // [D_out], may have null data if unused
  bool use_bias = true;
};

struct LinearGrads {
  Tensor2D<float> d_weight;  // [D_out, D_in]
  Tensor1D<float> d_bias;    // [D_out]
};

// Forward: Y = X @ W^T + b
Status linear_forward(Tensor2D<const float> X,
                      const LinearParams& params,
                      Tensor2D<float> Y,
                      const CudaStream& stream);

// Backward: given dY, compute dX, dW, db.
// X must be the saved input from forward.
Status linear_backward(Tensor2D<const float> X,
                       Tensor2D<const float> dY,
                       const LinearParams& params,
                       Tensor2D<float> dX,
                       LinearGrads& grads,
                       const CudaStream& stream);

}  // namespace ops
}  // namespace gpt
