#pragma once

// Layer 3 — Composite neural ops: GELU activation.
//
// Uses the tanh approximation:
//   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

#include "src/core/status.h"
#include "src/core/stream.h"
#include "src/tensor/tensor_view.h"

namespace gpt {
namespace ops {

// Forward: y = GELU(x)
Status gelu_forward(Tensor1D<const float> x, Tensor1D<float> y,
                    const CudaStream& stream);

// Backward: dx = dy * GELU'(x)
// Requires saved input x from forward.
Status gelu_backward(Tensor1D<const float> dy, Tensor1D<const float> x,
                     Tensor1D<float> dx, const CudaStream& stream);

}  // namespace ops
}  // namespace gpt
