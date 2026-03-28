#pragma once

// Layer 3 — Composite neural ops: masked row-wise softmax.
//
// Forward:
//   1. row_max over valid (unmasked) positions
//   2. subtract max
//   3. exp on valid positions
//   4. row_sum of exp values
//   5. divide by sum
//   6. masked positions stay exactly 0
//
// Backward:
//   dX_valid = P * (dP - sum(P * dP))  (row-wise)
//   dX_masked = 0

#include "src/core/status.h"
#include "src/core/stream.h"
#include "src/tensor/tensor_view.h"

namespace gpt {
namespace ops {

// Forward: row-wise masked softmax.
//   input:  [rows, cols]
//   mask:   [rows, cols]  int8, nonzero = masked (will be -inf)
//           Pass null data for no mask.
//   output: [rows, cols]  probability distribution on valid positions.
Status masked_softmax_forward(Tensor2D<const float> input,
                              Tensor2D<const int8_t> mask,  // nullable
                              Tensor2D<float> output,
                              const CudaStream& stream);

// Backward: given dP (grad of probs) and P (saved probs), compute dX.
//   dP:     [rows, cols]
//   P:      [rows, cols]  (saved output from forward)
//   dX:     [rows, cols]
Status masked_softmax_backward(Tensor2D<const float> dP,
                               Tensor2D<const float> P,
                               Tensor2D<float> dX,
                               const CudaStream& stream);

}  // namespace ops
}  // namespace gpt
