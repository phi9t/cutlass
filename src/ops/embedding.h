#pragma once

// Layer 3 — Composite neural ops: embedding lookup.
//
// Forward:  out[i] = table[indices[i]]   (gather)
// Backward: scatter-add dY into grad table.

#include "src/core/status.h"
#include "src/core/stream.h"
#include "src/tensor/tensor_view.h"

namespace gpt {
namespace ops {

// Forward: gather rows from embedding table.
//   table:   [vocab_size, d_model]
//   indices: [n]
//   output:  [n, d_model]
Status embedding_forward(Tensor2D<const float> table,
                         Tensor1D<const int32_t> indices,
                         Tensor2D<float> output,
                         const CudaStream& stream);

// Backward: scatter-add gradients into the embedding table gradient.
//   dY:      [n, d_model]
//   indices: [n]
//   d_table: [vocab_size, d_model]  (accumulated, not zeroed here)
Status embedding_backward(Tensor2D<const float> dY,
                          Tensor1D<const int32_t> indices,
                          Tensor2D<float> d_table,
                          const CudaStream& stream);

}  // namespace ops
}  // namespace gpt
