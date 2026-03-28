#pragma once

// Layer 3 — Composite neural ops: cross-entropy loss.
//
// Stable formulation using log-sum-exp:
//   loss_i = log_sum_exp(logits_i) - logits_i[target_i]
//   loss   = mean(loss_i)
//
// Backward:
//   dlogits_i[j] = (softmax(logits_i)[j] - 1{j == target_i}) / N

#include "src/core/status.h"
#include "src/core/stream.h"
#include "src/tensor/tensor_view.h"

namespace gpt {
namespace ops {

// Forward: cross-entropy loss over [N, V] logits with [N] targets.
//   logits:  [N, vocab_size]
//   targets: [N]
//   loss:    scalar (single float on device)
Status cross_entropy_forward(Tensor2D<const float> logits,
                             Tensor1D<const int32_t> targets,
                             float* d_loss,  // device pointer to scalar
                             const CudaStream& stream);

// Backward: compute gradient of loss w.r.t. logits.
//   logits:   [N, vocab_size]
//   targets:  [N]
//   d_logits: [N, vocab_size]
Status cross_entropy_backward(Tensor2D<const float> logits,
                              Tensor1D<const int32_t> targets,
                              Tensor2D<float> d_logits,
                              const CudaStream& stream);

}  // namespace ops
}  // namespace gpt
