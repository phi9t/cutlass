// Layer 4 — Attention subsystem: backward-specific CUDA kernels.
//
// Placeholder for attention backward kernels that are not covered
// by the generic composite ops (softmax backward, batched GEMM, etc.).

#include <cuda_runtime.h>
#include <cstdint>

namespace gpt {
namespace attention {

// TODO(m5): Implement any attention-backward-specific kernels here.
//
// Most backward stages reuse existing primitives:
//   - batched_gemm for dQ/dK/dV matmuls
//   - masked_softmax_backward for dS
//   - merge_heads / split_heads for gradient layout transforms
//   - linear_backward for projection gradients
//
// Attention-specific kernels that may live here:
//   - Fused softmax backward with causal mask
//   - Fused dQ/dK computation (later optimization)

}  // namespace attention
}  // namespace gpt
