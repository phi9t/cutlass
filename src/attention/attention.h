#pragma once

// Layer 4 — Attention subsystem.
//
// Causal multi-head self-attention, decomposed into explicit stages:
//
//   1. QKV projection:     QKV = X @ W_qkv + b_qkv
//   2. Split Q, K, V:      [B, T, D] → [B, H, T, Dh] each
//   3. Score computation:   S = Q @ K^T
//   4. Scale:               S *= 1/sqrt(Dh)
//   5. Causal mask:         S[i,j] = -inf where j > i
//   6. Softmax:             P = softmax(S)     (row-wise, masked)
//   7. Value aggregation:   C = P @ V
//   8. Merge heads:         [B, H, T, Dh] → [B, T, D]
//   9. Output projection:   O = C @ W_o + b_o
//
// V1: explicit intermediate tensors, no fusion.
// V2: replace steps 3-7 with a fused attention kernel.

#include <cmath>
#include <cstdint>
#include <vector>

#include "src/core/allocator.h"
#include "src/core/status.h"
#include "src/core/stream.h"
#include "src/tensor/tensor_view.h"

namespace gpt {
namespace attention {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct AttentionConfig {
  int64_t d_model;
  int64_t n_heads;
  int64_t head_dim;   // = d_model / n_heads
  bool use_bias;
  bool causal;

  float scale() const { return 1.0f / std::sqrt(static_cast<float>(head_dim)); }
};

// ---------------------------------------------------------------------------
// Parameters (weights)
// ---------------------------------------------------------------------------

struct AttentionParams {
  Tensor2D<float> W_qkv;    // [3*D, D]
  Tensor1D<float> b_qkv;    // [3*D], nullable
  Tensor2D<float> W_o;      // [D, D]
  Tensor1D<float> b_o;      // [D], nullable
};

// ---------------------------------------------------------------------------
// Gradients (same shapes as params)
// ---------------------------------------------------------------------------

struct AttentionGrads {
  Tensor2D<float> dW_qkv;
  Tensor1D<float> db_qkv;
  Tensor2D<float> dW_o;
  Tensor1D<float> db_o;
};

// ---------------------------------------------------------------------------
// Forward state — intermediates saved for backward.
// ---------------------------------------------------------------------------

struct AttentionForwardState {
  // Pre-allocated workspace tensors (caller provides).
  Tensor3D<float> qkv;         // [B, T, 3*D]
  Tensor4D<float> Q;            // [B, H, T, Dh]
  Tensor4D<float> K;            // [B, H, T, Dh]
  Tensor4D<float> V;            // [B, H, T, Dh]
  Tensor4D<float> scores;       // [B, H, T, T]
  Tensor4D<float> probs;        // [B, H, T, T]
  Tensor4D<float> context;      // [B, H, T, Dh]
  Tensor3D<float> context_merged; // [B, T, D]
};

class AttentionWorkspace {
 public:
  AttentionWorkspace() = default;

  Status ensure(const AttentionConfig& config, int64_t B, int64_t T);

  [[nodiscard]] AttentionForwardState& state() { return state_; }
  [[nodiscard]] const AttentionForwardState& state() const { return state_; }

  void release();
  ~AttentionWorkspace();

  AttentionWorkspace(const AttentionWorkspace&) = delete;
  AttentionWorkspace& operator=(const AttentionWorkspace&) = delete;
  AttentionWorkspace(AttentionWorkspace&& other) noexcept;
  AttentionWorkspace& operator=(AttentionWorkspace&& other) noexcept;

 private:
  AttentionForwardState state_;
  std::vector<float*> buffers_;
  int64_t capacity_B_ = 0;
  int64_t capacity_T_ = 0;
  int64_t capacity_d_model_ = 0;
  int64_t capacity_n_heads_ = 0;
  int64_t capacity_head_dim_ = 0;
};

// ---------------------------------------------------------------------------
// Forward
//   X:      [B, T, D]
//   output: [B, T, D]
// ---------------------------------------------------------------------------

Status attention_forward(Tensor3D<const float> X,
                         const AttentionConfig& config,
                         const AttentionParams& params,
                         Tensor3D<float> output,
                         AttentionWorkspace& workspace,
                         const CudaStream& stream);

// ---------------------------------------------------------------------------
// Backward
//   dO:     [B, T, D]   gradient from above
//   dX:     [B, T, D]   gradient to input
// ---------------------------------------------------------------------------

Status attention_backward(Tensor3D<const float> dO,
                          Tensor3D<const float> X,
                          const AttentionConfig& config,
                          const AttentionParams& params,
                          const AttentionForwardState& state,
                          Tensor3D<float> dX,
                          AttentionGrads& grads,
                          DeviceScratchArena& scratch,
                          const CudaStream& stream);

}  // namespace attention
}  // namespace gpt
