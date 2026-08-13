// Layer 5 — Model: GPT parameter registry implementation.
//
// Carves the flat parameter (and gradient) buffer into typed TensorView
// slices. The slice order MUST match GPTConfig::approx_param_count() so the
// buffer sized from that count is exactly consumed:
//
//   embeddings: token_embedding, position_embedding
//   per layer:  W_qkv, W_o, ln1(gamma,beta), fc1_w, fc2_w, ln2(gamma,beta)
//               [+ biases when use_bias: b_qkv, b_o, fc1_b, fc2_b]
//   final LN:   gamma, beta
//   lm_head:    aliases token_embedding when tie_embeddings, else its own slice
//
// All tensors are row-major contiguous views into the shared buffer.

#include "src/model/gpt_params.h"

namespace gpt {
namespace model {

namespace {

// A monotonic cursor over the flat buffer. Each hand-out returns a contiguous
// view of the requested shape and advances past it, so the layout is a simple
// linear pack with no gaps.
class Cursor {
 public:
  explicit Cursor(float* base) : base_(base), offset_(0) {}

  Tensor2D<float> mat(int64_t rows, int64_t cols) {
    Tensor2D<float> v = Tensor2D<float>::contiguous(base_ + offset_, {rows, cols});
    offset_ += rows * cols;
    return v;
  }

  Tensor1D<float> vec(int64_t n) {
    Tensor1D<float> v = Tensor1D<float>::contiguous(base_ + offset_, {n});
    offset_ += n;
    return v;
  }

  int64_t offset() const { return offset_; }

 private:
  float* base_;
  int64_t offset_;
};

}  // namespace

Status init_gpt_params(const GPTConfig& config,
                       float* buffer,
                       GPTParams& params) {
  if (buffer == nullptr) {
    return Status(StatusCode::kInvalidArgument, "init_gpt_params: null buffer");
  }

  const int64_t D = config.d_model;
  const int64_t V = config.vocab_size;
  const int64_t S = config.max_seq_len;
  const int64_t M = config.mlp_hidden;
  const bool bias = config.use_bias;

  Cursor c(buffer);

  // Embeddings.
  params.token_embedding = c.mat(V, D);
  params.position_embedding = c.mat(S, D);

  // Per-layer parameters.
  params.layers.resize(static_cast<size_t>(config.n_layers));
  for (auto& layer : params.layers) {
    // Attention projections: W_qkv [3D, D], W_o [D, D].
    layer.attn.W_qkv = c.mat(3 * D, D);
    layer.attn.W_o = c.mat(D, D);

    // Pre-attention LayerNorm.
    layer.ln1.gamma = c.vec(D);
    layer.ln1.beta = c.vec(D);
    layer.ln1.eps = config.layernorm_eps;

    // MLP: fc1 [M, D], fc2 [D, M].
    layer.fc1.weight = c.mat(M, D);
    layer.fc2.weight = c.mat(D, M);

    // Post-attention LayerNorm.
    layer.ln2.gamma = c.vec(D);
    layer.ln2.beta = c.vec(D);
    layer.ln2.eps = config.layernorm_eps;

    // Biases, when enabled.
    layer.attn.b_qkv = bias ? c.vec(3 * D) : Tensor1D<float>{};
    layer.attn.b_o = bias ? c.vec(D) : Tensor1D<float>{};
    layer.fc1.bias = bias ? c.vec(M) : Tensor1D<float>{};
    layer.fc2.bias = bias ? c.vec(D) : Tensor1D<float>{};
    layer.fc1.use_bias = bias;
    layer.fc2.use_bias = bias;
  }

  // Final LayerNorm.
  params.final_ln.gamma = c.vec(D);
  params.final_ln.beta = c.vec(D);
  params.final_ln.eps = config.layernorm_eps;

  // LM head: alias the token embedding when tied, else its own slice.
  params.lm_head = config.tie_embeddings ? params.token_embedding : c.mat(V, D);

  if (c.offset() > config.approx_param_count()) {
    return Status(StatusCode::kInvalidArgument,
                  "init_gpt_params: layout overran approx_param_count");
  }
  return Status::Ok();
}

Status init_gpt_grads(const GPTConfig& config,
                      float* buffer,
                      GPTGrads& grads) {
  if (buffer == nullptr) {
    return Status(StatusCode::kInvalidArgument, "init_gpt_grads: null buffer");
  }

  const int64_t D = config.d_model;
  const int64_t V = config.vocab_size;
  const int64_t S = config.max_seq_len;
  const int64_t M = config.mlp_hidden;
  const bool bias = config.use_bias;

  Cursor c(buffer);

  // Gradients mirror the parameter layout exactly (same order and shapes) so
  // grad_buffer[i] is the gradient of param_buffer[i].
  grads.d_token_embedding = c.mat(V, D);
  grads.d_position_embedding = c.mat(S, D);

  grads.layers.resize(static_cast<size_t>(config.n_layers));
  for (auto& layer : grads.layers) {
    layer.attn.dW_qkv = c.mat(3 * D, D);
    layer.attn.dW_o = c.mat(D, D);

    layer.d_ln1_gamma = c.vec(D);
    layer.d_ln1_beta = c.vec(D);

    layer.fc1.d_weight = c.mat(M, D);
    layer.fc2.d_weight = c.mat(D, M);

    layer.d_ln2_gamma = c.vec(D);
    layer.d_ln2_beta = c.vec(D);

    layer.attn.db_qkv = bias ? c.vec(3 * D) : Tensor1D<float>{};
    layer.attn.db_o = bias ? c.vec(D) : Tensor1D<float>{};
    layer.fc1.d_bias = bias ? c.vec(M) : Tensor1D<float>{};
    layer.fc2.d_bias = bias ? c.vec(D) : Tensor1D<float>{};
  }

  grads.d_final_ln_gamma = c.vec(D);
  grads.d_final_ln_beta = c.vec(D);

  grads.d_lm_head = config.tie_embeddings ? grads.d_token_embedding : c.mat(V, D);

  if (c.offset() > config.approx_param_count()) {
    return Status(StatusCode::kInvalidArgument,
                  "init_gpt_grads: layout overran approx_param_count");
  }
  return Status::Ok();
}

}  // namespace model
}  // namespace gpt
