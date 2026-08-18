// Layer 5 — Model: persistent forward workspace implementation.

#include "src/model/forward_workspace.h"

#include <cuda_runtime.h>

namespace gpt {
namespace model {

namespace {

Status AllocF32(int64_t count, float** ptr) {
  cudaError_t err =
      cudaMalloc(ptr, static_cast<size_t>(count) * sizeof(float));
  if (err != cudaSuccess) {
    *ptr = nullptr;
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));
  }
  return Status::Ok();
}

Tensor3D<float> Make3D(float* ptr, int64_t B, int64_t T, int64_t D) {
  return Tensor3D<float>{ptr, {B, T, D}, {T * D, D, 1}};
}

Tensor1D<float> Make1D(float* ptr, int64_t N) {
  return Tensor1D<float>{ptr, {N}, {1}};
}

}  // namespace

Status ForwardWorkspace::ensure(const GPTConfig& config, int64_t B,
                                int64_t T) {
  if (B <= 0 || T <= 0) {
    return Status(StatusCode::kInvalidArgument,
                  "ForwardWorkspace::ensure: empty batch");
  }
  if (T > config.max_seq_len) {
    return Status(StatusCode::kInvalidArgument,
                  "ForwardWorkspace::ensure: sequence length exceeds "
                  "max_seq_len");
  }
  const int64_t D = config.d_model;
  const int64_t mlp = config.mlp_hidden;
  const int64_t N = B * T;

  const int64_t H = config.n_heads;
  if (D <= 0 || H <= 0 || mlp <= 0 || config.vocab_size <= 0 ||
      config.n_layers < 0) {
    return Status(StatusCode::kInvalidArgument,
                  "ForwardWorkspace::ensure: invalid model shape");
  }
  const int64_t Dh = config.head_dim();
  if (Dh <= 0 || H * Dh != D) {
    return Status(StatusCode::kInvalidArgument,
                  "ForwardWorkspace::ensure: invalid model shape");
  }
  if (capacity_B_ == B && capacity_T_ == T &&
      capacity_d_model_ == config.d_model &&
      capacity_n_layers_ == config.n_layers &&
      capacity_n_heads_ == config.n_heads &&
      capacity_mlp_hidden_ == config.mlp_hidden &&
      capacity_vocab_size_ == config.vocab_size) {
    return Status::Ok();
  }

  release();

  state_.block_inputs.clear();
  state_.blocks.clear();
  attention_workspaces_.clear();

  auto add = [this](int64_t count) -> Result<float*> {
    float* ptr = nullptr;
    Status status = AllocF32(count, &ptr);
    if (!status.ok()) return status;
    buffers_.push_back(ptr);
    return ptr;
  };

  auto embed = add(N * D);
  if (!embed.ok()) return embed.status();
  state_.embed_out = Make3D(embed.value(), B, T, D);

  state_.block_inputs.resize(static_cast<size_t>(config.n_layers));
  state_.blocks.resize(static_cast<size_t>(config.n_layers));
  attention_workspaces_.resize(static_cast<size_t>(config.n_layers));
  for (int64_t layer = 0; layer < config.n_layers; ++layer) {
    auto block_input = add(N * D);
    if (!block_input.ok()) return block_input.status();
    state_.block_inputs[layer] = Make3D(block_input.value(), B, T, D);

    BlockForwardState& block = state_.blocks[layer];
    auto ln1_out = add(N * D);
    if (!ln1_out.ok()) return ln1_out.status();
    auto ln2_out = add(N * D);
    if (!ln2_out.ok()) return ln2_out.status();
    auto ln1_mean = add(N);
    if (!ln1_mean.ok()) return ln1_mean.status();
    auto ln1_inv = add(N);
    if (!ln1_inv.ok()) return ln1_inv.status();
    auto ln2_mean = add(N);
    if (!ln2_mean.ok()) return ln2_mean.status();
    auto ln2_inv = add(N);
    if (!ln2_inv.ok()) return ln2_inv.status();
    auto attn_out = add(N * D);
    if (!attn_out.ok()) return attn_out.status();
    auto fc1_out = add(N * mlp);
    if (!fc1_out.ok()) return fc1_out.status();
    auto gelu_out = add(N * mlp);
    if (!gelu_out.ok()) return gelu_out.status();
    auto fc2_out = add(N * D);
    if (!fc2_out.ok()) return fc2_out.status();

    attention::AttentionConfig attn_config;
    attn_config.d_model = config.d_model;
    attn_config.n_heads = config.n_heads;
    attn_config.head_dim = config.head_dim();
    attn_config.use_bias = config.use_bias;
    attn_config.causal = true;
    GPT_RETURN_IF_ERROR(
        attention_workspaces_[static_cast<size_t>(layer)].ensure(attn_config,
                                                                 B, T));

    block.ln1_out = Make3D(ln1_out.value(), B, T, D);
    block.ln2_out = Make3D(ln2_out.value(), B, T, D);
    block.ln1_state.mean = Make1D(ln1_mean.value(), N);
    block.ln1_state.inv_std = Make1D(ln1_inv.value(), N);
    block.ln2_state.mean = Make1D(ln2_mean.value(), N);
    block.ln2_state.inv_std = Make1D(ln2_inv.value(), N);
    block.attn_out = Make3D(attn_out.value(), B, T, D);
    block.fc1_out = Make3D(fc1_out.value(), B, T, mlp);
    block.gelu_out = Make3D(gelu_out.value(), B, T, mlp);
    block.fc2_out = Make3D(fc2_out.value(), B, T, D);
    block.attn_workspace = &attention_workspaces_[static_cast<size_t>(layer)];
    block.attn_state =
        attention_workspaces_[static_cast<size_t>(layer)].state();
  }

  auto final_ln = add(N * D);
  if (!final_ln.ok()) return final_ln.status();
  auto final_mean = add(N);
  if (!final_mean.ok()) return final_mean.status();
  auto final_inv = add(N);
  if (!final_inv.ok()) return final_inv.status();
  auto logits = add(N * config.vocab_size);
  if (!logits.ok()) return logits.status();
  state_.final_ln_out = Make3D(final_ln.value(), B, T, D);
  state_.final_ln_state.mean = Make1D(final_mean.value(), N);
  state_.final_ln_state.inv_std = Make1D(final_inv.value(), N);
  state_.logits =
      Tensor2D<float>{logits.value(), {N, config.vocab_size},
                      {config.vocab_size, 1}};

  capacity_B_ = B;
  capacity_T_ = T;
  capacity_d_model_ = config.d_model;
  capacity_n_layers_ = config.n_layers;
  capacity_n_heads_ = config.n_heads;
  capacity_mlp_hidden_ = config.mlp_hidden;
  capacity_vocab_size_ = config.vocab_size;
  return Status::Ok();
}

void ForwardWorkspace::release() {
  for (float* ptr : buffers_) {
    cudaFree(ptr);
  }
  buffers_.clear();
  attention_workspaces_.clear();
  state_ = GPTForwardState{};
  capacity_B_ = 0;
  capacity_T_ = 0;
  capacity_d_model_ = 0;
  capacity_n_layers_ = 0;
  capacity_n_heads_ = 0;
  capacity_mlp_hidden_ = 0;
  capacity_vocab_size_ = 0;
}

ForwardWorkspace::~ForwardWorkspace() { release(); }

}  // namespace model
}  // namespace gpt
