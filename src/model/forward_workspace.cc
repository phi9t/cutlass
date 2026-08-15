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

Tensor4D<float> Make4D(float* ptr, int64_t B, int64_t H, int64_t T,
                       int64_t Dh) {
  return Tensor4D<float>{ptr, {B, H, T, Dh},
                         {H * T * Dh, T * Dh, Dh, 1}};
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
  if (capacity_B_ == B && capacity_T_ == T) {
    return Status::Ok();
  }

  release();

  const int64_t D = config.d_model;
  const int64_t H = config.n_heads;
  const int64_t mlp = config.mlp_hidden;
  const int64_t N = B * T;

  if (D <= 0 || H <= 0 || mlp <= 0 || config.vocab_size <= 0) {
    return Status(StatusCode::kInvalidArgument,
                  "ForwardWorkspace::ensure: invalid model shape");
  }
  const int64_t Dh = config.head_dim();
  if (Dh <= 0 || H * Dh != D) {
    return Status(StatusCode::kInvalidArgument,
                  "ForwardWorkspace::ensure: invalid model shape");
  }

  state_.block_inputs.clear();
  state_.blocks.clear();

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
    auto qkv = add(N * 3 * D);
    if (!qkv.ok()) return qkv.status();
    auto Q = add(B * H * T * Dh);
    if (!Q.ok()) return Q.status();
    auto K = add(B * H * T * Dh);
    if (!K.ok()) return K.status();
    auto V = add(B * H * T * Dh);
    if (!V.ok()) return V.status();
    auto scores = add(B * H * T * T);
    if (!scores.ok()) return scores.status();
    auto probs = add(B * H * T * T);
    if (!probs.ok()) return probs.status();
    auto ctx = add(B * H * T * Dh);
    if (!ctx.ok()) return ctx.status();
    auto merged = add(N * D);
    if (!merged.ok()) return merged.status();

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
    block.attn_state.qkv = Make3D(qkv.value(), B, T, 3 * D);
    block.attn_state.Q = Make4D(Q.value(), B, H, T, Dh);
    block.attn_state.K = Make4D(K.value(), B, H, T, Dh);
    block.attn_state.V = Make4D(V.value(), B, H, T, Dh);
    block.attn_state.scores =
        Tensor4D<float>{scores.value(), {B, H, T, T},
                        {H * T * T, T * T, T, 1}};
    block.attn_state.probs =
        Tensor4D<float>{probs.value(), {B, H, T, T},
                        {H * T * T, T * T, T, 1}};
    block.attn_state.context = Make4D(ctx.value(), B, H, T, Dh);
    block.attn_state.context_merged = Make3D(merged.value(), B, T, D);
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
  return Status::Ok();
}

void ForwardWorkspace::release() {
  for (float* ptr : buffers_) {
    cudaFree(ptr);
  }
  buffers_.clear();
  state_ = GPTForwardState{};
  capacity_B_ = 0;
  capacity_T_ = 0;
}

ForwardWorkspace::~ForwardWorkspace() { release(); }

}  // namespace model
}  // namespace gpt
