// Layer 6 — Training runtime: trainer implementation.

#include "src/train/trainer.h"

#include <cuda_runtime.h>

#include <vector>

#include "src/model/gpt_model.h"
#include "src/ops/loss.h"

namespace gpt {
namespace train {

namespace {

float DeterministicInitValue(int64_t i) {
  uint32_t x = static_cast<uint32_t>(i) * 1664525u + 1013904223u;
  float u = static_cast<float>(x & 0x00FFFFFF) / 16777216.0f;
  return (2.0f * u - 1.0f) * 0.02f;
}

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

size_t F32Bytes(int64_t count) {
  return static_cast<size_t>(count) * sizeof(float);
}

size_t BackwardScratchBytes(const model::GPTConfig& config, int64_t B,
                            int64_t T) {
  const int64_t D = config.d_model;
  const int64_t H = config.n_heads;
  const int64_t M = config.mlp_hidden;
  const int64_t N = B * T;
  const int64_t block_scratch =
      15 * N * D + 2 * N * M + 2 * B * H * T * T;
  const int64_t model_scratch = 2 * N * D;
  const int64_t loss_scratch = 1;
  const size_t payload_bytes =
      F32Bytes(loss_scratch + model_scratch + config.n_layers * block_scratch);
  const int64_t allocation_count = 3 + config.n_layers * 17;
  return payload_bytes +
         static_cast<size_t>(allocation_count) *
             DeviceScratchArena::kDefaultAlignment;
}

}  // namespace

Status Trainer::init(const TrainerConfig& config) {
  config_ = config;

  // Create streams.
  auto cs = CudaStream::Create();
  if (!cs.ok()) return cs.status();
  compute_stream_ = cs.take();

  auto ms = CudaStream::Create();
  if (!ms.ok()) return ms.status();
  comm_stream_ = ms.take();

  // Allocate parameter buffer.
  param_count_ = config.model_config.approx_param_count();
  size_t bytes = static_cast<size_t>(param_count_) * sizeof(float);

  cudaError_t err = cudaMalloc(&param_buffer_, bytes);
  if (err != cudaSuccess) {
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));
  }
  std::vector<float> initial_params(static_cast<size_t>(param_count_));
  for (int64_t i = 0; i < param_count_; ++i) {
    initial_params[static_cast<size_t>(i)] = DeterministicInitValue(i);
  }
  err = cudaMemcpy(param_buffer_, initial_params.data(), bytes,
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));
  }

  // Allocate gradient buffer.
  err = cudaMalloc(&grad_buffer_, bytes);
  if (err != cudaSuccess) {
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));
  }
  cudaMemset(grad_buffer_, 0, bytes);

  // Initialize parameter views.
  GPT_RETURN_IF_ERROR(
      model::init_gpt_params(config.model_config, param_buffer_, params_));
  GPT_RETURN_IF_ERROR(
      model::init_gpt_grads(config.model_config, grad_buffer_, grads_));

  // Initialize optimizer.
  GPT_RETURN_IF_ERROR(optimizer_.init(param_count_, config.optimizer_config));

  // Initialize DDP.
  GPT_RETURN_IF_ERROR(ddp_.init(grad_buffer_, param_count_, config.ddp_config));

  // Initialize NCCL (if multi-GPU).
  if (config.nccl_config.world_size > 1) {
    GPT_RETURN_IF_ERROR(nccl_.init(config.nccl_config));
  }

  return Status::Ok();
}

Status Trainer::ensure_forward_state(int64_t B, int64_t T) {
  if (B <= 0 || T <= 0) {
    return Status(StatusCode::kInvalidArgument,
                  "Trainer::train_step: empty batch");
  }
  if (T > config_.model_config.max_seq_len) {
    return Status(StatusCode::kInvalidArgument,
                  "Trainer::train_step: sequence length exceeds max_seq_len");
  }
  if (state_capacity_B_ == B && state_capacity_T_ == T) {
    return Status::Ok();
  }

  release_forward_state();

  const model::GPTConfig& mc = config_.model_config;
  const int64_t D = mc.d_model;
  const int64_t H = mc.n_heads;
  const int64_t mlp = mc.mlp_hidden;
  const int64_t N = B * T;

  if (D <= 0 || H <= 0 || mlp <= 0 || mc.vocab_size <= 0) {
    return Status(StatusCode::kInvalidArgument,
                  "Trainer::ensure_forward_state: invalid model shape");
  }
  const int64_t Dh = mc.head_dim();
  if (Dh <= 0 || H * Dh != D) {
    return Status(StatusCode::kInvalidArgument,
                  "Trainer::ensure_forward_state: invalid model shape");
  }

  fwd_state_.block_inputs.clear();
  fwd_state_.blocks.clear();

  auto add = [this](int64_t count) -> Result<float*> {
    float* ptr = nullptr;
    Status status = AllocF32(count, &ptr);
    if (!status.ok()) return status;
    fwd_state_buffers_.push_back(ptr);
    return ptr;
  };

  auto embed = add(N * D);
  if (!embed.ok()) return embed.status();
  fwd_state_.embed_out = Make3D(embed.value(), B, T, D);

  fwd_state_.block_inputs.resize(static_cast<size_t>(mc.n_layers));
  fwd_state_.blocks.resize(static_cast<size_t>(mc.n_layers));
  for (int64_t layer = 0; layer < mc.n_layers; ++layer) {
    auto block_input = add(N * D);
    if (!block_input.ok()) return block_input.status();
    fwd_state_.block_inputs[layer] = Make3D(block_input.value(), B, T, D);

    model::BlockForwardState& block = fwd_state_.blocks[layer];
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
  auto logits = add(N * mc.vocab_size);
  if (!logits.ok()) return logits.status();
  fwd_state_.final_ln_out = Make3D(final_ln.value(), B, T, D);
  fwd_state_.final_ln_state.mean = Make1D(final_mean.value(), N);
  fwd_state_.final_ln_state.inv_std = Make1D(final_inv.value(), N);
  fwd_state_.logits =
      Tensor2D<float>{logits.value(), {N, mc.vocab_size}, {mc.vocab_size, 1}};

  GPT_RETURN_IF_ERROR(AllocF32(N * mc.vocab_size, &d_logits_buffer_));

  state_capacity_B_ = B;
  state_capacity_T_ = T;
  return Status::Ok();
}

Status Trainer::train_step(Tensor2D<const int32_t> input_ids,
                           Tensor2D<const int32_t> targets,
                           StepMetrics& metrics) {
  if (targets.shape[0] != input_ids.shape[0] ||
      targets.shape[1] != input_ids.shape[1]) {
    return Status(StatusCode::kInvalidArgument,
                  "Trainer::train_step: target shape mismatch");
  }
  int64_t B = input_ids.shape[0];
  int64_t T = input_ids.shape[1];
  int64_t N = B * T;
  GPT_RETURN_IF_ERROR(ensure_forward_state(B, T));
  GPT_RETURN_IF_ERROR(
      scratch_arena_.reserve_bytes(BackwardScratchBytes(config_.model_config,
                                                        B, T)));
  scratch_arena_.reset();

  // 1. Forward.
  GPT_RETURN_IF_ERROR(model::gpt_forward(
      input_ids, config_.model_config, params_, fwd_state_, compute_stream_));

  // 2. Loss.
  Tensor1D<const int32_t> targets_flat{targets.data, {B * T}, {1}};

  Result<float*> d_loss = scratch_arena_.alloc_f32(1);
  if (!d_loss.ok()) {
    return d_loss.status();
  }
  GPT_RETURN_IF_ERROR(ops::cross_entropy_forward(
      {fwd_state_.logits.data, fwd_state_.logits.shape, fwd_state_.logits.stride},
      targets_flat, d_loss.value(), compute_stream_));

  // Copy loss to host.
  GPT_RETURN_IF_ERROR(compute_stream_.synchronize());
  cudaError_t err =
      cudaMemcpy(&metrics.loss, d_loss.value(), sizeof(float),
                 cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));
  }

  // 3. Backward.
  err = cudaMemsetAsync(
      grad_buffer_, 0,
      static_cast<size_t>(param_count_) * sizeof(float),
      compute_stream_.get());
  if (err != cudaSuccess) {
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));
  }
  Tensor2D<float> d_logits{
      d_logits_buffer_, {N, config_.model_config.vocab_size},
      {config_.model_config.vocab_size, 1}};
  GPT_RETURN_IF_ERROR(ops::cross_entropy_backward(
      {fwd_state_.logits.data, fwd_state_.logits.shape, fwd_state_.logits.stride},
      targets_flat, d_logits, compute_stream_));
  Tensor2D<const float> d_logits_const{
      d_logits.data, d_logits.shape, d_logits.stride};
  GPT_RETURN_IF_ERROR(model::gpt_backward(
      d_logits_const, input_ids, config_.model_config, params_, fwd_state_,
      grads_, scratch_arena_, compute_stream_));

  // 4. DDP gradient sync.
  if (nccl_.is_initialized()) {
    GPT_RETURN_IF_ERROR(ddp_.all_reduce_grads(nccl_, comm_stream_));
    comm_stream_.synchronize();
  }

  // 5. Optimizer step.
  GPT_RETURN_IF_ERROR(
      optimizer_.step(param_buffer_, grad_buffer_, param_count_,
                      compute_stream_));
  optimizer_.advance_step();
  ++step_;
  scratch_arena_.reset();

  return Status::Ok();
}

Status Trainer::save_checkpoint(const std::string& dir,
                                int64_t dataset_cursor,
                                const std::string& config_json) {
  if (param_buffer_ == nullptr || optimizer_.first_moment() == nullptr ||
      optimizer_.second_moment() == nullptr) {
    return Status(StatusCode::kInvalidArgument,
                  "Trainer::save_checkpoint: trainer not initialized");
  }

  checkpoint::CheckpointMetadata meta;
  meta.step = step_;
  meta.param_count = param_count_;
  meta.dataset_cursor = dataset_cursor;
  meta.config_json = config_json;
  return checkpoint::save_checkpoint(dir, param_buffer_,
                                     optimizer_.first_moment(),
                                     optimizer_.second_moment(),
                                     param_count_, meta);
}

Status Trainer::load_checkpoint(const std::string& dir,
                                checkpoint::CheckpointMetadata& meta) {
  if (param_buffer_ == nullptr || optimizer_.first_moment() == nullptr ||
      optimizer_.second_moment() == nullptr) {
    return Status(StatusCode::kInvalidArgument,
                  "Trainer::load_checkpoint: trainer not initialized");
  }

  GPT_RETURN_IF_ERROR(checkpoint::load_checkpoint(
      dir, param_buffer_, optimizer_.first_moment(), optimizer_.second_moment(),
      param_count_, meta));
  if (meta.param_count != param_count_) {
    return Status(StatusCode::kInvalidArgument,
                  "Trainer::load_checkpoint: param_count mismatch");
  }
  step_ = meta.step;
  optimizer_.set_step(meta.step);
  return Status::Ok();
}

void Trainer::release_forward_state() {
  for (float* ptr : fwd_state_buffers_) {
    cudaFree(ptr);
  }
  fwd_state_buffers_.clear();
  fwd_state_ = model::GPTForwardState{};
  if (d_logits_buffer_) {
    cudaFree(d_logits_buffer_);
    d_logits_buffer_ = nullptr;
  }
  state_capacity_B_ = 0;
  state_capacity_T_ = 0;
}

void Trainer::release() {
  optimizer_.release();
  nccl_.destroy();
  scratch_arena_.release();
  release_forward_state();
  if (param_buffer_) { cudaFree(param_buffer_); param_buffer_ = nullptr; }
  if (grad_buffer_) { cudaFree(grad_buffer_); grad_buffer_ = nullptr; }
}

Trainer::~Trainer() { release(); }

}  // namespace train
}  // namespace gpt
