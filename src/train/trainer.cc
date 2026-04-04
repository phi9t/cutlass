// Layer 6 — Training runtime: trainer implementation.

#include "src/train/trainer.h"

#include "src/model/gpt_model.h"
#include "src/ops/loss.h"

namespace gpt {
namespace train {

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
  cudaMemset(param_buffer_, 0, bytes);

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

Status Trainer::train_step(Tensor2D<const int32_t> input_ids,
                           Tensor2D<const int32_t> targets,
                           StepMetrics& metrics) {
  // 1. Forward.
  GPT_RETURN_IF_ERROR(model::gpt_forward(
      input_ids, config_.model_config, params_, fwd_state_, compute_stream_));

  // 2. Loss.
  int64_t B = input_ids.shape[0];
  int64_t T = input_ids.shape[1];
  Tensor1D<const int32_t> targets_flat{targets.data, {B * T}, {1}};

  float* d_loss = nullptr;
  cudaMalloc(&d_loss, sizeof(float));
  GPT_RETURN_IF_ERROR(ops::cross_entropy_forward(
      {fwd_state_.logits.data, fwd_state_.logits.shape, fwd_state_.logits.stride},
      targets_flat, d_loss, compute_stream_));

  // Copy loss to host.
  compute_stream_.synchronize();
  cudaMemcpy(&metrics.loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_loss);

  // 3. Backward.
  // Zero grads.
  cudaMemsetAsync(grad_buffer_, 0,
                  static_cast<size_t>(param_count_) * sizeof(float),
                  compute_stream_.get());

  // Allocate d_logits buffer on first use.
  int64_t NT = B * T;
  int64_t V = config_.model_config.vocab_size;
  if (!d_logits_buffer_) {
    cudaError_t alloc_err = cudaMalloc(
        &d_logits_buffer_, static_cast<size_t>(NT * V) * sizeof(float));
    if (alloc_err != cudaSuccess) {
      return Status(StatusCode::kCudaError, cudaGetErrorString(alloc_err));
    }
  }

  // Compute d_logits from cross-entropy backward.
  {
    Tensor2D<const float> logits_view{fwd_state_.logits.data,
                                       fwd_state_.logits.shape,
                                       fwd_state_.logits.stride};
    Tensor2D<float> d_logits_view{d_logits_buffer_, {NT, V}, {V, 1}};
    GPT_RETURN_IF_ERROR(ops::cross_entropy_backward(
        logits_view, targets_flat, d_logits_view, compute_stream_));
  }

  // Model backward.
  {
    Tensor2D<const float> d_logits_const{d_logits_buffer_, {NT, V}, {V, 1}};
    GPT_RETURN_IF_ERROR(model::gpt_backward(
        d_logits_const, input_ids, config_.model_config, params_,
        fwd_state_, grads_, compute_stream_));
  }

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

  return Status::Ok();
}

void Trainer::release() {
  optimizer_.release();
  nccl_.destroy();
  if (param_buffer_) { cudaFree(param_buffer_); param_buffer_ = nullptr; }
  if (grad_buffer_) { cudaFree(grad_buffer_); grad_buffer_ = nullptr; }
  if (d_logits_buffer_) { cudaFree(d_logits_buffer_); d_logits_buffer_ = nullptr; }
}

Trainer::~Trainer() { release(); }

}  // namespace train
}  // namespace gpt
