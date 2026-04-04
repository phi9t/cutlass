#pragma once

// Layer 6 — Training runtime: step orchestration.
//
// Manages the training loop:
//   1. Forward pass
//   2. Loss computation
//   3. Backward pass
//   4. Gradient all-reduce (DDP)
//   5. Optimizer step
//   6. Metrics logging

#include <cstdint>

#include "src/core/status.h"
#include "src/core/stream.h"
#include "src/dist/ddp.h"
#include "src/dist/nccl_context.h"
#include "src/model/gpt_config.h"
#include "src/model/gpt_model.h"
#include "src/model/gpt_params.h"
#include "src/train/optimizer.h"

namespace gpt {
namespace train {

struct TrainerConfig {
  model::GPTConfig model_config;
  AdamWConfig optimizer_config;
  dist::DDPConfig ddp_config;
  dist::NcclConfig nccl_config;
  int64_t max_steps = 1000;
  int64_t log_interval = 10;
};

struct StepMetrics {
  float loss = 0.0f;
  float step_time_ms = 0.0f;
  float tokens_per_sec = 0.0f;
  float grad_norm = 0.0f;
};

class Trainer {
 public:
  Trainer() = default;

  // Initialize all components: model, optimizer, DDP, streams.
  Status init(const TrainerConfig& config);

  // Run a single training step on a batch.
  //   input_ids: [B, T]
  //   targets:   [B, T]
  Status train_step(Tensor2D<const int32_t> input_ids,
                    Tensor2D<const int32_t> targets,
                    StepMetrics& metrics);

  [[nodiscard]] int64_t current_step() const { return step_; }

  void release();
  ~Trainer();

  Trainer(const Trainer&) = delete;
  Trainer& operator=(const Trainer&) = delete;

 private:
  TrainerConfig config_;
  model::GPTParams params_;
  model::GPTGrads grads_;
  model::GPTForwardState fwd_state_;
  AdamW optimizer_;
  dist::NcclContext nccl_;
  dist::DDP ddp_;

  CudaStream compute_stream_;
  CudaStream comm_stream_;

  float* param_buffer_ = nullptr;
  float* grad_buffer_ = nullptr;
  float* d_logits_buffer_ = nullptr;
  int64_t param_count_ = 0;
  int64_t step_ = 0;
};

}  // namespace train
}  // namespace gpt
