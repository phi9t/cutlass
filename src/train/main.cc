// Layer 6 — Training runtime: main entry point.
//
// Usage: ./gpt_trainer [--config path_to_config]
//
// This is the top-level binary that:
//   1. Parses config
//   2. Initializes the trainer
//   3. Loads data
//   4. Runs the training loop
//   5. Saves checkpoints

#include <cstdio>
#include <cstdlib>

#include "src/core/device.h"
#include "src/train/trainer.h"

int main(int argc, char** argv) {
  // Minimal rank detection for single-node multi-GPU.
  // In production, use MPI or environment variables.
  int rank = 0;
  int world_size = 1;

  const char* rank_env = std::getenv("RANK");
  const char* world_env = std::getenv("WORLD_SIZE");
  if (rank_env) rank = std::atoi(rank_env);
  if (world_env) world_size = std::atoi(world_env);

  // Set CUDA device.
  gpt::DeviceGuard guard(rank);

  // Configure.
  gpt::train::TrainerConfig config;
  config.model_config.n_layers = 12;
  config.model_config.n_heads = 12;
  config.model_config.d_model = 768;
  config.model_config.mlp_hidden = 3072;
  config.model_config.vocab_size = 50257;
  config.model_config.max_seq_len = 1024;

  config.nccl_config.rank = rank;
  config.nccl_config.world_size = world_size;
  config.nccl_config.local_gpu_id = rank;

  config.max_steps = 1000;
  config.log_interval = 10;

  // Initialize trainer.
  gpt::train::Trainer trainer;
  auto status = trainer.init(config);
  if (!status.ok()) {
    std::fprintf(stderr, "[rank %d] Trainer init failed: %s\n",
                 rank, status.message().c_str());
    return 1;
  }

  std::printf("[rank %d/%d] Trainer initialized. Starting training...\n",
              rank, world_size);

  // TODO: Load data via //src/data:data and run training loop.
  // For now, this is a skeleton entry point.

  return 0;
}
