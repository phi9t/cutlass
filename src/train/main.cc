// Layer 6 — Training runtime: main entry point.
//
// Usage:
//   GPT_TOKENS_PATH=/path/to/int32_tokens.bin ./gpt_trainer
//
// Optional environment overrides:
//   GPT_MAX_STEPS, GPT_LOG_INTERVAL, GPT_BATCH_SIZE, GPT_SEQ_LEN,
//   GPT_N_LAYERS, GPT_N_HEADS, GPT_D_MODEL, GPT_MLP_HIDDEN, GPT_VOCAB_SIZE,
//   GPT_LR
//
// This is the top-level binary that:
//   1. Parses config
//   2. Initializes the trainer
//   3. Loads data
//   4. Runs the training loop

#include <cstdio>
#include <cstdlib>

#include "src/core/device.h"
#include "src/data/batcher.h"
#include "src/data/token_dataset.h"
#include "src/train/trainer.h"

namespace {

int64_t EnvInt64(char const* name, int64_t fallback) {
  char const* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  return std::atoll(value);
}

float EnvFloat(char const* name, float fallback) {
  char const* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  return std::atof(value);
}

char const* TokenPath(int argc, char** argv) {
  if (argc > 1 && argv[1] != nullptr && argv[1][0] != '\0') {
    return argv[1];
  }
  return std::getenv("GPT_TOKENS_PATH");
}

void PrintStatus(char const* context, gpt::Status const& status, int rank) {
  std::fprintf(stderr, "[rank %d] %s failed: %s\n", rank, context,
               status.message().c_str());
}

}  // namespace

int main(int argc, char** argv) {
  // Minimal rank detection for single-node multi-GPU.
  // In production, use MPI or environment variables.
  int rank = 0;
  int world_size = 1;

  char const* rank_env = std::getenv("RANK");
  char const* world_env = std::getenv("WORLD_SIZE");
  if (rank_env) {
    rank = std::atoi(rank_env);
  }
  if (world_env) {
    world_size = std::atoi(world_env);
  }

  // Set CUDA device.
  gpt::DeviceGuard guard(rank);

  char const* token_path = TokenPath(argc, argv);
  if (token_path == nullptr || token_path[0] == '\0') {
    std::fprintf(stderr,
                 "Usage: GPT_TOKENS_PATH=/path/to/int32_tokens.bin "
                 "%s [token_file]\n",
                 argv[0]);
    return 2;
  }

  // Configure.
  gpt::train::TrainerConfig config;
  config.model_config.n_layers = EnvInt64("GPT_N_LAYERS", 12);
  config.model_config.n_heads = EnvInt64("GPT_N_HEADS", 12);
  config.model_config.d_model = EnvInt64("GPT_D_MODEL", 768);
  config.model_config.mlp_hidden = EnvInt64("GPT_MLP_HIDDEN", 3072);
  config.model_config.vocab_size = EnvInt64("GPT_VOCAB_SIZE", 50257);
  config.model_config.max_seq_len = EnvInt64("GPT_SEQ_LEN", 1024);
  config.optimizer_config.lr = EnvFloat("GPT_LR", config.optimizer_config.lr);

  config.nccl_config.rank = rank;
  config.nccl_config.world_size = world_size;
  config.nccl_config.local_gpu_id = rank;

  config.max_steps = EnvInt64("GPT_MAX_STEPS", 1000);
  config.log_interval = EnvInt64("GPT_LOG_INTERVAL", 10);

  int64_t const batch_size = EnvInt64("GPT_BATCH_SIZE", 8);
  int64_t const seq_len = config.model_config.max_seq_len;

  gpt::data::DatasetConfig dataset_config;
  dataset_config.path = token_path;
  dataset_config.seq_len = seq_len;
  dataset_config.batch_size = batch_size;
  dataset_config.rank = rank;
  dataset_config.world_size = world_size;

  gpt::data::TokenDataset dataset;
  auto status = dataset.init(dataset_config);
  if (!status.ok()) {
    PrintStatus("Dataset init", status, rank);
    return 1;
  }

  gpt::data::Batcher batcher;
  status = batcher.init(batch_size, seq_len);
  if (!status.ok()) {
    PrintStatus("Batcher init", status, rank);
    return 1;
  }

  auto h2d_stream_result = gpt::CudaStream::Create();
  if (!h2d_stream_result.ok()) {
    PrintStatus("H2D stream creation", h2d_stream_result.status(), rank);
    return 1;
  }
  gpt::CudaStream h2d_stream = h2d_stream_result.take();

  // Initialize trainer.
  gpt::train::Trainer trainer;
  status = trainer.init(config);
  if (!status.ok()) {
    PrintStatus("Trainer init", status, rank);
    return 1;
  }

  std::printf("[rank %d/%d] Training %ld steps from %s "
              "(batch=%ld, seq=%ld, samples=%ld)\n",
              rank, world_size, config.max_steps, token_path, batch_size,
              seq_len, dataset.num_samples());

  for (int64_t step = 0; step < config.max_steps; ++step) {
    status = dataset.next_batch(batcher.host_input_ids(),
                                batcher.host_targets());
    if (!status.ok()) {
      dataset.reset();
      status = dataset.next_batch(batcher.host_input_ids(),
                                  batcher.host_targets());
    }
    if (!status.ok()) {
      PrintStatus("Dataset next_batch", status, rank);
      return 1;
    }

    status = batcher.stage_to_device(h2d_stream);
    if (!status.ok()) {
      PrintStatus("Batcher stage_to_device", status, rank);
      return 1;
    }
    status = h2d_stream.synchronize();
    if (!status.ok()) {
      PrintStatus("H2D synchronize", status, rank);
      return 1;
    }

    gpt::Tensor2D<const int32_t> input_ids{
        batcher.device_input_ids(), {batch_size, seq_len}, {seq_len, 1}};
    gpt::Tensor2D<const int32_t> targets{
        batcher.device_targets(), {batch_size, seq_len}, {seq_len, 1}};
    gpt::train::StepMetrics metrics;
    status = trainer.train_step(input_ids, targets, metrics);
    if (!status.ok()) {
      PrintStatus("Trainer train_step", status, rank);
      return 1;
    }

    if (rank == 0 &&
        (step == 0 || (step + 1) % config.log_interval == 0 ||
         step + 1 == config.max_steps)) {
      std::printf("[rank %d] step %ld/%ld loss=%.6f cursor=%ld\n",
                  rank, step + 1, config.max_steps, metrics.loss,
                  dataset.cursor());
    }
  }

  return 0;
}
