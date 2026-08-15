// Tier B — GPU unit tests for checkpointing.

#include "gtest/gtest.h"
#include "src/checkpoint/checkpoint.h"

#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace gpt;
using namespace gpt::checkpoint;

class CheckpointTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = "/tmp/gpt_ckpt_test";
    std::filesystem::remove_all(dir_);
  }

  void TearDown() override {
    std::filesystem::remove_all(dir_);
  }

  std::string dir_;
};

TEST_F(CheckpointTest, SaveLoadRoundTrip) {
  const int64_t n = 32;
  std::vector<float> h_params(n), h_m(n), h_v(n);
  for (int i = 0; i < n; ++i) {
    h_params[i] = static_cast<float>(i);
    h_m[i] = static_cast<float>(i) * 0.1f;
    h_v[i] = static_cast<float>(i) * 0.01f;
  }

  float *d_params, *d_m, *d_v;
  cudaMalloc(&d_params, n * sizeof(float));
  cudaMalloc(&d_m, n * sizeof(float));
  cudaMalloc(&d_v, n * sizeof(float));
  cudaMemcpy(d_params, h_params.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_m, h_m.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  CheckpointMetadata meta;
  meta.step = 42;
  meta.param_count = n;
  meta.dataset_cursor = 4096;
  meta.config_json = "{\"learning_rate\":0.001,\"run\":\"checkpoint_test\"}";

  ASSERT_TRUE(save_checkpoint(dir_, d_params, d_m, d_v, n, meta).ok());

  // Zero device buffers.
  cudaMemset(d_params, 0, n * sizeof(float));
  cudaMemset(d_m, 0, n * sizeof(float));
  cudaMemset(d_v, 0, n * sizeof(float));

  CheckpointMetadata loaded_meta;
  ASSERT_TRUE(load_checkpoint(dir_, d_params, d_m, d_v, n, loaded_meta).ok());

  EXPECT_EQ(loaded_meta.step, meta.step);
  EXPECT_EQ(loaded_meta.param_count, meta.param_count);
  EXPECT_EQ(loaded_meta.dataset_cursor, meta.dataset_cursor);
  EXPECT_EQ(loaded_meta.config_json, meta.config_json);

  // Verify params match.
  std::vector<float> h_loaded(n);
  cudaMemcpy(h_loaded.data(), d_params, n * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(h_loaded[i], h_params[i]);
  }

  cudaFree(d_params); cudaFree(d_m); cudaFree(d_v);
}

TEST_F(CheckpointTest, ParamCountMismatchDoesNotOverwriteBuffers) {
  const int64_t saved_n = 16;
  const int64_t target_n = 32;
  std::vector<float> h_saved(saved_n);
  std::vector<float> h_target(target_n);
  for (int i = 0; i < saved_n; ++i) {
    h_saved[i] = static_cast<float>(100 + i);
  }
  for (int i = 0; i < target_n; ++i) {
    h_target[i] = static_cast<float>(i);
  }

  float *d_saved_params, *d_saved_m, *d_saved_v;
  cudaMalloc(&d_saved_params, saved_n * sizeof(float));
  cudaMalloc(&d_saved_m, saved_n * sizeof(float));
  cudaMalloc(&d_saved_v, saved_n * sizeof(float));
  cudaMemcpy(d_saved_params, h_saved.data(), saved_n * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_saved_m, h_saved.data(), saved_n * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_saved_v, h_saved.data(), saved_n * sizeof(float),
             cudaMemcpyHostToDevice);

  CheckpointMetadata meta;
  meta.step = 7;
  meta.param_count = saved_n;
  meta.dataset_cursor = 11;
  meta.config_json = "{\"run\":\"mismatch\"}";
  ASSERT_TRUE(save_checkpoint(dir_, d_saved_params, d_saved_m, d_saved_v, saved_n,
                              meta).ok());

  float *d_params, *d_m, *d_v;
  cudaMalloc(&d_params, target_n * sizeof(float));
  cudaMalloc(&d_m, target_n * sizeof(float));
  cudaMalloc(&d_v, target_n * sizeof(float));
  cudaMemcpy(d_params, h_target.data(), target_n * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_m, h_target.data(), target_n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_target.data(), target_n * sizeof(float), cudaMemcpyHostToDevice);

  CheckpointMetadata loaded_meta;
  Status status = load_checkpoint(dir_, d_params, d_m, d_v, target_n, loaded_meta);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);

  std::vector<float> h_loaded(target_n);
  cudaMemcpy(h_loaded.data(), d_params, target_n * sizeof(float),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < target_n; ++i) {
    EXPECT_FLOAT_EQ(h_loaded[i], h_target[i]);
  }

  cudaFree(d_saved_params); cudaFree(d_saved_m); cudaFree(d_saved_v);
  cudaFree(d_params); cudaFree(d_m); cudaFree(d_v);
}

TEST_F(CheckpointTest, ShortBufferFileFailsLoad) {
  const int64_t n = 8;
  std::filesystem::create_directories(dir_);
  {
    std::ofstream meta(dir_ + "/meta.json");
    meta << "{\n"
         << "  \"step\": 1,\n"
         << "  \"param_count\": " << n << ",\n"
         << "  \"dataset_cursor\": 0,\n"
         << "  \"config\": {\"run\":\"short_file\"}\n"
         << "}\n";
  }
  {
    std::ofstream params(dir_ + "/params.bin", std::ios::binary);
    float value = 1.0f;
    params.write(reinterpret_cast<char const*>(&value), sizeof(float));
  }
  {
    std::ofstream opt_m(dir_ + "/opt_m.bin", std::ios::binary);
    std::ofstream opt_v(dir_ + "/opt_v.bin", std::ios::binary);
    std::vector<float> full(n, 0.0f);
    opt_m.write(reinterpret_cast<char const*>(full.data()), n * sizeof(float));
    opt_v.write(reinterpret_cast<char const*>(full.data()), n * sizeof(float));
  }

  float *d_params, *d_m, *d_v;
  cudaMalloc(&d_params, n * sizeof(float));
  cudaMalloc(&d_m, n * sizeof(float));
  cudaMalloc(&d_v, n * sizeof(float));

  CheckpointMetadata loaded_meta;
  Status status = load_checkpoint(dir_, d_params, d_m, d_v, n, loaded_meta);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);

  cudaFree(d_params); cudaFree(d_m); cudaFree(d_v);
}
