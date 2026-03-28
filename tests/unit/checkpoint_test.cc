// Tier B — GPU unit tests for checkpointing.

#include "gtest/gtest.h"
#include "src/checkpoint/checkpoint.h"

#include <cuda_runtime.h>
#include <filesystem>
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
  meta.config_json = "{}";

  ASSERT_TRUE(save_checkpoint(dir_, d_params, d_m, d_v, n, meta).ok());

  // Zero device buffers.
  cudaMemset(d_params, 0, n * sizeof(float));
  cudaMemset(d_m, 0, n * sizeof(float));
  cudaMemset(d_v, 0, n * sizeof(float));

  CheckpointMetadata loaded_meta;
  ASSERT_TRUE(load_checkpoint(dir_, d_params, d_m, d_v, n, loaded_meta).ok());

  // Verify params match.
  std::vector<float> h_loaded(n);
  cudaMemcpy(h_loaded.data(), d_params, n * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(h_loaded[i], h_params[i]);
  }

  cudaFree(d_params); cudaFree(d_m); cudaFree(d_v);
}
