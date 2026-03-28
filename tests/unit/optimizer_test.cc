// Tier B — GPU unit tests for AdamW optimizer.

#include "gtest/gtest.h"
#include "src/train/optimizer.h"

#include <cuda_runtime.h>
#include <vector>

using namespace gpt;
using namespace gpt::train;

class AdamWTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto s = CudaStream::Create();
    ASSERT_TRUE(s.ok());
    stream_ = s.take();
  }
  CudaStream stream_;
};

TEST_F(AdamWTest, InitAndStep) {
  const int64_t n = 16;
  AdamWConfig cfg;
  cfg.lr = 0.01f;

  AdamW opt;
  ASSERT_TRUE(opt.init(n, cfg).ok());
  EXPECT_EQ(opt.current_step(), 0);

  // Create params and constant grads on device.
  float *d_params, *d_grads;
  cudaMalloc(&d_params, n * sizeof(float));
  cudaMalloc(&d_grads, n * sizeof(float));

  std::vector<float> h_params(n, 1.0f);
  std::vector<float> h_grads(n, 0.1f);
  cudaMemcpy(d_params, h_params.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_grads, h_grads.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  // Run one step.
  ASSERT_TRUE(opt.step(d_params, d_grads, n, stream_).ok());
  opt.advance_step();
  EXPECT_EQ(opt.current_step(), 1);

  stream_.synchronize();

  // Params should have changed.
  std::vector<float> h_result(n);
  cudaMemcpy(h_result.data(), d_params, n * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; ++i) {
    EXPECT_NE(h_result[i], 1.0f) << "param[" << i << "] unchanged after step";
  }

  cudaFree(d_params); cudaFree(d_grads);
  opt.release();
}
