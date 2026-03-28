// Tier B — GPU unit tests for GELU activation.

#include "gtest/gtest.h"
#include "src/ops/gelu.h"
#include "tests/test_utils.h"

#include <cuda_runtime.h>
#include <vector>

using namespace gpt;
using namespace gpt::ops;

class GeluTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto s = CudaStream::Create();
    ASSERT_TRUE(s.ok());
    stream_ = s.take();
  }
  CudaStream stream_;
};

TEST_F(GeluTest, ForwardMatchesCPUReference) {
  const int64_t n = 16;
  std::vector<float> h_x(n);
  for (int i = 0; i < n; ++i) h_x[i] = static_cast<float>(i) - 8.0f;

  float *d_x, *d_y;
  cudaMalloc(&d_x, n * sizeof(float));
  cudaMalloc(&d_y, n * sizeof(float));
  cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  Tensor1D<const float> x{d_x, {n}, {1}};
  Tensor1D<float> y{d_y, {n}, {1}};

  ASSERT_TRUE(gelu_forward(x, y, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_y(n);
  cudaMemcpy(h_y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; ++i) {
    float expected = test::cpu_gelu(h_x[i]);
    EXPECT_NEAR(h_y[i], expected, 1e-5f) << "at index " << i;
  }

  cudaFree(d_x); cudaFree(d_y);
}

TEST_F(GeluTest, ZeroInputGivesZeroOutput) {
  float h_x = 0.0f;
  float *d_x, *d_y;
  cudaMalloc(&d_x, sizeof(float));
  cudaMalloc(&d_y, sizeof(float));
  cudaMemcpy(d_x, &h_x, sizeof(float), cudaMemcpyHostToDevice);

  Tensor1D<const float> x{d_x, {1}, {1}};
  Tensor1D<float> y{d_y, {1}, {1}};

  ASSERT_TRUE(gelu_forward(x, y, stream_).ok());
  stream_.synchronize();

  float h_y;
  cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);
  EXPECT_NEAR(h_y, 0.0f, 1e-7f);

  cudaFree(d_x); cudaFree(d_y);
}
