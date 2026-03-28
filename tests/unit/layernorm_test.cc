// Tier B — GPU unit tests for LayerNorm.

#include "gtest/gtest.h"
#include "src/ops/layernorm.h"

#include <cuda_runtime.h>
#include <cmath>
#include <vector>

using namespace gpt;
using namespace gpt::ops;

class LayerNormTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto s = CudaStream::Create();
    ASSERT_TRUE(s.ok());
    stream_ = s.take();
  }
  CudaStream stream_;
};

TEST_F(LayerNormTest, OutputHasZeroMeanUnitVariance) {
  const int64_t rows = 2, D = 8;
  std::vector<float> h_x(rows * D);
  for (int i = 0; i < rows * D; ++i) h_x[i] = static_cast<float>(i);

  // gamma=1, beta=0 → pure normalization.
  std::vector<float> h_gamma(D, 1.0f), h_beta(D, 0.0f);

  float *d_x, *d_y, *d_gamma, *d_beta, *d_mean, *d_inv_std;
  cudaMalloc(&d_x, rows * D * sizeof(float));
  cudaMalloc(&d_y, rows * D * sizeof(float));
  cudaMalloc(&d_gamma, D * sizeof(float));
  cudaMalloc(&d_beta, D * sizeof(float));
  cudaMalloc(&d_mean, rows * sizeof(float));
  cudaMalloc(&d_inv_std, rows * sizeof(float));

  cudaMemcpy(d_x, h_x.data(), rows * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gamma, h_gamma.data(), D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_beta, h_beta.data(), D * sizeof(float), cudaMemcpyHostToDevice);

  LayerNormParams params;
  params.gamma = {d_gamma, {D}, {1}};
  params.beta = {d_beta, {D}, {1}};
  params.eps = 1e-5f;

  LayerNormState state;
  state.mean = {d_mean, {rows}, {1}};
  state.inv_std = {d_inv_std, {rows}, {1}};

  Tensor2D<const float> x{d_x, {rows, D}, {D, 1}};
  Tensor2D<float> y{d_y, {rows, D}, {D, 1}};

  ASSERT_TRUE(layernorm_forward(x, params, y, state, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_y(rows * D);
  cudaMemcpy(h_y.data(), d_y, rows * D * sizeof(float), cudaMemcpyDeviceToHost);

  // Each row should have ~zero mean and ~unit variance.
  for (int64_t i = 0; i < rows; ++i) {
    float mean = 0.0f, var = 0.0f;
    for (int64_t j = 0; j < D; ++j) mean += h_y[i * D + j];
    mean /= D;
    for (int64_t j = 0; j < D; ++j) {
      float diff = h_y[i * D + j] - mean;
      var += diff * diff;
    }
    var /= D;
    EXPECT_NEAR(mean, 0.0f, 1e-4f);
    EXPECT_NEAR(var, 1.0f, 1e-3f);
  }

  cudaFree(d_x); cudaFree(d_y); cudaFree(d_gamma);
  cudaFree(d_beta); cudaFree(d_mean); cudaFree(d_inv_std);
}
