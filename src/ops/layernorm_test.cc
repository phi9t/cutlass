// Tier B — GPU unit tests for Layer Normalization.
//
// Tests cover:
//   - Forward: output has zero mean and unit variance (gamma=1, beta=0)
//   - Forward: CPU reference match (with nontrivial gamma/beta)
//   - Forward: saved mean and inv_std match CPU
//   - Forward: single-row input
//   - Backward: CPU reference match for dx, dgamma, dbeta
//   - Backward: finite-difference gradient check (CPU-only)

#include "gtest/gtest.h"
#include "src/ops/layernorm.h"
#include "tests/test_utils.h"

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

// --- Forward tests ---

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

TEST_F(LayerNormTest, ForwardMatchesCPUReference) {
  const int64_t rows = 3, D = 16;
  test::SimpleRng rng(42);
  std::vector<float> h_x(rows * D), h_gamma(D), h_beta(D);
  rng.fill(h_x, 2.0f);
  rng.fill(h_gamma, 1.0f);
  rng.fill(h_beta, 0.5f);
  // Make gamma positive (scale should be positive for typical usage).
  for (auto& g : h_gamma) g = std::abs(g) + 0.1f;

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

  const float eps = 1e-5f;
  LayerNormParams params;
  params.gamma = {d_gamma, {D}, {1}};
  params.beta = {d_beta, {D}, {1}};
  params.eps = eps;

  LayerNormState state;
  state.mean = {d_mean, {rows}, {1}};
  state.inv_std = {d_inv_std, {rows}, {1}};

  Tensor2D<const float> x{d_x, {rows, D}, {D, 1}};
  Tensor2D<float> y{d_y, {rows, D}, {D, 1}};

  ASSERT_TRUE(layernorm_forward(x, params, y, state, stream_).ok());
  stream_.synchronize();

  // Read GPU results.
  std::vector<float> h_y(rows * D), h_mean(rows), h_inv_std(rows);
  cudaMemcpy(h_y.data(), d_y, rows * D * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_mean.data(), d_mean, rows * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_inv_std.data(), d_inv_std, rows * sizeof(float), cudaMemcpyDeviceToHost);

  // CPU reference.
  auto ref = test::cpu_layernorm_forward(h_x.data(), h_gamma.data(),
                                          h_beta.data(), eps, rows, D);

  EXPECT_TRUE(test::vectors_near(h_y, ref.y, 1e-4f, 1e-3f));
  EXPECT_TRUE(test::vectors_near(h_mean, ref.mean, 1e-5f));
  EXPECT_TRUE(test::vectors_near(h_inv_std, ref.inv_std, 1e-4f));

  cudaFree(d_x); cudaFree(d_y); cudaFree(d_gamma);
  cudaFree(d_beta); cudaFree(d_mean); cudaFree(d_inv_std);
}

TEST_F(LayerNormTest, ForwardGammaBetaAffineTransform) {
  // With gamma=2, beta=3, output should be 2*normalized + 3.
  const int64_t rows = 1, D = 4;
  std::vector<float> h_x = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> h_gamma(D, 2.0f), h_beta(D, 3.0f);

  float *d_x, *d_y, *d_gamma, *d_beta, *d_mean, *d_inv_std;
  cudaMalloc(&d_x, D * sizeof(float));
  cudaMalloc(&d_y, D * sizeof(float));
  cudaMalloc(&d_gamma, D * sizeof(float));
  cudaMalloc(&d_beta, D * sizeof(float));
  cudaMalloc(&d_mean, sizeof(float));
  cudaMalloc(&d_inv_std, sizeof(float));

  cudaMemcpy(d_x, h_x.data(), D * sizeof(float), cudaMemcpyHostToDevice);
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

  std::vector<float> h_y(D);
  cudaMemcpy(h_y.data(), d_y, D * sizeof(float), cudaMemcpyDeviceToHost);

  auto ref = test::cpu_layernorm_forward(h_x.data(), h_gamma.data(),
                                          h_beta.data(), 1e-5f, rows, D);
  for (int j = 0; j < D; ++j) {
    EXPECT_NEAR(h_y[j], ref.y[j], 1e-5f) << "at j=" << j;
  }

  // Mean of output should be 3 (beta) since normalized mean is 0.
  float out_mean = 0.0f;
  for (int j = 0; j < D; ++j) out_mean += h_y[j];
  out_mean /= D;
  EXPECT_NEAR(out_mean, 3.0f, 1e-3f);

  cudaFree(d_x); cudaFree(d_y); cudaFree(d_gamma);
  cudaFree(d_beta); cudaFree(d_mean); cudaFree(d_inv_std);
}

TEST_F(LayerNormTest, SingleRowInput) {
  const int64_t rows = 1, D = 8;
  test::SimpleRng rng(99);
  std::vector<float> h_x(D), h_gamma(D, 1.0f), h_beta(D, 0.0f);
  rng.fill(h_x, 5.0f);

  float *d_x, *d_y, *d_gamma, *d_beta, *d_mean, *d_inv_std;
  cudaMalloc(&d_x, D * sizeof(float));
  cudaMalloc(&d_y, D * sizeof(float));
  cudaMalloc(&d_gamma, D * sizeof(float));
  cudaMalloc(&d_beta, D * sizeof(float));
  cudaMalloc(&d_mean, sizeof(float));
  cudaMalloc(&d_inv_std, sizeof(float));

  cudaMemcpy(d_x, h_x.data(), D * sizeof(float), cudaMemcpyHostToDevice);
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

  std::vector<float> h_y(D);
  cudaMemcpy(h_y.data(), d_y, D * sizeof(float), cudaMemcpyDeviceToHost);

  auto ref = test::cpu_layernorm_forward(h_x.data(), h_gamma.data(),
                                          h_beta.data(), 1e-5f, rows, D);
  EXPECT_TRUE(test::vectors_near(h_y, ref.y, 1e-4f));

  cudaFree(d_x); cudaFree(d_y); cudaFree(d_gamma);
  cudaFree(d_beta); cudaFree(d_mean); cudaFree(d_inv_std);
}

// --- Backward tests ---

TEST_F(LayerNormTest, BackwardMatchesCPUReference) {
  const int64_t rows = 2, D = 8;
  test::SimpleRng rng(42);
  std::vector<float> h_x(rows * D), h_gamma(D), h_beta(D);
  rng.fill(h_x, 2.0f);
  rng.fill(h_gamma, 1.0f);
  rng.fill(h_beta, 0.5f);
  for (auto& g : h_gamma) g = std::abs(g) + 0.1f;

  const float eps = 1e-5f;

  // Forward on CPU to get mean, inv_std.
  auto fwd = test::cpu_layernorm_forward(h_x.data(), h_gamma.data(),
                                          h_beta.data(), eps, rows, D);

  // Random upstream gradient.
  std::vector<float> h_dy(rows * D);
  rng.fill(h_dy, 1.0f);

  // Allocate GPU buffers.
  float *d_dy, *d_x, *d_gamma, *d_mean, *d_inv_std;
  float *d_dx, *d_dgamma, *d_dbeta;
  cudaMalloc(&d_dy, rows * D * sizeof(float));
  cudaMalloc(&d_x, rows * D * sizeof(float));
  cudaMalloc(&d_gamma, D * sizeof(float));
  cudaMalloc(&d_mean, rows * sizeof(float));
  cudaMalloc(&d_inv_std, rows * sizeof(float));
  cudaMalloc(&d_dx, rows * D * sizeof(float));
  cudaMalloc(&d_dgamma, D * sizeof(float));
  cudaMalloc(&d_dbeta, D * sizeof(float));

  cudaMemcpy(d_dy, h_dy.data(), rows * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, h_x.data(), rows * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gamma, h_gamma.data(), D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mean, fwd.mean.data(), rows * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_inv_std, fwd.inv_std.data(), rows * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_dgamma, 0, D * sizeof(float));
  cudaMemset(d_dbeta, 0, D * sizeof(float));

  LayerNormParams params;
  params.gamma = {d_gamma, {D}, {1}};
  params.beta = {nullptr, {D}, {1}};  // beta not needed for backward
  params.eps = eps;

  LayerNormState state;
  state.mean = {d_mean, {rows}, {1}};
  state.inv_std = {d_inv_std, {rows}, {1}};

  Tensor2D<const float> dy{d_dy, {rows, D}, {D, 1}};
  Tensor2D<const float> x{d_x, {rows, D}, {D, 1}};
  Tensor2D<float> dx{d_dx, {rows, D}, {D, 1}};
  Tensor1D<float> dgamma{d_dgamma, {D}, {1}};
  Tensor1D<float> dbeta{d_dbeta, {D}, {1}};

  ASSERT_TRUE(layernorm_backward(dy, x, params, state, dx, dgamma, dbeta,
                                  stream_).ok());
  stream_.synchronize();

  // Read results.
  std::vector<float> h_dx(rows * D), h_dgamma(D), h_dbeta(D);
  cudaMemcpy(h_dx.data(), d_dx, rows * D * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_dgamma.data(), d_dgamma, D * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_dbeta.data(), d_dbeta, D * sizeof(float), cudaMemcpyDeviceToHost);

  // CPU reference.
  auto ref = test::cpu_layernorm_backward(h_dy.data(), h_x.data(),
                                           h_gamma.data(), fwd.mean.data(),
                                           fwd.inv_std.data(), rows, D);

  EXPECT_TRUE(test::vectors_near(h_dx, ref.dx, 1e-4f, 1e-3f))
      << "dx mismatch";
  EXPECT_TRUE(test::vectors_near(h_dgamma, ref.dgamma, 1e-4f, 1e-3f))
      << "dgamma mismatch";
  EXPECT_TRUE(test::vectors_near(h_dbeta, ref.dbeta, 1e-4f, 1e-3f))
      << "dbeta mismatch";

  cudaFree(d_dy); cudaFree(d_x); cudaFree(d_gamma);
  cudaFree(d_mean); cudaFree(d_inv_std);
  cudaFree(d_dx); cudaFree(d_dgamma); cudaFree(d_dbeta);
}

// CPU-only finite difference gradient check for LayerNorm backward.
TEST(LayerNormCPUTest, BackwardFiniteDifference) {
  const int64_t rows = 2, D = 4;
  test::SimpleRng rng(77);
  std::vector<float> h_x(rows * D), h_gamma(D), h_beta(D);
  rng.fill(h_x, 2.0f);
  for (int j = 0; j < D; ++j) {
    h_gamma[j] = 0.5f + rng.uniform(0.5f);  // positive
    h_beta[j] = rng.uniform(0.3f);
  }
  const float eps = 1e-5f;

  auto fwd = test::cpu_layernorm_forward(h_x.data(), h_gamma.data(),
                                          h_beta.data(), eps, rows, D);

  // Use dy = 1 for simple gradient extraction.
  std::vector<float> h_dy(rows * D, 1.0f);
  auto ref = test::cpu_layernorm_backward(h_dy.data(), h_x.data(),
                                           h_gamma.data(), fwd.mean.data(),
                                           fwd.inv_std.data(), rows, D);

  // Numerical gradient for dx: perturb each x[i] by ±h.
  const float h = 1e-3f;
  for (int64_t idx = 0; idx < rows * D; ++idx) {
    float orig = h_x[idx];

    h_x[idx] = orig + h;
    auto fwd_p = test::cpu_layernorm_forward(h_x.data(), h_gamma.data(),
                                              h_beta.data(), eps, rows, D);
    float loss_p = 0.0f;
    for (auto v : fwd_p.y) loss_p += v;

    h_x[idx] = orig - h;
    auto fwd_m = test::cpu_layernorm_forward(h_x.data(), h_gamma.data(),
                                              h_beta.data(), eps, rows, D);
    float loss_m = 0.0f;
    for (auto v : fwd_m.y) loss_m += v;

    h_x[idx] = orig;

    float numerical = (loss_p - loss_m) / (2.0f * h);
    EXPECT_NEAR(numerical, ref.dx[idx], 1e-2f)
        << "dx finite difference at index " << idx;
  }
}
