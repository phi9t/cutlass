// Tier B — GPU unit tests for GELU activation.
//
// Tests cover:
//   - Forward: CPU reference match across negative/zero/positive range
//   - Forward: zero input → zero output
//   - Forward: large magnitude saturation
//   - Forward: negative inputs (GELU is asymmetric)
//   - Backward: CPU reference match for GELU derivative
//   - Backward: gradient at zero

#include "gtest/gtest.h"
#include "src/ops/gelu.h"
#include "tests/test_utils.h"

#include <cuda_runtime.h>
#include <cmath>
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

// --- Forward tests ---

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

TEST_F(GeluTest, ForwardLargePositiveSaturates) {
  // For large positive x, GELU(x) ≈ x.
  const int64_t n = 4;
  std::vector<float> h_x = {5.0f, 10.0f, 20.0f, 50.0f};

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
    // GELU(x) should be very close to x for large positive x.
    EXPECT_NEAR(h_y[i], h_x[i], 0.01f) << "at index " << i;
  }

  cudaFree(d_x); cudaFree(d_y);
}

TEST_F(GeluTest, ForwardLargeNegativeNearZero) {
  // For large negative x, GELU(x) ≈ 0.
  const int64_t n = 4;
  std::vector<float> h_x = {-5.0f, -10.0f, -20.0f, -50.0f};

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
    EXPECT_NEAR(h_y[i], 0.0f, 0.01f) << "at index " << i;
  }

  cudaFree(d_x); cudaFree(d_y);
}

TEST_F(GeluTest, ForwardRandomInputsCPUReference) {
  // Test with random inputs to cover a wider range.
  const int64_t n = 256;
  test::SimpleRng rng(123);
  std::vector<float> h_x(n);
  rng.fill(h_x, 4.0f);

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

// --- Backward tests ---

TEST_F(GeluTest, BackwardMatchesCPUReference) {
  const int64_t n = 64;
  test::SimpleRng rng(42);
  std::vector<float> h_dy(n), h_x(n);
  rng.fill(h_x, 3.0f);
  rng.fill(h_dy, 1.0f);

  float *d_dy, *d_x, *d_dx;
  cudaMalloc(&d_dy, n * sizeof(float));
  cudaMalloc(&d_x, n * sizeof(float));
  cudaMalloc(&d_dx, n * sizeof(float));
  cudaMemcpy(d_dy, h_dy.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  Tensor1D<const float> dy{d_dy, {n}, {1}};
  Tensor1D<const float> x{d_x, {n}, {1}};
  Tensor1D<float> dx{d_dx, {n}, {1}};

  ASSERT_TRUE(gelu_backward(dy, x, dx, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_dx(n);
  cudaMemcpy(h_dx.data(), d_dx, n * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; ++i) {
    float expected = h_dy[i] * test::cpu_gelu_backward(h_x[i]);
    EXPECT_NEAR(h_dx[i], expected, 1e-4f) << "at index " << i;
  }

  cudaFree(d_dy); cudaFree(d_x); cudaFree(d_dx);
}

TEST_F(GeluTest, BackwardAtZero) {
  // GELU'(0) = 0.5 (since tanh(0)=0, sech²(0)=1).
  float h_dy = 1.0f, h_x = 0.0f;
  float *d_dy, *d_x, *d_dx;
  cudaMalloc(&d_dy, sizeof(float));
  cudaMalloc(&d_x, sizeof(float));
  cudaMalloc(&d_dx, sizeof(float));
  cudaMemcpy(d_dy, &h_dy, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, &h_x, sizeof(float), cudaMemcpyHostToDevice);

  Tensor1D<const float> dy{d_dy, {1}, {1}};
  Tensor1D<const float> x{d_x, {1}, {1}};
  Tensor1D<float> dx{d_dx, {1}, {1}};

  ASSERT_TRUE(gelu_backward(dy, x, dx, stream_).ok());
  stream_.synchronize();

  float h_dx;
  cudaMemcpy(&h_dx, d_dx, sizeof(float), cudaMemcpyDeviceToHost);
  EXPECT_NEAR(h_dx, 0.5f, 1e-5f);

  cudaFree(d_dy); cudaFree(d_x); cudaFree(d_dx);
}

TEST_F(GeluTest, BackwardFiniteDifferenceCheck) {
  // Numerical gradient check: d/dx GELU(x) ≈ (GELU(x+h) - GELU(x-h)) / 2h
  const int64_t n = 16;
  test::SimpleRng rng(99);
  std::vector<float> h_x(n);
  rng.fill(h_x, 3.0f);

  const float eps = 1e-3f;
  for (int i = 0; i < n; ++i) {
    float x = h_x[i];
    float numerical = (test::cpu_gelu(x + eps) - test::cpu_gelu(x - eps)) /
                      (2.0f * eps);
    float analytical = test::cpu_gelu_backward(x);
    EXPECT_NEAR(numerical, analytical, 1e-3f) << "at x=" << x;
  }
}
