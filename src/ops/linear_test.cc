// Tier B — GPU unit tests for linear projection.
//
// Tests cover:
//   - Forward: CPU reference match (no bias)
//   - Forward: CPU reference match (with bias)
//   - Forward: identity weight produces copy
//   - Forward: shape validation
//   - Backward: dX matches CPU reference
//   - Backward: dW matches CPU reference
//   - Backward: random inputs CPU reference match

#include "gtest/gtest.h"
#include "src/ops/linear.h"
#include "tests/test_utils.h"

#include <cuda_runtime.h>
#include <vector>

using namespace gpt;
using namespace gpt::ops;

class LinearTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto s = CudaStream::Create();
    ASSERT_TRUE(s.ok());
    stream_ = s.take();
  }
  CudaStream stream_;
};

// --- Forward tests ---

TEST_F(LinearTest, ForwardNoBiasMatchesCPU) {
  // X: [2, 3], W: [4, 3] → Y: [2, 4]
  const int64_t N = 2, D_in = 3, D_out = 4;
  test::SimpleRng rng(42);
  std::vector<float> h_X(N * D_in), h_W(D_out * D_in);
  rng.fill(h_X, 1.0f);
  rng.fill(h_W, 1.0f);

  float *d_X, *d_W, *d_Y;
  cudaMalloc(&d_X, N * D_in * sizeof(float));
  cudaMalloc(&d_W, D_out * D_in * sizeof(float));
  cudaMalloc(&d_Y, N * D_out * sizeof(float));
  cudaMemcpy(d_X, h_X.data(), N * D_in * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W, h_W.data(), D_out * D_in * sizeof(float), cudaMemcpyHostToDevice);

  LinearParams params;
  params.weight = {d_W, {D_out, D_in}, {D_in, 1}};
  params.bias = {nullptr, {D_out}, {1}};
  params.use_bias = false;

  Tensor2D<const float> X{d_X, {N, D_in}, {D_in, 1}};
  Tensor2D<float> Y{d_Y, {N, D_out}, {D_out, 1}};

  ASSERT_TRUE(linear_forward(X, params, Y, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_Y(N * D_out);
  cudaMemcpy(h_Y.data(), d_Y, N * D_out * sizeof(float), cudaMemcpyDeviceToHost);

  auto expected = test::cpu_linear_forward(h_X.data(), h_W.data(), nullptr,
                                            N, D_in, D_out, false);
  EXPECT_TRUE(test::vectors_near(h_Y, expected, 1e-4f, 1e-3f));

  cudaFree(d_X); cudaFree(d_W); cudaFree(d_Y);
}

TEST_F(LinearTest, ForwardWithBiasMatchesCPU) {
  const int64_t N = 3, D_in = 4, D_out = 2;
  test::SimpleRng rng(77);
  std::vector<float> h_X(N * D_in), h_W(D_out * D_in), h_b(D_out);
  rng.fill(h_X, 1.0f);
  rng.fill(h_W, 1.0f);
  rng.fill(h_b, 0.5f);

  float *d_X, *d_W, *d_b, *d_Y;
  cudaMalloc(&d_X, N * D_in * sizeof(float));
  cudaMalloc(&d_W, D_out * D_in * sizeof(float));
  cudaMalloc(&d_b, D_out * sizeof(float));
  cudaMalloc(&d_Y, N * D_out * sizeof(float));
  cudaMemcpy(d_X, h_X.data(), N * D_in * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W, h_W.data(), D_out * D_in * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), D_out * sizeof(float), cudaMemcpyHostToDevice);

  LinearParams params;
  params.weight = {d_W, {D_out, D_in}, {D_in, 1}};
  params.bias = {d_b, {D_out}, {1}};
  params.use_bias = true;

  Tensor2D<const float> X{d_X, {N, D_in}, {D_in, 1}};
  Tensor2D<float> Y{d_Y, {N, D_out}, {D_out, 1}};

  ASSERT_TRUE(linear_forward(X, params, Y, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_Y(N * D_out);
  cudaMemcpy(h_Y.data(), d_Y, N * D_out * sizeof(float), cudaMemcpyDeviceToHost);

  auto expected = test::cpu_linear_forward(h_X.data(), h_W.data(), h_b.data(),
                                            N, D_in, D_out, true);
  EXPECT_TRUE(test::vectors_near(h_Y, expected, 1e-4f, 1e-3f));

  cudaFree(d_X); cudaFree(d_W); cudaFree(d_b); cudaFree(d_Y);
}

TEST_F(LinearTest, ForwardIdentityWeight) {
  // With identity weight and no bias, Y should equal X.
  const int64_t N = 2, D = 3;
  std::vector<float> h_X = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> h_W = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // 3x3 identity

  float *d_X, *d_W, *d_Y;
  cudaMalloc(&d_X, N * D * sizeof(float));
  cudaMalloc(&d_W, D * D * sizeof(float));
  cudaMalloc(&d_Y, N * D * sizeof(float));
  cudaMemcpy(d_X, h_X.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W, h_W.data(), D * D * sizeof(float), cudaMemcpyHostToDevice);

  LinearParams params;
  params.weight = {d_W, {D, D}, {D, 1}};
  params.bias = {nullptr, {D}, {1}};
  params.use_bias = false;

  Tensor2D<const float> X{d_X, {N, D}, {D, 1}};
  Tensor2D<float> Y{d_Y, {N, D}, {D, 1}};

  ASSERT_TRUE(linear_forward(X, params, Y, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_Y(N * D);
  cudaMemcpy(h_Y.data(), d_Y, N * D * sizeof(float), cudaMemcpyDeviceToHost);

  EXPECT_TRUE(test::vectors_near(h_Y, h_X, 1e-5f));

  cudaFree(d_X); cudaFree(d_W); cudaFree(d_Y);
}

TEST_F(LinearTest, ForwardShapeMismatchReturnsError) {
  // Weight D_in doesn't match X D_in → should fail.
  float *d_X, *d_W, *d_Y;
  cudaMalloc(&d_X, 6 * sizeof(float));
  cudaMalloc(&d_W, 8 * sizeof(float));
  cudaMalloc(&d_Y, 4 * sizeof(float));

  LinearParams params;
  params.weight = {d_W, {2, 4}, {4, 1}};  // D_in=4
  params.bias = {nullptr, {2}, {1}};
  params.use_bias = false;

  Tensor2D<const float> X{d_X, {3, 2}, {2, 1}};  // D_in=2 (mismatch!)
  Tensor2D<float> Y{d_Y, {3, 2}, {2, 1}};

  auto status = linear_forward(X, params, Y, stream_);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);

  cudaFree(d_X); cudaFree(d_W); cudaFree(d_Y);
}

// --- Backward tests ---

TEST_F(LinearTest, BackwardDXMatchesCPU) {
  const int64_t N = 2, D_in = 3, D_out = 4;
  test::SimpleRng rng(42);
  std::vector<float> h_X(N * D_in), h_W(D_out * D_in), h_dY(N * D_out);
  rng.fill(h_X, 1.0f);
  rng.fill(h_W, 1.0f);
  rng.fill(h_dY, 1.0f);

  float *d_X, *d_W, *d_dY, *d_dX, *d_dW, *d_db;
  cudaMalloc(&d_X, N * D_in * sizeof(float));
  cudaMalloc(&d_W, D_out * D_in * sizeof(float));
  cudaMalloc(&d_dY, N * D_out * sizeof(float));
  cudaMalloc(&d_dX, N * D_in * sizeof(float));
  cudaMalloc(&d_dW, D_out * D_in * sizeof(float));
  cudaMalloc(&d_db, D_out * sizeof(float));
  cudaMemcpy(d_X, h_X.data(), N * D_in * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W, h_W.data(), D_out * D_in * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dY, h_dY.data(), N * D_out * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_dW, 0, D_out * D_in * sizeof(float));
  cudaMemset(d_db, 0, D_out * sizeof(float));

  LinearParams params;
  params.weight = {d_W, {D_out, D_in}, {D_in, 1}};
  params.bias = {nullptr, {D_out}, {1}};
  params.use_bias = false;

  LinearGrads grads;
  grads.d_weight = {d_dW, {D_out, D_in}, {D_in, 1}};
  grads.d_bias = {d_db, {D_out}, {1}};

  Tensor2D<const float> X{d_X, {N, D_in}, {D_in, 1}};
  Tensor2D<const float> dY{d_dY, {N, D_out}, {D_out, 1}};
  Tensor2D<float> dX{d_dX, {N, D_in}, {D_in, 1}};

  ASSERT_TRUE(linear_backward(X, dY, params, dX, grads, stream_).ok());
  stream_.synchronize();

  // Read dX.
  std::vector<float> h_dX(N * D_in);
  cudaMemcpy(h_dX.data(), d_dX, N * D_in * sizeof(float), cudaMemcpyDeviceToHost);

  auto ref = test::cpu_linear_backward(h_X.data(), h_dY.data(), h_W.data(),
                                        N, D_in, D_out);
  EXPECT_TRUE(test::vectors_near(h_dX, ref.dX, 1e-4f, 1e-3f)) << "dX mismatch";

  // Read dW.
  std::vector<float> h_dW(D_out * D_in);
  cudaMemcpy(h_dW.data(), d_dW, D_out * D_in * sizeof(float), cudaMemcpyDeviceToHost);
  EXPECT_TRUE(test::vectors_near(h_dW, ref.dW, 1e-4f, 1e-3f)) << "dW mismatch";

  // Read db.
  std::vector<float> h_db(D_out);
  cudaMemcpy(h_db.data(), d_db, D_out * sizeof(float), cudaMemcpyDeviceToHost);
  EXPECT_TRUE(test::vectors_near(h_db, ref.db, 1e-4f, 1e-3f)) << "db mismatch";

  cudaFree(d_X); cudaFree(d_W); cudaFree(d_dY);
  cudaFree(d_dX); cudaFree(d_dW); cudaFree(d_db);
}

TEST_F(LinearTest, BackwardRandomInputsCPUReference) {
  const int64_t N = 4, D_in = 8, D_out = 6;
  test::SimpleRng rng(99);
  std::vector<float> h_X(N * D_in), h_W(D_out * D_in), h_dY(N * D_out);
  rng.fill(h_X, 2.0f);
  rng.fill(h_W, 0.5f);
  rng.fill(h_dY, 1.0f);

  float *d_X, *d_W, *d_dY, *d_dX, *d_dW, *d_db;
  cudaMalloc(&d_X, N * D_in * sizeof(float));
  cudaMalloc(&d_W, D_out * D_in * sizeof(float));
  cudaMalloc(&d_dY, N * D_out * sizeof(float));
  cudaMalloc(&d_dX, N * D_in * sizeof(float));
  cudaMalloc(&d_dW, D_out * D_in * sizeof(float));
  cudaMalloc(&d_db, D_out * sizeof(float));
  cudaMemcpy(d_X, h_X.data(), N * D_in * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W, h_W.data(), D_out * D_in * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dY, h_dY.data(), N * D_out * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_dW, 0, D_out * D_in * sizeof(float));
  cudaMemset(d_db, 0, D_out * sizeof(float));

  LinearParams params;
  params.weight = {d_W, {D_out, D_in}, {D_in, 1}};
  params.bias = {nullptr, {D_out}, {1}};
  params.use_bias = false;

  LinearGrads grads;
  grads.d_weight = {d_dW, {D_out, D_in}, {D_in, 1}};
  grads.d_bias = {d_db, {D_out}, {1}};

  Tensor2D<const float> X{d_X, {N, D_in}, {D_in, 1}};
  Tensor2D<const float> dY{d_dY, {N, D_out}, {D_out, 1}};
  Tensor2D<float> dX{d_dX, {N, D_in}, {D_in, 1}};

  ASSERT_TRUE(linear_backward(X, dY, params, dX, grads, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_dX(N * D_in), h_dW(D_out * D_in), h_db(D_out);
  cudaMemcpy(h_dX.data(), d_dX, N * D_in * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_dW.data(), d_dW, D_out * D_in * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_db.data(), d_db, D_out * sizeof(float), cudaMemcpyDeviceToHost);

  auto ref = test::cpu_linear_backward(h_X.data(), h_dY.data(), h_W.data(),
                                        N, D_in, D_out);
  EXPECT_TRUE(test::vectors_near(h_dX, ref.dX, 1e-3f, 1e-2f)) << "dX mismatch";
  EXPECT_TRUE(test::vectors_near(h_dW, ref.dW, 1e-3f, 1e-2f)) << "dW mismatch";
  EXPECT_TRUE(test::vectors_near(h_db, ref.db, 1e-3f, 1e-2f)) << "db mismatch";

  cudaFree(d_X); cudaFree(d_W); cudaFree(d_dY);
  cudaFree(d_dX); cudaFree(d_dW); cudaFree(d_db);
}

// CPU-only test for linear forward/backward reference correctness.
TEST(LinearCPUTest, ForwardBackwardConsistency) {
  // Verify that d/dX (sum(Y)) matches dX with dY = all-ones.
  const int64_t N = 2, D_in = 3, D_out = 4;
  test::SimpleRng rng(42);
  std::vector<float> h_X(N * D_in), h_W(D_out * D_in);
  rng.fill(h_X, 1.0f);
  rng.fill(h_W, 1.0f);

  auto Y = test::cpu_linear_forward(h_X.data(), h_W.data(), nullptr,
                                     N, D_in, D_out, false);

  std::vector<float> h_dY(N * D_out, 1.0f);
  auto grads = test::cpu_linear_backward(h_X.data(), h_dY.data(), h_W.data(),
                                          N, D_in, D_out);

  // Numerical gradient check for dX.
  const float eps = 1e-3f;
  for (int64_t idx = 0; idx < N * D_in; ++idx) {
    float orig = h_X[idx];

    h_X[idx] = orig + eps;
    auto Y_p = test::cpu_linear_forward(h_X.data(), h_W.data(), nullptr,
                                         N, D_in, D_out, false);
    float sum_p = 0.0f;
    for (float v : Y_p) sum_p += v;

    h_X[idx] = orig - eps;
    auto Y_m = test::cpu_linear_forward(h_X.data(), h_W.data(), nullptr,
                                         N, D_in, D_out, false);
    float sum_m = 0.0f;
    for (float v : Y_m) sum_m += v;

    h_X[idx] = orig;

    float numerical = (sum_p - sum_m) / (2.0f * eps);
    EXPECT_NEAR(numerical, grads.dX[idx], 1e-2f)
        << "dX numerical gradient at index " << idx;
  }
}
