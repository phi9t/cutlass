// Tier B — GPU unit tests for cross-entropy loss.
//
// Tests cover:
//   - Forward: CPU reference match
//   - Forward: perfect prediction (one-hot logits)
//   - Forward: uniform logits → log(V) loss
//   - Backward: gradient sums to zero per sample
//   - Backward: CPU reference match for gradient values
//   - Backward: target position gets negative gradient

#include "gtest/gtest.h"
#include "src/ops/loss.h"
#include "tests/test_utils.h"

#include <cuda_runtime.h>
#include <cmath>
#include <vector>

using namespace gpt;
using namespace gpt::ops;

class LossTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto s = CudaStream::Create();
    ASSERT_TRUE(s.ok());
    stream_ = s.take();
  }
  CudaStream stream_;
};

// --- Forward tests ---

TEST_F(LossTest, ForwardMatchesCPU) {
  const int64_t N = 2, V = 4;
  std::vector<float> h_logits = {1.0f, 2.0f, 3.0f, 4.0f,
                                   0.5f, 1.5f, 2.5f, 3.5f};
  std::vector<int32_t> h_targets = {2, 0};

  float* d_logits; int32_t* d_targets; float* d_loss;
  cudaMalloc(&d_logits, N * V * sizeof(float));
  cudaMalloc(&d_targets, N * sizeof(int32_t));
  cudaMalloc(&d_loss, sizeof(float));
  cudaMemcpy(d_logits, h_logits.data(), N * V * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_targets, h_targets.data(), N * sizeof(int32_t),
             cudaMemcpyHostToDevice);

  Tensor2D<const float> logits{d_logits, {N, V}, {V, 1}};
  Tensor1D<const int32_t> targets{d_targets, {N}, {1}};

  ASSERT_TRUE(cross_entropy_forward(logits, targets, d_loss, stream_).ok());
  stream_.synchronize();

  float h_loss;
  cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

  // CPU reference.
  float expected = 0.0f;
  expected += test::cpu_cross_entropy(h_logits.data(), h_targets[0], V);
  expected += test::cpu_cross_entropy(h_logits.data() + V, h_targets[1], V);
  expected /= static_cast<float>(N);

  EXPECT_NEAR(h_loss, expected, 1e-4f);

  cudaFree(d_logits); cudaFree(d_targets); cudaFree(d_loss);
}

TEST_F(LossTest, ForwardPerfectPredictionLowLoss) {
  // If the correct class has a very large logit, loss should be near zero.
  const int64_t N = 1, V = 4;
  std::vector<float> h_logits = {-10.0f, -10.0f, 100.0f, -10.0f};
  std::vector<int32_t> h_targets = {2};

  float* d_logits; int32_t* d_targets; float* d_loss;
  cudaMalloc(&d_logits, V * sizeof(float));
  cudaMalloc(&d_targets, sizeof(int32_t));
  cudaMalloc(&d_loss, sizeof(float));
  cudaMemcpy(d_logits, h_logits.data(), V * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_targets, h_targets.data(), sizeof(int32_t), cudaMemcpyHostToDevice);

  Tensor2D<const float> logits{d_logits, {N, V}, {V, 1}};
  Tensor1D<const int32_t> targets{d_targets, {N}, {1}};

  ASSERT_TRUE(cross_entropy_forward(logits, targets, d_loss, stream_).ok());
  stream_.synchronize();

  float h_loss;
  cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
  EXPECT_NEAR(h_loss, 0.0f, 1e-3f);

  cudaFree(d_logits); cudaFree(d_targets); cudaFree(d_loss);
}

TEST_F(LossTest, ForwardUniformLogitsGivesLogV) {
  // If all logits are equal, loss = log(V).
  const int64_t N = 1, V = 8;
  std::vector<float> h_logits(V, 1.0f);
  std::vector<int32_t> h_targets = {3};

  float* d_logits; int32_t* d_targets; float* d_loss;
  cudaMalloc(&d_logits, V * sizeof(float));
  cudaMalloc(&d_targets, sizeof(int32_t));
  cudaMalloc(&d_loss, sizeof(float));
  cudaMemcpy(d_logits, h_logits.data(), V * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_targets, h_targets.data(), sizeof(int32_t), cudaMemcpyHostToDevice);

  Tensor2D<const float> logits{d_logits, {N, V}, {V, 1}};
  Tensor1D<const int32_t> targets{d_targets, {N}, {1}};

  ASSERT_TRUE(cross_entropy_forward(logits, targets, d_loss, stream_).ok());
  stream_.synchronize();

  float h_loss;
  cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
  EXPECT_NEAR(h_loss, std::log(static_cast<float>(V)), 1e-4f);

  cudaFree(d_logits); cudaFree(d_targets); cudaFree(d_loss);
}

TEST_F(LossTest, ForwardRandomInputsCPUReference) {
  const int64_t N = 4, V = 16;
  test::SimpleRng rng(42);
  std::vector<float> h_logits(N * V);
  rng.fill(h_logits, 3.0f);
  std::vector<int32_t> h_targets = {0, 5, 15, 8};

  float* d_logits; int32_t* d_targets; float* d_loss;
  cudaMalloc(&d_logits, N * V * sizeof(float));
  cudaMalloc(&d_targets, N * sizeof(int32_t));
  cudaMalloc(&d_loss, sizeof(float));
  cudaMemcpy(d_logits, h_logits.data(), N * V * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_targets, h_targets.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);

  Tensor2D<const float> logits{d_logits, {N, V}, {V, 1}};
  Tensor1D<const int32_t> targets{d_targets, {N}, {1}};

  ASSERT_TRUE(cross_entropy_forward(logits, targets, d_loss, stream_).ok());
  stream_.synchronize();

  float h_loss;
  cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

  float expected = 0.0f;
  for (int64_t i = 0; i < N; ++i)
    expected += test::cpu_cross_entropy(h_logits.data() + i * V, h_targets[i], V);
  expected /= N;

  EXPECT_NEAR(h_loss, expected, 1e-3f);

  cudaFree(d_logits); cudaFree(d_targets); cudaFree(d_loss);
}

// --- Backward tests ---

TEST_F(LossTest, BackwardGradientsSumToZero) {
  // For cross-entropy, per-sample gradient should sum to 0 across vocab.
  const int64_t N = 1, V = 4;
  std::vector<float> h_logits = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int32_t> h_targets = {1};

  float* d_logits; int32_t* d_targets; float* d_grad;
  cudaMalloc(&d_logits, V * sizeof(float));
  cudaMalloc(&d_targets, sizeof(int32_t));
  cudaMalloc(&d_grad, V * sizeof(float));
  cudaMemcpy(d_logits, h_logits.data(), V * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_targets, h_targets.data(), sizeof(int32_t), cudaMemcpyHostToDevice);

  Tensor2D<const float> logits{d_logits, {N, V}, {V, 1}};
  Tensor1D<const int32_t> targets{d_targets, {N}, {1}};
  Tensor2D<float> d_logits_out{d_grad, {N, V}, {V, 1}};

  ASSERT_TRUE(cross_entropy_backward(logits, targets, d_logits_out, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_grad(V);
  cudaMemcpy(h_grad.data(), d_grad, V * sizeof(float), cudaMemcpyDeviceToHost);

  float grad_sum = 0.0f;
  for (float g : h_grad) grad_sum += g;
  EXPECT_NEAR(grad_sum, 0.0f, 1e-5f);

  cudaFree(d_logits); cudaFree(d_targets); cudaFree(d_grad);
}

TEST_F(LossTest, BackwardMatchesCPUReference) {
  const int64_t N = 3, V = 8;
  test::SimpleRng rng(42);
  std::vector<float> h_logits(N * V);
  rng.fill(h_logits, 3.0f);
  std::vector<int32_t> h_targets = {2, 7, 0};

  float* d_logits; int32_t* d_targets; float* d_grad;
  cudaMalloc(&d_logits, N * V * sizeof(float));
  cudaMalloc(&d_targets, N * sizeof(int32_t));
  cudaMalloc(&d_grad, N * V * sizeof(float));
  cudaMemcpy(d_logits, h_logits.data(), N * V * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_targets, h_targets.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);

  Tensor2D<const float> logits{d_logits, {N, V}, {V, 1}};
  Tensor1D<const int32_t> targets{d_targets, {N}, {1}};
  Tensor2D<float> d_logits_out{d_grad, {N, V}, {V, 1}};

  ASSERT_TRUE(cross_entropy_backward(logits, targets, d_logits_out, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_grad(N * V);
  cudaMemcpy(h_grad.data(), d_grad, N * V * sizeof(float), cudaMemcpyDeviceToHost);

  auto expected = test::cpu_cross_entropy_backward(h_logits.data(),
                                                    h_targets.data(), N, V);
  EXPECT_TRUE(test::vectors_near(h_grad, expected, 1e-5f, 1e-4f));

  cudaFree(d_logits); cudaFree(d_targets); cudaFree(d_grad);
}

TEST_F(LossTest, BackwardTargetGetsNegativeGradient) {
  // The gradient at the target position should be (softmax - 1)/N, which is negative.
  const int64_t N = 1, V = 4;
  std::vector<float> h_logits = {0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<int32_t> h_targets = {2};

  float* d_logits; int32_t* d_targets; float* d_grad;
  cudaMalloc(&d_logits, V * sizeof(float));
  cudaMalloc(&d_targets, sizeof(int32_t));
  cudaMalloc(&d_grad, V * sizeof(float));
  cudaMemcpy(d_logits, h_logits.data(), V * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_targets, h_targets.data(), sizeof(int32_t), cudaMemcpyHostToDevice);

  Tensor2D<const float> logits{d_logits, {N, V}, {V, 1}};
  Tensor1D<const int32_t> targets{d_targets, {N}, {1}};
  Tensor2D<float> d_logits_out{d_grad, {N, V}, {V, 1}};

  ASSERT_TRUE(cross_entropy_backward(logits, targets, d_logits_out, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_grad(V);
  cudaMemcpy(h_grad.data(), d_grad, V * sizeof(float), cudaMemcpyDeviceToHost);

  // For uniform logits: softmax = 1/V = 0.25
  // Target grad = (0.25 - 1)/1 = -0.75
  // Non-target grad = 0.25/1 = 0.25
  EXPECT_NEAR(h_grad[2], -0.75f, 1e-5f);
  EXPECT_NEAR(h_grad[0], 0.25f, 1e-5f);
  EXPECT_NEAR(h_grad[1], 0.25f, 1e-5f);
  EXPECT_NEAR(h_grad[3], 0.25f, 1e-5f);

  cudaFree(d_logits); cudaFree(d_targets); cudaFree(d_grad);
}
