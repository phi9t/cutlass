// Tier B — GPU unit tests for cross-entropy loss.

#include "gtest/gtest.h"
#include "src/ops/loss.h"
#include "tests/test_utils.h"

#include <cuda_runtime.h>
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
