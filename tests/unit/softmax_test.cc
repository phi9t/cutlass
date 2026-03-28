// Tier B — GPU unit tests for masked softmax.
//
// Tests cover:
//   - Forward: rows sum to one (no mask)
//   - Forward: CPU reference match (no mask)
//   - Forward: masked positions are zero, valid positions sum to one
//   - Forward: masked CPU reference match
//   - Forward: causal mask pattern
//   - Forward: uniform input → uniform output
//   - Backward: CPU reference match
//   - Backward: masked backward

#include "gtest/gtest.h"
#include "src/ops/softmax.h"
#include "tests/test_utils.h"

#include <cuda_runtime.h>
#include <vector>

using namespace gpt;
using namespace gpt::ops;

class SoftmaxTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto s = CudaStream::Create();
    ASSERT_TRUE(s.ok());
    stream_ = s.take();
  }
  CudaStream stream_;
};

// --- Forward tests ---

TEST_F(SoftmaxTest, RowsSumToOne) {
  const int64_t rows = 4, cols = 8;
  std::vector<float> h_input(rows * cols);
  for (int i = 0; i < rows * cols; ++i) h_input[i] = static_cast<float>(i) * 0.1f;

  float *d_input, *d_output;
  cudaMalloc(&d_input, rows * cols * sizeof(float));
  cudaMalloc(&d_output, rows * cols * sizeof(float));
  cudaMemcpy(d_input, h_input.data(), rows * cols * sizeof(float),
             cudaMemcpyHostToDevice);

  Tensor2D<const float> input{d_input, {rows, cols}, {cols, 1}};
  Tensor2D<const int8_t> null_mask{nullptr, {rows, cols}, {cols, 1}};
  Tensor2D<float> output{d_output, {rows, cols}, {cols, 1}};

  ASSERT_TRUE(masked_softmax_forward(input, null_mask, output, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_output(rows * cols);
  cudaMemcpy(h_output.data(), d_output, rows * cols * sizeof(float),
             cudaMemcpyDeviceToHost);

  for (int64_t i = 0; i < rows; ++i) {
    float sum = 0.0f;
    for (int64_t j = 0; j < cols; ++j) {
      sum += h_output[i * cols + j];
      EXPECT_GE(h_output[i * cols + j], 0.0f);
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
  }

  cudaFree(d_input); cudaFree(d_output);
}

TEST_F(SoftmaxTest, MatchesCPUReference) {
  const int64_t rows = 2, cols = 4;
  std::vector<float> h_input = {1.0f, 2.0f, 3.0f, 4.0f,
                                  -1.0f, 0.0f, 1.0f, 2.0f};

  float *d_input, *d_output;
  cudaMalloc(&d_input, rows * cols * sizeof(float));
  cudaMalloc(&d_output, rows * cols * sizeof(float));
  cudaMemcpy(d_input, h_input.data(), rows * cols * sizeof(float),
             cudaMemcpyHostToDevice);

  Tensor2D<const float> input{d_input, {rows, cols}, {cols, 1}};
  Tensor2D<const int8_t> null_mask{nullptr, {rows, cols}, {cols, 1}};
  Tensor2D<float> output{d_output, {rows, cols}, {cols, 1}};

  ASSERT_TRUE(masked_softmax_forward(input, null_mask, output, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_output(rows * cols);
  cudaMemcpy(h_output.data(), d_output, rows * cols * sizeof(float),
             cudaMemcpyDeviceToHost);

  auto expected = test::cpu_softmax(h_input.data(), rows, cols);
  EXPECT_TRUE(test::vectors_near(h_output, expected, 1e-5f));

  cudaFree(d_input); cudaFree(d_output);
}

TEST_F(SoftmaxTest, MaskedPositionsAreZero) {
  const int64_t rows = 1, cols = 4;
  std::vector<float> h_input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int8_t> h_mask = {0, 1, 0, 1};  // mask positions 1 and 3

  float* d_input; int8_t* d_mask; float* d_output;
  cudaMalloc(&d_input, 4 * sizeof(float));
  cudaMalloc(&d_mask, 4 * sizeof(int8_t));
  cudaMalloc(&d_output, 4 * sizeof(float));
  cudaMemcpy(d_input, h_input.data(), 4 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask, h_mask.data(), 4 * sizeof(int8_t), cudaMemcpyHostToDevice);

  Tensor2D<const float> input{d_input, {rows, cols}, {cols, 1}};
  Tensor2D<const int8_t> mask{d_mask, {rows, cols}, {cols, 1}};
  Tensor2D<float> output{d_output, {rows, cols}, {cols, 1}};

  ASSERT_TRUE(masked_softmax_forward(input, mask, output, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_output(4);
  cudaMemcpy(h_output.data(), d_output, 4 * sizeof(float), cudaMemcpyDeviceToHost);

  EXPECT_EQ(h_output[1], 0.0f);  // masked
  EXPECT_EQ(h_output[3], 0.0f);  // masked
  EXPECT_GT(h_output[0], 0.0f);  // valid
  EXPECT_GT(h_output[2], 0.0f);  // valid

  // Valid positions should sum to 1.
  EXPECT_NEAR(h_output[0] + h_output[2], 1.0f, 1e-5f);

  cudaFree(d_input); cudaFree(d_mask); cudaFree(d_output);
}

TEST_F(SoftmaxTest, MaskedForwardMatchesCPUReference) {
  const int64_t rows = 2, cols = 4;
  std::vector<float> h_input = {1.0f, 2.0f, 3.0f, 4.0f,
                                  0.5f, 1.5f, 2.5f, 3.5f};
  // Row 0: mask positions 1 and 3.  Row 1: mask position 2.
  std::vector<int8_t> h_mask = {0, 1, 0, 1,
                                 0, 0, 1, 0};

  float* d_input; int8_t* d_mask; float* d_output;
  cudaMalloc(&d_input, rows * cols * sizeof(float));
  cudaMalloc(&d_mask, rows * cols * sizeof(int8_t));
  cudaMalloc(&d_output, rows * cols * sizeof(float));
  cudaMemcpy(d_input, h_input.data(), rows * cols * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask, h_mask.data(), rows * cols * sizeof(int8_t),
             cudaMemcpyHostToDevice);

  Tensor2D<const float> input{d_input, {rows, cols}, {cols, 1}};
  Tensor2D<const int8_t> mask{d_mask, {rows, cols}, {cols, 1}};
  Tensor2D<float> output{d_output, {rows, cols}, {cols, 1}};

  ASSERT_TRUE(masked_softmax_forward(input, mask, output, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_output(rows * cols);
  cudaMemcpy(h_output.data(), d_output, rows * cols * sizeof(float),
             cudaMemcpyDeviceToHost);

  auto expected = test::cpu_masked_softmax(h_input.data(), h_mask.data(),
                                            rows, cols);
  EXPECT_TRUE(test::vectors_near(h_output, expected, 1e-5f));

  cudaFree(d_input); cudaFree(d_mask); cudaFree(d_output);
}

TEST_F(SoftmaxTest, CausalMaskPattern) {
  // Verify causal mask: for T=4, each query row i can only attend to positions 0..i.
  const int64_t T = 4;
  test::SimpleRng rng(77);
  std::vector<float> h_input(T * T);
  rng.fill(h_input, 2.0f);

  auto h_mask = test::cpu_causal_mask(T);

  float* d_input; int8_t* d_mask; float* d_output;
  cudaMalloc(&d_input, T * T * sizeof(float));
  cudaMalloc(&d_mask, T * T * sizeof(int8_t));
  cudaMalloc(&d_output, T * T * sizeof(float));
  cudaMemcpy(d_input, h_input.data(), T * T * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask, h_mask.data(), T * T * sizeof(int8_t), cudaMemcpyHostToDevice);

  Tensor2D<const float> input{d_input, {T, T}, {T, 1}};
  Tensor2D<const int8_t> mask{d_mask, {T, T}, {T, 1}};
  Tensor2D<float> output{d_output, {T, T}, {T, 1}};

  ASSERT_TRUE(masked_softmax_forward(input, mask, output, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_output(T * T);
  cudaMemcpy(h_output.data(), d_output, T * T * sizeof(float), cudaMemcpyDeviceToHost);

  // Verify causal structure: upper triangle should be zero.
  for (int64_t i = 0; i < T; ++i) {
    float valid_sum = 0.0f;
    for (int64_t j = 0; j < T; ++j) {
      if (j > i) {
        EXPECT_EQ(h_output[i * T + j], 0.0f)
            << "masked position (" << i << "," << j << ") should be 0";
      } else {
        EXPECT_GT(h_output[i * T + j], 0.0f);
        valid_sum += h_output[i * T + j];
      }
    }
    EXPECT_NEAR(valid_sum, 1.0f, 1e-5f) << "row " << i;
  }

  // Verify against CPU reference.
  auto expected = test::cpu_masked_softmax(h_input.data(), h_mask.data(), T, T);
  EXPECT_TRUE(test::vectors_near(h_output, expected, 1e-5f));

  cudaFree(d_input); cudaFree(d_mask); cudaFree(d_output);
}

TEST_F(SoftmaxTest, UniformInputGivesUniformOutput) {
  // If all inputs in a row are the same, softmax should give 1/cols each.
  const int64_t rows = 2, cols = 5;
  std::vector<float> h_input(rows * cols, 3.0f);

  float *d_input, *d_output;
  cudaMalloc(&d_input, rows * cols * sizeof(float));
  cudaMalloc(&d_output, rows * cols * sizeof(float));
  cudaMemcpy(d_input, h_input.data(), rows * cols * sizeof(float),
             cudaMemcpyHostToDevice);

  Tensor2D<const float> input{d_input, {rows, cols}, {cols, 1}};
  Tensor2D<const int8_t> null_mask{nullptr, {rows, cols}, {cols, 1}};
  Tensor2D<float> output{d_output, {rows, cols}, {cols, 1}};

  ASSERT_TRUE(masked_softmax_forward(input, null_mask, output, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_output(rows * cols);
  cudaMemcpy(h_output.data(), d_output, rows * cols * sizeof(float),
             cudaMemcpyDeviceToHost);

  float expected = 1.0f / static_cast<float>(cols);
  for (int i = 0; i < rows * cols; ++i) {
    EXPECT_NEAR(h_output[i], expected, 1e-6f);
  }

  cudaFree(d_input); cudaFree(d_output);
}

// --- Backward tests ---

TEST_F(SoftmaxTest, BackwardMatchesCPUReference) {
  const int64_t rows = 3, cols = 6;
  test::SimpleRng rng(42);

  // Generate random input, compute forward probs on CPU.
  std::vector<float> h_input(rows * cols);
  rng.fill(h_input, 2.0f);
  auto h_P = test::cpu_softmax(h_input.data(), rows, cols);

  // Random upstream gradient.
  std::vector<float> h_dP(rows * cols);
  rng.fill(h_dP, 1.0f);

  float *d_dP, *d_P, *d_dX;
  cudaMalloc(&d_dP, rows * cols * sizeof(float));
  cudaMalloc(&d_P, rows * cols * sizeof(float));
  cudaMalloc(&d_dX, rows * cols * sizeof(float));
  cudaMemcpy(d_dP, h_dP.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_P, h_P.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);

  Tensor2D<const float> dP{d_dP, {rows, cols}, {cols, 1}};
  Tensor2D<const float> P{d_P, {rows, cols}, {cols, 1}};
  Tensor2D<float> dX{d_dX, {rows, cols}, {cols, 1}};

  ASSERT_TRUE(masked_softmax_backward(dP, P, dX, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_dX(rows * cols);
  cudaMemcpy(h_dX.data(), d_dX, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

  auto expected = test::cpu_softmax_backward(h_dP.data(), h_P.data(), rows, cols);
  EXPECT_TRUE(test::vectors_near(h_dX, expected, 1e-5f));

  cudaFree(d_dP); cudaFree(d_P); cudaFree(d_dX);
}

TEST_F(SoftmaxTest, BackwardGradientRowSumsToZero) {
  // For softmax, sum_j(dX[i,j]) should be ~0 for each row.
  const int64_t rows = 4, cols = 8;
  test::SimpleRng rng(55);

  std::vector<float> h_input(rows * cols);
  rng.fill(h_input, 2.0f);
  auto h_P = test::cpu_softmax(h_input.data(), rows, cols);

  std::vector<float> h_dP(rows * cols);
  rng.fill(h_dP, 1.0f);

  float *d_dP, *d_P, *d_dX;
  cudaMalloc(&d_dP, rows * cols * sizeof(float));
  cudaMalloc(&d_P, rows * cols * sizeof(float));
  cudaMalloc(&d_dX, rows * cols * sizeof(float));
  cudaMemcpy(d_dP, h_dP.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_P, h_P.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);

  Tensor2D<const float> dP{d_dP, {rows, cols}, {cols, 1}};
  Tensor2D<const float> P{d_P, {rows, cols}, {cols, 1}};
  Tensor2D<float> dX{d_dX, {rows, cols}, {cols, 1}};

  ASSERT_TRUE(masked_softmax_backward(dP, P, dX, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_dX(rows * cols);
  cudaMemcpy(h_dX.data(), d_dX, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

  for (int64_t i = 0; i < rows; ++i) {
    float row_sum = 0.0f;
    for (int64_t j = 0; j < cols; ++j)
      row_sum += h_dX[i * cols + j];
    EXPECT_NEAR(row_sum, 0.0f, 1e-5f) << "row " << i;
  }

  cudaFree(d_dP); cudaFree(d_P); cudaFree(d_dX);
}
