// Tier B — GPU unit tests for masked softmax.

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
