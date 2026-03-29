// Tier B — GPU unit tests for reduction kernels.

#include "gtest/gtest.h"
#include "src/kernels/reductions.h"
#include "tests/test_utils.h"

#include <cuda_runtime.h>
#include <random>
#include <vector>

using namespace gpt;
using namespace gpt::kernels;

class ReductionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto s = CudaStream::Create();
    ASSERT_TRUE(s.ok());
    stream_ = s.take();
  }

  CudaStream stream_;
};

TEST_F(ReductionTest, RowMaxMatchesCPU) {
  const int64_t rows = 8, cols = 32;
  std::vector<float> h_input(rows * cols);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  for (auto& v : h_input) v = dist(rng);

  float *d_input, *d_output;
  cudaMalloc(&d_input, rows * cols * sizeof(float));
  cudaMalloc(&d_output, rows * sizeof(float));
  cudaMemcpy(d_input, h_input.data(), rows * cols * sizeof(float),
             cudaMemcpyHostToDevice);

  Tensor2D<const float> input{d_input, {rows, cols}, {cols, 1}};
  Tensor1D<float> output{d_output, {rows}, {1}};

  auto status = row_max_f32(input, output, stream_);
  ASSERT_TRUE(status.ok()) << status.message();
  stream_.synchronize();

  std::vector<float> h_output(rows);
  cudaMemcpy(h_output.data(), d_output, rows * sizeof(float),
             cudaMemcpyDeviceToHost);

  auto expected = test::cpu_row_max(h_input.data(), rows, cols);
  EXPECT_TRUE(test::vectors_near(h_output, expected));

  cudaFree(d_input);
  cudaFree(d_output);
}

TEST_F(ReductionTest, RowSumMatchesCPU) {
  const int64_t rows = 4, cols = 16;
  std::vector<float> h_input(rows * cols);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
  for (auto& v : h_input) v = dist(rng);

  float *d_input, *d_output;
  cudaMalloc(&d_input, rows * cols * sizeof(float));
  cudaMalloc(&d_output, rows * sizeof(float));
  cudaMemcpy(d_input, h_input.data(), rows * cols * sizeof(float),
             cudaMemcpyHostToDevice);

  Tensor2D<const float> input{d_input, {rows, cols}, {cols, 1}};
  Tensor1D<float> output{d_output, {rows}, {1}};

  auto status = row_sum_f32(input, output, stream_);
  ASSERT_TRUE(status.ok()) << status.message();
  stream_.synchronize();

  std::vector<float> h_output(rows);
  cudaMemcpy(h_output.data(), d_output, rows * sizeof(float),
             cudaMemcpyDeviceToHost);

  auto expected = test::cpu_row_sum(h_input.data(), rows, cols);
  EXPECT_TRUE(test::vectors_near(h_output, expected, 1e-4f));

  cudaFree(d_input);
  cudaFree(d_output);
}

TEST_F(ReductionTest, RowMeanMatchesCPU) {
  const int64_t rows = 4, cols = 8;
  std::vector<float> h_input(rows * cols);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : h_input) v = dist(rng);

  float *d_input, *d_output;
  cudaMalloc(&d_input, rows * cols * sizeof(float));
  cudaMalloc(&d_output, rows * sizeof(float));
  cudaMemcpy(d_input, h_input.data(), rows * cols * sizeof(float),
             cudaMemcpyHostToDevice);

  Tensor2D<const float> input{d_input, {rows, cols}, {cols, 1}};
  Tensor1D<float> output{d_output, {rows}, {1}};

  auto status = row_mean_f32(input, output, stream_);
  ASSERT_TRUE(status.ok()) << status.message();
  stream_.synchronize();

  std::vector<float> h_output(rows);
  cudaMemcpy(h_output.data(), d_output, rows * sizeof(float),
             cudaMemcpyDeviceToHost);

  auto expected = test::cpu_row_mean(h_input.data(), rows, cols);
  EXPECT_TRUE(test::vectors_near(h_output, expected, 1e-5f));

  cudaFree(d_input);
  cudaFree(d_output);
}
