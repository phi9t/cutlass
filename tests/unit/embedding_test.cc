// Tier B — GPU unit tests for embedding ops.

#include "gtest/gtest.h"
#include "src/ops/embedding.h"

#include <cuda_runtime.h>
#include <vector>

using namespace gpt;
using namespace gpt::ops;

class EmbeddingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto s = CudaStream::Create();
    ASSERT_TRUE(s.ok());
    stream_ = s.take();
  }
  CudaStream stream_;
};

TEST_F(EmbeddingTest, ForwardGathersCorrectRows) {
  // table: [4, 3], indices: [2] = {1, 3}
  std::vector<float> h_table = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int32_t> h_idx = {1, 3};

  float* d_table; int32_t* d_idx; float* d_out;
  cudaMalloc(&d_table, 12 * sizeof(float));
  cudaMalloc(&d_idx, 2 * sizeof(int32_t));
  cudaMalloc(&d_out, 6 * sizeof(float));
  cudaMemcpy(d_table, h_table.data(), 12 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_idx, h_idx.data(), 2 * sizeof(int32_t), cudaMemcpyHostToDevice);

  Tensor2D<const float> table{d_table, {4, 3}, {3, 1}};
  Tensor1D<const int32_t> indices{d_idx, {2}, {1}};
  Tensor2D<float> output{d_out, {2, 3}, {3, 1}};

  ASSERT_TRUE(embedding_forward(table, indices, output, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_out(6);
  cudaMemcpy(h_out.data(), d_out, 6 * sizeof(float), cudaMemcpyDeviceToHost);

  // Row 0 = table[1] = {4, 5, 6}
  EXPECT_NEAR(h_out[0], 4.0f, 1e-6f);
  EXPECT_NEAR(h_out[1], 5.0f, 1e-6f);
  EXPECT_NEAR(h_out[2], 6.0f, 1e-6f);

  // Row 1 = table[3] = {10, 11, 12}
  EXPECT_NEAR(h_out[3], 10.0f, 1e-6f);

  cudaFree(d_table); cudaFree(d_idx); cudaFree(d_out);
}
