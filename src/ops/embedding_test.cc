// Tier B — GPU unit tests for embedding ops.
//
// Tests cover:
//   - Forward: correct row gathering (manual check)
//   - Forward: CPU reference match with random data
//   - Forward: duplicate indices
//   - Backward: scatter-add matches CPU reference
//   - Backward: duplicate indices accumulate correctly

#include "gtest/gtest.h"
#include "src/ops/embedding.h"
#include "tests/test_utils.h"

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

// --- Forward tests ---

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
  EXPECT_NEAR(h_out[4], 11.0f, 1e-6f);
  EXPECT_NEAR(h_out[5], 12.0f, 1e-6f);

  cudaFree(d_table); cudaFree(d_idx); cudaFree(d_out);
}

TEST_F(EmbeddingTest, ForwardMatchesCPUReference) {
  const int64_t vocab = 8, d_model = 4, n = 5;
  test::SimpleRng rng(42);
  std::vector<float> h_table(vocab * d_model);
  rng.fill(h_table, 2.0f);
  std::vector<int32_t> h_idx = {0, 3, 7, 1, 3};

  float* d_table; int32_t* d_idx; float* d_out;
  cudaMalloc(&d_table, vocab * d_model * sizeof(float));
  cudaMalloc(&d_idx, n * sizeof(int32_t));
  cudaMalloc(&d_out, n * d_model * sizeof(float));
  cudaMemcpy(d_table, h_table.data(), vocab * d_model * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_idx, h_idx.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice);

  Tensor2D<const float> table{d_table, {vocab, d_model}, {d_model, 1}};
  Tensor1D<const int32_t> indices{d_idx, {n}, {1}};
  Tensor2D<float> output{d_out, {n, d_model}, {d_model, 1}};

  ASSERT_TRUE(embedding_forward(table, indices, output, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_out(n * d_model);
  cudaMemcpy(h_out.data(), d_out, n * d_model * sizeof(float),
             cudaMemcpyDeviceToHost);

  auto expected = test::cpu_embedding_forward(h_table.data(), h_idx.data(),
                                               n, d_model);
  EXPECT_TRUE(test::vectors_near(h_out, expected, 1e-6f));

  cudaFree(d_table); cudaFree(d_idx); cudaFree(d_out);
}

TEST_F(EmbeddingTest, ForwardDuplicateIndices) {
  // All indices point to the same row → all output rows should be identical.
  const int64_t vocab = 4, d_model = 3, n = 3;
  std::vector<float> h_table = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int32_t> h_idx = {2, 2, 2};

  float* d_table; int32_t* d_idx; float* d_out;
  cudaMalloc(&d_table, vocab * d_model * sizeof(float));
  cudaMalloc(&d_idx, n * sizeof(int32_t));
  cudaMalloc(&d_out, n * d_model * sizeof(float));
  cudaMemcpy(d_table, h_table.data(), vocab * d_model * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_idx, h_idx.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice);

  Tensor2D<const float> table{d_table, {vocab, d_model}, {d_model, 1}};
  Tensor1D<const int32_t> indices{d_idx, {n}, {1}};
  Tensor2D<float> output{d_out, {n, d_model}, {d_model, 1}};

  ASSERT_TRUE(embedding_forward(table, indices, output, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_out(n * d_model);
  cudaMemcpy(h_out.data(), d_out, n * d_model * sizeof(float),
             cudaMemcpyDeviceToHost);

  // table[2] = {7, 8, 9}
  for (int64_t i = 0; i < n; ++i) {
    EXPECT_NEAR(h_out[i * d_model + 0], 7.0f, 1e-6f);
    EXPECT_NEAR(h_out[i * d_model + 1], 8.0f, 1e-6f);
    EXPECT_NEAR(h_out[i * d_model + 2], 9.0f, 1e-6f);
  }

  cudaFree(d_table); cudaFree(d_idx); cudaFree(d_out);
}

// --- Backward tests ---

TEST_F(EmbeddingTest, BackwardScatterAddMatchesCPU) {
  const int64_t vocab = 6, d_model = 4, n = 3;
  test::SimpleRng rng(42);

  std::vector<float> h_dY(n * d_model);
  rng.fill(h_dY, 1.0f);
  std::vector<int32_t> h_idx = {1, 4, 1};  // index 1 appears twice

  float* d_dY; int32_t* d_idx; float* d_dtable;
  cudaMalloc(&d_dY, n * d_model * sizeof(float));
  cudaMalloc(&d_idx, n * sizeof(int32_t));
  cudaMalloc(&d_dtable, vocab * d_model * sizeof(float));
  cudaMemcpy(d_dY, h_dY.data(), n * d_model * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_idx, h_idx.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemset(d_dtable, 0, vocab * d_model * sizeof(float));

  Tensor2D<const float> dY{d_dY, {n, d_model}, {d_model, 1}};
  Tensor1D<const int32_t> indices{d_idx, {n}, {1}};
  Tensor2D<float> d_table{d_dtable, {vocab, d_model}, {d_model, 1}};

  ASSERT_TRUE(embedding_backward(dY, indices, d_table, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_dtable(vocab * d_model);
  cudaMemcpy(h_dtable.data(), d_dtable, vocab * d_model * sizeof(float),
             cudaMemcpyDeviceToHost);

  auto expected = test::cpu_embedding_backward(h_dY.data(), h_idx.data(),
                                                n, d_model, vocab);
  EXPECT_TRUE(test::vectors_near(h_dtable, expected, 1e-5f));

  // Verify duplicate index accumulation: d_table[1] should be dY[0] + dY[2].
  for (int64_t j = 0; j < d_model; ++j) {
    float expect_j = h_dY[0 * d_model + j] + h_dY[2 * d_model + j];
    EXPECT_NEAR(h_dtable[1 * d_model + j], expect_j, 1e-5f);
  }

  // Unused rows should be zero.
  for (int64_t j = 0; j < d_model; ++j) {
    EXPECT_EQ(h_dtable[0 * d_model + j], 0.0f);  // index 0 unused
    EXPECT_EQ(h_dtable[2 * d_model + j], 0.0f);  // index 2 unused
    EXPECT_EQ(h_dtable[3 * d_model + j], 0.0f);  // index 3 unused
    EXPECT_EQ(h_dtable[5 * d_model + j], 0.0f);  // index 5 unused
  }

  cudaFree(d_dY); cudaFree(d_idx); cudaFree(d_dtable);
}

TEST_F(EmbeddingTest, BackwardAllSameIndex) {
  // All gradients go to the same row → should be sum of all dY rows.
  const int64_t vocab = 3, d_model = 2, n = 4;
  std::vector<float> h_dY = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<int32_t> h_idx = {0, 0, 0, 0};

  float* d_dY; int32_t* d_idx; float* d_dtable;
  cudaMalloc(&d_dY, n * d_model * sizeof(float));
  cudaMalloc(&d_idx, n * sizeof(int32_t));
  cudaMalloc(&d_dtable, vocab * d_model * sizeof(float));
  cudaMemcpy(d_dY, h_dY.data(), n * d_model * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_idx, h_idx.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemset(d_dtable, 0, vocab * d_model * sizeof(float));

  Tensor2D<const float> dY{d_dY, {n, d_model}, {d_model, 1}};
  Tensor1D<const int32_t> indices{d_idx, {n}, {1}};
  Tensor2D<float> d_table{d_dtable, {vocab, d_model}, {d_model, 1}};

  ASSERT_TRUE(embedding_backward(dY, indices, d_table, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_dtable(vocab * d_model);
  cudaMemcpy(h_dtable.data(), d_dtable, vocab * d_model * sizeof(float),
             cudaMemcpyDeviceToHost);

  // d_table[0] = sum of all 4 rows = {1+3+5+7, 2+4+6+8} = {16, 20}
  EXPECT_NEAR(h_dtable[0], 16.0f, 1e-5f);
  EXPECT_NEAR(h_dtable[1], 20.0f, 1e-5f);

  cudaFree(d_dY); cudaFree(d_idx); cudaFree(d_dtable);
}
