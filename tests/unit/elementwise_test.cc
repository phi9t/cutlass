// Tier B — GPU unit tests for elementwise kernels.

#include "gtest/gtest.h"
#include "src/kernels/elementwise.h"
#include "tests/test_utils.h"

#include <cuda_runtime.h>
#include <cmath>
#include <vector>

using namespace gpt;
using namespace gpt::kernels;

class ElementwiseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto s = CudaStream::Create();
    ASSERT_TRUE(s.ok());
    stream_ = s.take();
  }

  CudaStream stream_;
};

TEST_F(ElementwiseTest, VecAddCorrect) {
  const int64_t n = 128;
  std::vector<float> h_a(n, 1.0f), h_b(n, 2.0f);

  float *d_a, *d_b, *d_out;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));
  cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  Tensor1D<const float> a{d_a, {n}, {1}};
  Tensor1D<const float> b{d_b, {n}, {1}};
  Tensor1D<float> out{d_out, {n}, {1}};

  ASSERT_TRUE(vec_add_f32(a, b, out, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_out(n);
  cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(h_out[i], 3.0f, 1e-6f);
  }

  cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
}

TEST_F(ElementwiseTest, VecScaleCorrect) {
  const int64_t n = 64;
  std::vector<float> h_x(n, 4.0f);

  float *d_x, *d_out;
  cudaMalloc(&d_x, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));
  cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  Tensor1D<const float> x{d_x, {n}, {1}};
  Tensor1D<float> out{d_out, {n}, {1}};

  ASSERT_TRUE(vec_scale_f32(x, 0.5f, out, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_out(n);
  cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(h_out[i], 2.0f, 1e-6f);
  }

  cudaFree(d_x); cudaFree(d_out);
}

TEST_F(ElementwiseTest, MaskedFillCorrect) {
  const int64_t n = 8;
  std::vector<float> h_x = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int8_t> h_mask = {0, 1, 0, 1, 0, 0, 1, 0};

  float* d_x; int8_t* d_mask; float* d_out;
  cudaMalloc(&d_x, n * sizeof(float));
  cudaMalloc(&d_mask, n * sizeof(int8_t));
  cudaMalloc(&d_out, n * sizeof(float));
  cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask, h_mask.data(), n * sizeof(int8_t), cudaMemcpyHostToDevice);

  Tensor1D<const float> x{d_x, {n}, {1}};
  Tensor1D<const int8_t> mask{d_mask, {n}, {1}};
  Tensor1D<float> out{d_out, {n}, {1}};

  ASSERT_TRUE(masked_fill_f32(x, mask, -999.0f, out, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_out(n);
  cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  EXPECT_NEAR(h_out[0], 1.0f, 1e-6f);
  EXPECT_NEAR(h_out[1], -999.0f, 1e-6f);
  EXPECT_NEAR(h_out[2], 3.0f, 1e-6f);
  EXPECT_NEAR(h_out[3], -999.0f, 1e-6f);

  cudaFree(d_x); cudaFree(d_mask); cudaFree(d_out);
}

TEST_F(ElementwiseTest, GatherCorrect) {
  // table: [4, 3], indices: [3], out: [3, 3]
  std::vector<float> h_table = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    10, 11, 12
  };
  std::vector<int32_t> h_idx = {2, 0, 3};

  float* d_table; int32_t* d_idx; float* d_out;
  cudaMalloc(&d_table, 12 * sizeof(float));
  cudaMalloc(&d_idx, 3 * sizeof(int32_t));
  cudaMalloc(&d_out, 9 * sizeof(float));
  cudaMemcpy(d_table, h_table.data(), 12 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_idx, h_idx.data(), 3 * sizeof(int32_t), cudaMemcpyHostToDevice);

  Tensor2D<const float> table{d_table, {4, 3}, {3, 1}};
  Tensor1D<const int32_t> indices{d_idx, {3}, {1}};
  Tensor2D<float> out{d_out, {3, 3}, {3, 1}};

  ASSERT_TRUE(gather_f32(table, indices, out, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_out(9);
  cudaMemcpy(h_out.data(), d_out, 9 * sizeof(float), cudaMemcpyDeviceToHost);

  // Row 0 of out should be table[2] = {7, 8, 9}
  EXPECT_NEAR(h_out[0], 7.0f, 1e-6f);
  EXPECT_NEAR(h_out[1], 8.0f, 1e-6f);
  EXPECT_NEAR(h_out[2], 9.0f, 1e-6f);
  // Row 2 of out should be table[3] = {10, 11, 12}
  EXPECT_NEAR(h_out[6], 10.0f, 1e-6f);

  cudaFree(d_table); cudaFree(d_idx); cudaFree(d_out);
}
