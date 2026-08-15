// Tier B — GPU unit tests for GEMM kernels.
// Tests validation logic; actual GEMM execution pending kernel wiring.

#include "gtest/gtest.h"
#include "src/kernels/gemm.h"
#include "tests/test_utils.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector>

using namespace gpt;
using namespace gpt::kernels;

class GemmTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto s = CudaStream::Create();
    ASSERT_TRUE(s.ok());
    stream_ = s.take();
  }
  CudaStream stream_;
};

TEST_F(GemmTest, ShapeMismatchReturnsError) {
  float* d_buf;
  cudaMalloc(&d_buf, 256 * sizeof(float));

  // A: [4, 8], B: [16, 4] → K-dim mismatch (8 != 16)
  Tensor2D<float> A{d_buf, {4, 8}, {8, 1}};
  Tensor2D<float> B{d_buf, {16, 4}, {4, 1}};
  Tensor2D<float> C{d_buf, {4, 4}, {4, 1}};

  auto status = gemm_f32(A, B, C, 1.0f, 0.0f, stream_);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);

  cudaFree(d_buf);
}

TEST_F(GemmTest, ValidShapeAccepted) {
  float* d_buf;
  cudaMalloc(&d_buf, 256 * sizeof(float));

  // A: [4, 8], B: [8, 4], C: [4, 4] → valid
  Tensor2D<float> A{d_buf, {4, 8}, {8, 1}};
  Tensor2D<float> B{d_buf, {8, 4}, {4, 1}};
  Tensor2D<float> C{d_buf, {4, 4}, {4, 1}};

  auto status = gemm_f32(A, B, C, 1.0f, 0.0f, stream_);
  // Will return kNotImplemented (kernel not wired yet) but NOT kInvalidArgument.
  EXPECT_NE(status.code(), StatusCode::kInvalidArgument);

  cudaFree(d_buf);
}

TEST_F(GemmTest, Bf16GemmMatchesCPUReference) {
  const int64_t M = 4, K = 5, N = 3;
  std::vector<float> h_a = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
      -1.0f, 0.5f, 2.0f, -0.5f, 1.5f,
      3.0f, -2.0f, 1.0f, 0.0f, 2.5f,
      0.25f, 1.25f, -1.5f, 2.0f, -0.75f,
  };
  std::vector<float> h_b = {
      1.0f, -2.0f, 0.5f,
      0.0f, 1.5f, -1.0f,
      -0.5f, 2.0f, 1.0f,
      2.0f, -1.0f, 0.0f,
      1.5f, 0.5f, -2.0f,
  };
  std::vector<__nv_bfloat16> h_a_bf16(M * K);
  std::vector<__nv_bfloat16> h_b_bf16(K * N);
  std::vector<__nv_bfloat16> h_c_bf16(M * N);
  for (int64_t i = 0; i < M * K; ++i) h_a_bf16[i] = __float2bfloat16(h_a[i]);
  for (int64_t i = 0; i < K * N; ++i) h_b_bf16[i] = __float2bfloat16(h_b[i]);
  for (int64_t i = 0; i < M * N; ++i) h_c_bf16[i] = __float2bfloat16(0.0f);

  __nv_bfloat16 *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, h_a_bf16.size() * sizeof(__nv_bfloat16));
  cudaMalloc(&d_b, h_b_bf16.size() * sizeof(__nv_bfloat16));
  cudaMalloc(&d_c, h_c_bf16.size() * sizeof(__nv_bfloat16));
  cudaMemcpy(d_a, h_a_bf16.data(), h_a_bf16.size() * sizeof(__nv_bfloat16),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b_bf16.data(), h_b_bf16.size() * sizeof(__nv_bfloat16),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c_bf16.data(), h_c_bf16.size() * sizeof(__nv_bfloat16),
             cudaMemcpyHostToDevice);

  Tensor2D<__nv_bfloat16> A{d_a, {M, K}, {K, 1}};
  Tensor2D<__nv_bfloat16> B{d_b, {K, N}, {N, 1}};
  Tensor2D<__nv_bfloat16> C{d_c, {M, N}, {N, 1}};

  auto status = gemm_bf16(A, B, C, 1.0f, 0.0f, stream_);
  ASSERT_TRUE(status.ok()) << status.message();
  stream_.synchronize();

  std::vector<__nv_bfloat16> h_out_bf16(M * N);
  cudaMemcpy(h_out_bf16.data(), d_c, h_out_bf16.size() * sizeof(__nv_bfloat16),
             cudaMemcpyDeviceToHost);

  std::vector<float> h_output(M * N);
  for (int64_t i = 0; i < M * N; ++i) h_output[i] = __bfloat162float(h_out_bf16[i]);

  std::vector<float> expected(M * N, 0.0f);
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t k = 0; k < K; ++k) {
        expected[m * N + n] += h_a[m * K + k] * h_b[k * N + n];
      }
    }
  }

  EXPECT_TRUE(test::vectors_near(h_output, expected, 4e-2f, 1e-3f));

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

TEST_F(GemmTest, Bf16GemmAcceptsColumnMajorOperandView) {
  const int64_t M = 3, K = 4, N = 2;
  std::vector<float> h_a = {
      1.0f, -2.0f, 3.0f, 0.5f,
      2.0f, 1.0f, -1.0f, 4.0f,
      -0.5f, 3.0f, 2.0f, -1.5f,
  };
  std::vector<float> h_b_logical = {
      1.0f, -1.0f,
      0.5f, 2.0f,
      -2.0f, 1.5f,
      3.0f, -0.5f,
  };
  std::vector<__nv_bfloat16> h_a_bf16(M * K);
  std::vector<__nv_bfloat16> h_b_colmajor(K * N);
  std::vector<__nv_bfloat16> h_c_bf16(M * N);
  for (int64_t i = 0; i < M * K; ++i) h_a_bf16[i] = __float2bfloat16(h_a[i]);
  for (int64_t k = 0; k < K; ++k) {
    for (int64_t n = 0; n < N; ++n) {
      h_b_colmajor[k + n * K] = __float2bfloat16(h_b_logical[k * N + n]);
    }
  }
  for (int64_t i = 0; i < M * N; ++i) h_c_bf16[i] = __float2bfloat16(0.0f);

  __nv_bfloat16 *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, h_a_bf16.size() * sizeof(__nv_bfloat16));
  cudaMalloc(&d_b, h_b_colmajor.size() * sizeof(__nv_bfloat16));
  cudaMalloc(&d_c, h_c_bf16.size() * sizeof(__nv_bfloat16));
  cudaMemcpy(d_a, h_a_bf16.data(), h_a_bf16.size() * sizeof(__nv_bfloat16),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b_colmajor.data(), h_b_colmajor.size() * sizeof(__nv_bfloat16),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c_bf16.data(), h_c_bf16.size() * sizeof(__nv_bfloat16),
             cudaMemcpyHostToDevice);

  Tensor2D<__nv_bfloat16> A{d_a, {M, K}, {K, 1}};
  Tensor2D<__nv_bfloat16> B{d_b, {K, N}, {1, K}};
  Tensor2D<__nv_bfloat16> C{d_c, {M, N}, {N, 1}};

  auto status = gemm_bf16(A, B, C, 1.0f, 0.0f, stream_);
  ASSERT_TRUE(status.ok()) << status.message();
  stream_.synchronize();

  std::vector<__nv_bfloat16> h_out_bf16(M * N);
  cudaMemcpy(h_out_bf16.data(), d_c, h_out_bf16.size() * sizeof(__nv_bfloat16),
             cudaMemcpyDeviceToHost);

  std::vector<float> h_output(M * N);
  for (int64_t i = 0; i < M * N; ++i) h_output[i] = __bfloat162float(h_out_bf16[i]);

  std::vector<float> expected(M * N, 0.0f);
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t k = 0; k < K; ++k) {
        expected[m * N + n] += h_a[m * K + k] * h_b_logical[k * N + n];
      }
    }
  }

  EXPECT_TRUE(test::vectors_near(h_output, expected, 4e-2f, 1e-3f));

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
