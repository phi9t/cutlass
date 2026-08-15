// Tier C — Integration: cross-kernel composition tests.
//
// Verifies that primitives compose correctly, e.g.:
//   transpose → gemm → reduction pipeline.

#include "gtest/gtest.h"
#include "src/kernels/layout_kernels.h"
#include "src/kernels/reductions.h"
#include "tests/test_utils.h"

#include <cuda_runtime.h>
#include <vector>

using namespace gpt;

class KernelIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto s = CudaStream::Create();
    ASSERT_TRUE(s.ok());
    stream_ = s.take();
  }
  CudaStream stream_;
};

TEST_F(KernelIntegrationTest, SplitMergeHeadsRoundTrip) {
  // [B=1, T=4, D=8] → split_heads(H=2) → [1, 2, 4, 4] → merge_heads → [1, 4, 8]
  const int64_t B = 1, T = 4, D = 8, H = 2, Dh = 4;
  std::vector<float> h_input(B * T * D);
  for (int i = 0; i < B * T * D; ++i) h_input[i] = static_cast<float>(i);

  float *d_input, *d_split, *d_merged;
  cudaMalloc(&d_input, B * T * D * sizeof(float));
  cudaMalloc(&d_split, B * H * T * Dh * sizeof(float));
  cudaMalloc(&d_merged, B * T * D * sizeof(float));
  cudaMemcpy(d_input, h_input.data(), B * T * D * sizeof(float),
             cudaMemcpyHostToDevice);

  Tensor3D<const float> input{d_input, {B, T, D}, {T * D, D, 1}};
  Tensor4D<float> split{d_split, {B, H, T, Dh}, {H * T * Dh, T * Dh, Dh, 1}};
  Tensor3D<float> merged{d_merged, {B, T, D}, {T * D, D, 1}};

  ASSERT_TRUE(kernels::split_heads_f32(input, split, H, stream_).ok());
  Tensor4D<const float> split_c{d_split, {B, H, T, Dh}, {H * T * Dh, T * Dh, Dh, 1}};
  ASSERT_TRUE(kernels::merge_heads_f32(split_c, merged, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_merged(B * T * D);
  cudaMemcpy(h_merged.data(), d_merged, B * T * D * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Round trip should be identity.
  EXPECT_TRUE(test::vectors_near(h_merged, h_input, 1e-6f));

  cudaFree(d_input); cudaFree(d_split); cudaFree(d_merged);
}

TEST_F(KernelIntegrationTest, Transpose2DInverseRoundTrip) {
  const int64_t M = 4, N = 6;
  std::vector<float> h_input(M * N);
  for (int i = 0; i < M * N; ++i) h_input[i] = static_cast<float>(i);

  float *d_input, *d_transposed, *d_back;
  cudaMalloc(&d_input, M * N * sizeof(float));
  cudaMalloc(&d_transposed, N * M * sizeof(float));
  cudaMalloc(&d_back, M * N * sizeof(float));
  cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

  Tensor2D<const float> input{d_input, {M, N}, {N, 1}};
  Tensor2D<float> trans{d_transposed, {N, M}, {M, 1}};
  Tensor2D<const float> trans_c{d_transposed, {N, M}, {M, 1}};
  Tensor2D<float> back{d_back, {M, N}, {N, 1}};

  ASSERT_TRUE(kernels::transpose_2d_f32(input, trans, stream_).ok());
  ASSERT_TRUE(kernels::transpose_2d_f32(trans_c, back, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_back(M * N);
  cudaMemcpy(h_back.data(), d_back, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  EXPECT_TRUE(test::vectors_near(h_back, h_input, 1e-6f));

  cudaFree(d_input); cudaFree(d_transposed); cudaFree(d_back);
}

TEST_F(KernelIntegrationTest, RepackContiguousCopiesStridedRows) {
  const int64_t M = 3, N = 4, pitch = 6;
  std::vector<float> h_input(M * pitch, -1000.0f);
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      h_input[i * pitch + j] = static_cast<float>(i * 10 + j);
    }
  }
  std::vector<float> h_expected{
      0.0f, 1.0f, 2.0f, 3.0f,
      10.0f, 11.0f, 12.0f, 13.0f,
      20.0f, 21.0f, 22.0f, 23.0f,
  };

  float *d_input, *d_output;
  cudaMalloc(&d_input, h_input.size() * sizeof(float));
  cudaMalloc(&d_output, M * N * sizeof(float));
  cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  Tensor2D<const float> input{d_input, {M, N}, {pitch, 1}};
  Tensor2D<float> output{d_output, {M, N}, {N, 1}};

  ASSERT_TRUE(kernels::repack_contiguous_f32(input, output, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_output(M * N);
  cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float),
             cudaMemcpyDeviceToHost);

  EXPECT_TRUE(test::vectors_near(h_output, h_expected, 1e-6f));

  cudaFree(d_input);
  cudaFree(d_output);
}
