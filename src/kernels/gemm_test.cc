// Tier B — GPU unit tests for GEMM kernels.
// Tests validation logic; actual GEMM execution pending kernel wiring.

#include "gtest/gtest.h"
#include "src/kernels/gemm.h"

#include <cuda_runtime.h>

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
