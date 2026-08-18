// Tier C — Integration: attention forward GPU test.

#include "gtest/gtest.h"
#include "src/attention/attention.h"
#include "tests/test_utils.h"

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

using namespace gpt;
using namespace gpt::attention;

namespace {

void FreeAll(std::vector<float*> ptrs) {
  for (float* ptr : ptrs) {
    cudaFree(ptr);
  }
}

}  // namespace

TEST(AttentionIntegrationTest, ForwardProducesCausalProbabilityRows) {
  const int64_t B = 1;
  const int64_t T = 4;
  const int64_t D = 8;
  const int64_t H = 2;
  const int64_t Dh = 4;

  auto stream_result = CudaStream::Create();
  ASSERT_TRUE(stream_result.ok());
  CudaStream stream = stream_result.take();

  test::SimpleRng rng(99);
  std::vector<float> h_X(B * T * D);
  std::vector<float> h_W_qkv(3 * D * D);
  std::vector<float> h_b_qkv(3 * D);
  std::vector<float> h_W_o(D * D);
  std::vector<float> h_b_o(D);
  rng.fill(h_X, 0.5f);
  rng.fill(h_W_qkv, 0.2f);
  rng.fill(h_b_qkv, 0.05f);
  rng.fill(h_W_o, 0.2f);
  rng.fill(h_b_o, 0.05f);

  float* d_X = nullptr;
  float* d_out = nullptr;
  float* d_W_qkv = nullptr;
  float* d_b_qkv = nullptr;
  float* d_W_o = nullptr;
  float* d_b_o = nullptr;
  cudaMalloc(&d_X, B * T * D * sizeof(float));
  cudaMalloc(&d_out, B * T * D * sizeof(float));
  cudaMalloc(&d_W_qkv, 3 * D * D * sizeof(float));
  cudaMalloc(&d_b_qkv, 3 * D * sizeof(float));
  cudaMalloc(&d_W_o, D * D * sizeof(float));
  cudaMalloc(&d_b_o, D * sizeof(float));

  cudaMemcpy(d_X, h_X.data(), h_X.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_W_qkv, h_W_qkv.data(), h_W_qkv.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_qkv, h_b_qkv.data(), h_b_qkv.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_W_o, h_W_o.data(), h_W_o.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_o, h_b_o.data(), h_b_o.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  AttentionConfig config;
  config.d_model = D;
  config.n_heads = H;
  config.head_dim = Dh;
  config.use_bias = true;
  config.causal = true;

  AttentionParams params;
  params.W_qkv = Tensor2D<float>{d_W_qkv, {3 * D, D}, {D, 1}};
  params.b_qkv = Tensor1D<float>{d_b_qkv, {3 * D}, {1}};
  params.W_o = Tensor2D<float>{d_W_o, {D, D}, {D, 1}};
  params.b_o = Tensor1D<float>{d_b_o, {D}, {1}};

  Tensor3D<const float> X{d_X, {B, T, D}, {T * D, D, 1}};
  Tensor3D<float> output{d_out, {B, T, D}, {T * D, D, 1}};
  AttentionWorkspace workspace;
  auto status = attention_forward(X, config, params, output, workspace, stream);
  ASSERT_TRUE(status.ok()) << status.message();
  ASSERT_TRUE(stream.synchronize().ok());

  std::vector<float> h_output(B * T * D);
  std::vector<float> h_probs(B * H * T * T);
  cudaMemcpy(h_output.data(), d_out, h_output.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_probs.data(), workspace.state().probs.data,
             h_probs.size() * sizeof(float), cudaMemcpyDeviceToHost);

  for (float value : h_output) {
    EXPECT_TRUE(std::isfinite(value));
  }
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t q = 0; q < T; ++q) {
        float row_sum = 0.0f;
        for (int64_t k = 0; k < T; ++k) {
          float p = h_probs[((b * H + h) * T + q) * T + k];
          EXPECT_TRUE(std::isfinite(p));
          if (k > q) {
            EXPECT_NEAR(p, 0.0f, 1e-6f)
                << "b=" << b << " h=" << h << " q=" << q << " k=" << k;
          } else {
            EXPECT_GE(p, 0.0f);
          }
          row_sum += p;
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-5f)
            << "b=" << b << " h=" << h << " q=" << q;
      }
    }
  }

  FreeAll({d_X, d_out, d_W_qkv, d_b_qkv, d_W_o, d_b_o});
}
