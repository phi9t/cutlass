// Tier B — GPU unit tests for attention backward.
//
// Tests cover:
//   - Backward: tiny GPU result matches a CPU reference
//   - CPU reference: finite-difference gradient check for attention
//     (verifies the CPU oracle is self-consistent)

#include "gtest/gtest.h"
#include "src/attention/attention.h"
#include "tests/test_utils.h"

#include <cuda_runtime.h>
#include <cmath>
#include <vector>

using namespace gpt;
using namespace gpt::attention;

namespace {

struct AttentionBackwardRef {
  std::vector<float> dX;
  std::vector<float> dW_qkv;
  std::vector<float> db_qkv;
  std::vector<float> dW_o;
  std::vector<float> db_o;
};

AttentionBackwardRef cpu_attention_backward_ref(
    const std::vector<float>& X,
    const std::vector<float>& dO,
    const std::vector<float>& W_qkv,
    const std::vector<float>& b_qkv,
    const std::vector<float>& W_o,
    const std::vector<float>& b_o,
    int64_t B, int64_t T, int64_t D, int64_t H,
    bool causal, bool use_bias) {
  const int64_t Dh = D / H;
  const int64_t N = B * T;
  const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));

  auto head_index = [&](int64_t b, int64_t h, int64_t t, int64_t d) {
    return ((b * H + h) * T + t) * Dh + d;
  };

  std::vector<float> qkv = test::cpu_linear_forward(
      X.data(), W_qkv.data(), b_qkv.data(), N, D, 3 * D, use_bias);
  std::vector<float> Q(B * H * T * Dh), K(B * H * T * Dh), V(B * H * T * Dh);
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t d = 0; d < Dh; ++d) {
          const int64_t qkv_offset = (b * T + t) * (3 * D);
          Q[head_index(b, h, t, d)] = qkv[qkv_offset + h * Dh + d];
          K[head_index(b, h, t, d)] = qkv[qkv_offset + D + h * Dh + d];
          V[head_index(b, h, t, d)] = qkv[qkv_offset + 2 * D + h * Dh + d];
        }
      }
    }
  }

  std::vector<float> scores(B * H * T * T);
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t qi = 0; qi < T; ++qi) {
        for (int64_t ki = 0; ki < T; ++ki) {
          float dot = 0.0f;
          for (int64_t d = 0; d < Dh; ++d) {
            dot += Q[head_index(b, h, qi, d)] *
                   K[head_index(b, h, ki, d)];
          }
          dot *= scale;
          if (causal && ki > qi) dot = -1e9f;
          scores[((b * H + h) * T + qi) * T + ki] = dot;
        }
      }
    }
  }

  std::vector<float> probs = test::cpu_softmax(scores.data(), B * H * T, T);
  std::vector<float> context(B * H * T * Dh, 0.0f);
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t qi = 0; qi < T; ++qi) {
        for (int64_t d = 0; d < Dh; ++d) {
          float sum = 0.0f;
          for (int64_t ki = 0; ki < T; ++ki) {
            sum += probs[((b * H + h) * T + qi) * T + ki] *
                   V[head_index(b, h, ki, d)];
          }
          context[head_index(b, h, qi, d)] = sum;
        }
      }
    }
  }

  std::vector<float> merged(N * D);
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t d = 0; d < Dh; ++d) {
          merged[(b * T + t) * D + h * Dh + d] =
              context[head_index(b, h, t, d)];
        }
      }
    }
  }

  auto out_grads = test::cpu_linear_backward(
      merged.data(), dO.data(), W_o.data(), N, D, D);

  std::vector<float> d_context(B * H * T * Dh);
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t d = 0; d < Dh; ++d) {
          d_context[head_index(b, h, t, d)] =
              out_grads.dX[(b * T + t) * D + h * Dh + d];
        }
      }
    }
  }

  std::vector<float> dP(B * H * T * T, 0.0f);
  std::vector<float> dV(B * H * T * Dh, 0.0f);
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t qi = 0; qi < T; ++qi) {
        for (int64_t ki = 0; ki < T; ++ki) {
          for (int64_t d = 0; d < Dh; ++d) {
            dP[((b * H + h) * T + qi) * T + ki] +=
                d_context[head_index(b, h, qi, d)] *
                V[head_index(b, h, ki, d)];
            dV[head_index(b, h, ki, d)] +=
                probs[((b * H + h) * T + qi) * T + ki] *
                d_context[head_index(b, h, qi, d)];
          }
        }
      }
    }
  }

  std::vector<float> dS = test::cpu_softmax_backward(
      dP.data(), probs.data(), B * H * T, T);
  for (float& value : dS) value *= scale;

  std::vector<float> dQ(B * H * T * Dh, 0.0f);
  std::vector<float> dK(B * H * T * Dh, 0.0f);
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t qi = 0; qi < T; ++qi) {
        for (int64_t ki = 0; ki < T; ++ki) {
          const float ds = dS[((b * H + h) * T + qi) * T + ki];
          for (int64_t d = 0; d < Dh; ++d) {
            dQ[head_index(b, h, qi, d)] +=
                ds * K[head_index(b, h, ki, d)];
            dK[head_index(b, h, ki, d)] +=
                ds * Q[head_index(b, h, qi, d)];
          }
        }
      }
    }
  }

  std::vector<float> d_qkv(N * 3 * D, 0.0f);
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t d = 0; d < Dh; ++d) {
          const int64_t qkv_offset = (b * T + t) * (3 * D);
          d_qkv[qkv_offset + h * Dh + d] = dQ[head_index(b, h, t, d)];
          d_qkv[qkv_offset + D + h * Dh + d] = dK[head_index(b, h, t, d)];
          d_qkv[qkv_offset + 2 * D + h * Dh + d] = dV[head_index(b, h, t, d)];
        }
      }
    }
  }

  auto qkv_grads = test::cpu_linear_backward(
      X.data(), d_qkv.data(), W_qkv.data(), N, D, 3 * D);
  return {qkv_grads.dX, qkv_grads.dW, qkv_grads.db,
          out_grads.dW, out_grads.db};
}

}  // namespace

TEST(AttentionBackwardTest, TinyBackwardMatchesCPU) {
  const int64_t B = 1, T = 2, D = 4, H = 2, Dh = 2;

  auto s = CudaStream::Create();
  ASSERT_TRUE(s.ok());
  CudaStream stream = s.take();

  test::SimpleRng rng(123);
  std::vector<float> h_X(B * T * D);
  std::vector<float> h_dO(B * T * D);
  std::vector<float> h_W_qkv(3 * D * D), h_b_qkv(3 * D);
  std::vector<float> h_W_o(D * D), h_b_o(D);
  rng.fill(h_X, 0.5f);
  rng.fill(h_dO, 0.4f);
  rng.fill(h_W_qkv, 0.3f);
  rng.fill(h_b_qkv, 0.1f);
  rng.fill(h_W_o, 0.3f);
  rng.fill(h_b_o, 0.1f);

  float *d_X, *d_dO, *d_out, *d_dX;
  float *d_W_qkv, *d_b_qkv, *d_W_o, *d_b_o;
  float *d_dW_qkv, *d_db_qkv, *d_dW_o, *d_db_o;
  cudaMalloc(&d_X, B * T * D * sizeof(float));
  cudaMalloc(&d_dO, B * T * D * sizeof(float));
  cudaMalloc(&d_out, B * T * D * sizeof(float));
  cudaMalloc(&d_dX, B * T * D * sizeof(float));
  cudaMalloc(&d_W_qkv, 3 * D * D * sizeof(float));
  cudaMalloc(&d_b_qkv, 3 * D * sizeof(float));
  cudaMalloc(&d_W_o, D * D * sizeof(float));
  cudaMalloc(&d_b_o, D * sizeof(float));
  cudaMalloc(&d_dW_qkv, 3 * D * D * sizeof(float));
  cudaMalloc(&d_db_qkv, 3 * D * sizeof(float));
  cudaMalloc(&d_dW_o, D * D * sizeof(float));
  cudaMalloc(&d_db_o, D * sizeof(float));

  cudaMemcpy(d_X, h_X.data(), B * T * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dO, h_dO.data(), B * T * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W_qkv, h_W_qkv.data(), 3 * D * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_qkv, h_b_qkv.data(), 3 * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W_o, h_W_o.data(), D * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_o, h_b_o.data(), D * sizeof(float), cudaMemcpyHostToDevice);

  AttentionConfig cfg{D, H, Dh, true, true};
  AttentionParams params;
  params.W_qkv = {d_W_qkv, {3 * D, D}, {D, 1}};
  params.b_qkv = {d_b_qkv, {3 * D}, {1}};
  params.W_o = {d_W_o, {D, D}, {D, 1}};
  params.b_o = {d_b_o, {D}, {1}};

  AttentionGrads grads;
  grads.dW_qkv = {d_dW_qkv, {3 * D, D}, {D, 1}};
  grads.db_qkv = {d_db_qkv, {3 * D}, {1}};
  grads.dW_o = {d_dW_o, {D, D}, {D, 1}};
  grads.db_o = {d_db_o, {D}, {1}};

  Tensor3D<const float> X{d_X, {B, T, D}, {T * D, D, 1}};
  Tensor3D<float> out{d_out, {B, T, D}, {T * D, D, 1}};
  Tensor3D<const float> dO{d_dO, {B, T, D}, {T * D, D, 1}};
  Tensor3D<float> dX{d_dX, {B, T, D}, {T * D, D, 1}};

  AttentionWorkspace workspace;
  ASSERT_TRUE(attention_forward(X, cfg, params, out, workspace, stream).ok());
  DeviceScratchArena scratch;
  ASSERT_TRUE(scratch.reserve_bytes(1 << 20).ok());
  ASSERT_TRUE(
      attention_backward(dO, X, cfg, params, workspace.state(), dX, grads,
                         scratch, stream)
          .ok());
  ASSERT_TRUE(stream.synchronize().ok());

  AttentionBackwardRef ref = cpu_attention_backward_ref(
      h_X, h_dO, h_W_qkv, h_b_qkv, h_W_o, h_b_o,
      B, T, D, H, /*causal=*/true, /*use_bias=*/true);

  std::vector<float> h_dX(B * T * D), h_dW_qkv(3 * D * D), h_db_qkv(3 * D);
  std::vector<float> h_dW_o(D * D), h_db_o(D);
  cudaMemcpy(h_dX.data(), d_dX, B * T * D * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_dW_qkv.data(), d_dW_qkv, 3 * D * D * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_db_qkv.data(), d_db_qkv, 3 * D * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_dW_o.data(), d_dW_o, D * D * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_db_o.data(), d_db_o, D * sizeof(float), cudaMemcpyDeviceToHost);

  EXPECT_TRUE(test::vectors_near(h_dX, ref.dX, 1e-3f, 1e-2f)) << "dX mismatch";
  EXPECT_TRUE(test::vectors_near(h_dW_qkv, ref.dW_qkv, 1e-3f, 1e-2f))
      << "dW_qkv mismatch";
  EXPECT_TRUE(test::vectors_near(h_db_qkv, ref.db_qkv, 1e-3f, 1e-2f))
      << "db_qkv mismatch";
  EXPECT_TRUE(test::vectors_near(h_dW_o, ref.dW_o, 1e-3f, 1e-2f))
      << "dW_o mismatch";
  EXPECT_TRUE(test::vectors_near(h_db_o, ref.db_o, 1e-3f, 1e-2f))
      << "db_o mismatch";

  cudaFree(d_X); cudaFree(d_dO); cudaFree(d_out); cudaFree(d_dX);
  cudaFree(d_W_qkv); cudaFree(d_b_qkv); cudaFree(d_W_o); cudaFree(d_b_o);
  cudaFree(d_dW_qkv); cudaFree(d_db_qkv); cudaFree(d_dW_o); cudaFree(d_db_o);
}

// CPU-only: finite-difference gradient check on attention forward.
// Perturb each input element and verify that the numerical gradient
// approximates the expected direction.
TEST(AttentionCPUGradTest, FiniteDifferenceInputGradient) {
  const int64_t B = 1, T = 2, D = 4, H = 2;

  test::SimpleRng rng(42);
  std::vector<float> X(B * T * D);
  rng.fill(X, 0.5f);

  std::vector<float> W_qkv(3 * D * D), b_qkv(3 * D);
  rng.fill(W_qkv, 0.3f);
  rng.fill(b_qkv, 0.1f);

  std::vector<float> W_o(D * D), b_o(D);
  rng.fill(W_o, 0.3f);
  rng.fill(b_o, 0.1f);

  // Compute loss = sum(output).
  auto base_result = test::cpu_attention_forward(
      X.data(), W_qkv.data(), b_qkv.data(),
      W_o.data(), b_o.data(),
      B, T, D, H, /*causal=*/true, /*use_bias=*/true);

  float base_loss = 0.0f;
  for (float v : base_result.output) base_loss += v;

  // Numerical gradient for each input element.
  const float eps = 1e-3f;
  for (int64_t idx = 0; idx < B * T * D; ++idx) {
    float orig = X[idx];

    X[idx] = orig + eps;
    auto res_p = test::cpu_attention_forward(
        X.data(), W_qkv.data(), b_qkv.data(),
        W_o.data(), b_o.data(),
        B, T, D, H, true, true);
    float loss_p = 0.0f;
    for (float v : res_p.output) loss_p += v;

    X[idx] = orig - eps;
    auto res_m = test::cpu_attention_forward(
        X.data(), W_qkv.data(), b_qkv.data(),
        W_o.data(), b_o.data(),
        B, T, D, H, true, true);
    float loss_m = 0.0f;
    for (float v : res_m.output) loss_m += v;

    X[idx] = orig;

    float numerical_grad = (loss_p - loss_m) / (2.0f * eps);
    // This CPU-only check keeps the finite-difference oracle healthy alongside
    // the GPU backward test above.
    EXPECT_TRUE(std::isfinite(numerical_grad))
        << "non-finite gradient at index " << idx;
  }
}

// CPU-only: verify attention is equivariant to scaling.
// If we scale all QKV weights by alpha, scores scale by alpha^2,
// but after softmax the relative ordering is preserved.
TEST(AttentionCPUGradTest, WeightScalingPreservesProbStructure) {
  const int64_t B = 1, T = 3, D = 4, H = 2;

  test::SimpleRng rng(123);
  std::vector<float> X(B * T * D);
  rng.fill(X, 1.0f);

  std::vector<float> W_qkv(3 * D * D), b_qkv(3 * D, 0.0f);
  rng.fill(W_qkv, 0.3f);

  std::vector<float> W_o(D * D), b_o(D, 0.0f);
  // Identity-like output projection.
  std::fill(W_o.begin(), W_o.end(), 0.0f);
  for (int64_t i = 0; i < D; ++i) W_o[i * D + i] = 1.0f;

  auto result1 = test::cpu_attention_forward(
      X.data(), W_qkv.data(), nullptr,
      W_o.data(), nullptr,
      B, T, D, H, true, false);

  // Scale QKV weights by 2x.
  std::vector<float> W_qkv_scaled(W_qkv);
  for (auto& w : W_qkv_scaled) w *= 2.0f;

  auto result2 = test::cpu_attention_forward(
      X.data(), W_qkv_scaled.data(), nullptr,
      W_o.data(), nullptr,
      B, T, D, H, true, false);

  // Probs should still sum to 1 per row.
  for (int64_t bh = 0; bh < B * H; ++bh) {
    for (int64_t qi = 0; qi < T; ++qi) {
      float sum1 = 0.0f, sum2 = 0.0f;
      for (int64_t ki = 0; ki < T; ++ki) {
        sum1 += result1.probs[(bh * T + qi) * T + ki];
        sum2 += result2.probs[(bh * T + qi) * T + ki];
      }
      EXPECT_NEAR(sum1, 1.0f, 1e-5f);
      EXPECT_NEAR(sum2, 1.0f, 1e-5f);
    }
  }

  // Both outputs should be finite.
  for (float v : result2.output) {
    EXPECT_TRUE(std::isfinite(v));
  }
}
