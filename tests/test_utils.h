#pragma once

// Shared test utilities: CPU reference oracles and comparison helpers.
//
// Every GPU op has a corresponding cpu_* reference here so that unit tests
// can compare CUDA results against a known-good C++ implementation.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace gpt {
namespace test {

// ---------------------------------------------------------------------------
// Comparison helpers.
// ---------------------------------------------------------------------------

inline bool near_equal(float a, float b, float atol = 1e-5f,
                        float rtol = 1e-4f) {
  return std::abs(a - b) <= atol + rtol * std::abs(b);
}

inline bool vectors_near(const std::vector<float>& a,
                          const std::vector<float>& b,
                          float atol = 1e-5f, float rtol = 1e-4f) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!near_equal(a[i], b[i], atol, rtol)) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// CPU reference: row reductions.
// ---------------------------------------------------------------------------

inline std::vector<float> cpu_row_max(const float* data, int64_t rows,
                                       int64_t cols) {
  std::vector<float> out(rows);
  for (int64_t i = 0; i < rows; ++i) {
    float mx = data[i * cols];
    for (int64_t j = 1; j < cols; ++j) {
      mx = std::max(mx, data[i * cols + j]);
    }
    out[i] = mx;
  }
  return out;
}

inline std::vector<float> cpu_row_sum(const float* data, int64_t rows,
                                       int64_t cols) {
  std::vector<float> out(rows);
  for (int64_t i = 0; i < rows; ++i) {
    float s = 0.0f;
    for (int64_t j = 0; j < cols; ++j) s += data[i * cols + j];
    out[i] = s;
  }
  return out;
}

inline std::vector<float> cpu_row_mean(const float* data, int64_t rows,
                                        int64_t cols) {
  auto sums = cpu_row_sum(data, rows, cols);
  for (auto& s : sums) s /= static_cast<float>(cols);
  return sums;
}

// ---------------------------------------------------------------------------
// CPU reference: GELU activation.
//   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ---------------------------------------------------------------------------

inline float cpu_gelu(float x) {
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  constexpr float kCoeff = 0.044715f;
  float inner = kSqrt2OverPi * (x + kCoeff * x * x * x);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

// GELU derivative: d/dx GELU(x).
// Using the chain rule on the tanh-approximation form.
inline float cpu_gelu_backward(float x) {
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  constexpr float kCoeff = 0.044715f;
  float x3 = x * x * x;
  float inner = kSqrt2OverPi * (x + kCoeff * x3);
  float tanh_inner = std::tanh(inner);
  float sech2 = 1.0f - tanh_inner * tanh_inner;
  float d_inner = kSqrt2OverPi * (1.0f + 3.0f * kCoeff * x * x);
  return 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * d_inner;
}

// ---------------------------------------------------------------------------
// CPU reference: softmax (row-wise).
// ---------------------------------------------------------------------------

inline std::vector<float> cpu_softmax(const float* data, int64_t rows,
                                       int64_t cols) {
  std::vector<float> out(rows * cols);
  for (int64_t i = 0; i < rows; ++i) {
    float mx = data[i * cols];
    for (int64_t j = 1; j < cols; ++j)
      mx = std::max(mx, data[i * cols + j]);

    float sum = 0.0f;
    for (int64_t j = 0; j < cols; ++j) {
      out[i * cols + j] = std::exp(data[i * cols + j] - mx);
      sum += out[i * cols + j];
    }
    for (int64_t j = 0; j < cols; ++j) {
      out[i * cols + j] /= sum;
    }
  }
  return out;
}

// Masked softmax: mask[i*cols+j] != 0 means position j is masked in row i.
// Masked positions get probability 0; valid positions share the row sum.
inline std::vector<float> cpu_masked_softmax(const float* data,
                                              const int8_t* mask,
                                              int64_t rows, int64_t cols) {
  if (mask == nullptr) return cpu_softmax(data, rows, cols);

  std::vector<float> out(rows * cols, 0.0f);
  for (int64_t i = 0; i < rows; ++i) {
    float mx = -1e30f;
    for (int64_t j = 0; j < cols; ++j) {
      if (mask[i * cols + j] == 0)
        mx = std::max(mx, data[i * cols + j]);
    }
    float sum = 0.0f;
    for (int64_t j = 0; j < cols; ++j) {
      if (mask[i * cols + j] == 0) {
        out[i * cols + j] = std::exp(data[i * cols + j] - mx);
        sum += out[i * cols + j];
      }
    }
    if (sum > 0.0f) {
      for (int64_t j = 0; j < cols; ++j) {
        out[i * cols + j] /= sum;
      }
    }
  }
  return out;
}

// Softmax backward: given dP (grad of output) and P (saved probs), compute dX.
//   dX[i,j] = P[i,j] * (dP[i,j] - sum_k(P[i,k] * dP[i,k]))
inline std::vector<float> cpu_softmax_backward(const float* dP, const float* P,
                                                int64_t rows, int64_t cols) {
  std::vector<float> dX(rows * cols);
  for (int64_t i = 0; i < rows; ++i) {
    float dot = 0.0f;
    for (int64_t j = 0; j < cols; ++j)
      dot += P[i * cols + j] * dP[i * cols + j];
    for (int64_t j = 0; j < cols; ++j)
      dX[i * cols + j] = P[i * cols + j] * (dP[i * cols + j] - dot);
  }
  return dX;
}

// ---------------------------------------------------------------------------
// CPU reference: layer normalization.
// ---------------------------------------------------------------------------

struct LayerNormResult {
  std::vector<float> y;        // [rows * D]
  std::vector<float> mean;     // [rows]
  std::vector<float> inv_std;  // [rows]
};

inline LayerNormResult cpu_layernorm_forward(const float* x,
                                              const float* gamma,
                                              const float* beta,
                                              float eps,
                                              int64_t rows, int64_t D) {
  LayerNormResult r;
  r.y.resize(rows * D);
  r.mean.resize(rows);
  r.inv_std.resize(rows);

  for (int64_t i = 0; i < rows; ++i) {
    // Compute mean.
    float mu = 0.0f;
    for (int64_t j = 0; j < D; ++j) mu += x[i * D + j];
    mu /= static_cast<float>(D);
    r.mean[i] = mu;

    // Compute variance.
    float var = 0.0f;
    for (int64_t j = 0; j < D; ++j) {
      float diff = x[i * D + j] - mu;
      var += diff * diff;
    }
    var /= static_cast<float>(D);

    float inv_s = 1.0f / std::sqrt(var + eps);
    r.inv_std[i] = inv_s;

    // Normalize and apply affine transform.
    for (int64_t j = 0; j < D; ++j) {
      float x_hat = (x[i * D + j] - mu) * inv_s;
      r.y[i * D + j] = gamma[j] * x_hat + beta[j];
    }
  }
  return r;
}

struct LayerNormGrads {
  std::vector<float> dx;      // [rows * D]
  std::vector<float> dgamma;  // [D]
  std::vector<float> dbeta;   // [D]
};

inline LayerNormGrads cpu_layernorm_backward(const float* dy, const float* x,
                                              const float* gamma,
                                              const float* mean,
                                              const float* inv_std,
                                              int64_t rows, int64_t D) {
  LayerNormGrads g;
  g.dx.resize(rows * D, 0.0f);
  g.dgamma.assign(D, 0.0f);
  g.dbeta.assign(D, 0.0f);

  for (int64_t i = 0; i < rows; ++i) {
    float mu = mean[i];
    float is = inv_std[i];

    // Compute x_hat for this row.
    std::vector<float> x_hat(D);
    for (int64_t j = 0; j < D; ++j)
      x_hat[j] = (x[i * D + j] - mu) * is;

    // Accumulate dgamma, dbeta.
    for (int64_t j = 0; j < D; ++j) {
      g.dgamma[j] += dy[i * D + j] * x_hat[j];
      g.dbeta[j] += dy[i * D + j];
    }

    // dx for this row.
    // dx = (1/D) * inv_std * (D * dy_hat - sum(dy_hat) - x_hat * sum(dy_hat * x_hat))
    // where dy_hat = gamma * dy
    float sum_dy_hat = 0.0f;
    float sum_dy_hat_xhat = 0.0f;
    for (int64_t j = 0; j < D; ++j) {
      float dy_hat = gamma[j] * dy[i * D + j];
      sum_dy_hat += dy_hat;
      sum_dy_hat_xhat += dy_hat * x_hat[j];
    }
    float inv_D = 1.0f / static_cast<float>(D);
    for (int64_t j = 0; j < D; ++j) {
      float dy_hat = gamma[j] * dy[i * D + j];
      g.dx[i * D + j] = is * inv_D *
          (static_cast<float>(D) * dy_hat - sum_dy_hat -
           x_hat[j] * sum_dy_hat_xhat);
    }
  }
  return g;
}

// ---------------------------------------------------------------------------
// CPU reference: embedding.
// ---------------------------------------------------------------------------

// Forward: gather rows from table.
inline std::vector<float> cpu_embedding_forward(const float* table,
                                                 const int32_t* indices,
                                                 int64_t n, int64_t d_model) {
  std::vector<float> out(n * d_model);
  for (int64_t i = 0; i < n; ++i) {
    int32_t idx = indices[i];
    for (int64_t j = 0; j < d_model; ++j) {
      out[i * d_model + j] = table[idx * d_model + j];
    }
  }
  return out;
}

// Backward: scatter-add dY into d_table.
inline std::vector<float> cpu_embedding_backward(const float* dY,
                                                  const int32_t* indices,
                                                  int64_t n, int64_t d_model,
                                                  int64_t vocab_size) {
  std::vector<float> d_table(vocab_size * d_model, 0.0f);
  for (int64_t i = 0; i < n; ++i) {
    int32_t idx = indices[i];
    for (int64_t j = 0; j < d_model; ++j) {
      d_table[idx * d_model + j] += dY[i * d_model + j];
    }
  }
  return d_table;
}

// ---------------------------------------------------------------------------
// CPU reference: linear projection.
//   Forward:  Y = X @ W^T + b
//   X: [N, D_in], W: [D_out, D_in], b: [D_out], Y: [N, D_out]
// ---------------------------------------------------------------------------

inline std::vector<float> cpu_linear_forward(const float* X, const float* W,
                                              const float* bias,
                                              int64_t N, int64_t D_in,
                                              int64_t D_out, bool use_bias) {
  std::vector<float> Y(N * D_out, 0.0f);
  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = 0; j < D_out; ++j) {
      float sum = 0.0f;
      for (int64_t k = 0; k < D_in; ++k) {
        // Y[i,j] = sum_k X[i,k] * W[j,k]  (W is [D_out, D_in] row-major)
        sum += X[i * D_in + k] * W[j * D_in + k];
      }
      Y[i * D_out + j] = sum + (use_bias ? bias[j] : 0.0f);
    }
  }
  return Y;
}

struct LinearGradsResult {
  std::vector<float> dX;  // [N, D_in]
  std::vector<float> dW;  // [D_out, D_in]
  std::vector<float> db;  // [D_out]
};

// Backward:
//   dX = dY @ W           [N, D_out] @ [D_out, D_in] → [N, D_in]
//   dW = dY^T @ X         [D_out, N] @ [N, D_in]     → [D_out, D_in]
//   db = col_sum(dY)      sum over N → [D_out]
inline LinearGradsResult cpu_linear_backward(const float* X, const float* dY,
                                              const float* W,
                                              int64_t N, int64_t D_in,
                                              int64_t D_out) {
  LinearGradsResult g;
  g.dX.assign(N * D_in, 0.0f);
  g.dW.assign(D_out * D_in, 0.0f);
  g.db.assign(D_out, 0.0f);

  // dX = dY @ W
  for (int64_t i = 0; i < N; ++i) {
    for (int64_t k = 0; k < D_in; ++k) {
      float sum = 0.0f;
      for (int64_t j = 0; j < D_out; ++j) {
        sum += dY[i * D_out + j] * W[j * D_in + k];
      }
      g.dX[i * D_in + k] = sum;
    }
  }

  // dW = dY^T @ X
  for (int64_t j = 0; j < D_out; ++j) {
    for (int64_t k = 0; k < D_in; ++k) {
      float sum = 0.0f;
      for (int64_t i = 0; i < N; ++i) {
        sum += dY[i * D_out + j] * X[i * D_in + k];
      }
      g.dW[j * D_in + k] = sum;
    }
  }

  // db = col_sum(dY)
  for (int64_t j = 0; j < D_out; ++j) {
    float sum = 0.0f;
    for (int64_t i = 0; i < N; ++i) {
      sum += dY[i * D_out + j];
    }
    g.db[j] = sum;
  }

  return g;
}

// ---------------------------------------------------------------------------
// CPU reference: cross-entropy loss.
// ---------------------------------------------------------------------------

inline float cpu_cross_entropy(const float* logits, int32_t target,
                                int64_t V) {
  float mx = *std::max_element(logits, logits + V);
  float sum_exp = 0.0f;
  for (int64_t j = 0; j < V; ++j) sum_exp += std::exp(logits[j] - mx);
  return mx + std::log(sum_exp) - logits[target];
}

// Backward: dlogits[j] = (softmax(logits)[j] - 1{j == target}) / N
inline std::vector<float> cpu_cross_entropy_backward(const float* logits,
                                                      const int32_t* targets,
                                                      int64_t N, int64_t V) {
  std::vector<float> dlogits(N * V);
  for (int64_t i = 0; i < N; ++i) {
    // Compute softmax for row i.
    float mx = *std::max_element(logits + i * V, logits + (i + 1) * V);
    float sum_exp = 0.0f;
    for (int64_t j = 0; j < V; ++j)
      sum_exp += std::exp(logits[i * V + j] - mx);
    for (int64_t j = 0; j < V; ++j) {
      float p = std::exp(logits[i * V + j] - mx) / sum_exp;
      float indicator = (j == targets[i]) ? 1.0f : 0.0f;
      dlogits[i * V + j] = (p - indicator) / static_cast<float>(N);
    }
  }
  return dlogits;
}

// ---------------------------------------------------------------------------
// CPU reference: multi-head attention (causal).
//
// Full attention forward pass for testing. Steps:
//   1. QKV = X @ W_qkv^T + b_qkv
//   2. Split into Q, K, V heads: [B, T, D] → per-head [B, H, T, Dh]
//   3. Scores = Q @ K^T * scale
//   4. Apply causal mask (upper triangle → -inf)
//   5. Probs = softmax(scores)
//   6. Context = Probs @ V
//   7. Merge heads: [B, H, T, Dh] → [B, T, D]
//   8. Output = Context @ W_o^T + b_o
// ---------------------------------------------------------------------------

struct AttentionResult {
  std::vector<float> output;   // [B*T*D]
  std::vector<float> probs;    // [B*H*T*T] (for verification)
};

inline AttentionResult cpu_attention_forward(
    const float* X,       // [B, T, D]
    const float* W_qkv,   // [3*D, D]
    const float* b_qkv,   // [3*D], may be nullptr
    const float* W_o,     // [D, D]
    const float* b_o,     // [D], may be nullptr
    int64_t B, int64_t T, int64_t D, int64_t H,
    bool causal, bool use_bias) {

  int64_t Dh = D / H;
  float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
  int64_t N = B * T;

  // Step 1: QKV projection — [N, D] @ [3D, D]^T → [N, 3D]
  std::vector<float> qkv = cpu_linear_forward(X, W_qkv, b_qkv,
                                               N, D, 3 * D, use_bias);

  // Step 2: Split into Q, K, V and reshape to [B, H, T, Dh].
  // qkv layout: [B, T, 3*D] with Q at [0:D], K at [D:2D], V at [2D:3D]
  auto head_index = [&](int64_t b, int64_t h, int64_t t, int64_t d) {
    return ((b * H + h) * T + t) * Dh + d;
  };

  int64_t BHTDh = B * H * T * Dh;
  std::vector<float> Q(BHTDh), K(BHTDh), V(BHTDh);

  for (int64_t b = 0; b < B; ++b) {
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t d = 0; d < Dh; ++d) {
          int64_t qkv_offset = (b * T + t) * (3 * D);
          Q[head_index(b, h, t, d)] = qkv[qkv_offset + h * Dh + d];
          K[head_index(b, h, t, d)] = qkv[qkv_offset + D + h * Dh + d];
          V[head_index(b, h, t, d)] = qkv[qkv_offset + 2 * D + h * Dh + d];
        }
      }
    }
  }

  // Step 3-4: Scores = Q @ K^T * scale, then causal mask.
  int64_t BHTT = B * H * T * T;
  std::vector<float> scores(BHTT);
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t qi = 0; qi < T; ++qi) {
        for (int64_t ki = 0; ki < T; ++ki) {
          float dot = 0.0f;
          for (int64_t d = 0; d < Dh; ++d) {
            dot += Q[head_index(b, h, qi, d)] * K[head_index(b, h, ki, d)];
          }
          dot *= scale;
          if (causal && ki > qi) dot = -1e9f;
          scores[((b * H + h) * T + qi) * T + ki] = dot;
        }
      }
    }
  }

  // Step 5: Softmax over last dimension (keys) for each (b, h, q).
  std::vector<float> probs = cpu_softmax(scores.data(), B * H * T, T);

  // Step 6: Context = Probs @ V  → [B, H, T, Dh]
  std::vector<float> context(BHTDh, 0.0f);
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

  // Step 7: Merge heads [B, H, T, Dh] → [B, T, D].
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

  // Step 8: Output projection — [N, D] @ [D, D]^T → [N, D]
  std::vector<float> output = cpu_linear_forward(merged.data(), W_o, b_o,
                                                  N, D, D, use_bias);

  return {output, probs};
}

// ---------------------------------------------------------------------------
// CPU reference: causal mask generation.
//   mask[i, j] = 1 if j > i (upper triangle), 0 otherwise.
//   Used for testing masked softmax with causal attention.
// ---------------------------------------------------------------------------

inline std::vector<int8_t> cpu_causal_mask(int64_t T) {
  std::vector<int8_t> mask(T * T, 0);
  for (int64_t i = 0; i < T; ++i)
    for (int64_t j = i + 1; j < T; ++j)
      mask[i * T + j] = 1;
  return mask;
}

// ---------------------------------------------------------------------------
// Simple PRNG for deterministic test data (avoids <random> header bloat).
// ---------------------------------------------------------------------------

class SimpleRng {
 public:
  explicit SimpleRng(uint32_t seed = 42) : state_(seed) {}

  // Returns a float in [-range, +range].
  float uniform(float range = 1.0f) {
    state_ = state_ * 1664525u + 1013904223u;  // LCG
    float u = static_cast<float>(state_ & 0x00FFFFFF) / 16777216.0f;
    return (2.0f * u - 1.0f) * range;
  }

  // Fill a vector with uniform random values.
  void fill(std::vector<float>& v, float range = 1.0f) {
    for (auto& x : v) x = uniform(range);
  }

 private:
  uint32_t state_;
};

}  // namespace test
}  // namespace gpt
