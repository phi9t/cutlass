#pragma once

// Shared test utilities: CPU reference oracles and comparison helpers.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace gpt {
namespace test {

// ---------------------------------------------------------------------------
// CPU reference oracles for primitive ops.
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

inline float cpu_gelu(float x) {
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  constexpr float kCoeff = 0.044715f;
  float inner = kSqrt2OverPi * (x + kCoeff * x * x * x);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

inline float cpu_cross_entropy(const float* logits, int32_t target,
                                int64_t V) {
  float mx = *std::max_element(logits, logits + V);
  float sum_exp = 0.0f;
  for (int64_t j = 0; j < V; ++j) sum_exp += std::exp(logits[j] - mx);
  return mx + std::log(sum_exp) - logits[target];
}

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

}  // namespace test
}  // namespace gpt
