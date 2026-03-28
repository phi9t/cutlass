#pragma once

// Layer 1 — Tensor substrate: data type enum and traits.

#include <cstddef>
#include <cstdint>

namespace gpt {

// ---------------------------------------------------------------------------
// Dtype — the small set of types we actually support for v1.
// ---------------------------------------------------------------------------

enum class Dtype : uint8_t {
  kFloat32 = 0,
  kBFloat16,
  kFloat16,
  // Extend later: kFloat8E4M3, kFloat8E5M2, kInt32, ...
};

// ---------------------------------------------------------------------------
// Size-of lookup (bytes per element).
// ---------------------------------------------------------------------------

constexpr size_t dtype_size(Dtype dt) {
  switch (dt) {
    case Dtype::kFloat32:  return 4;
    case Dtype::kBFloat16: return 2;
    case Dtype::kFloat16:  return 2;
  }
  return 0;  // unreachable
}

// ---------------------------------------------------------------------------
// Human-readable name.
// ---------------------------------------------------------------------------

constexpr const char* dtype_name(Dtype dt) {
  switch (dt) {
    case Dtype::kFloat32:  return "float32";
    case Dtype::kBFloat16: return "bfloat16";
    case Dtype::kFloat16:  return "float16";
  }
  return "unknown";
}

}  // namespace gpt
