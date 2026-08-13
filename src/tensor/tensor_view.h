#pragma once

// Layer 1 — Tensor substrate: non-owning, typed tensor view.
//
// TensorView is the vocabulary type passed to every kernel and composite op.
// It is non-owning, trivially copyable, and carries shape + stride metadata.

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "src/tensor/dtype.h"

namespace gpt {

// ---------------------------------------------------------------------------
// TensorView<T, Rank>
// ---------------------------------------------------------------------------

template <typename T, int Rank>
struct TensorView {
  static_assert(Rank > 0 && Rank <= 8, "Rank must be in [1, 8]");

  T* data = nullptr;
  std::array<int64_t, Rank> shape{};
  std::array<int64_t, Rank> stride{};

  // ------- const conversion -------

  // Implicitly convert a mutable view to its const-element counterpart, so a
  // Tensor2D<float> can be passed where a Tensor2D<const float> is expected.
  // Only present when T is non-const (guards against a self-conversion). Kept
  // as a conversion operator rather than a constructor so TensorView stays an
  // aggregate and brace-initialization keeps working.
  template <typename U = T,
            typename = std::enable_if_t<!std::is_const_v<U>>>
  operator TensorView<const T, Rank>() const {  // NOLINT(google-explicit-constructor)
    TensorView<const T, Rank> v;
    v.data = data;
    v.shape = shape;
    v.stride = stride;
    return v;
  }

  // ------- construction helpers -------

  // Create a contiguous view (row-major / C-order).
  static TensorView contiguous(T* data, std::array<int64_t, Rank> shape) {
    TensorView v;
    v.data = data;
    v.shape = shape;
    v.stride = compute_contiguous_strides(shape);
    return v;
  }

  // ------- element access (host-side debug only) -------

  template <typename... Indices>
  [[nodiscard]] int64_t linear_offset(Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank);
    std::array<int64_t, Rank> idx{static_cast<int64_t>(indices)...};
    int64_t off = 0;
    for (int i = 0; i < Rank; ++i) {
      assert(idx[i] >= 0 && idx[i] < shape[i]);
      off += idx[i] * stride[i];
    }
    return off;
  }

  // ------- metadata queries -------

  [[nodiscard]] int64_t numel() const {
    int64_t n = 1;
    for (int i = 0; i < Rank; ++i) n *= shape[i];
    return n;
  }

  [[nodiscard]] size_t size_bytes() const {
    return static_cast<size_t>(numel()) * sizeof(T);
  }

  [[nodiscard]] bool is_contiguous() const {
    auto expected = compute_contiguous_strides(shape);
    return stride == expected;
  }

  // ------- slicing / reshaping helpers -------

  // Return a view of a single element along the outermost dimension.
  // e.g. for [B, T, D] with Rank=3, slice(b) gives a Rank-2 view [T, D].
  // Enabled only for Rank > 1 (SFINAE, since C++17 has no requires-clauses).
  template <int R = Rank, typename = std::enable_if_t<(R > 1)>>
  TensorView<T, R - 1> slice(int64_t index) const {
    assert(index >= 0 && index < shape[0]);
    TensorView<T, R - 1> sub;
    sub.data = data + index * stride[0];
    for (int i = 0; i < R - 1; ++i) {
      sub.shape[i] = shape[i + 1];
      sub.stride[i] = stride[i + 1];
    }
    return sub;
  }

 private:
  static std::array<int64_t, Rank> compute_contiguous_strides(
      const std::array<int64_t, Rank>& shape) {
    std::array<int64_t, Rank> s{};
    s[Rank - 1] = 1;
    for (int i = Rank - 2; i >= 0; --i) {
      s[i] = s[i + 1] * shape[i + 1];
    }
    return s;
  }
};

// ---------------------------------------------------------------------------
// Common type aliases for transformer tensors.
// ---------------------------------------------------------------------------

// Activations: [B, T, D]
template <typename T>
using Tensor3D = TensorView<T, 3>;

// QKV / attention intermediates: [B, H, T, Dh]
template <typename T>
using Tensor4D = TensorView<T, 4>;

// Flat parameter / gradient buffers.
template <typename T>
using Tensor1D = TensorView<T, 1>;

// Weight matrices: [D_out, D_in] or [D_in, D_out].
template <typename T>
using Tensor2D = TensorView<T, 2>;

}  // namespace gpt
