#pragma once

// Layer 1 — Tensor substrate: layout utilities.
//
// Canonical layout conventions for the GPT trainer:
//
//   activations     [B, T, D]
//   packed QKV      [B, T, 3, H, Dh]  or flat [B, T, 3*D]
//   per-head        [B, H, T, Dh]
//   MLP hidden      [B, T, 4*D]
//   weight matrix   [D_out, D_in]   (row-major)
//
// These helpers perform the split/merge/transpose bookkeeping that attention
// and the transformer block need.  They do not copy data — they return new
// TensorView metadata pointing at the same buffer.

#include <cassert>
#include <cstdint>
#include <utility>

#include "src/tensor/tensor_view.h"

namespace gpt {

// ---------------------------------------------------------------------------
// QKV split: [B, T, 3*D] → three [B, T, D] views.
// ---------------------------------------------------------------------------

template <typename T>
struct QKVSplit {
  Tensor3D<T> Q;
  Tensor3D<T> K;
  Tensor3D<T> V;
};

template <typename T>
QKVSplit<T> split_qkv(Tensor3D<T> packed, int64_t d_model) {
  // packed.shape = [B, T, 3*D], contiguous.
  assert(packed.shape[2] == 3 * d_model);
  QKVSplit<T> out;
  auto make = [&](int64_t offset) -> Tensor3D<T> {
    Tensor3D<T> v;
    v.data = packed.data + offset;
    v.shape = {packed.shape[0], packed.shape[1], d_model};
    // Stride along dim-2 is 1 (innermost), but outer strides stay the same
    // because Q/K/V are interleaved in the packed buffer.
    v.stride = packed.stride;
    return v;
  };
  out.Q = make(0);
  out.K = make(d_model);
  out.V = make(2 * d_model);
  return out;
}

// ---------------------------------------------------------------------------
// Reshape to head layout: [B, T, D] → [B, H, T, Dh]  (logical view).
//
// This is a metadata-only reinterpretation.  The returned view is NOT
// contiguous in the usual sense — it has a transposed stride pattern.
// Callers that need physically contiguous [B,H,T,Dh] data must copy.
// ---------------------------------------------------------------------------

template <typename T>
Tensor4D<T> reshape_to_heads(Tensor3D<T> x, int64_t n_heads) {
  int64_t B = x.shape[0];
  int64_t T_len = x.shape[1];
  int64_t D = x.shape[2];
  assert(D % n_heads == 0);
  int64_t Dh = D / n_heads;

  // Physical layout: [B, T, H, Dh] contiguous.
  // We want logical layout: [B, H, T, Dh].
  // stride mapping:
  //   dim B → same as before
  //   dim H → Dh  (stepping across heads within a row)
  //   dim T → D   (stepping across sequence positions)
  //   dim Dh → 1
  Tensor4D<T> out;
  out.data = x.data;
  out.shape = {B, n_heads, T_len, Dh};
  out.stride = {x.stride[0], Dh, x.stride[1], 1};
  return out;
}

// ---------------------------------------------------------------------------
// Merge heads: [B, H, T, Dh] → [B, T, D]  (metadata only).
// Inverse of reshape_to_heads.  Caller must ensure the data was originally
// laid out as [B, T, H*Dh] contiguous.
// ---------------------------------------------------------------------------

template <typename T>
Tensor3D<T> merge_heads(Tensor4D<T> x) {
  int64_t B = x.shape[0];
  int64_t H = x.shape[1];
  int64_t T_len = x.shape[2];
  int64_t Dh = x.shape[3];
  int64_t D = H * Dh;

  Tensor3D<T> out;
  out.data = x.data;
  out.shape = {B, T_len, D};
  // Assumes the underlying physical layout is [B, T, H*Dh] contiguous.
  out.stride = {T_len * D, D, 1};
  return out;
}

// ---------------------------------------------------------------------------
// Transpose last two dims: [..., M, N] → [..., N, M].
// Metadata-only stride swap.
// ---------------------------------------------------------------------------

template <typename T, int Rank>
TensorView<T, Rank> transpose_last2(TensorView<T, Rank> x) {
  static_assert(Rank >= 2);
  TensorView<T, Rank> out = x;
  std::swap(out.shape[Rank - 1], out.shape[Rank - 2]);
  std::swap(out.stride[Rank - 1], out.stride[Rank - 2]);
  return out;
}

}  // namespace gpt
