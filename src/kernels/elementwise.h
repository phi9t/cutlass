#pragma once

// Layer 2 — Primitive kernels: elementwise operations.
//
// All operate on flat 1D views (caller reshapes as needed).

#include <cstdint>

#include "src/core/status.h"
#include "src/core/stream.h"
#include "src/tensor/tensor_view.h"

namespace gpt {
namespace kernels {

// ---------------------------------------------------------------------------
// Elementwise add:  out[i] = a[i] + b[i]
// ---------------------------------------------------------------------------
Status vec_add_f32(Tensor1D<const float> a, Tensor1D<const float> b,
                   Tensor1D<float> out, const CudaStream& stream);

// ---------------------------------------------------------------------------
// Elementwise mul:  out[i] = a[i] * b[i]
// ---------------------------------------------------------------------------
Status vec_mul_f32(Tensor1D<const float> a, Tensor1D<const float> b,
                   Tensor1D<float> out, const CudaStream& stream);

// ---------------------------------------------------------------------------
// Scale:  out[i] = alpha * x[i]
// ---------------------------------------------------------------------------
Status vec_scale_f32(Tensor1D<const float> x, float alpha,
                     Tensor1D<float> out, const CudaStream& stream);

// ---------------------------------------------------------------------------
// Exp:  out[i] = exp(x[i])
// ---------------------------------------------------------------------------
Status vec_exp_f32(Tensor1D<const float> x, Tensor1D<float> out,
                   const CudaStream& stream);

// ---------------------------------------------------------------------------
// Masked fill:  out[i] = mask[i] ? fill_value : x[i]
//   mask is int8: nonzero = masked.
// ---------------------------------------------------------------------------
Status masked_fill_f32(Tensor1D<const float> x,
                       Tensor1D<const int8_t> mask,
                       float fill_value,
                       Tensor1D<float> out,
                       const CudaStream& stream);

// ---------------------------------------------------------------------------
// Gather:  out[i] = table[indices[i]]
//   table:   [vocab_size, dim]
//   indices: [n]
//   out:     [n, dim]
// ---------------------------------------------------------------------------
Status gather_f32(Tensor2D<const float> table,
                  Tensor1D<const int32_t> indices,
                  Tensor2D<float> out,
                  const CudaStream& stream);

}  // namespace kernels
}  // namespace gpt
