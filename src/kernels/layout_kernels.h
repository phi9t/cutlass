#pragma once

// Layer 2 — Primitive kernels: data movement / layout transforms.
//
// Physical copy operations that complement the metadata-only layout
// helpers in //src/tensor:layout.  Use these when a kernel requires
// physically contiguous data in a specific order.

#include "src/core/status.h"
#include "src/core/stream.h"
#include "src/tensor/tensor_view.h"

namespace gpt {
namespace kernels {

// ---------------------------------------------------------------------------
// Transpose 2D:  out[j, i] = input[i, j]
//   input:  [M, N]
//   output: [N, M]
// ---------------------------------------------------------------------------

Status transpose_2d_f32(Tensor2D<const float> input,
                        Tensor2D<float> output,
                        const CudaStream& stream);

// ---------------------------------------------------------------------------
// Contiguous repack: copy a strided tensor into a contiguous buffer.
//   Both views must have the same shape and element count.
// ---------------------------------------------------------------------------

Status repack_contiguous_f32(Tensor2D<const float> input,
                             Tensor2D<float> output,
                             const CudaStream& stream);

// ---------------------------------------------------------------------------
// Head split + contiguous copy:
//   [B, T, H*Dh] → [B, H, T, Dh]  (physically contiguous output).
// ---------------------------------------------------------------------------

Status split_heads_f32(Tensor3D<const float> input,  // [B, T, D]
                       Tensor4D<float> output,        // [B, H, T, Dh]
                       int64_t n_heads,
                       const CudaStream& stream);

// ---------------------------------------------------------------------------
// Head merge + contiguous copy:
//   [B, H, T, Dh] → [B, T, H*Dh]  (physically contiguous output).
// ---------------------------------------------------------------------------

Status merge_heads_f32(Tensor4D<const float> input,  // [B, H, T, Dh]
                       Tensor3D<float> output,        // [B, T, D]
                       const CudaStream& stream);

}  // namespace kernels
}  // namespace gpt
