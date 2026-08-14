// Layer 3 — Composite neural ops: linear projection implementation.
//
// Delegates to //src/kernels:gemm for the matmul and
// //src/kernels:reductions for bias-gradient reduction.

#include "src/ops/linear.h"

#include "src/kernels/elementwise.h"
#include "src/kernels/gemm.h"
#include "src/kernels/reductions.h"

namespace gpt {
namespace ops {

Status linear_forward(Tensor2D<const float> X,
                      const LinearParams& params,
                      Tensor2D<float> Y,
                      const CudaStream& stream) {
  // Y = X @ W^T
  // X: [N, D_in], W: [D_out, D_in], Y: [N, D_out]
  // For GEMM: we need X * W^T.
  // Treat W^T as a matrix with shape [D_in, D_out] and transposed strides.
  int64_t N = X.shape[0];
  int64_t D_in = X.shape[1];
  int64_t D_out = params.weight.shape[0];

  if (params.weight.shape[1] != D_in) {
    return Status(StatusCode::kInvalidArgument,
                  "linear_forward: weight D_in mismatch");
  }
  if (Y.shape[0] != N || Y.shape[1] != D_out) {
    return Status(StatusCode::kInvalidArgument,
                  "linear_forward: output shape mismatch");
  }

  // Construct transposed view of W: [D_in, D_out] with swapped strides.
  Tensor2D<float> W_T;
  W_T.data = params.weight.data;
  W_T.shape = {D_in, D_out};
  W_T.stride = {1, D_in};  // transposed row-major

  // Cast away const for the GEMM interface (GEMM reads only).
  auto X_mut = Tensor2D<float>{const_cast<float*>(X.data), X.shape, X.stride};

  GPT_RETURN_IF_ERROR(
      kernels::gemm_f32(X_mut, W_T, Y, 1.0f, 0.0f, stream));

  // Add bias: broadcast [D_out] over each of the N rows of Y ([N, D_out]).
  if (params.use_bias && params.bias.data != nullptr) {
    GPT_RETURN_IF_ERROR(kernels::bias_add_f32(Y, params.bias, stream));
  }

  return Status::Ok();
}

Status linear_backward(Tensor2D<const float> X,
                       Tensor2D<const float> dY,
                       const LinearParams& params,
                       Tensor2D<float> dX,
                       LinearGrads& grads,
                       const CudaStream& stream) {
  int64_t N = X.shape[0];
  int64_t D_out = params.weight.shape[0];

  // dX = dY @ W      (dY: [N, D_out], W: [D_out, D_in] → dX: [N, D_in])
  auto dY_mut = Tensor2D<float>{const_cast<float*>(dY.data), dY.shape, dY.stride};
  GPT_RETURN_IF_ERROR(
      kernels::gemm_f32(dY_mut, params.weight, dX, 1.0f, 0.0f, stream));

  // dW = dY^T @ X    (dY^T: [D_out, N], X: [N, D_in] → dW: [D_out, D_in])
  Tensor2D<float> dY_T;
  dY_T.data = const_cast<float*>(dY.data);
  dY_T.shape = {D_out, N};
  dY_T.stride = {1, D_out};  // transposed

  auto X_mut = Tensor2D<float>{const_cast<float*>(X.data), X.shape, X.stride};
  GPT_RETURN_IF_ERROR(
      kernels::gemm_f32(dY_T, X_mut, grads.d_weight, 1.0f, 0.0f, stream));

  // db = col_sum(dY)
  if (grads.d_bias.data != nullptr) {
    GPT_RETURN_IF_ERROR(kernels::col_sum_f32(dY, grads.d_bias, stream));
  }

  return Status::Ok();
}

}  // namespace ops
}  // namespace gpt
