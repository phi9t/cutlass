#pragma once

// Layer 2 — Primitive kernels: CUDA-free GEMM problem descriptors.

#include <cstdint>

#include "src/core/status.h"
#include "src/tensor/tensor_view.h"

namespace gpt {
namespace kernels {

enum class MatrixOrientation {
  kRowMajor,
  kColumnMajor,
};

struct MatrixLayout {
  MatrixOrientation orientation;
  int64_t leading_dim;
};

struct GemmProblem {
  int64_t m;
  int64_t n;
  int64_t k;
  MatrixLayout a;
  MatrixLayout b;
  MatrixLayout c;
  int64_t batch_count = 1;
  int64_t batch_stride_a = 0;
  int64_t batch_stride_b = 0;
  int64_t batch_stride_c = 0;
};

template <typename T>
Result<MatrixLayout> classify_matrix_layout(Tensor2D<T> matrix) {
  if (matrix.stride[1] == 1) {
    return MatrixLayout{MatrixOrientation::kRowMajor, matrix.stride[0]};
  }
  if (matrix.stride[0] == 1) {
    return MatrixLayout{MatrixOrientation::kColumnMajor, matrix.stride[1]};
  }
  return Status(StatusCode::kInvalidArgument,
                "GEMM layout needs one unit-stride matrix axis");
}

template <typename T>
Result<GemmProblem> make_gemm_problem(Tensor2D<T> A, Tensor2D<T> B,
                                      Tensor2D<T> C) {
  if (A.shape[1] != B.shape[0]) {
    return Status(StatusCode::kInvalidArgument, "GEMM K-dim mismatch");
  }
  if (A.shape[0] != C.shape[0] || B.shape[1] != C.shape[1]) {
    return Status(StatusCode::kInvalidArgument, "GEMM output shape mismatch");
  }
  Result<MatrixLayout> a = classify_matrix_layout(A);
  if (!a.ok()) return a.status();
  Result<MatrixLayout> b = classify_matrix_layout(B);
  if (!b.ok()) return b.status();
  Result<MatrixLayout> c = classify_matrix_layout(C);
  if (!c.ok()) return c.status();

  GemmProblem problem;
  problem.m = A.shape[0];
  problem.n = B.shape[1];
  problem.k = A.shape[1];
  problem.a = a.value();
  problem.b = b.value();
  problem.c = c.value();
  return problem;
}

template <typename T>
Result<GemmProblem> make_batched_gemm_problem(Tensor3D<T> A, Tensor3D<T> B,
                                              Tensor3D<T> C) {
  if (A.shape[0] != B.shape[0] || A.shape[0] != C.shape[0]) {
    return Status(StatusCode::kInvalidArgument,
                  "Batched GEMM batch-dim mismatch");
  }
  Tensor2D<T> a{A.data, {A.shape[1], A.shape[2]},
                {A.stride[1], A.stride[2]}};
  Tensor2D<T> b{B.data, {B.shape[1], B.shape[2]},
                {B.stride[1], B.stride[2]}};
  Tensor2D<T> c{C.data, {C.shape[1], C.shape[2]},
                {C.stride[1], C.stride[2]}};
  Result<GemmProblem> problem = make_gemm_problem(a, b, c);
  if (!problem.ok()) return problem.status();
  problem.value().batch_count = A.shape[0];
  problem.value().batch_stride_a = A.stride[0];
  problem.value().batch_stride_b = B.stride[0];
  problem.value().batch_stride_c = C.stride[0];
  return problem;
}

}  // namespace kernels
}  // namespace gpt
