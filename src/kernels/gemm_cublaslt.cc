// Layer 2 — Primitive kernels: cuBLASLt GEMM fallback backend.
//
// Used for correctness isolation and backend comparison against the CUTLASS
// path. The first slice keeps descriptors per-call and supports fp32 GEMM.

#include "src/kernels/gemm_cublaslt_internal.h"

#include <string>

#include <cublasLt.h>

namespace gpt {
namespace kernels {

namespace {

enum class MatrixOrder { kRowMajor, kColumnMajor, kUnsupported };

MatrixOrder classify_strides(int64_t row_stride, int64_t col_stride,
                             int64_t* leading_dim) {
  if (col_stride == 1) {
    *leading_dim = row_stride;
    return MatrixOrder::kRowMajor;
  }
  if (row_stride == 1) {
    *leading_dim = col_stride;
    return MatrixOrder::kColumnMajor;
  }
  return MatrixOrder::kUnsupported;
}

Status cublas_status(cublasStatus_t status, const char* context) {
  if (status == CUBLAS_STATUS_SUCCESS) return Status::Ok();
  return Status(StatusCode::kCudaError,
                std::string(context) + " failed with cuBLAS status " +
                    std::to_string(static_cast<int>(status)));
}

Status create_layout(cublasLtMatrixLayout_t* layout, cudaDataType type,
                     int64_t rows, int64_t cols, int64_t leading_dim,
                     MatrixOrder order) {
  GPT_RETURN_IF_ERROR(cublas_status(
      cublasLtMatrixLayoutCreate(layout, type, static_cast<uint64_t>(rows),
                                 static_cast<uint64_t>(cols), leading_dim),
      "cublasLtMatrixLayoutCreate"));

  cublasLtOrder_t cublas_order =
      (order == MatrixOrder::kRowMajor) ? CUBLASLT_ORDER_ROW : CUBLASLT_ORDER_COL;
  GPT_RETURN_IF_ERROR(cublas_status(
      cublasLtMatrixLayoutSetAttribute(*layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                       &cublas_order, sizeof(cublas_order)),
      "cublasLtMatrixLayoutSetAttribute(order)"));
  return Status::Ok();
}

}  // namespace

Status gemm_f32_cublaslt(Tensor2D<float> A, Tensor2D<float> B,
                         Tensor2D<float> C, float alpha, float beta,
                         const CudaStream& stream) {
  if (A.shape[1] != B.shape[0]) {
    return Status(StatusCode::kInvalidArgument, "GEMM K-dim mismatch");
  }
  if (A.shape[0] != C.shape[0] || B.shape[1] != C.shape[1]) {
    return Status(StatusCode::kInvalidArgument, "GEMM output shape mismatch");
  }

  int64_t lda = 0, ldb = 0, ldc = 0;
  const MatrixOrder a_order = classify_strides(A.stride[0], A.stride[1], &lda);
  const MatrixOrder b_order = classify_strides(B.stride[0], B.stride[1], &ldb);
  const MatrixOrder c_order = classify_strides(C.stride[0], C.stride[1], &ldc);
  if (a_order == MatrixOrder::kUnsupported ||
      b_order == MatrixOrder::kUnsupported ||
      c_order == MatrixOrder::kUnsupported) {
    return Status(StatusCode::kInvalidArgument,
                  "gemm_f32_cublaslt: unsupported strides (need unit-stride "
                  "axis per operand)");
  }

  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t matmul = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr;
  cublasLtMatrixLayout_t b_layout = nullptr;
  cublasLtMatrixLayout_t c_layout = nullptr;
  cublasStatus_t status = cublasLtCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_status(status, "cublasLtCreate");
  }

  auto cleanup = [&]() {
    if (c_layout) cublasLtMatrixLayoutDestroy(c_layout);
    if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
    if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
    if (matmul) cublasLtMatmulDescDestroy(matmul);
    if (handle) cublasLtDestroy(handle);
  };

  Status result = cublas_status(
      cublasLtMatmulDescCreate(&matmul, CUBLAS_COMPUTE_32F, CUDA_R_32F),
      "cublasLtMatmulDescCreate");
  if (!result.ok()) {
    cleanup();
    return result;
  }

  result = create_layout(&a_layout, CUDA_R_32F, A.shape[0], A.shape[1], lda,
                         a_order);
  if (!result.ok()) {
    cleanup();
    return result;
  }
  result = create_layout(&b_layout, CUDA_R_32F, B.shape[0], B.shape[1], ldb,
                         b_order);
  if (!result.ok()) {
    cleanup();
    return result;
  }
  result = create_layout(&c_layout, CUDA_R_32F, C.shape[0], C.shape[1], ldc,
                         c_order);
  if (!result.ok()) {
    cleanup();
    return result;
  }

  status = cublasLtMatmul(handle, matmul, &alpha, A.data, a_layout, B.data,
                          b_layout, &beta, C.data, c_layout, C.data, c_layout,
                          nullptr, nullptr, 0, stream.get());
  result = cublas_status(status, "cublasLtMatmul");
  cleanup();
  if (!result.ok()) return result;
  GPT_CHECK_CUDA(cudaGetLastError());
  return Status::Ok();
}

}  // namespace kernels
}  // namespace gpt
