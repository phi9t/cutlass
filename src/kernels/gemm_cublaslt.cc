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

Status set_batch_layout(cublasLtMatrixLayout_t layout, int32_t batch_count,
                        int64_t batch_stride) {
  GPT_RETURN_IF_ERROR(cublas_status(
      cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                       &batch_count, sizeof(batch_count)),
      "cublasLtMatrixLayoutSetAttribute(batch_count)"));
  GPT_RETURN_IF_ERROR(cublas_status(
      cublasLtMatrixLayoutSetAttribute(
          layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride,
          sizeof(batch_stride)),
      "cublasLtMatrixLayoutSetAttribute(strided_batch_offset)"));
  return Status::Ok();
}

template <typename T>
Status gemm_cublaslt_impl(Tensor2D<T> A, Tensor2D<T> B, Tensor2D<T> C,
                          cudaDataType type, const char* context, float alpha,
                          float beta, const CudaStream& stream,
                          int32_t batch_count = 1, int64_t batch_a = 0,
                          int64_t batch_b = 0, int64_t batch_c = 0) {
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
                  std::string(context) +
                      ": unsupported strides (need unit-stride axis per operand)");
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

  result = create_layout(&a_layout, type, A.shape[0], A.shape[1], lda, a_order);
  if (!result.ok()) {
    cleanup();
    return result;
  }
  result = create_layout(&b_layout, type, B.shape[0], B.shape[1], ldb, b_order);
  if (!result.ok()) {
    cleanup();
    return result;
  }
  result = create_layout(&c_layout, type, C.shape[0], C.shape[1], ldc, c_order);
  if (!result.ok()) {
    cleanup();
    return result;
  }
  if (batch_count > 1) {
    result = set_batch_layout(a_layout, batch_count, batch_a);
    if (!result.ok()) {
      cleanup();
      return result;
    }
    result = set_batch_layout(b_layout, batch_count, batch_b);
    if (!result.ok()) {
      cleanup();
      return result;
    }
    result = set_batch_layout(c_layout, batch_count, batch_c);
    if (!result.ok()) {
      cleanup();
      return result;
    }
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

}  // namespace

Status gemm_f32_cublaslt(Tensor2D<float> A, Tensor2D<float> B,
                         Tensor2D<float> C, float alpha, float beta,
                         const CudaStream& stream) {
  return gemm_cublaslt_impl(A, B, C, CUDA_R_32F, "gemm_f32_cublaslt", alpha,
                            beta, stream);
}

Status gemm_bf16_cublaslt(Tensor2D<__nv_bfloat16> A,
                          Tensor2D<__nv_bfloat16> B,
                          Tensor2D<__nv_bfloat16> C, float alpha, float beta,
                          const CudaStream& stream) {
  return gemm_cublaslt_impl(A, B, C, CUDA_R_16BF, "gemm_bf16_cublaslt", alpha,
                            beta, stream);
}

Status batched_gemm_f32_cublaslt(Tensor3D<float> A, Tensor3D<float> B,
                                 Tensor3D<float> C, float alpha, float beta,
                                 const CudaStream& stream) {
  if (A.shape[0] != B.shape[0] || A.shape[0] != C.shape[0]) {
    return Status(StatusCode::kInvalidArgument,
                  "Batched GEMM batch-dim mismatch");
  }
  Tensor2D<float> a{A.data,
                    {A.shape[1], A.shape[2]},
                    {A.stride[1], A.stride[2]}};
  Tensor2D<float> b{B.data,
                    {B.shape[1], B.shape[2]},
                    {B.stride[1], B.stride[2]}};
  Tensor2D<float> c{C.data,
                    {C.shape[1], C.shape[2]},
                    {C.stride[1], C.stride[2]}};
  return gemm_cublaslt_impl(a, b, c, CUDA_R_32F, "batched_gemm_f32_cublaslt",
                            alpha, beta, stream,
                            static_cast<int32_t>(A.shape[0]), A.stride[0],
                            B.stride[0], C.stride[0]);
}

Status batched_gemm_bf16_cublaslt(Tensor3D<__nv_bfloat16> A,
                                  Tensor3D<__nv_bfloat16> B,
                                  Tensor3D<__nv_bfloat16> C, float alpha,
                                  float beta, const CudaStream& stream) {
  if (A.shape[0] != B.shape[0] || A.shape[0] != C.shape[0]) {
    return Status(StatusCode::kInvalidArgument,
                  "Batched GEMM batch-dim mismatch");
  }
  Tensor2D<__nv_bfloat16> a{A.data,
                            {A.shape[1], A.shape[2]},
                            {A.stride[1], A.stride[2]}};
  Tensor2D<__nv_bfloat16> b{B.data,
                            {B.shape[1], B.shape[2]},
                            {B.stride[1], B.stride[2]}};
  Tensor2D<__nv_bfloat16> c{C.data,
                            {C.shape[1], C.shape[2]},
                            {C.stride[1], C.stride[2]}};
  return gemm_cublaslt_impl(a, b, c, CUDA_R_16BF,
                            "batched_gemm_bf16_cublaslt", alpha, beta, stream,
                            static_cast<int32_t>(A.shape[0]), A.stride[0],
                            B.stride[0], C.stride[0]);
}

}  // namespace kernels
}  // namespace gpt
