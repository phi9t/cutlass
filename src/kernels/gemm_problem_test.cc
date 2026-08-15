// Tier A — Host tests for GEMM shape and stride descriptors.

#include "gtest/gtest.h"
#include "src/kernels/gemm_problem.h"

using namespace gpt;
using namespace gpt::kernels;

TEST(GemmProblemTest, ClassifiesRowAndColumnMajorLayouts) {
  float data[16] = {};
  Tensor2D<float> row_major{data, {2, 3}, {3, 1}};
  Tensor2D<float> column_major{data, {2, 3}, {1, 2}};

  Result<MatrixLayout> row = classify_matrix_layout(row_major);
  ASSERT_TRUE(row.ok()) << row.status().message();
  EXPECT_EQ(row.value().orientation, MatrixOrientation::kRowMajor);
  EXPECT_EQ(row.value().leading_dim, 3);

  Result<MatrixLayout> column = classify_matrix_layout(column_major);
  ASSERT_TRUE(column.ok()) << column.status().message();
  EXPECT_EQ(column.value().orientation, MatrixOrientation::kColumnMajor);
  EXPECT_EQ(column.value().leading_dim, 2);
}

TEST(GemmProblemTest, RejectsUnsupportedStrides) {
  float data[16] = {};
  Tensor2D<float> unsupported{data, {2, 3}, {6, 2}};

  Result<MatrixLayout> layout = classify_matrix_layout(unsupported);
  ASSERT_FALSE(layout.ok());
  EXPECT_EQ(layout.status().code(), StatusCode::kInvalidArgument);
}

TEST(GemmProblemTest, BuildsGemmProblem) {
  float data[64] = {};
  Tensor2D<float> A{data, {2, 3}, {3, 1}};
  Tensor2D<float> B{data, {3, 4}, {1, 3}};
  Tensor2D<float> C{data, {2, 4}, {4, 1}};

  Result<GemmProblem> problem = make_gemm_problem(A, B, C);
  ASSERT_TRUE(problem.ok()) << problem.status().message();
  EXPECT_EQ(problem.value().m, 2);
  EXPECT_EQ(problem.value().n, 4);
  EXPECT_EQ(problem.value().k, 3);
  EXPECT_EQ(problem.value().a.orientation, MatrixOrientation::kRowMajor);
  EXPECT_EQ(problem.value().b.orientation, MatrixOrientation::kColumnMajor);
  EXPECT_EQ(problem.value().c.orientation, MatrixOrientation::kRowMajor);
}

TEST(GemmProblemTest, RejectsShapeMismatch) {
  float data[64] = {};
  Tensor2D<float> A{data, {2, 3}, {3, 1}};
  Tensor2D<float> B{data, {2, 4}, {4, 1}};
  Tensor2D<float> C{data, {2, 4}, {4, 1}};

  Result<GemmProblem> problem = make_gemm_problem(A, B, C);
  ASSERT_FALSE(problem.ok());
  EXPECT_EQ(problem.status().code(), StatusCode::kInvalidArgument);
}

TEST(GemmProblemTest, BuildsBatchedGemmProblem) {
  float data[256] = {};
  Tensor3D<float> A{data, {5, 2, 3}, {6, 3, 1}};
  Tensor3D<float> B{data, {5, 3, 4}, {12, 4, 1}};
  Tensor3D<float> C{data, {5, 2, 4}, {8, 4, 1}};

  Result<GemmProblem> problem = make_batched_gemm_problem(A, B, C);
  ASSERT_TRUE(problem.ok()) << problem.status().message();
  EXPECT_EQ(problem.value().batch_count, 5);
  EXPECT_EQ(problem.value().batch_stride_a, 6);
  EXPECT_EQ(problem.value().batch_stride_b, 12);
  EXPECT_EQ(problem.value().batch_stride_c, 8);
  EXPECT_EQ(problem.value().m, 2);
  EXPECT_EQ(problem.value().n, 4);
  EXPECT_EQ(problem.value().k, 3);
}
