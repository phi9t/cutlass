// Tier A — Host-only unit tests for tensor substrate.

#include "gtest/gtest.h"
#include "src/tensor/dtype.h"
#include "src/tensor/tensor_view.h"

using namespace gpt;

TEST(DtypeTest, Sizes) {
  EXPECT_EQ(dtype_size(Dtype::kFloat32), 4u);
  EXPECT_EQ(dtype_size(Dtype::kBFloat16), 2u);
  EXPECT_EQ(dtype_size(Dtype::kFloat16), 2u);
}

TEST(DtypeTest, Names) {
  EXPECT_STREQ(dtype_name(Dtype::kFloat32), "float32");
  EXPECT_STREQ(dtype_name(Dtype::kBFloat16), "bfloat16");
}

TEST(TensorViewTest, ContiguousStrides) {
  float dummy[24];
  auto v = TensorView<float, 3>::contiguous(dummy, {2, 3, 4});
  EXPECT_EQ(v.stride[0], 12);
  EXPECT_EQ(v.stride[1], 4);
  EXPECT_EQ(v.stride[2], 1);
  EXPECT_TRUE(v.is_contiguous());
}

TEST(TensorViewTest, Numel) {
  float dummy[60];
  auto v = TensorView<float, 3>::contiguous(dummy, {3, 4, 5});
  EXPECT_EQ(v.numel(), 60);
}

TEST(TensorViewTest, SizeBytes) {
  float dummy[12];
  auto v = TensorView<float, 2>::contiguous(dummy, {3, 4});
  EXPECT_EQ(v.size_bytes(), 48u);
}

TEST(TensorViewTest, LinearOffset) {
  float dummy[24];
  auto v = TensorView<float, 3>::contiguous(dummy, {2, 3, 4});
  // v[1, 2, 3] = 1*12 + 2*4 + 3 = 23
  EXPECT_EQ(v.linear_offset(1, 2, 3), 23);
}

TEST(TensorViewTest, Slice) {
  float dummy[24];
  auto v = TensorView<float, 3>::contiguous(dummy, {2, 3, 4});
  auto sub = v.slice(1);  // [3, 4] starting at offset 12
  EXPECT_EQ(sub.shape[0], 3);
  EXPECT_EQ(sub.shape[1], 4);
  EXPECT_EQ(sub.data, dummy + 12);
}

TEST(TensorViewTest, IsContiguousAfterSlice) {
  float dummy[24];
  auto v = TensorView<float, 3>::contiguous(dummy, {2, 3, 4});
  auto sub = v.slice(0);
  EXPECT_TRUE(sub.is_contiguous());
}
