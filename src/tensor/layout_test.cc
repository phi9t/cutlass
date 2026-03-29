// Tier A — Host-only unit tests for layout utilities.

#include "gtest/gtest.h"
#include "src/tensor/layout.h"

using namespace gpt;

TEST(QKVSplitTest, ShapesCorrect) {
  float dummy[2 * 4 * 6];  // [B=2, T=4, 3*D=6] → D=2
  auto packed = Tensor3D<float>::contiguous(dummy, {2, 4, 6});
  auto split = split_qkv(packed, 2);

  EXPECT_EQ(split.Q.shape[0], 2);
  EXPECT_EQ(split.Q.shape[1], 4);
  EXPECT_EQ(split.Q.shape[2], 2);

  EXPECT_EQ(split.K.data, dummy + 2);
  EXPECT_EQ(split.V.data, dummy + 4);
}

TEST(ReshapeToHeadsTest, ShapeAndStride) {
  float dummy[2 * 4 * 8];  // [B=2, T=4, D=8], H=2, Dh=4
  auto x = Tensor3D<float>::contiguous(dummy, {2, 4, 8});
  auto heads = reshape_to_heads(x, 2);

  EXPECT_EQ(heads.shape[0], 2);  // B
  EXPECT_EQ(heads.shape[1], 2);  // H
  EXPECT_EQ(heads.shape[2], 4);  // T
  EXPECT_EQ(heads.shape[3], 4);  // Dh

  // Stride: B=T*D=32, H=Dh=4, T=D=8, Dh=1
  EXPECT_EQ(heads.stride[0], 32);
  EXPECT_EQ(heads.stride[1], 4);
  EXPECT_EQ(heads.stride[2], 8);
  EXPECT_EQ(heads.stride[3], 1);
}

TEST(MergeHeadsTest, RoundTrip) {
  float dummy[2 * 4 * 8];
  auto x = Tensor3D<float>::contiguous(dummy, {2, 4, 8});
  auto heads = reshape_to_heads(x, 2);
  auto merged = merge_heads(heads);

  EXPECT_EQ(merged.shape[0], 2);
  EXPECT_EQ(merged.shape[1], 4);
  EXPECT_EQ(merged.shape[2], 8);
  EXPECT_EQ(merged.data, dummy);
}

TEST(TransposeLast2Test, SwapsShapeAndStride) {
  float dummy[12];
  auto v = TensorView<float, 2>::contiguous(dummy, {3, 4});
  auto t = transpose_last2(v);

  EXPECT_EQ(t.shape[0], 4);
  EXPECT_EQ(t.shape[1], 3);
  EXPECT_EQ(t.stride[0], 1);
  EXPECT_EQ(t.stride[1], 4);
}
