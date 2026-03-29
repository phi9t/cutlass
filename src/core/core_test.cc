// Tier A — Host-only unit tests for core runtime.

#include "gtest/gtest.h"
#include "src/core/status.h"

using namespace gpt;

TEST(StatusTest, DefaultIsOk) {
  Status s;
  EXPECT_TRUE(s.ok());
  EXPECT_EQ(s.code(), StatusCode::kSuccess);
}

TEST(StatusTest, ErrorCarriesMessage) {
  Status s(StatusCode::kInvalidArgument, "bad shape");
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(s.message(), "bad shape");
}

TEST(ResultTest, HoldsValue) {
  Result<int> r(42);
  EXPECT_TRUE(r.ok());
  EXPECT_EQ(r.value(), 42);
}

TEST(ResultTest, HoldsError) {
  Result<int> r(Status(StatusCode::kOutOfMemory, "oom"));
  EXPECT_FALSE(r.ok());
  EXPECT_EQ(r.status().code(), StatusCode::kOutOfMemory);
}

TEST(ResultTest, TakeMovesValue) {
  Result<std::string> r(std::string("hello"));
  EXPECT_TRUE(r.ok());
  std::string val = r.take();
  EXPECT_EQ(val, "hello");
}
