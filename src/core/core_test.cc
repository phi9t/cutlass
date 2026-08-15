// Tier A — Host-only unit tests for core runtime.

#include "gtest/gtest.h"
#include "src/core/allocator.h"
#include "src/core/status.h"

#include <cstdint>

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

TEST(DeviceScratchArenaTest, ResetReusesBackingAllocation) {
  DeviceScratchArena arena;
  ASSERT_TRUE(arena.reserve_bytes(1024).ok());
  EXPECT_EQ(arena.capacity(), static_cast<size_t>(1024));

  Result<float*> first = arena.alloc_f32(16);
  ASSERT_TRUE(first.ok()) << first.status().message();
  EXPECT_GT(arena.used(), static_cast<size_t>(0));

  arena.reset();
  EXPECT_EQ(arena.used(), static_cast<size_t>(0));

  Result<float*> second = arena.alloc_f32(16);
  ASSERT_TRUE(second.ok()) << second.status().message();
  EXPECT_EQ(second.value(), first.value());

  Result<void*> aligned = arena.alloc_bytes(1, 512);
  ASSERT_TRUE(aligned.ok()) << aligned.status().message();
  EXPECT_EQ(reinterpret_cast<uintptr_t>(aligned.value()) % 512,
            static_cast<uintptr_t>(0));
}

TEST(DeviceScratchArenaTest, ExhaustionReturnsOutOfMemory) {
  DeviceScratchArena arena;
  ASSERT_TRUE(arena.reserve_bytes(16).ok());

  Result<void*> allocation = arena.alloc_bytes(32);
  ASSERT_FALSE(allocation.ok());
  EXPECT_EQ(allocation.status().code(), StatusCode::kOutOfMemory);
}
