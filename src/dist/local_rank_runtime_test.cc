// Tier A — Host tests for local-rank environment parsing.

#include "gtest/gtest.h"
#include "src/dist/local_rank_runtime.h"

#include <string>

using namespace gpt;
using namespace gpt::dist;

TEST(LocalRankRuntimeTest, DefaultsToSingleRank) {
  Result<LocalRankEnv> env = parse_local_rank_env(nullptr, nullptr, nullptr,
                                                 nullptr);
  ASSERT_TRUE(env.ok()) << env.status().message();
  EXPECT_EQ(env.value().rank, 0);
  EXPECT_EQ(env.value().world_size, 1);
  EXPECT_EQ(env.value().local_gpu_id, 0);
  EXPECT_FALSE(env.value().has_unique_id);

  Result<NcclConfig> config = make_nccl_config(env.value());
  ASSERT_TRUE(config.ok()) << config.status().message();
  EXPECT_EQ(config.value().rank, 0);
  EXPECT_EQ(config.value().world_size, 1);
  EXPECT_EQ(config.value().local_gpu_id, 0);
  EXPECT_FALSE(config.value().has_unique_id());
}

TEST(LocalRankRuntimeTest, ParsesMultiRankWithUniqueId) {
  std::string hex(256, '0');
  hex[255] = 'a';
  Result<LocalRankEnv> env = parse_local_rank_env("1", "2", "7", hex.c_str());
  ASSERT_TRUE(env.ok()) << env.status().message();
  EXPECT_EQ(env.value().rank, 1);
  EXPECT_EQ(env.value().world_size, 2);
  EXPECT_EQ(env.value().local_gpu_id, 7);
  EXPECT_TRUE(env.value().has_unique_id);
  EXPECT_EQ(env.value().unique_id[127], 0x0a);

  Result<NcclConfig> config = make_nccl_config(env.value());
  ASSERT_TRUE(config.ok()) << config.status().message();
  EXPECT_EQ(config.value().rank, 1);
  EXPECT_EQ(config.value().world_size, 2);
  EXPECT_EQ(config.value().local_gpu_id, 7);
  EXPECT_TRUE(config.value().has_unique_id());
  EXPECT_EQ(config.value().unique_id[127], 0x0a);
}

TEST(LocalRankRuntimeTest, RejectsMultiRankWithoutUniqueId) {
  Result<LocalRankEnv> env = parse_local_rank_env("0", "2", "0", nullptr);
  ASSERT_FALSE(env.ok());
  EXPECT_EQ(env.status().code(), StatusCode::kInvalidArgument);
}

TEST(LocalRankRuntimeTest, RejectsMalformedUniqueId) {
  Result<LocalRankEnv> env = parse_local_rank_env("0", "2", "0", "abc");
  ASSERT_FALSE(env.ok());
  EXPECT_EQ(env.status().code(), StatusCode::kInvalidArgument);
}

TEST(LocalRankRuntimeTest, RejectsRankOutOfRange) {
  Result<LocalRankEnv> env = parse_local_rank_env("2", "2", "0", nullptr);
  ASSERT_FALSE(env.ok());
  EXPECT_EQ(env.status().code(), StatusCode::kInvalidArgument);
}
