// Tier A — Host-only unit tests for data loading.

#include "gtest/gtest.h"
#include "src/data/token_dataset.h"

#include <cstdio>
#include <cstdint>
#include <vector>

using namespace gpt::data;

class TokenDatasetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a small binary token file.
    path_ = "/tmp/gpt_test_tokens.bin";
    std::vector<int32_t> tokens(200);
    for (int i = 0; i < 200; ++i) tokens[i] = i % 50;
    FILE* f = std::fopen(path_.c_str(), "wb");
    std::fwrite(tokens.data(), sizeof(int32_t), tokens.size(), f);
    std::fclose(f);
  }

  void TearDown() override {
    std::remove(path_.c_str());
  }

  std::string path_;
};

TEST_F(TokenDatasetTest, InitSucceeds) {
  TokenDataset ds;
  DatasetConfig cfg;
  cfg.path = path_;
  cfg.seq_len = 16;
  cfg.batch_size = 2;
  auto status = ds.init(cfg);
  EXPECT_TRUE(status.ok()) << status.message();
  EXPECT_GT(ds.num_samples(), 0);
}

TEST_F(TokenDatasetTest, NextBatchFillsBuffers) {
  TokenDataset ds;
  DatasetConfig cfg;
  cfg.path = path_;
  cfg.seq_len = 8;
  cfg.batch_size = 2;
  ASSERT_TRUE(ds.init(cfg).ok());

  std::vector<int32_t> input(16), target(16);
  auto status = ds.next_batch(input.data(), target.data());
  EXPECT_TRUE(status.ok()) << status.message();

  // Target should be shifted by 1 from input.
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(target[i], (input[i] + 1) % 50);
  }
}

TEST_F(TokenDatasetTest, ResetRewindsCursor) {
  TokenDataset ds;
  DatasetConfig cfg;
  cfg.path = path_;
  cfg.seq_len = 8;
  cfg.batch_size = 2;
  ASSERT_TRUE(ds.init(cfg).ok());

  std::vector<int32_t> input(16), target(16);
  ds.next_batch(input.data(), target.data());
  EXPECT_GT(ds.cursor(), 0);

  ds.reset();
  EXPECT_EQ(ds.cursor(), 0);
}
