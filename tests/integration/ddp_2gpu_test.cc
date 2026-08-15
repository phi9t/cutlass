// Tier C — Integration: 2-GPU DDP correctness test.

#include "gtest/gtest.h"
#include "src/core/stream.h"
#include "src/dist/ddp.h"
#include "src/dist/local_rank_runtime.h"
#include "src/dist/nccl_context.h"

#include <cuda_runtime.h>
#include <cstring>
#include <thread>
#include <vector>

using namespace gpt;

namespace {

std::string UniqueIdHex(unsigned char const unique_id[128]) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.resize(256);
  for (int i = 0; i < 128; ++i) {
    out[2 * i] = kHex[unique_id[i] >> 4];
    out[2 * i + 1] = kHex[unique_id[i] & 0x0f];
  }
  return out;
}

}  // namespace

TEST(DDP2GPUTest, TwoRanksAverageGradientBuckets) {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  ASSERT_EQ(err, cudaSuccess);
  if (device_count < 2) {
    GTEST_SKIP() << "DDP 2-GPU test requires at least two visible GPUs";
  }

  unsigned char unique_id[128] = {};
  auto status = dist::create_nccl_unique_id(unique_id);
  ASSERT_TRUE(status.ok()) << status.message();
  std::string unique_id_hex = UniqueIdHex(unique_id);

  const int64_t count = 4;
  std::vector<float> rank0 = {1.0f, 3.0f, 5.0f, 7.0f};
  std::vector<float> rank1 = {9.0f, 7.0f, 5.0f, 3.0f};
  std::vector<float> expected = {5.0f, 5.0f, 5.0f, 5.0f};

  float* d_rank0 = nullptr;
  float* d_rank1 = nullptr;
  cudaSetDevice(0);
  cudaMalloc(&d_rank0, count * sizeof(float));
  cudaMemcpy(d_rank0, rank0.data(), count * sizeof(float),
             cudaMemcpyHostToDevice);
  auto stream0_result = CudaStream::Create();
  ASSERT_TRUE(stream0_result.ok()) << stream0_result.status().message();
  CudaStream stream0 = stream0_result.take();

  cudaSetDevice(1);
  cudaMalloc(&d_rank1, count * sizeof(float));
  cudaMemcpy(d_rank1, rank1.data(), count * sizeof(float),
             cudaMemcpyHostToDevice);
  auto stream1_result = CudaStream::Create();
  ASSERT_TRUE(stream1_result.ok()) << stream1_result.status().message();
  CudaStream stream1 = stream1_result.take();

  auto env0 = dist::parse_local_rank_env("0", "2", "0", unique_id_hex.c_str());
  ASSERT_TRUE(env0.ok()) << env0.status().message();
  auto env1 = dist::parse_local_rank_env("1", "2", "1", unique_id_hex.c_str());
  ASSERT_TRUE(env1.ok()) << env1.status().message();
  auto config0_result = dist::make_nccl_config(env0.value());
  ASSERT_TRUE(config0_result.ok()) << config0_result.status().message();
  auto config1_result = dist::make_nccl_config(env1.value());
  ASSERT_TRUE(config1_result.ok()) << config1_result.status().message();
  dist::NcclConfig config0 = config0_result.value();
  dist::NcclConfig config1 = config1_result.value();

  dist::NcclContext nccl0;
  dist::NcclContext nccl1;
  Status init0;
  Status init1;
  std::thread init_thread0([&] {
    init0 = nccl0.init(config0);
  });
  std::thread init_thread1([&] {
    init1 = nccl1.init(config1);
  });
  init_thread0.join();
  init_thread1.join();
  ASSERT_TRUE(init0.ok()) << init0.message();
  ASSERT_TRUE(init1.ok()) << init1.message();

  dist::DDP ddp0;
  dist::DDP ddp1;
  dist::DDPConfig ddp_config;
  status = ddp0.init(d_rank0, count, ddp_config);
  ASSERT_TRUE(status.ok()) << status.message();
  status = ddp1.init(d_rank1, count, ddp_config);
  ASSERT_TRUE(status.ok()) << status.message();

  Status reduce0;
  Status reduce1;
  std::thread reduce_thread0([&] {
    cudaSetDevice(0);
    reduce0 = ddp0.all_reduce_grads(nccl0, stream0);
  });
  std::thread reduce_thread1([&] {
    cudaSetDevice(1);
    reduce1 = ddp1.all_reduce_grads(nccl1, stream1);
  });
  reduce_thread0.join();
  reduce_thread1.join();
  ASSERT_TRUE(reduce0.ok()) << reduce0.message();
  ASSERT_TRUE(reduce1.ok()) << reduce1.message();
  ASSERT_TRUE(stream0.synchronize().ok());
  ASSERT_TRUE(stream1.synchronize().ok());

  std::vector<float> out0(count);
  std::vector<float> out1(count);
  cudaSetDevice(0);
  cudaMemcpy(out0.data(), d_rank0, count * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaSetDevice(1);
  cudaMemcpy(out1.data(), d_rank1, count * sizeof(float),
             cudaMemcpyDeviceToHost);

  EXPECT_EQ(out0, expected);
  EXPECT_EQ(out1, expected);

  EXPECT_TRUE(nccl0.destroy().ok());
  EXPECT_TRUE(nccl1.destroy().ok());
  cudaSetDevice(0);
  cudaFree(d_rank0);
  cudaSetDevice(1);
  cudaFree(d_rank1);
}

TEST(NcclContextTest, SingleRankAllReduceIsIdentity) {
  auto stream_result = CudaStream::Create();
  ASSERT_TRUE(stream_result.ok());
  CudaStream stream = stream_result.take();

  dist::NcclContext nccl;
  dist::NcclConfig config;
  config.world_size = 1;
  config.rank = 0;
  config.local_gpu_id = 0;

  auto status = nccl.init(config);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_TRUE(nccl.is_initialized());

  std::vector<float> h_input = {1.0f, -2.0f, 3.5f, 4.0f};
  float* d_buffer = nullptr;
  cudaMalloc(&d_buffer, h_input.size() * sizeof(float));
  cudaMemcpy(d_buffer, h_input.data(), h_input.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  status = nccl.all_reduce_sum(d_buffer, static_cast<int64_t>(h_input.size()),
                               stream);
  ASSERT_TRUE(status.ok()) << status.message();
  stream.synchronize();

  std::vector<float> h_output(h_input.size());
  cudaMemcpy(h_output.data(), d_buffer, h_output.size() * sizeof(float),
             cudaMemcpyDeviceToHost);

  EXPECT_EQ(h_output, h_input);

  cudaFree(d_buffer);
  EXPECT_TRUE(nccl.destroy().ok());
  EXPECT_FALSE(nccl.is_initialized());
}
