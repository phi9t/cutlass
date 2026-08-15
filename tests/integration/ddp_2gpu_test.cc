// Tier C — Integration: 2-GPU DDP correctness test.
// Placeholder — will be filled in Milestone 7.

#include "gtest/gtest.h"
#include "src/core/stream.h"
#include "src/dist/nccl_context.h"

#include <cuda_runtime.h>
#include <vector>

using namespace gpt;

TEST(DDP2GPUTest, Placeholder) {
  // TODO(m7): Spawn 2 ranks, run 1 training step each,
  // verify averaged gradients match single-GPU reference.
  SUCCEED() << "DDP 2-GPU test pending NCCL wiring";
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
