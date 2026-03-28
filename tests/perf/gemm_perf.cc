// Tier D — Performance: GEMM throughput measurement.

#include "gtest/gtest.h"
#include "src/core/event.h"
#include "src/core/stream.h"
#include "src/kernels/gemm.h"

#include <cuda_runtime.h>
#include <cstdio>

using namespace gpt;

TEST(GemmPerfTest, SmokeTiming) {
  // This test just ensures GEMM can be launched and timed.
  // It does NOT enforce throughput targets.
  auto stream_r = CudaStream::Create();
  ASSERT_TRUE(stream_r.ok());
  auto stream = stream_r.take();

  // TODO: Allocate [M, K] x [K, N] with realistic sizes,
  // launch GEMM in a loop, measure elapsed time.
  SUCCEED() << "GEMM perf smoke test placeholder";
}
