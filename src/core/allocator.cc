// Layer 1 — Core runtime: allocator implementations.

#include "src/core/allocator.h"

#include <cuda_runtime.h>

namespace gpt {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

// ---------------------------------------------------------------------------
// PersistentArena
// ---------------------------------------------------------------------------

Status PersistentArena::init(size_t capacity_bytes) {
  if (base_) {
    return Status(StatusCode::kInvalidArgument,
                  "PersistentArena already initialized");
  }
  cudaError_t err = cudaMalloc(&base_, capacity_bytes);
  if (err != cudaSuccess) {
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));
  }
  capacity_ = capacity_bytes;
  offset_ = 0;
  return Status::Ok();
}

Result<void*> PersistentArena::allocate(size_t size, size_t alignment) {
  size_t aligned_offset = align_up(offset_, alignment);
  if (aligned_offset + size > capacity_) {
    return Status(StatusCode::kOutOfMemory, "PersistentArena exhausted");
  }
  void* ptr = static_cast<uint8_t*>(base_) + aligned_offset;
  offset_ = aligned_offset + size;
  return ptr;
}

void PersistentArena::release() {
  if (base_) {
    cudaFree(base_);
    base_ = nullptr;
    capacity_ = 0;
    offset_ = 0;
  }
}

PersistentArena::~PersistentArena() { release(); }

// ---------------------------------------------------------------------------
// WorkspaceArena
// ---------------------------------------------------------------------------

Status WorkspaceArena::init(size_t capacity_bytes) {
  if (base_) {
    return Status(StatusCode::kInvalidArgument,
                  "WorkspaceArena already initialized");
  }
  cudaError_t err = cudaMalloc(&base_, capacity_bytes);
  if (err != cudaSuccess) {
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));
  }
  capacity_ = capacity_bytes;
  offset_ = 0;
  return Status::Ok();
}

Result<void*> WorkspaceArena::allocate(size_t size, size_t alignment) {
  size_t aligned_offset = align_up(offset_, alignment);
  if (aligned_offset + size > capacity_) {
    return Status(StatusCode::kOutOfMemory, "WorkspaceArena exhausted");
  }
  void* ptr = static_cast<uint8_t*>(base_) + aligned_offset;
  offset_ = aligned_offset + size;
  return ptr;
}

void WorkspaceArena::release() {
  if (base_) {
    cudaFree(base_);
    base_ = nullptr;
    capacity_ = 0;
    offset_ = 0;
  }
}

WorkspaceArena::~WorkspaceArena() { release(); }

}  // namespace gpt
