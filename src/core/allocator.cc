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

// ---------------------------------------------------------------------------
// DeviceScratchArena
// ---------------------------------------------------------------------------

Status DeviceScratchArena::reserve_bytes(size_t bytes) {
  if (bytes <= capacity_) {
    return Status::Ok();
  }

  release();
  cudaError_t err = cudaMalloc(&base_, bytes);
  if (err != cudaSuccess) {
    base_ = nullptr;
    capacity_ = 0;
    offset_ = 0;
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));
  }
  capacity_ = bytes;
  offset_ = 0;
  return Status::Ok();
}

Result<float*> DeviceScratchArena::alloc_f32(int64_t count) {
  if (count < 0) {
    return Status(StatusCode::kInvalidArgument,
                  "DeviceScratchArena negative allocation count");
  }
  Result<void*> ptr = alloc_bytes(static_cast<size_t>(count) * sizeof(float));
  if (!ptr.ok()) {
    return ptr.status();
  }
  return static_cast<float*>(ptr.value());
}

Result<void*> DeviceScratchArena::alloc_bytes(size_t bytes, size_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return Status(StatusCode::kInvalidArgument,
                  "DeviceScratchArena alignment must be a power of two");
  }
  size_t aligned_offset = align_up(offset_, alignment);
  if (aligned_offset + bytes > capacity_) {
    return Status(StatusCode::kOutOfMemory, "DeviceScratchArena exhausted");
  }
  void* ptr = static_cast<uint8_t*>(base_) + aligned_offset;
  offset_ = aligned_offset + bytes;
  return ptr;
}

void DeviceScratchArena::release() {
  if (base_) {
    cudaFree(base_);
    base_ = nullptr;
    capacity_ = 0;
    offset_ = 0;
  }
}

DeviceScratchArena::~DeviceScratchArena() { release(); }

}  // namespace gpt
