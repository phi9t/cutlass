/***************************************************************************************************
 * CUTLASS Tutorial 04: CuTe Layouts - The Foundation of CUTLASS 3.x
 *
 * This tutorial explores CuTe (Composable Unreified Template Expressions), the mathematical
 * foundation that powers CUTLASS 3.x. CuTe provides a compile-time tensor algebra system
 * where all indexing, layout transformations, and memory access patterns are resolved at
 * compile time, resulting in zero runtime overhead.
 *
 * EDUCATIONAL FOCUS:
 * CuTe replaces ad-hoc indexing with a principled algebra of Shapes, Strides, and Layouts.
 * Instead of manually calculating offsets like `ptr[row * lda + col]`, you compose layouts
 * that describe the mathematical structure, and let the compiler generate optimal code.
 *
 * LEARNING OBJECTIVES:
 * 1. Understand Shape: Compile-time sizes (e.g., Shape<_4, _8> = 4×8 matrix)
 * 2. Understand Stride: Memory traversal pattern (e.g., row-major vs column-major)
 * 3. Learn Layout composition: Shape ⊗ Stride = complete coordinate→address mapping
 * 4. Explore hierarchical layouts: Nested structures for tiling and blocking
 * 5. See practical examples: How CuTe enables CUTLASS 3.x's flexibility
 *
 * KEY CONCEPTS:
 * - Integral constants (_1, _2, _4, etc.): Compile-time values for optimization
 * - Layout algebra: Compose, slice, dice, reshape layouts
 * - Coordinate spaces: Logical (how you think) vs Physical (how memory is laid out)
 * - Hierarchical composition: Threadblock → Warp → Thread mappings
 *
 * COMPILE:
 *   nvcc -arch=sm_80 -std=c++17 -I../../include 04_cute_layouts.cu -o 04_cute_layouts
 *
 * RUN:
 *   ./04_cute_layouts
 *
 * EXPECTED OUTPUT:
 *   Various layout examples demonstrating Shape, Stride, and composition patterns
 ***************************************************************************************************/

#include <iostream>
#include <iomanip>

// CuTe headers - the foundation of CUTLASS 3.x
#include "cute/tensor.hpp"
#include "cute/layout.hpp"

using namespace cute;  // CuTe lives in the cute:: namespace

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 1: BASIC CONCEPTS - Shape and Stride
//
// EDUCATIONAL NOTE - What is a Shape?
// A Shape describes the size of each dimension at compile time.
// Examples:
//   Shape<_4>           - 1D shape of size 4
//   Shape<_4, _8>       - 2D shape of size 4×8
//   Shape<_2, _3, _4>   - 3D shape of size 2×3×4
//
// The underscore prefix (_4, _8, etc.) indicates integral_constant<int, N>.
// This means the value is known at compile time and can be used for template
// instantiation, loop unrolling, and other optimizations.
//
// EDUCATIONAL NOTE - What is a Stride?
// A Stride describes how to traverse memory to access elements.
// For a 2D array stored in memory:
//   Row-major: stride = <N, 1> (increment column first)
//   Column-major: stride = <1, M> (increment row first)
//
// Examples for a 4×8 matrix:
//   Row-major: Stride<_8, _1>   - row*8 + col*1
//   Column-major: Stride<_1, _4> - row*1 + col*4
//
///////////////////////////////////////////////////////////////////////////////////////////////////

void example_01_basic_shapes() {
  std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║   Example 1: Basic Shapes and Strides                         ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝\n" << std::endl;

  // 1D shape
  auto shape_1d = Shape<_8>{};
  std::cout << "1D Shape<_8>: " << shape_1d << std::endl;
  std::cout << "  Size: " << size(shape_1d) << " elements" << std::endl;

  // 2D shape
  auto shape_2d = Shape<_4, _8>{};
  std::cout << "\n2D Shape<_4, _8>: " << shape_2d << std::endl;
  std::cout << "  Size: " << size(shape_2d) << " elements" << std::endl;
  std::cout << "  Rank: " << rank(shape_2d) << " dimensions" << std::endl;

  // 3D hierarchical shape
  auto shape_3d = Shape<_2, _3, _4>{};
  std::cout << "\n3D Shape<_2, _3, _4>: " << shape_3d << std::endl;
  std::cout << "  Size: " << size(shape_3d) << " elements" << std::endl;
  std::cout << "  Rank: " << rank(shape_3d) << " dimensions" << std::endl;

  // Row-major stride for 4×8 matrix
  auto stride_rowmajor = Stride<_8, _1>{};
  std::cout << "\nRow-major Stride<_8, _1>: " << stride_rowmajor << std::endl;
  std::cout << "  Element [2,3] offset: " << stride_rowmajor(_2{}, _3{}) << std::endl;
  std::cout << "  Calculation: 2*8 + 3*1 = " << 2*8 + 3*1 << std::endl;

  // Column-major stride for 4×8 matrix
  auto stride_colmajor = Stride<_1, _4>{};
  std::cout << "\nColumn-major Stride<_1, _4>: " << stride_colmajor << std::endl;
  std::cout << "  Element [2,3] offset: " << stride_colmajor(_2{}, _3{}) << std::endl;
  std::cout << "  Calculation: 2*1 + 3*4 = " << 2*1 + 3*4 << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 2: LAYOUTS - Composing Shape and Stride
//
// EDUCATIONAL NOTE - What is a Layout?
// A Layout is the composition of a Shape and a Stride: Layout = (Shape, Stride)
// It provides a complete mapping from logical coordinates to linear memory offsets.
//
// Key operations:
//   size(layout)       - Total number of elements
//   shape(layout)      - Extract the shape
//   stride(layout)     - Extract the stride
//   layout(coord...)   - Map coordinates to linear offset
//   layout(offset)     - Map linear offset to coordinates (inverse)
//
// CuTe uses layouts everywhere in CUTLASS 3.x:
//   - Tensor layouts for global memory
//   - Shared memory layouts with swizzling
//   - Thread→Data mappings
//   - Tile decompositions
//
///////////////////////////////////////////////////////////////////////////////////////////////////

void example_02_layouts() {
  std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║   Example 2: Layouts - Composing Shape and Stride             ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝\n" << std::endl;

  // Row-major 4×8 layout
  auto layout_rowmajor = make_layout(Shape<_4, _8>{}, Stride<_8, _1>{});
  std::cout << "Row-major Layout (4×8):" << std::endl;
  std::cout << "  Layout: " << layout_rowmajor << std::endl;
  std::cout << "  Size: " << size(layout_rowmajor) << " elements" << std::endl;
  std::cout << "  Shape: " << shape(layout_rowmajor) << std::endl;
  std::cout << "  Stride: " << stride(layout_rowmajor) << std::endl;

  // Access element [2, 3]
  std::cout << "\n  Accessing element [2, 3]:" << std::endl;
  std::cout << "    Offset: " << layout_rowmajor(_2{}, _3{}) << std::endl;

  // Column-major 4×8 layout
  auto layout_colmajor = make_layout(Shape<_4, _8>{}, Stride<_1, _4>{});
  std::cout << "\nColumn-major Layout (4×8):" << std::endl;
  std::cout << "  Layout: " << layout_colmajor << std::endl;
  std::cout << "  Accessing element [2, 3]: offset " << layout_colmajor(_2{}, _3{}) << std::endl;

  // EDUCATIONAL NOTE - Compact Layout:
  // A compact layout has no gaps in memory. You can create one with make_layout(shape)
  // which automatically deduces compact strides.
  auto layout_compact = make_layout(Shape<_4, _8>{});
  std::cout << "\nCompact Layout (auto-generated strides):" << std::endl;
  std::cout << "  Layout: " << layout_compact << std::endl;
  std::cout << "  Note: Automatically uses column-major stride for better coalescing" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 3: HIERARCHICAL LAYOUTS - Tiling and Blocking
//
// EDUCATIONAL NOTE - Hierarchical Composition:
// CuTe's real power comes from hierarchical layouts, which enable multi-level tiling.
// A hierarchical shape looks like: Shape<Shape<_2, _3>, Shape<_4, _5>>
//
// This represents a 2-level hierarchy:
//   Outer level: 2×3 blocks
//   Inner level: Each block is 4×5 elements
//   Total size: (2×3) × (4×5) = 6 × 20 = 120 elements
//
// This is how CUTLASS 3.x represents:
//   - Threadblock tiles within global memory
//   - Warp tiles within threadblock tiles
//   - Thread tiles within warp tiles
//
// Example: Threadblock tile of 128×128, subdivided into 4 warp tiles of 64×64 each
//   Shape<Shape<_2, _2>, Shape<_64, _64>>
//   = 2×2 warps, each processing 64×64 elements
//
///////////////////////////////////////////////////////////////////////////////////////////////////

void example_03_hierarchical_layouts() {
  std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║   Example 3: Hierarchical Layouts - Multi-level Tiling        ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝\n" << std::endl;

  // Simple hierarchical shape: 2×3 blocks, each 4×5 elements
  auto shape_hier = Shape<Shape<_2, _3>, Shape<_4, _5>>{};
  std::cout << "Hierarchical Shape:" << std::endl;
  std::cout << "  Shape<Shape<_2, _3>, Shape<_4, _5>>: " << shape_hier << std::endl;
  std::cout << "  Interpretation: (2×3 blocks) × (4×5 elements per block)" << std::endl;
  std::cout << "  Total size: " << size(shape_hier) << " elements" << std::endl;
  std::cout << "  Outer shape: " << shape<0>(shape_hier) << " (blocks)" << std::endl;
  std::cout << "  Inner shape: " << shape<1>(shape_hier) << " (elements per block)" << std::endl;

  // EDUCATIONAL NOTE - Practical Example: Threadblock Tiling
  // In a real GEMM, you might have:
  //   - Global matrix: 1024×1024
  //   - Threadblock tile: 128×128
  //   - Number of threadblocks: (1024/128) × (1024/128) = 8×8 = 64 CTAs
  //
  // This can be represented as:
  //   Shape<Shape<_8, _8>, Shape<_128, _128>>
  //   = 8×8 threadblocks, each processing 128×128 elements

  auto shape_gemm_tiling = Shape<Shape<_8, _8>, Shape<_128, _128>>{};
  std::cout << "\nGEMM Tiling Example:" << std::endl;
  std::cout << "  Global matrix decomposition: " << shape_gemm_tiling << std::endl;
  std::cout << "  8×8 threadblocks, each 128×128 elements" << std::endl;
  std::cout << "  Total coverage: " << size(shape_gemm_tiling) << " elements" << std::endl;
  std::cout << "  = " << (8*128) << "×" << (8*128) << " matrix" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 4: LAYOUT ALGEBRA - Composition and Transformation
//
// EDUCATIONAL NOTE - Layout Transformations:
// CuTe provides powerful operations to manipulate layouts:
//
//   composition(layoutA, layoutB) - Compose two layouts
//   complement(layout, shape)     - Create complement layout for the remaining space
//   make_tile(layout, shape)      - Tile a layout into blocks
//   coalesce(layout)              - Merge consecutive dimensions
//   filter(layout)                - Remove dimensions of size 1
//
// These operations are used heavily in CUTLASS 3.x to:
//   - Transform global layouts to threadblock layouts
//   - Apply swizzling patterns to shared memory
//   - Map thread IDs to data elements
//
///////////////////////////////////////////////////////////////////////////////////////////////////

void example_04_layout_algebra() {
  std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║   Example 4: Layout Algebra - Composition and Transformation  ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝\n" << std::endl;

  // Start with a simple 8×8 row-major layout
  auto layout_base = make_layout(Shape<_8, _8>{}, Stride<_8, _1>{});
  std::cout << "Base layout (8×8 row-major): " << layout_base << std::endl;

  // Tile it into 2×2 blocks of 4×4 elements each
  auto layout_tiled = make_layout(
    Shape<Shape<_2, _2>, Shape<_4, _4>>{},
    Stride<Stride<_32, _4>, Stride<_8, _1>>{}
  );
  std::cout << "\nTiled layout (2×2 blocks of 4×4):" << std::endl;
  std::cout << "  " << layout_tiled << std::endl;
  std::cout << "  Outer blocks: 2×2 with stride (32, 4)" << std::endl;
  std::cout << "  Inner elements: 4×4 with stride (8, 1)" << std::endl;

  // EDUCATIONAL NOTE - Understanding Hierarchical Strides:
  // For the tiled layout above:
  //   - To move to next block in first dimension: +32 (skip 4 rows × 8 cols)
  //   - To move to next block in second dimension: +4 (skip 4 columns)
  //   - Within a block, use standard row-major: stride (8, 1)
  //
  // This enables efficient blocked access patterns used in:
  //   - Shared memory tiling
  //   - Register blocking
  //   - Swizzled layouts for bank conflict avoidance

  std::cout << "\nKey insight: Hierarchical strides enable multi-level access patterns!" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 5: TENSORS - Layouts + Data
//
// EDUCATIONAL NOTE - What is a Tensor in CuTe?
// A Tensor combines a Layout with a data pointer: Tensor = (Layout, Pointer)
// This gives you a mathematically-principled view of memory where:
//   - Layout describes the coordinate→address mapping
//   - Pointer provides the base address
//   - Indexing is automatic via operator()
//
// Example:
//   Tensor t = make_tensor(ptr, layout);
//   auto elem = t(2, 3);  // Automatically computes: ptr[layout(2, 3)]
//
// CuTe tensors are used for:
//   - Global memory tensors (input/output matrices)
//   - Shared memory tensors (threadblock tiles)
//   - Register tensors (thread-local data)
//   - Predication tensors (boundary handling)
//
///////////////////////////////////////////////////////////////////////////////////////////////////

void example_05_tensors() {
  std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║   Example 5: Tensors - Layouts + Data                         ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝\n" << std::endl;

  // Allocate a 4×8 matrix in row-major order
  constexpr int M = 4;
  constexpr int N = 8;
  float data[M * N];

  // Initialize with sequential values
  for (int i = 0; i < M * N; ++i) {
    data[i] = static_cast<float>(i);
  }

  // Create a CuTe tensor with row-major layout
  auto layout = make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{});
  auto tensor = make_tensor(data, layout);

  std::cout << "Row-major Tensor (4×8):" << std::endl;
  std::cout << "  Layout: " << tensor.layout() << std::endl;
  std::cout << "  Size: " << size(tensor) << " elements" << std::endl;

  // Access elements using multi-dimensional indexing
  std::cout << "\nAccessing elements:" << std::endl;
  std::cout << "  tensor(0, 0) = " << tensor(_0{}, _0{}) << std::endl;
  std::cout << "  tensor(1, 2) = " << tensor(_1{}, _2{}) << std::endl;
  std::cout << "  tensor(2, 5) = " << tensor(_2{}, _5{}) << std::endl;

  // Print the full matrix
  std::cout << "\nFull matrix visualization:" << std::endl;
  for (int i = 0; i < M; ++i) {
    std::cout << "  ";
    for (int j = 0; j < N; ++j) {
      std::cout << std::setw(5) << std::fixed << std::setprecision(1) << tensor(i, j) << " ";
    }
    std::cout << std::endl;
  }

  // EDUCATIONAL NOTE - Thread→Data Mapping:
  // In CUTLASS 3.x, tensors are combined with thread layouts to create automatic
  // distribution of data to threads. For example:
  //
  //   auto thread_layout = make_layout(Shape<_4, _8>{});  // 32 threads
  //   auto data_layout = make_layout(Shape<_64, _64>{});  // 64×64 data
  //   auto thr_tensor = local_partition(data_layout, thread_layout, thread_id);
  //
  // This automatically computes which elements each thread should process!

  std::cout << "\nKey insight: Tensors = Layouts + Pointers, enabling automatic indexing!" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 6: PRACTICAL EXAMPLE - GEMM Tile Mapping
//
// EDUCATIONAL NOTE - How CuTe Powers CUTLASS 3.x GEMM:
// Let's see how layouts are used to map a GEMM operation to hardware.
//
// Given: C[M,N] = A[M,K] × B[K,N]
//
// 1. Global tile: Shape<_128, _128, _32> (M×N×K threadblock tile)
// 2. Thread layout: Shape<_32, _4> (32 threads × 4 warps = 128 threads)
// 3. Data partition: Each thread gets a portion of the tile
//
// CuTe automatically computes:
//   - Which elements each thread loads from global memory
//   - How to arrange data in shared memory (with swizzling)
//   - Which elements each thread computes on
//
///////////////////////////////////////////////////////////////////////////////////////////////////

void example_06_gemm_mapping() {
  std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║   Example 6: GEMM Tile Mapping (Conceptual)                   ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝\n" << std::endl;

  // Threadblock tile shape for GEMM: 128×128×32 (M×N×K)
  auto tile_shape_m = _128{};
  auto tile_shape_n = _128{};
  auto tile_shape_k = _32{};

  std::cout << "GEMM Threadblock Tile:" << std::endl;
  std::cout << "  M dimension: " << tile_shape_m << " elements" << std::endl;
  std::cout << "  N dimension: " << tile_shape_n << " elements" << std::endl;
  std::cout << "  K dimension: " << tile_shape_k << " elements" << std::endl;

  // Thread layout: 32×4 = 128 threads (1 warp × 4 warps)
  auto thread_layout = make_layout(Shape<_32, _4>{});
  std::cout << "\nThread Layout:" << std::endl;
  std::cout << "  Shape: " << thread_layout << std::endl;
  std::cout << "  Total threads: " << size(thread_layout) << std::endl;

  // A matrix tile layout (M×K): 128×32
  auto layout_A = make_layout(Shape<_128, _32>{});
  std::cout << "\nA Matrix Tile (M×K): " << layout_A << std::endl;
  std::cout << "  Elements per thread: " << size(layout_A) / size(thread_layout) << std::endl;

  // B matrix tile layout (K×N): 32×128
  auto layout_B = make_layout(Shape<_32, _128>{});
  std::cout << "\nB Matrix Tile (K×N): " << layout_B << std::endl;
  std::cout << "  Elements per thread: " << size(layout_B) / size(thread_layout) << std::endl;

  // C matrix tile layout (M×N): 128×128
  auto layout_C = make_layout(Shape<_128, _128>{});
  std::cout << "\nC Matrix Tile (M×N): " << layout_C << std::endl;
  std::cout << "  Elements per thread: " << size(layout_C) / size(thread_layout) << std::endl;

  std::cout << "\nEDUCATIONAL NOTE:" << std::endl;
  std::cout << "In CUTLASS 3.x, these layouts are combined with thread IDs to automatically:" << std::endl;
  std::cout << "  1. Partition data across threads (local_partition)" << std::endl;
  std::cout << "  2. Generate global→shared copies (copy with TiledCopy)" << std::endl;
  std::cout << "  3. Perform Tensor Core operations (MMA with TiledMMA)" << std::endl;
  std::cout << "  4. Store results back to global memory (copy with TiledCopy)" << std::endl;
  std::cout << "\nAll coordination is automatic via layout algebra - no manual indexing!" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// MAIN - Run all examples
//
///////////////////////////////////////////////////////////////////////////////////////////////////

int main() {
  std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║   CUTLASS Tutorial 04: CuTe Layouts                            ║" << std::endl;
  std::cout << "║   The Mathematical Foundation of CUTLASS 3.x                   ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;

  // Run examples
  example_01_basic_shapes();
  example_02_layouts();
  example_03_hierarchical_layouts();
  example_04_layout_algebra();
  example_05_tensors();
  example_06_gemm_mapping();

  std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║   KEY TAKEAWAYS                                                ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  1. Shape: Compile-time sizes for dimensions                   ║" << std::endl;
  std::cout << "║     (_4, _8, etc. = integral_constant for optimization)        ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  2. Stride: Memory traversal pattern                           ║" << std::endl;
  std::cout << "║     Row-major: <N, 1>, Column-major: <1, M>                    ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  3. Layout = Shape ⊗ Stride: Complete coordinate mapping       ║" << std::endl;
  std::cout << "║     All indexing resolved at compile time                      ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  4. Hierarchical composition: Multi-level tiling               ║" << std::endl;
  std::cout << "║     Shape<Shape<_2, _2>, Shape<_64, _64>> = 2×2 blocks         ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  5. Tensor = Layout + Pointer: Automatic indexing              ║" << std::endl;
  std::cout << "║     tensor(i, j) = ptr[layout(i, j)] computed automatically    ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  6. Layout algebra enables CUTLASS 3.x's flexibility           ║" << std::endl;
  std::cout << "║     Partition, transform, swizzle all via composable layouts   ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║   WHY CUTE MATTERS                                             ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  Before CuTe (CUTLASS 2.x):                                    ║" << std::endl;
  std::cout << "║    - Manual offset calculations: ptr[row*lda + col]            ║" << std::endl;
  std::cout << "║    - Hard-coded thread→data mappings                           ║" << std::endl;
  std::cout << "║    - Difficult to add new features (swizzling, layouts, etc.)  ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  After CuTe (CUTLASS 3.x):                                     ║" << std::endl;
  std::cout << "║    - Declarative layouts: describe structure, not computation  ║" << std::endl;
  std::cout << "║    - Automatic thread→data partitioning via layout algebra     ║" << std::endl;
  std::cout << "║    - Easy to extend: new layouts → new capabilities            ║" << std::endl;
  std::cout << "║    - Zero runtime cost: everything resolved at compile time    ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║   NEXT STEPS                                                   ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  • Next: 05_collective_builders.cu - See how CuTe powers       ║" << std::endl;
  std::cout << "║    the Collective pattern with automatic layout generation     ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  • Deep dive: Read include/cute/layout.hpp for full algebra    ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  • Experiment: Try creating your own hierarchical layouts      ║" << std::endl;
  std::cout << "║    - 3-level hierarchy: CTA → Warp → Thread                    ║" << std::endl;
  std::cout << "║    - Swizzled patterns for bank conflict avoidance             ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// APPENDIX: ADVANCED CUTE CONCEPTS
//
// EDUCATIONAL NOTE - Beyond the Basics:
//
// 1. SWIZZLING:
//    CuTe supports complex swizzling patterns to avoid shared memory bank conflicts.
//    Example: Swizzle<3, 0, 3> creates a XOR-based swizzle pattern.
//
// 2. COALESCE AND FILTER:
//    coalesce(layout) - Merges consecutive dimensions for simpler representation
//    filter(layout)   - Removes modes of size 1
//
// 3. COMPOSITION:
//    composition(layoutA, layoutB) - Composes two layouts
//    Used for multi-level transformations: Global → Shared → Register
//
// 4. COMPLEMENT:
//    complement(layout, shape) - Creates a layout for the "other" elements
//    Useful for partitioning data across threads
//
// 5. LOCAL_PARTITION:
//    local_partition(tensor, thread_layout, thread_id)
//    Automatically partitions a tensor across threads based on thread ID
//    This is the magic behind CUTLASS 3.x's automatic thread coordination!
//
// 6. TILED_COPY AND TILED_MMA:
//    TiledCopy - Describes how threads cooperate to copy data
//    TiledMMA  - Describes how threads cooperate to perform MMA operations
//    Both use layout algebra to avoid manual coordination
//
// These advanced concepts enable CUTLASS 3.x to support:
//    - Arbitrary matrix layouts (row-major, col-major, blocked, swizzled)
//    - Custom memory access patterns
//    - Automatic thread coordination
//    - Architecture-specific optimizations (TMA, cp.async, etc.)
//    - All with zero runtime overhead!
//
///////////////////////////////////////////////////////////////////////////////////////////////////
