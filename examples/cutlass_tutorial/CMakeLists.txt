# CUTLASS Tutorial CMakeLists.txt
#
# This CMake file builds the CUTLASS tutorial examples.
# It integrates with the CUTLASS build system and handles dependencies automatically.
#
# Usage:
#   mkdir build && cd build
#   cmake .. -DCUTLASS_NVCC_ARCHS="80;90"
#   make -j
#
# Or from CUTLASS root:
#   cmake -B build -DCUTLASS_ENABLE_EXAMPLES=ON
#   cmake --build build --target cutlass_tutorial_examples

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(cutlass_tutorial LANGUAGES CXX CUDA)

# Find CUTLASS
if(NOT TARGET CUTLASS)
  # If building standalone, find CUTLASS
  set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "CUTLASS root directory")

  if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR
      "CUTLASS not found at ${CUTLASS_DIR}. "
      "Please set CUTLASS_DIR to the CUTLASS root directory.")
  endif()

  message(STATUS "CUTLASS root: ${CUTLASS_DIR}")

  # Add CUTLASS include directories
  include_directories(${CUTLASS_DIR}/include)
  include_directories(${CUTLASS_DIR}/tools/util/include)
  include_directories(${CUTLASS_DIR}/tools/library/include)
endif()

# CUDA architecture settings
if(NOT DEFINED CUTLASS_NVCC_ARCHS)
  # Default to Ampere (SM80)
  set(CUTLASS_NVCC_ARCHS "80" CACHE STRING "CUDA architectures to compile for")
endif()

message(STATUS "Building for CUDA architectures: ${CUTLASS_NVCC_ARCHS}")

# Convert architecture list to gencode flags
set(CUDA_ARCH_FLAGS "")
foreach(ARCH ${CUTLASS_NVCC_ARCHS})
  list(APPEND CUDA_ARCH_FLAGS "-gencode arch=compute_${ARCH},code=sm_${ARCH}")
endforeach()

# CUDA flags
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -lineinfo")

# CXX flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Tutorial sources
set(TUTORIAL_SOURCES
    01_basic_gemm_volta.cu
    02_ampere_async.cu
    03_hopper_tma_gmma.cu
    04_cute_layouts.cu
    05_collective_builders.cu
    06_mixed_precision.cu
    07_epilogue_fusion.cu
)

# Create a target for each tutorial
foreach(TUTORIAL_SOURCE ${TUTORIAL_SOURCES})
  # Extract tutorial name (remove .cu extension)
  get_filename_component(TUTORIAL_NAME ${TUTORIAL_SOURCE} NAME_WE)

  # Create executable target
  add_executable(${TUTORIAL_NAME} ${TUTORIAL_SOURCE})

  # Link against CUDA libraries
  target_link_libraries(${TUTORIAL_NAME}
    PRIVATE
      CUDA::cudart
      CUDA::cuda_driver
  )

  # Set target properties
  set_target_properties(${TUTORIAL_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CUTLASS_NVCC_ARCHS}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )

  # Add to tutorial list
  list(APPEND TUTORIAL_TARGETS ${TUTORIAL_NAME})

  message(STATUS "Configured tutorial: ${TUTORIAL_NAME}")
endforeach()

# Create a target to build all tutorials
add_custom_target(cutlass_tutorial_examples
  DEPENDS ${TUTORIAL_TARGETS}
  COMMENT "Building all CUTLASS tutorial examples"
)

# Install targets (optional)
if(CUTLASS_INSTALL_EXAMPLES)
  install(
    TARGETS ${TUTORIAL_TARGETS}
    RUNTIME DESTINATION bin/examples/cutlass_tutorial
  )

  install(
    FILES README.md
    DESTINATION share/cutlass/examples/cutlass_tutorial
  )
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "CUTLASS Tutorial Configuration:")
message(STATUS "  CUDA Architectures: ${CUTLASS_NVCC_ARCHS}")
message(STATUS "  CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Tutorials: ${TUTORIAL_TARGETS}")
message(STATUS "")

# Add tests (optional)
option(CUTLASS_ENABLE_TUTORIAL_TESTS "Enable tutorial tests" OFF)

if(CUTLASS_ENABLE_TUTORIAL_TESTS AND BUILD_TESTING)
  enable_testing()

  foreach(TUTORIAL_TARGET ${TUTORIAL_TARGETS})
    add_test(
      NAME test_${TUTORIAL_TARGET}
      COMMAND ${TUTORIAL_TARGET}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    # Set test properties
    set_tests_properties(test_${TUTORIAL_TARGET} PROPERTIES
      LABELS "tutorial"
      TIMEOUT 300  # 5 minutes
    )
  endforeach()

  message(STATUS "Tutorial tests enabled")
endif()

# Print build instructions
message(STATUS "")
message(STATUS "To build tutorials:")
message(STATUS "  cmake --build ${CMAKE_BINARY_DIR} --target cutlass_tutorial_examples")
message(STATUS "")
message(STATUS "To run a specific tutorial:")
message(STATUS "  ${CMAKE_BINARY_DIR}/bin/01_basic_gemm_volta")
message(STATUS "")
