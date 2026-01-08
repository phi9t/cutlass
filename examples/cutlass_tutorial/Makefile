# CUTLASS Tutorial Makefile
#
# This Makefile compiles the CUTLASS tutorial examples.
# It assumes CUTLASS is located in ../../ (two directories up from this Makefile).
#
# Usage:
#   make                  # Build all tutorials
#   make 01               # Build specific tutorial
#   make clean            # Clean all build artifacts
#   make run-01           # Build and run tutorial 01
#
# Requirements:
#   - CUDA Toolkit (nvcc)
#   - CUTLASS headers (../../include)
#   - GPU with compute capability 7.0+ (Volta or newer)

# Compiler and flags
NVCC := nvcc
CXX_STD := c++17

# CUTLASS include path (adjust if needed)
CUTLASS_ROOT := ../..
CUTLASS_INCLUDE := -I$(CUTLASS_ROOT)/include
CUTLASS_TOOLS := -I$(CUTLASS_ROOT)/tools/util/include

# CUDA architecture flags
# SM70: Volta (V100)
# SM80: Ampere (A100, A30)
# SM90: Hopper (H100, H200)
ARCH_FLAGS := -arch=sm_80

# Compiler flags
NVCC_FLAGS := $(ARCH_FLAGS) \
              -std=$(CXX_STD) \
              $(CUTLASS_INCLUDE) \
              $(CUTLASS_TOOLS) \
              -O3 \
              --expt-relaxed-constexpr \
              --use_fast_math

# Debug flags (uncomment for debugging)
# NVCC_FLAGS += -g -G -lineinfo

# Tutorial sources
TUTORIALS := 01_basic_gemm_volta \
             02_ampere_async \
             03_hopper_tma_gmma \
             04_cute_layouts \
             05_collective_builders \
             06_mixed_precision \
             07_epilogue_fusion

# Binaries
BINS := $(addprefix bin/,$(TUTORIALS))

# Default target: build all
all: $(BINS)

# Create bin directory
bin:
	@mkdir -p bin

# Pattern rule for building tutorials
bin/%: %.cu | bin
	@echo "Compiling $<..."
	$(NVCC) $(NVCC_FLAGS) $< -o $@
	@echo "Built: $@"

# Convenience targets for individual tutorials
01: bin/01_basic_gemm_volta
02: bin/02_ampere_async
03: bin/03_hopper_tma_gmma
04: bin/04_cute_layouts
05: bin/05_collective_builders
06: bin/06_mixed_precision
07: bin/07_epilogue_fusion

# Run targets
run-01: bin/01_basic_gemm_volta
	@echo "Running Tutorial 01: Volta Basics"
	@./bin/01_basic_gemm_volta

run-02: bin/02_ampere_async
	@echo "Running Tutorial 02: Ampere Async"
	@./bin/02_ampere_async

run-03: bin/03_hopper_tma_gmma
	@echo "Running Tutorial 03: Hopper TMA/GMMA"
	@./bin/03_hopper_tma_gmma

run-04: bin/04_cute_layouts
	@echo "Running Tutorial 04: CuTe Layouts"
	@./bin/04_cute_layouts

run-05: bin/05_collective_builders
	@echo "Running Tutorial 05: Collective Builders"
	@./bin/05_collective_builders

run-06: bin/06_mixed_precision
	@echo "Running Tutorial 06: Mixed Precision"
	@./bin/06_mixed_precision

run-07: bin/07_epilogue_fusion
	@echo "Running Tutorial 07: Epilogue Fusion"
	@./bin/07_epilogue_fusion

# Run all tutorials
run-all: $(BINS)
	@for tutorial in $(BINS); do \
		echo ""; \
		echo "========================================"; \
		echo "Running $$tutorial"; \
		echo "========================================"; \
		./$$tutorial || exit 1; \
	done

# Clean build artifacts
clean:
	@echo "Cleaning..."
	@rm -rf bin
	@echo "Clean complete"

# Help target
help:
	@echo "CUTLASS Tutorial Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build all tutorials (default)"
	@echo "  01-07      - Build specific tutorial"
	@echo "  run-01     - Build and run tutorial 01"
	@echo "  run-all    - Build and run all tutorials"
	@echo "  clean      - Remove build artifacts"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make               # Build everything"
	@echo "  make 03            # Build tutorial 03 only"
	@echo "  make run-01        # Run tutorial 01"
	@echo "  make ARCH_FLAGS=-arch=sm_90 all  # Build for Hopper"
	@echo ""
	@echo "Architecture flags:"
	@echo "  -arch=sm_70   Volta (V100)"
	@echo "  -arch=sm_80   Ampere (A100, A30) [default]"
	@echo "  -arch=sm_90   Hopper (H100, H200)"

# Check environment
check:
	@echo "Checking build environment..."
	@echo "NVCC: $(shell which nvcc)"
	@echo "NVCC version: $(shell nvcc --version | grep release)"
	@echo "CUTLASS root: $(CUTLASS_ROOT)"
	@echo "Architecture: $(ARCH_FLAGS)"
	@if [ ! -d "$(CUTLASS_ROOT)/include/cutlass" ]; then \
		echo "ERROR: CUTLASS headers not found at $(CUTLASS_ROOT)/include"; \
		exit 1; \
	fi
	@echo "Environment check passed!"

.PHONY: all clean help check run-all run-01 run-02 run-03 run-04 run-05 run-06 run-07 \
        01 02 03 04 05 06 07
