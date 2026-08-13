#!/bin/bash
# In-rootfs driver for the hermetic CUDA build/verify path. This is meant to be
# run *inside* the bwrap rootfs (via enter_rootfs.sh); it hard-fails if the
# rootfs marker is absent, then delegates to the local GPU smoke verifier.
#
# The rootfs (built by build_rootfs.sh) bakes a pinned CUDA 12.8 devel toolchain
# and Bazel 9.2.0, and enter_rootfs.sh binds the local GPUs and host driver
# userspace. So here there are no CC/CUDA_HOME overrides to apply: CUDA_HOME is
# already the devel base's /usr/local/cuda, and `bazel --config=cuda` finds nvcc
# on PATH. This script exists so the verifier can re-exec itself into the rootfs
# with a single repo-relative entrypoint (ADR 0002).
#
# Usage (from inside the rootfs):
#   scripts/rootfs/run_in_rootfs.sh [args passed to run_local_gpu_smoke.sh]

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

die() { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

[[ "${CUTLASS_IN_ROOTFS:-0}" == "1" ]] || \
  die "not inside the rootfs; run via scripts/rootfs/enter_rootfs.sh -- scripts/rootfs/run_in_rootfs.sh"

command -v bazel >/dev/null || die "bazel not found in rootfs"
command -v nvcc >/dev/null || die "nvcc not found in rootfs (CUDA_HOME=$CUDA_HOME)"

exec "$REPO_ROOT/scripts/run_local_gpu_smoke.sh" --in-rootfs "$@"
