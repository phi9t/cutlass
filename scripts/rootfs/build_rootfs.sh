#!/bin/bash
# Build a flattened rootfs directory for local GPU builds/runs of the fork.
#
# This host ships no nvcc and no guaranteed C++17 host toolchain, and its build
# environment may not match the loaded NVIDIA driver. Rather than patch the host
# per machine, this builds a consistent Ubuntu userland from a pinned CUDA 12.8
# *devel* image (real nvcc + C++17 g++ out of the box — no synthetic CUDA) and
# exports it to a plain directory that enter_rootfs.sh runs under bwrap. Inside
# it, Bazel builds with --config=cuda against the pinned CUDA (ADR 0002/0003/0004).
#
# Unlike monarch's rootfs, this one has NO Python/torch/uv/Rust: the fork trainer
# is C++/CUTLASS. It carries a devel CUDA (so CUDA_HOME points at /usr/local/cuda)
# plus Bazel, git, and the build essentials.
#
# Usage:
#   scripts/rootfs/build_rootfs.sh [options]
#
# Options:
#   --rebuild       Force a docker rebuild even if the image tag exists.
#   --tag TAG       Docker image tag to build/use (default: cutlass-rootfs:local).
#   --dest DIR      Managed directory under scripts/rootfs/ named rootfs or
#                   rootfs-* (default: scripts/rootfs/rootfs).
#   -h, --help      Show this help and exit.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOTFS_DIR="$REPO_ROOT/scripts/rootfs"

TAG="cutlass-rootfs:local"
DEST="$ROOTFS_DIR/rootfs"
REBUILD=0
# Pinned CUDA 12.8 devel image (ADR 0004). 12.8 supports Blackwell sm_100 and
# matches cutlass's recommended toolkit. Pinned by digest for reproducibility.
BASE_IMAGE="nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97"
# Bazel version pinned to match .bazelversion (ADR 0001). Fetched via the
# official apt release so the rootfs has a real 9.2.0 binary on PATH.
BAZEL_VERSION="9.2.0"

usage() { sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild) REBUILD=1; shift ;;
    --tag) TAG="$2"; shift 2 ;;
    --tag=*) TAG="${1#*=}"; shift ;;
    --dest) DEST="$2"; shift 2 ;;
    --dest=*) DEST="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

log() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
die() { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }
usage_error() { printf 'error: %s\n' "$*" >&2; exit 2; }

ROOTFS_DIR="$(realpath -e "$ROOTFS_DIR")"
dest_parent="$(realpath -m "$(dirname -- "$DEST")")"
dest_name="$(basename -- "$DEST")"
if [[ "$dest_parent" != "$ROOTFS_DIR" || ! "$dest_name" =~ ^rootfs(-[A-Za-z0-9._-]+)?$ ]]; then
  usage_error "--dest must be a managed rootfs path under $ROOTFS_DIR named rootfs or rootfs-*"
fi
DEST="$dest_parent/$dest_name"
[[ ! -L "$DEST" ]] || usage_error "--dest must not be a symbolic link: $DEST"

command -v docker >/dev/null || die "docker not found on host"

image_exists() { docker image inspect "$TAG" >/dev/null 2>&1; }

if image_exists && [[ "$REBUILD" -eq 0 ]]; then
  log "image $TAG already exists (use --rebuild to force)"
else
  log "building $TAG from $BASE_IMAGE"
  # Inline Dockerfile via stdin. Keep the layer minimal: C++ build essentials,
  # git, and a pinned Bazel. CUDA (nvcc + headers + libs) is already in the
  # devel base at /usr/local/cuda, so CUDA_HOME points straight at it.
  docker build -t "$TAG" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg BAZEL_VERSION="$BAZEL_VERSION" \
    -f - "$ROOTFS_DIR" <<'DOCKERFILE'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG BAZEL_VERSION
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Build essentials + git + curl. build-essential brings a C++17-capable g++,
# which cutlass's headers require (README.md). No Python/torch/uv/Rust.
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        build-essential g++ git curl ca-certificates rsync \
        pkg-config python3-minimal && \
    rm -rf /var/lib/apt/lists/*

# Pinned Bazel binary matching .bazelversion. Install the official static
# release binary directly (no bazelisk network fetch at run time).
RUN curl -fLsS -o /usr/local/bin/bazel \
        "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-x86_64" && \
    chmod +x /usr/local/bin/bazel && \
    bazel --version

# CUDA is already present in the devel base at /usr/local/cuda (real nvcc).
# Point CUDA_HOME/CUDA_PATH at it so rules_cuda's local toolkit + local_cuda
# both resolve to this one pinned CUDA (ADR 0003).
RUN test -x /usr/local/cuda/bin/nvcc || { echo "nvcc missing in devel base" >&2; exit 1; } && \
    test -f /usr/local/cuda/include/cuda_runtime.h || { echo "cuda headers missing" >&2; exit 1; } && \
    /usr/local/cuda/bin/nvcc --version
ENV CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
DOCKERFILE
fi

# Export the image filesystem to a flattened directory. Export to a temp dir and
# swap into place so an interrupted run never leaves a half-populated rootfs.
log "exporting $TAG to $DEST"
mkdir -p "$ROOTFS_DIR"
STAGE="$(mktemp -d "$ROOTFS_DIR/.rootfs.stage.XXXXXX")"
[[ -d "$STAGE" && ! -L "$STAGE" && "$(dirname -- "$STAGE")" == "$ROOTFS_DIR" ]] || \
  die "mktemp produced an invalid stage directory: $STAGE"
cid="$(docker create "$TAG")"
[[ "$cid" =~ ^[0-9a-f]+$ ]] || die "docker create returned an invalid container id"
cleanup() {
  docker rm -f -- "${cid:?}" >/dev/null 2>&1 || true
  if [[ -d "${STAGE:-}" && ! -L "$STAGE" && \
        "$(dirname -- "$STAGE")" == "$ROOTFS_DIR" && \
        "$(basename -- "$STAGE")" == .rootfs.stage.* ]]; then
    rm -rf -- "${STAGE:?}"
  fi
}
trap cleanup EXIT
docker export "$cid" | tar -C "$STAGE" -xf -
[[ -x "$STAGE/bin/bash" ]] || die "export produced an unusable rootfs (no bin/bash)"
[[ "$(dirname -- "$DEST")" == "$ROOTFS_DIR" && ! -L "$DEST" ]] || \
  die "refusing to replace invalid rootfs destination: $DEST"
if [[ -e "$DEST" ]]; then
  [[ -d "$DEST" ]] || die "rootfs destination is not a directory: $DEST"
  rm -rf -- "${DEST:?}"
fi
mv "$STAGE" "$DEST"

log "rootfs ready: $DEST"
