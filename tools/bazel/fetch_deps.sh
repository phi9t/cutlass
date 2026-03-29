#!/usr/bin/env bash
# Fetch and extract Bazel module dependencies for offline builds.
# Run from the repository root: bash tools/bazel/fetch_deps.sh
set -euo pipefail

DEPS_DIR=".bazel_deps"
TARBALLS="$DEPS_DIR"
EXTRACTED="$DEPS_DIR/extracted"

mkdir -p "$TARBALLS" "$EXTRACTED"

fetch() {
  local name="$1" url="$2" dir="$3" strip="${4:-0}"
  echo "Fetching $name ..."
  local tarball="$TARBALLS/$(basename "$url")"
  [ -f "$tarball" ] || curl -fsSL "$url" -o "$tarball"
  rm -rf "$EXTRACTED/$dir"
  mkdir -p "$EXTRACTED/$dir"
  tar -xzf "$tarball" --strip-components="$strip" -C "$EXTRACTED/$dir"
}

stub() {
  local name="$1"
  echo "Creating stub: $name"
  mkdir -p "$EXTRACTED/$name"
  echo "module(name = \"$name\", version = \"0.0.0\")" > "$EXTRACTED/$name/MODULE.bazel"
  echo "# Stub" > "$EXTRACTED/$name/BUILD.bazel"
}

# Real dependencies
fetch bazel_skylib   "https://github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz"           bazel_skylib   0
fetch googletest     "https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz"                googletest-1.15.2 1
fetch platforms      "https://github.com/bazelbuild/platforms/releases/download/0.0.10/platforms-0.0.10.tar.gz"               platforms      0
fetch abseil-cpp     "https://github.com/abseil/abseil-cpp/releases/download/20240116.2/abseil-cpp-20240116.2.tar.gz"         abseil-cpp-20240116.2 1
fetch re2            "https://github.com/google/re2/releases/download/2024-07-02/re2-2024-07-02.tar.gz"                       re2-2024-07-02 1
fetch apple_support  "https://github.com/bazelbuild/apple_support/releases/download/1.15.1/apple_support.1.15.1.tar.gz"      apple_support  0
fetch rules_cc       "https://github.com/bazelbuild/rules_cc/releases/download/0.1.1/rules_cc-0.1.1.tar.gz"                  rules_cc-0.1.1 1
fetch rules_license  "https://github.com/bazelbuild/rules_license/releases/download/1.0.0/rules_license-1.0.0.tar.gz"        rules_license  0
fetch bazel_features "https://github.com/bazel-contrib/bazel_features/releases/download/v1.19.0/bazel_features-v1.19.0.tar.gz" bazel_features 1
fetch rules_shell    "https://github.com/bazelbuild/rules_shell/releases/download/v0.2.0/rules_shell-v0.2.0.tar.gz"          rules_shell    1

# Stubs for transitive deps not used at build time
stub pybind11_bazel
stub rules_python
stub buildozer
stub zlib
stub rules_proto
stub protobuf
stub rules_java
stub stardoc

# Fix protobuf stub (needs bazel/ package for autoload compatibility)
mkdir -p "$EXTRACTED/protobuf/bazel"
echo "# Stub" > "$EXTRACTED/protobuf/bazel/BUILD.bazel"
for f in proto_library cc_proto_library java_lite_proto_library java_proto_library proto_lang_toolchain; do
  echo "def ${f}(**kwargs): pass" > "$EXTRACTED/protobuf/bazel/${f}.bzl"
done

echo "Done. All dependencies cached in $DEPS_DIR/"
