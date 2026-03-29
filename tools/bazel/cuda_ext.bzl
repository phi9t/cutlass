"""
Module extension for discovering system-local CUDA and NCCL installations.

This is the Bazel 8+ / bzlmod way to perform repo-rule-like setup from
MODULE.bazel, including filesystem probing and environment variable reads.
"""

def _find_cuda_path(ctx):
    """Resolve CUDA toolkit root from environment or well-known paths."""
    cuda_home = ctx.os.environ.get("CUDA_HOME", "")
    if cuda_home:
        return cuda_home
    cuda_path = ctx.os.environ.get("CUDA_PATH", "")
    if cuda_path:
        return cuda_path
    # Fallback to well-known default.
    return "/usr/local/cuda"

def _find_nccl_path(ctx):
    """Resolve NCCL root from environment or well-known paths."""
    nccl_home = ctx.os.environ.get("NCCL_HOME", "")
    if nccl_home:
        return nccl_home
    return "/usr"

# ---------------------------------------------------------------------------
# Repository rules
# ---------------------------------------------------------------------------

def _local_cuda_impl(rctx):
    cuda_path = _find_cuda_path(rctx)
    rctx.file("BUILD.bazel", content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cuda_headers",
    hdrs = glob(["include/**/*.h", "include/**/*.hpp"], allow_empty = True),
    includes = ["include"],
)

cc_import(
    name = "cudart",
    shared_library = "lib64/libcudart.so",
    deps = [":cuda_headers"],
)

cc_import(
    name = "cublas",
    shared_library = "lib64/libcublas.so",
    deps = [":cuda_headers"],
)

cc_import(
    name = "cublaslt",
    shared_library = "lib64/libcublasLt.so",
    deps = [":cuda_headers"],
)
""")
    # Symlink the CUDA toolkit contents into the repo if they exist.
    for subdir in ["include", "lib64"]:
        path = cuda_path + "/" + subdir
        if rctx.path(path).exists:
            rctx.symlink(path, subdir)
        else:
            rctx.file(subdir + "/.gitkeep", content = "")

local_cuda = repository_rule(
    implementation = _local_cuda_impl,
    environ = ["CUDA_HOME", "CUDA_PATH"],
    local = True,
)

def _local_nccl_impl(rctx):
    nccl_path = _find_nccl_path(rctx)
    rctx.file("BUILD.bazel", content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "nccl_headers",
    hdrs = glob(["include/**/*.h"], allow_empty = True),
    includes = ["include"],
)

cc_import(
    name = "nccl",
    shared_library = "lib/libnccl.so",
    deps = [":nccl_headers"],
)
""")
    for subdir in ["include", "lib"]:
        path = nccl_path + "/" + subdir
        if rctx.path(path).exists:
            rctx.symlink(path, subdir)
        else:
            rctx.file(subdir + "/.gitkeep", content = "")

local_nccl = repository_rule(
    implementation = _local_nccl_impl,
    environ = ["NCCL_HOME"],
    local = True,
)

# ---------------------------------------------------------------------------
# Module extension
# ---------------------------------------------------------------------------

def _cuda_extension_impl(module_ctx):
    for mod in module_ctx.modules:
        for tag in mod.tags.local_cuda:
            local_cuda(name = tag.name)
        for tag in mod.tags.local_nccl:
            local_nccl(name = tag.name)

_local_cuda_tag = tag_class(attrs = {
    "name": attr.string(mandatory = True),
})

_local_nccl_tag = tag_class(attrs = {
    "name": attr.string(mandatory = True),
})

cuda = module_extension(
    implementation = _cuda_extension_impl,
    tag_classes = {
        "local_cuda": _local_cuda_tag,
        "local_nccl": _local_nccl_tag,
    },
)
