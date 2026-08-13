# ADR 0001 — Bazel module root at the repo root

- Status: Accepted
- Date: 2026-08-13
- Context repos: reconciles the pre-existing `src/` scaffold with root-relative labels.

## Context

The pre-existing scaffold placed `MODULE.bazel` and `.bazelrc` under `src/`, but every
BUILD file uses **root-relative labels** (`//third_party/cuda`, `//tests`,
`//tools/bazel:cuda_ext.bzl`, `//src/core`). Those packages (`third_party/`, `tests/`,
`tools/bazel/`) live at the **repo root**, and there is no `MODULE.bazel`/`.bazelrc`/
`.bazelversion` at the root. As laid out, the labels do not resolve — the scaffold is
unbuildable. Two ways to make root and labels agree:

1. Move the Bazel module files up to the repo root.
2. Restructure everything to be `src/`-rooted (rewrite every label).

## Decision

The Bazel **module root is the cutlass repo root**. Move `MODULE.bazel`/`.bazelrc` up and
add `.bazelversion`. cutlass-proper's `include/{cute,cutlass}`, `third_party/`, `tests/`,
`tools/bazel/`, and `src/` all sit under one module.

## Consequences

- All existing root-relative labels resolve without edits.
- The fork's Bazel tree coexists with cutlass's CMake build at the same root; Bazel only
  ever *builds* `src/` and *exposes* cutlass headers (see ADR 0002 scope).
- A `src/`-rooted module (rejected) would have been self-contained but blind to
  `//third_party` and the real `//include` header tree.
- Rebase risk: root-level Bazel files are fork-only additions, disjoint from upstream
  cutlass, so they replay cleanly onto new release tags.
