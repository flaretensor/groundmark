#!/usr/bin/env bash
# Keep interpreter, tooling and model writes in the Groundmark workspace.
groundmark_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="$groundmark_root/.cache/uv"
export HF_HOME="$groundmark_root/.cache/huggingface"
export UV_PYTHON_INSTALL_DIR="$groundmark_root/.cache/python"
export UV_PYTHON_BIN_DIR="$groundmark_root/.cache/bin"
export UV_TOOL_DIR="$groundmark_root/.cache/tools"
export XDG_CACHE_HOME="$groundmark_root/.cache"
export PIP_CACHE_DIR="$groundmark_root/.cache/pip"
export TMPDIR="$groundmark_root/.cache/tmp"
export CUDA_VISIBLE_DEVICES=""
mkdir -p "$TMPDIR"
