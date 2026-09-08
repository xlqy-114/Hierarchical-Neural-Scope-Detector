#!/usr/bin/env bash
set -euo pipefail
export UNIAD_ROOT=/root/autodl-tmp/UniAD2
export REPO_ROOT=/root/autodl-tmp/UniAD2/code/UniAD
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/projects:${PYTHONPATH:-}"
export PIP_CACHE_DIR=/root/autodl-tmp/UniAD2/pip_cache
export TMPDIR=/root/autodl-tmp/UniAD2/tmp
export TEMP=/root/autodl-tmp/UniAD2/tmp
export TMP=/root/autodl-tmp/UniAD2/tmp
export XDG_CACHE_HOME=/root/autodl-tmp/UniAD2/cache
export TORCH_HOME=/root/autodl-tmp/UniAD2/cache/torch
export CUDA_CACHE_PATH=/root/autodl-tmp/UniAD2/cache/cuda
mkdir -p /root/autodl-tmp/UniAD2/{outputs,logs,tmp,cache,code_pack}
cd /root/autodl-tmp/UniAD2
