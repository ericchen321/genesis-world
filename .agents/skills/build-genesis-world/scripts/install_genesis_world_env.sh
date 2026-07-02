#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(pwd)"
env_root="/media/eric/data/conda_envs"
env_name="genesis_world-main"
symlink_path=".conda/genesis_world-main"
python_version="3.11"
torch_index="https://download.pytorch.org/whl/cu126"
torch_package="torch"
extras="dev"
skip_torch=0
run_hello=0

usage() {
  cat <<'USAGE'
Usage: install_genesis_world_env.sh [options]

Options:
  --repo-dir PATH       Genesis World checkout. Defaults to current directory.
  --env-root PATH       Directory containing conda env prefixes. Defaults to /media/eric/data/conda_envs.
  --env-name NAME       Conda env directory name. Defaults to genesis_world-main.
  --env-prefix PATH     Full conda env prefix. Overrides --env-root/--env-name.
  --symlink PATH        Repo-local symlink path. Defaults to .conda/genesis_world-main.
  --python VERSION      Python version for conda create. Defaults to 3.11.
  --torch-index URL     PyTorch wheel index. Use CUDA or CPU index. Defaults to cu126.
  --torch-package SPEC  Torch package spec. Defaults to torch.
  --extras NAME         Editable extra to install. Defaults to dev. Use empty string for none.
  --skip-torch          Do not install torch.
  --run-hello           Run examples/tutorials/hello_genesis.py after import checks.
  -h, --help            Show this help.
USAGE
}

env_prefix=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      repo_dir="$2"; shift 2 ;;
    --env-root)
      env_root="$2"; shift 2 ;;
    --env-name)
      env_name="$2"; shift 2 ;;
    --env-prefix)
      env_prefix="$2"; shift 2 ;;
    --symlink)
      symlink_path="$2"; shift 2 ;;
    --python)
      python_version="$2"; shift 2 ;;
    --torch-index)
      torch_index="$2"; shift 2 ;;
    --torch-package)
      torch_package="$2"; shift 2 ;;
    --extras)
      extras="$2"; shift 2 ;;
    --skip-torch)
      skip_torch=1; shift ;;
    --run-hello)
      run_hello=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

repo_dir="$(cd "$repo_dir" && pwd)"
if [[ -z "$env_prefix" ]]; then
  env_prefix="${env_root%/}/${env_name}"
fi
mkdir -p "$(dirname "$env_prefix")"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not on PATH" >&2
  exit 1
fi

if [[ ! -x "$env_prefix/bin/python" ]]; then
  conda create -y -p "$env_prefix" "python=$python_version" pip
fi

mkdir -p "$repo_dir/$(dirname "$symlink_path")"
link_abs="$repo_dir/$symlink_path"
if [[ -L "$link_abs" ]]; then
  current_target="$(readlink "$link_abs")"
  if [[ "$current_target" != "$env_prefix" ]]; then
    echo "Refusing to rewrite $link_abs: points to $current_target, expected $env_prefix" >&2
    exit 1
  fi
elif [[ -e "$link_abs" ]]; then
  echo "Refusing to replace existing non-symlink path: $link_abs" >&2
  exit 1
else
  ln -s "$env_prefix" "$link_abs"
fi

python="$env_prefix/bin/python"
export PYTHONNOUSERSITE=1

"$python" -m pip install --upgrade pip wheel 'setuptools>=77,<82'
if [[ "$skip_torch" -eq 0 ]]; then
  "$python" -m pip install "$torch_package" --index-url "$torch_index"
fi

cd "$repo_dir"
if [[ -n "$extras" ]]; then
  "$python" -m pip install -e ".[${extras}]"
else
  "$python" -m pip install -e .
fi

"$python" -m pip install 'setuptools>=77,<82'
"$python" -m pip check

"$python" - <<'PY'
import importlib.metadata as md
import pathlib
import torch
import genesis as gs

print("genesis-world", md.version("genesis-world"))
print("genesis_file", pathlib.Path(gs.__file__).resolve())
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY

if [[ "$run_hello" -eq 1 ]]; then
  "$python" examples/tutorials/hello_genesis.py
fi
