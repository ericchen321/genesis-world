---
name: build-genesis-world
description: Install or validate the Genesis World repository in a conda environment, especially when setting up editable installs on a local workstation, cluster node, or Runpod machine. Use this when asked to create a branch-named Genesis env, wire a repo-local .conda symlink, install PyTorch plus Genesis dependencies from README.md or pyproject.toml, or capture a reproducible Genesis deployment workflow.
---

# Build Genesis World

Use this skill to set up Genesis World from a live checkout with a real conda prefix and an editable install.

## Default Local Workflow

From the repository root:

```bash
bash .agents/skills/build-genesis-world/scripts/install_genesis_world_env.sh \
  --env-root /media/eric/data/conda_envs \
  --env-name genesis_world-main \
  --symlink .conda/genesis_world-main \
  --python 3.11 \
  --torch-index https://download.pytorch.org/whl/cu126 \
  --extras dev \
  --run-hello
```

This creates `/media/eric/data/conda_envs/genesis_world-main`, links `.conda/genesis_world-main` to it, installs CUDA 12.6 PyTorch, runs `pip install -e ".[dev]"`, repairs the known `setuptools` constraint window, runs `pip check`, verifies imports, and optionally runs `examples/tutorials/hello_genesis.py`.

## Deployment Variants

- Local workstation with NVIDIA CUDA 12.6 wheels: keep the default `--torch-index https://download.pytorch.org/whl/cu126`.
- CPU-only Linux or login-node setup: pass `--torch-index https://download.pytorch.org/whl/cpu`.
- Runpod or cluster scratch storage: pass an env root on persistent storage, for example `--env-root /workspace/conda_envs` or a project scratch path.
- Existing env recreation is intentionally not automatic. If the prefix exists, the script reuses it. If a symlink exists and points somewhere else, stop and inspect before changing it.

## Validation Contract

Treat installation as complete only after all of these pass:

```bash
PYTHONNOUSERSITE=1 .conda/genesis_world-main/bin/python -m pip check
PYTHONNOUSERSITE=1 .conda/genesis_world-main/bin/python - <<'PY'
import importlib.metadata as md
import pathlib
import torch
import genesis as gs
print(md.version("genesis-world"))
print(pathlib.Path(gs.__file__).resolve())
print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)
PY
```

For a stronger README-level smoke test, run:

```bash
PYTHONNOUSERSITE=1 .conda/genesis_world-main/bin/python examples/tutorials/hello_genesis.py
```

Expected benign warnings include MJCF tendon approximation, neutral joint-limit warnings, and constraint time-constant adjustment from the tutorial asset. A nonzero exit code is not benign.

## Notes

- Use Python 3.11 unless the target platform requires otherwise. It satisfies `requires-python >=3.10,<3.14` while preserving broad wheel availability for scientific packages.
- Always run package commands through the env Python, for example `.conda/genesis_world-main/bin/python -m pip`, and set `PYTHONNOUSERSITE=1` to avoid user-site contamination.
- Genesis should be installed in editable mode with `pip install -e ".[dev]"` for contribution or active development checkouts.
- If `pip install -e ".[dev]"` upgrades `setuptools` to a version rejected by torch, install `setuptools>=77,<82` and rerun `pip check`.
