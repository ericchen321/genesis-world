---
name: build-hag4r
description: Build and validate the HAG4R local desktop workspace, including third-party submodules, branch-scoped conda environments, repo-local .conda symlinks, and an end-to-end kitchen-tong asset-generation run with RealSim diagnostics enabled.
---

# Build HAG4R

Use this skill when setting up a fresh or incomplete HAG4R workspace for local development or end-to-end validation.

## Assumptions

- Build on a local desktop with GPU access, not on a cluster.
- HAG4R is the current repository root.
- RealSim has already been built under `~/research/RealSim-Hetero_Asset_Gen/`.
- If RealSim lives elsewhere, was not built yet, or the machine uses different CUDA/driver/toolchain versions, adjust the paths and package installation commands before running validation.
- Do not use the system Python. Run Python only through the branch-scoped conda prefixes.
- Keep model caches inside the repo-local `.cache/` directory so offline model-loading behavior is reproducible across runs.
- If another GPU-heavy workload is running, treat CUDA allocation failures in SAM3 or OmniPart as environmental until logs show a code or input issue.

## 1. Install System Xvfb

Install Xvfb as an OS-native package before setting up the conda envs. On Ubuntu or Debian:

```bash
sudo apt install xvfb
```

Use the equivalent package from the system package manager on other Linux distributions. HAG4R's mesh-transfer and OmniPart paths use `xvfb-run`, so the installation must provide both `Xvfb` and `xvfb-run` on `PATH`.

Confirm the commands are available:

```bash
command -v Xvfb
command -v xvfb-run
```

## 2. Prepare Submodules

Initialize the third-party code needed by HAG4R:

```bash
git submodule sync --recursive
git submodule update --init --recursive third_party/newton third_party/OmniPart third_party/sam3
```

If manuscript/docs submodules are needed for the task, initialize them separately. For build validation, the required code submodules are `third_party/newton`, `third_party/OmniPart`, and `third_party/sam3`.

Confirm the checkouts are populated:

```bash
test -d third_party/newton/.git || test -f third_party/newton/.git
test -d third_party/OmniPart/.git || test -f third_party/OmniPart/.git
test -d third_party/sam3/.git || test -f third_party/sam3/.git
```

## 3. Define Branch-Scoped Paths and Caches

Use one real env per logical runtime and branch. Sanitize branch names because `/` in Git branch names cannot be used literally inside a single path component.

```bash
BRANCH_NAME="$(git branch --show-current | sed 's#[^A-Za-z0-9_.-]#_#g')"
ENV_ROOT=/media/eric/data/conda_envs
mkdir -p "${ENV_ROOT}" .conda

export HF_HOME="${PWD}/.cache/huggingface"
export TORCH_HOME="${PWD}/.cache/torch"
export XDG_CACHE_HOME="${PWD}/.cache"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"
```

The required HAG4R envs are:

- `hag4r-${BRANCH_NAME}` for orchestration, agentic runtime, VLM calls, and lightweight tests.
- `hag4r_mesh-${BRANCH_NAME}` for mesh-transfer utilities.
- `hag4r_segmentation-${BRANCH_NAME}` for SAM3 segmentation.
- `omnipart-${BRANCH_NAME}` for OmniPart.
- RealSim uses `realsim_py` from the RealSim checkout, preferably `~/research/RealSim-Hetero_Asset_Gen/.conda/realsim_py`.

Create `.conda/<env_name>` symlinks after each real env is created:

```bash
ln -sfn "${ENV_ROOT}/hag4r-${BRANCH_NAME}" .conda/hag4r
ln -sfn "${ENV_ROOT}/hag4r_mesh-${BRANCH_NAME}" .conda/hag4r_mesh
ln -sfn "${ENV_ROOT}/hag4r_segmentation-${BRANCH_NAME}" .conda/hag4r_segmentation
ln -sfn "${ENV_ROOT}/omnipart-${BRANCH_NAME}" .conda/omnipart
```

## 4. Create HAG4R Runtime Env

```bash
conda create -p "${ENV_ROOT}/hag4r-${BRANCH_NAME}" python=3.12 -y
ln -sfn "${ENV_ROOT}/hag4r-${BRANCH_NAME}" .conda/hag4r
conda run -p .conda/hag4r python -m pip install -r requirements.txt
conda run -p .conda/hag4r python -m pip install -r requirements-agentic.txt
```

Install `requirements.txt` before `requirements-agentic.txt`; the agentic file intentionally owns the active `openai` and `numpy` pins for orchestration.

Ensure gateway credentials are available before VLM-backed stages:

```bash
test -n "${NHR_FAU_API_KEY:-}" && test -n "${OPENAI_API_HAG4R_KEY:-}"
```

## 5. Create Mesh Env

Clone the main env, then reinstall mesh requirements:

```bash
conda create -p "${ENV_ROOT}/hag4r_mesh-${BRANCH_NAME}" --clone "${ENV_ROOT}/hag4r-${BRANCH_NAME}" -y
ln -sfn "${ENV_ROOT}/hag4r_mesh-${BRANCH_NAME}" .conda/hag4r_mesh
conda run -p .conda/hag4r_mesh python -m pip install -r requirements.txt
```

The runtime resolves this env through `.conda/hag4r_mesh`; `scripts/assign_monolithic_mesh_params.sh` also honors `HAG4R_MESH_CONDA_ENV` when an override is needed.

## 6. Create SAM3 Segmentation Env

Install a Torch build compatible with the local CUDA driver first. Then install SAM3 and the image/mask utilities:

```bash
conda create -p "${ENV_ROOT}/hag4r_segmentation-${BRANCH_NAME}" python=3.12 -y
ln -sfn "${ENV_ROOT}/hag4r_segmentation-${BRANCH_NAME}" .conda/hag4r_segmentation

# Choose the torch/torchvision command for this desktop's CUDA stack.
conda run -p .conda/hag4r_segmentation python -m pip install torch torchvision

conda run -p .conda/hag4r_segmentation python -m pip install -e third_party/sam3
conda run -p .conda/hag4r_segmentation python -m pip install opencv-python pycocotools supervision pillow numpy
```

Keep SAM3 model weights out of git. Place the SAM3 checkpoint under the SAM3 checkout and route the runtime to it explicitly:

```bash
test -f third_party/sam3/ckpt/sam3.pt
export HAG4R_SAM3_CHECKPOINT_PATH="${PWD}/third_party/sam3/ckpt/sam3.pt"
```

Preflight the local checkpoint before the full pipeline:

```bash
conda run -p .conda/hag4r_segmentation python -c "from sam3.model_builder import build_sam3_image_model; build_sam3_image_model(checkpoint_path='third_party/sam3/ckpt/sam3.pt', device='cuda')"
```

## 7. Create OmniPart Env

Set up OmniPart according to `third_party/OmniPart`'s README, but put the env under the branch-scoped prefix:

```bash
conda create -p "${ENV_ROOT}/omnipart-${BRANCH_NAME}" python=<OmniPart-required-python> -y
ln -sfn "${ENV_ROOT}/omnipart-${BRANCH_NAME}" .conda/omnipart
```

Then run the OmniPart install steps from its README inside `.conda/omnipart`. Do not guess around missing OmniPart-specific dependencies; inspect the OmniPart checkout and install exactly what that branch requires.

If OmniPart's requirements contain conflicting pins, resolve them by matching the checked-out OmniPart code and validating the env with `pip check`. On the current desktop setup, `pyvista==0.47.0` is the working pin.

Required OmniPart checkpoints:

```text
third_party/OmniPart/ckpt/partfield_encoder.ckpt
third_party/OmniPart/ckpt/bbox_gen.ckpt
```

The HAG4R OmniPart stage runs with offline Hugging Face flags, so prefetch the OmniPart model snapshot into the repo-local cache before end-to-end validation:

```bash
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
  conda run -p .conda/omnipart python -c "from huggingface_hub import snapshot_download; print(snapshot_download('omnipart/OmniPart', allow_patterns=['pipeline.json', '*/*.json', '*/*.safetensors']))"
```

## 8. Check RealSim

This skill assumes RealSim is already built here:

```bash
REALSIM_ROOT="${HOME}/research/RealSim-Hetero_Asset_Gen"
test -x "${REALSIM_ROOT}/build/bin/InteractiveRealSim"
test -x "${REALSIM_ROOT}/build/bin/RealSimLiveServer"
test -e "${REALSIM_ROOT}/.conda/realsim_py"
```

If RealSim uses a different root, pass `--realsim-root <path>` during validation and adjust any RealSim-local `.conda/realsim_py` setup. The HAG4R live diagnostic client resolves `realsim_py` through the RealSim checkout when that prefix exists.

## 9. Sanity Tests

Before an expensive full run, execute focused tests:

```bash
conda run -p .conda/hag4r python -m pytest \
  tests/test_agentic_react_pipeline.py \
  tests/test_agentic_react_orchestrator.py \
  tests/test_agentic_react_stage_agents.py \
  tests/test_agentic_react_diagnostics.py \
  tests/test_segmentation_step16.py \
  -q

conda run -p .conda/hag4r pip check
conda run -p .conda/hag4r_mesh pip check
conda run -p .conda/hag4r_segmentation pip check
conda run -p .conda/omnipart pip check
command -v Xvfb
command -v xvfb-run

git diff --check
git -C third_party/OmniPart diff --check
```

## 10. Kitchen-Tong End-To-End Validation

Run raw image to final volumetric asset with RealSim diagnostics enabled:

```bash
RUN_ID="react_e2e_kitchen_tong_diag_$(date +%Y%m%d_%H%M%S)"
SOURCE_IMAGE=assets/internet_pics/kitchen_tong/kitchen_tong.jpg
REALSIM_ROOT="${HOME}/research/RealSim-Hetero_Asset_Gen"

export HAG4R_SAM3_CHECKPOINT_PATH="${PWD}/third_party/sam3/ckpt/sam3.pt"
export HF_HOME="${PWD}/.cache/huggingface"
export TORCH_HOME="${PWD}/.cache/torch"
export XDG_CACHE_HOME="${PWD}/.cache"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

conda run -p .conda/hag4r python -m hag4r.agentic.run_pipeline \
  --source_image "${SOURCE_IMAGE}" \
  --run_id "${RUN_ID}" \
  --representation volumetric \
  --fidelity low \
  --config configs/config_kimi_k26.yaml \
  --sim-diagnostics-max-runs 2 \
  --sim-diagnostics-max-episodes 3 \
  --sim-diagnostics-max-actions-per-episode 4 \
  --realsim-root "${REALSIM_ROOT}"
```

Full permission is granted for running RealSim binaries.

Expected artifacts:

```bash
PIPELINE_TAG="${RUN_ID}_volumetric_agentic"
test -f "outputs/run_pipeline/${PIPELINE_TAG}/final_mesh.mesh"
test -f "outputs/run_pipeline/${PIPELINE_TAG}/heterogeneous_params.npz"
test -f "outputs/run_pipeline/${PIPELINE_TAG}/homogeneous_params.npz"
test -f "outputs/run_pipeline/${PIPELINE_TAG}/final_export_manifest.json"
test -f "outputs/agentic_asset_refinement/${RUN_ID}/state.json"
test -f "outputs/agentic_asset_refinement/${RUN_ID}/sim_diagnostics/diagnostic_summary.json"
```

Optionally verify the numeric payloads load and contain finite values:

```bash
RUN_ID="${RUN_ID}" conda run -p .conda/hag4r python -c $'import os\nfrom pathlib import Path\nimport numpy as np\nrun_id = os.environ["RUN_ID"]\nroot = Path("outputs/run_pipeline") / f"{run_id}_volumetric_agentic"\nfor path in (root / "heterogeneous_params.npz", root / "homogeneous_params.npz"):\n    payload = np.load(path)\n    assert payload.files, f"empty npz: {path}"\n    for key in payload.files:\n        arr = payload[key]\n        assert arr.size, f"empty array {key} in {path}"\n        assert np.isfinite(arr).all(), f"non-finite array {key} in {path}"'
```

If validation fails, read the stage logs under `outputs/agentic_asset_refinement/${RUN_ID}/logs/` and the run state at `outputs/agentic_asset_refinement/${RUN_ID}/state.json` before changing code. Missing envs, empty submodules, missing checkpoints, an empty offline Hugging Face cache, GPU memory pressure from other workloads, or a different RealSim path are setup problems, not HAG4R runtime fixes.

For a partially completed run where a failed stage was manually recovered and its required artifacts are present, update the run state only after confirming those artifacts. Then resume the orchestrator directly from the existing run root:

```bash
conda run -p .conda/hag4r python -c "import json; from hag4r.agentic.react_orchestrator import run_react_orchestrator; print(json.dumps(run_react_orchestrator('outputs/agentic_asset_refinement/${RUN_ID}'), indent=2))"
```
