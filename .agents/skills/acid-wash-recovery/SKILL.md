---
name: acid-wash-recovery
description: Recover HAG4R and RealSim on a Runpod machine after the root filesystem or OS disk was wiped and reinstalled as fresh Ubuntu 24.04 while /workspace remains intact. Use when restoring host packages, CUDA, conda PATH, credentials, repo-local caches, preserved HAG4R conda prefixes, RealSim builds, model checkpoints, and a final HAG4R plus RealSim smoke test from /workspace/HAG4R-runpod and /workspace/RealSim-Hetero_Asset_Gen.
---

# Acid-Wash Recovery

Use this skill to recover the Runpod HAG4R workspace after an acid-wash reset: `/` is a clean Ubuntu 24.04 install, but `/workspace` still contains the repos, conda prefixes, caches, and generated artifacts.

The goal is not to rebuild everything for sport. Restore host-level dependencies, reconnect the preserved `/workspace` runtime, validate the preserved environments and RealSim build, rebuild only what fails validation, then run one end-to-end HAG4R smoke test with RealSim diagnostics.

## Ground Rules

- Treat `/workspace/HAG4R-runpod` and `/workspace/RealSim-Hetero_Asset_Gen` as the source of truth. Do not reclone either repo by default.
- Keep Python execution inside conda prefixes. Do not run HAG4R or RealSim Python code with the system Python.
- Prefer preserved repo-local prefixes before rebuilding: `/workspace/HAG4R-runpod/.conda/*` and `/workspace/RealSim-Hetero_Asset_Gen/.conda/realsim_py`.
- Fail loudly on missing credentials, missing checkpoints, missing conda prefixes, missing CUDA, missing RealSim binaries, or unavailable data/config paths.
- OS-native package installation and NVIDIA driver/CUDA setup may require sudo. If sudo or host-level changes are not already authorized, stop and ask for permission before running them.
- Do not write outside `/workspace` except for required host-level OS setup or user-approved shell startup restoration.

## Define The Recovery Context

Start every recovery session by exporting the canonical paths and checking that the preserved workspace really exists:

```bash
export HAG4R_ROOT=/workspace/HAG4R-runpod
export REALSIM_ROOT=/workspace/RealSim-Hetero_Asset_Gen
export HAG4R_ENV="$HAG4R_ROOT/.conda/hag4r"
export HAG4R_MESH_ENV="$HAG4R_ROOT/.conda/hag4r_mesh"
export HAG4R_SEGMENTATION_ENV="$HAG4R_ROOT/.conda/hag4r_segmentation"
export OMNIPART_ENV="$HAG4R_ROOT/.conda/omnipart"
export REALSIM_ENV="$REALSIM_ROOT/.conda/realsim_py"

test -d "$HAG4R_ROOT"
test -d "$REALSIM_ROOT"
cd "$HAG4R_ROOT"
```

Restore repo-local caches in the current shell:

```bash
export HF_HOME="$HAG4R_ROOT/.cache/huggingface"
export TORCH_HOME="$HAG4R_ROOT/.cache/torch"
export XDG_CACHE_HOME="$HAG4R_ROOT/.cache"
```

## Restore Host-Level Tools

Install the Ubuntu packages needed by HAG4R, OmniPart, RealSim, OpenGL, and headless rendering:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git wget curl pkg-config xvfb ffmpeg \
  libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
  libxext-dev libxrender-dev libxfixes-dev \
  libgl1-mesa-dev libglu1-mesa-dev libegl1-mesa-dev \
  libwayland-dev libxkbcommon-dev libglm-dev libeigen3-dev
```

Install or restore the NVIDIA driver and CUDA Toolkit 12.8. RealSim expects an NVIDIA driver at least `570.86.10` and `nvcc` from CUDA 12.8.

Validate the host tools:

```bash
nvidia-smi
nvcc --version
command -v Xvfb
command -v xvfb-run
```

If `/workspace/miniconda3` exists, put it back on `PATH`:

```bash
export PATH=/workspace/miniconda3/bin:$PATH
conda info --base
```

If `/workspace/miniconda3` is missing, install Miniconda or Miniforge under `/workspace`, not under `/root` or another wiped home directory.

## Restore Credentials And Runtime Variables

Anything that lived in `/home`, `.bashrc`, `.profile`, `.env`, or a shell startup file was wiped. Recreate the required exports before running HAG4R:

```bash
cd "$HAG4R_ROOT"

export PATH=/workspace/miniconda3/bin:$PATH
export HF_HOME="$HAG4R_ROOT/.cache/huggingface"
export TORCH_HOME="$HAG4R_ROOT/.cache/torch"
export XDG_CACHE_HOME="$HAG4R_ROOT/.cache"
export REALSIM_ROOT=/workspace/RealSim-Hetero_Asset_Gen

export OPENAI_API_HAG4R_KEY=...
export OPENROUTER_API_HAG4R_KEY=...
export NHR_FAU_API_KEY=...
```

Always pass `--realsim-root "$REALSIM_ROOT"` to pipeline runs. Some code defaults may point at desktop paths such as `/home/eric/research/RealSim-Hetero_Asset_Gen`; do not rely on those defaults on Runpod.

Verify credentials through the HAG4R code path when possible:

```bash
conda run -p "$HAG4R_ENV" python -c "from hag4r.agentic.vlm import credential_env_value; assert credential_env_value('OPENAI_API_HAG4R_KEY'), 'OPENAI_API_HAG4R_KEY is unavailable'; print('OPENAI_API_HAG4R_KEY available')"
```

## Validate Preserved HAG4R Envs

Try the preserved conda prefixes before rebuilding anything:

```bash
cd "$HAG4R_ROOT"

test -d "$HAG4R_ENV/conda-meta"
test -d "$HAG4R_MESH_ENV/conda-meta"
test -d "$HAG4R_SEGMENTATION_ENV/conda-meta"
test -d "$OMNIPART_ENV/conda-meta"

conda run -p "$HAG4R_ENV" python -m pytest tests/test_agentic_react_diagnostics.py -q
conda run -p "$HAG4R_ENV" pip check
conda run -p "$HAG4R_MESH_ENV" pip check
conda run -p "$HAG4R_SEGMENTATION_ENV" pip check
conda run -p "$OMNIPART_ENV" pip check
```

If only the base HAG4R env is broken, rebuild it through conda:

```bash
cd "$HAG4R_ROOT"
scripts/install_hag4r_base_env.sh .conda/hag4r
```

If the full HAG4R environment set is missing or badly broken, read and follow `.agents/skills/build-hag4r/SKILL.md`, adapting paths for Runpod and keeping all envs under `/workspace/HAG4R-runpod/.conda/`.

## Validate Preserved RealSim

Check the preserved RealSim Python env, CUDA visibility, and live server binary:

```bash
cd "$REALSIM_ROOT"

test -d "$REALSIM_ENV/conda-meta"
conda run -p "$REALSIM_ENV" python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
test -x build/bin/RealSimLiveServer
```

For HAG4R diagnostics, `build/bin/RealSimLiveServer` is the key binary. `InteractiveRealSim` is useful but not the primary requirement for the agentic diagnostic flow.

If RealSim must be rebuilt, run the rebuild inside the RealSim conda prefix:

```bash
cd "$REALSIM_ROOT"
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate "$REALSIM_ENV"

./build_metis.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j"$(nproc)"
python -m pip install -e .
test -x build/bin/RealSimLiveServer
```

If the RealSim conda prefix is missing or irrecoverable:

```bash
cd "$REALSIM_ROOT"
conda env create -p .conda/realsim_py -f environment.yml
```

Then activate the new prefix and rebuild RealSim as above.

## Check Model Assets And Caches

Validate OmniPart checkpoints:

```bash
cd "$HAG4R_ROOT"
test -f third_party/OmniPart/ckpt/partfield_encoder.ckpt
test -f third_party/OmniPart/ckpt/bbox_gen.ckpt
```

Validate the SAM3 checkpoint and route HAG4R to it explicitly:

```bash
cd "$HAG4R_ROOT"
test -f third_party/sam3/ckpt/sam3.pt
export HAG4R_SAM3_CHECKPOINT_PATH="$HAG4R_ROOT/third_party/sam3/ckpt/sam3.pt"
```

If the SAM3 checkpoint test fails, stop and download or place the checkpoint at `third_party/sam3/ckpt/sam3.pt` before running segmentation.

If the Hugging Face cache is incomplete, prefetch the OmniPart snapshot from inside the OmniPart env:

```bash
cd "$HAG4R_ROOT"
export HF_HOME="$HAG4R_ROOT/.cache/huggingface"
export TORCH_HOME="$HAG4R_ROOT/.cache/torch"
export XDG_CACHE_HOME="$HAG4R_ROOT/.cache"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
  conda run -p "$OMNIPART_ENV" python -c "from huggingface_hub import snapshot_download; print(snapshot_download('omnipart/OmniPart', allow_patterns=['pipeline.json', '*/*.json', '*/*.safetensors']))"
```

## Run The Recovery Smoke Test

Run one low-fidelity kitchen-tong pipeline with RealSim diagnostics. Use a fresh run id and headless rendering:

```bash
cd "$HAG4R_ROOT"

RUN_ID="recovery_smoke_$(date +%Y%m%d_%H%M%S)"
export REALSIM_ROOT=/workspace/RealSim-Hetero_Asset_Gen
export HAG4R_SAM3_CHECKPOINT_PATH="$HAG4R_ROOT/third_party/sam3/ckpt/sam3.pt"
export HF_HOME="$HAG4R_ROOT/.cache/huggingface"
export TORCH_HOME="$HAG4R_ROOT/.cache/torch"
export XDG_CACHE_HOME="$HAG4R_ROOT/.cache"

xvfb-run -a conda run -p "$HAG4R_ENV" python -m hag4r.agentic.run_pipeline \
  --source_image assets/internet_pics/kitchen_tong/kitchen_tong.jpg \
  --run_id "$RUN_ID" \
  --representation volumetric \
  --fidelity low \
  --config configs/config_kimi_k26.yaml \
  --sim-diagnostics-max-runs 2 \
  --sim-diagnostics-max-episodes 3 \
  --sim-diagnostics-max-actions-per-episode 4 \
  --realsim-root "$REALSIM_ROOT"
```

Check the expected outputs:

```bash
PIPELINE_TAG="${RUN_ID}_volumetric_agentic"
test -f "outputs/run_pipeline/${PIPELINE_TAG}/final_mesh.mesh"
test -f "outputs/run_pipeline/${PIPELINE_TAG}/heterogeneous_params.npz"
test -f "outputs/run_pipeline/${PIPELINE_TAG}/homogeneous_params.npz"
test -f "outputs/run_pipeline/${PIPELINE_TAG}/final_export_manifest.json"
test -f "outputs/agentic_asset_refinement/${RUN_ID}/state.json"
test -f "outputs/agentic_asset_refinement/${RUN_ID}/sim_diagnostics/diagnostic_summary.json"
```

When the smoke test fails, classify the first hard blocker before changing anything:

- Missing `nvidia-smi`, CUDA, or `nvcc`: fix host GPU/CUDA installation.
- Missing `conda`: restore or install Miniconda/Miniforge under `/workspace`.
- Missing HAG4R conda prefixes: rebuild HAG4R envs under `/workspace/HAG4R-runpod/.conda/`.
- Missing RealSim env or `RealSimLiveServer`: rebuild RealSim under `/workspace/RealSim-Hetero_Asset_Gen`.
- Missing API keys: stop and ask the user for credentials.
- Missing SAM3 or OmniPart checkpoints: stop and restore the checkpoint files.
- Runtime pipeline failure after all prerequisites validate: debug the actual stage logs and preserve the original error context.

## Completion Criteria

Consider recovery complete only after these are true:

- Host tools validate: `nvidia-smi`, `nvcc --version`, `Xvfb`, and `xvfb-run`.
- `conda` is available from `/workspace/miniconda3` or another `/workspace` conda install.
- HAG4R credentials and `REALSIM_ROOT=/workspace/RealSim-Hetero_Asset_Gen` are exported.
- Preserved or rebuilt HAG4R conda prefixes pass focused validation.
- Preserved or rebuilt RealSim has a working `.conda/realsim_py` and executable `build/bin/RealSimLiveServer`.
- Required OmniPart and SAM3 checkpoints exist.
- The low-fidelity kitchen-tong smoke run produces final mesh, params, export manifest, refinement state, and diagnostic summary artifacts.
