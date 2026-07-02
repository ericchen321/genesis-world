---
name: deploy-runpod-dev16-deformable-exp
description: Deploy the HAG4R MV2-Dev16Deformables experiment on this Runpod machine. Use when an agent must run or resume the local Runpod Dev16-deformables pipeline from `/workspace/HAG4R-runpod` with RealSim diagnostics, including mandatory pre-run updates of HAG4R branch `guanxiong/runpod` and RealSim branch `Hetero_Asset_Gen` under `/workspace/RealSim-Hetero_Asset_Gen`, rebuilding `RealSimLiveServer`, validating local conda/data/credential paths, launching a smoke asset from `/workspace/mvi2_00_mv2-dev16-deformables`, proceeding to the full 16-asset batch even if smoke fails unless dependencies or credentials are missing, monitoring progress, and archiving or reporting results without using Slurm.
---

# Deploy Runpod Dev16 Deformable Exp

Use this skill for the Runpod deployment only. The core invariant is ordering: pull latest HAG4R `guanxiong/runpod`, pull latest RealSim-Hetero `Hetero_Asset_Gen`, rebuild `RealSimLiveServer`, then run the experiment. Do not start a smoke or full Dev16-deformables run until those three update/build steps have succeeded in the current session. Treat smoke as an early signal, not as a hard gate: proceed to the full Dev16-deformables batch after smoke unless smoke proves missing dependencies or missing credentials.

## Runpod Defaults

Resolve these paths before running commands, then use them unless the user explicitly overrides them:

- HAG4R repo: `/workspace/HAG4R-runpod`
- RealSim repo: `/workspace/RealSim-Hetero_Asset_Gen`
- Dev16-deformables input root: `/workspace/mvi2_00_mv2-dev16-deformables`
- HAG4R env: `/workspace/HAG4R-runpod/.conda/hag4r`
- RealSim env: `/workspace/RealSim-Hetero_Asset_Gen/.conda/realsim_py`
- Default config: `configs/config_gpt55_for_all.yaml`
- Dev16-deformables manifest: `outputs/build_mvimgnet_MV2_Dev16Deformables/MV2-Dev16Deformables.json`
- Runs root: `outputs/agentic_asset_refinement`
- Records root: `run_pipeline`
- RealSim live binary: `/workspace/RealSim-Hetero_Asset_Gen/build/bin/RealSimLiveServer`

Run HAG4R Python commands through `conda run -p "$HAG4R_ENV"`. Run RealSim build/test/server commands through `conda run -p "$REALSIM_ENV"` or through the HAG4R launcher that internally uses the RealSim env. Use `xvfb-run -a` around pipeline launches on this headless Runpod host.

Keep model/cache writes repo-local unless the user explicitly authorizes another cache:

```bash
export HAG4R_ROOT=/workspace/HAG4R-runpod
export REALSIM_ROOT=/workspace/RealSim-Hetero_Asset_Gen
export DEV16_DEFORMABLE_INPUT_ROOT=/workspace/mvi2_00_mv2-dev16-deformables
export HAG4R_ENV="$HAG4R_ROOT/.conda/hag4r"
export REALSIM_ENV="$REALSIM_ROOT/.conda/realsim_py"
export HF_HOME="$HAG4R_ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$HAG4R_ROOT/.cache"
export HF_HUB_ENABLE_HF_TRANSFER=0
unset HF_ENDPOINT
cd "$HAG4R_ROOT"
```

The referenced Dev16-deformables manifest includes the balloon asset:

```text
/workspace/mvi2_00_mv2-dev16-deformables/class_366_balloon_instance_3707ded1/004.jpg
```

Use it as the default smoke asset unless the user asks for a random or specific smoke asset.

## Mandatory Pre-Run Update

Do these steps in order. If a tracked dirty worktree blocks `git switch` or `git pull --ff-only`, stop and report the exact dirty files instead of stashing, resetting, or overwriting user changes.

Update HAG4R to latest `origin/guanxiong/runpod`:

```bash
cd "$HAG4R_ROOT"
git status --short
git fetch origin guanxiong/runpod
if git show-ref --verify --quiet refs/heads/guanxiong/runpod; then
  git switch guanxiong/runpod
else
  git switch --track -c guanxiong/runpod origin/guanxiong/runpod
fi
git pull --ff-only origin guanxiong/runpod
test "$(git branch --show-current)" = "guanxiong/runpod"
git rev-parse --short HEAD
```

Update RealSim-Hetero to latest `origin/Hetero_Asset_Gen`:

```bash
cd "$REALSIM_ROOT"
git status --short
git fetch origin Hetero_Asset_Gen
if git show-ref --verify --quiet refs/heads/Hetero_Asset_Gen; then
  git switch Hetero_Asset_Gen
else
  git switch --track -c Hetero_Asset_Gen origin/Hetero_Asset_Gen
fi
git pull --ff-only origin Hetero_Asset_Gen
test "$(git branch --show-current)" = "Hetero_Asset_Gen"
git rev-parse --short HEAD
```

Rebuild `RealSimLiveServer` from the updated RealSim worktree:

```bash
cd "$REALSIM_ROOT"
conda run -p "$REALSIM_ENV" bash -lc \
  'cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build build --target RealSimLiveServer -j"$(nproc)"'
test -x "$REALSIM_ROOT/build/bin/RealSimLiveServer"
```

This writes to `/workspace/RealSim-Hetero_Asset_Gen`. If the current user request does not clearly authorize updating and rebuilding that external repo, ask for permission before the RealSim update/build step.

## Preflight

Fail loudly on missing environment, data, credentials, or GPU resources. Do not guess substitute paths for Dev16-deformables input data or RealSim.

```bash
cd "$HAG4R_ROOT"
test -d "$HAG4R_ENV/conda-meta"
test -d "$REALSIM_ENV/conda-meta"
test -d "$DEV16_DEFORMABLE_INPUT_ROOT"
test -f configs/config_gpt55_for_all.yaml
test -f outputs/build_mvimgnet_MV2_Dev16Deformables/MV2-Dev16Deformables.json
test -x "$REALSIM_ROOT/build/bin/RealSimLiveServer"
find "$DEV16_DEFORMABLE_INPUT_ROOT" -mindepth 2 -maxdepth 2 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort | wc -l
```

Verify credentials through the code path used by HAG4R:

```bash
conda run -p "$HAG4R_ENV" python -c "from hag4r.agentic.vlm import credential_env_value; assert credential_env_value('OPENAI_API_HAG4R_KEY'), 'OPENAI_API_HAG4R_KEY is unavailable'; print('OPENAI_API_HAG4R_KEY available')"
```

Verify the CLI and RealSim live runtime before launching an experiment:

```bash
mkdir -p .agents/logs
conda run -p "$HAG4R_ENV" python -m hag4r.agentic.run_pipeline --help > .agents/logs/hag4r_run_pipeline_help.txt
conda run -p "$HAG4R_ENV" python - <<'PY'
from pathlib import Path
from hag4r.tools.realsim.live_client import verify_realsim_live_runtime
verify_realsim_live_runtime(realsim_root=Path("/workspace/RealSim-Hetero_Asset_Gen"))
print("RealSim live runtime preflight OK")
PY
```

Check GPUs:

```bash
nvidia-smi -L
```

The checked-in batch runner is named `scripts/run_mvimgnet2_dev32_pipeline.py`, but it accepts `--manifest`, `--input-root`, and `--expected-asset-count`. For Dev16-deformables, use that runner with `--expected-asset-count 16`, Dev16-deformables manifest, Dev16-deformables input root, and a Dev16 worker prefix. Require visible GPUs in pairs. Use one two-GPU worker on a 2-GPU Runpod. On a 4- or 8-GPU Runpod, use one shard per two-GPU pair. If exactly one GPU is visible, proceed with the slower direct single-pipeline fallback from the prior Runpod procedure instead of stopping. If no GPU is visible, treat it as a missing dependency and stop before full Dev16-deformables.

## Smoke Run

Run a smoke asset after the mandatory update/build/preflight. Use a fresh run id and capture logs.

```bash
cd "$HAG4R_ROOT"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
SMOKE_IMAGE="$DEV16_DEFORMABLE_INPUT_ROOT/class_366_balloon_instance_3707ded1/004.jpg"
SMOKE_RUN_ID="dev16_deformable_runpod_smoke_${STAMP}_balloon"
mkdir -p .agents/logs

xvfb-run -a conda run -p "$HAG4R_ENV" python -m hag4r.agentic.run_pipeline \
  --source_image "$SMOKE_IMAGE" \
  --run_id "$SMOKE_RUN_ID" \
  --config configs/config_gpt55_for_all.yaml \
  --fidelity low \
  --output_tag "$SMOKE_RUN_ID" \
  --realsim-root "$REALSIM_ROOT" \
  > ".agents/logs/${SMOKE_RUN_ID}.log" 2>&1
```

The smoke artifact check passes only when the expected final export and diagnostics artifacts exist and are non-empty:

```text
outputs/run_pipeline/<SMOKE_RUN_ID>/final_export_manifest.json
outputs/run_pipeline/<SMOKE_RUN_ID>/final_mesh.mesh or final_mesh.obj
outputs/run_pipeline/<SMOKE_RUN_ID>/heterogeneous_params.npz
outputs/run_pipeline/<SMOKE_RUN_ID>/homogeneous_params.npz
outputs/run_pipeline/<SMOKE_RUN_ID>/inferred_params.json
outputs/run_pipeline/<SMOKE_RUN_ID>/cleaned_image.png
outputs/run_pipeline/<SMOKE_RUN_ID>/vlm_image_cleanup.json
outputs/run_pipeline/<SMOKE_RUN_ID>/asset_refinement/state.json
outputs/run_pipeline/<SMOKE_RUN_ID>/asset_refinement/sim_diagnostics/diagnostic_summary.json
```

Also verify every `exported_artifacts` entry in `final_export_manifest.json` exists. Extract and report diagnostic recommendation fields such as `recommended_route`, `recommended_ready`, reason fields, episode count, and video paths when present.

If smoke fails, classify the failure before deciding the next step:

- Stop before full Dev16-deformables only when the smoke failure proves missing dependencies or missing credentials, such as absent conda envs, unavailable required Python packages, missing model/checkpoint assets, missing `OPENAI_API_HAG4R_KEY`, missing RealSim binary/build output, or an unavailable required data/config path.
- Proceed to the full Dev16-deformables batch for all other smoke failures. This includes asset-specific failures, model/tool mistakes, timeouts, segmentation or OmniPart failures, RealSim diagnostics failures, final artifact mismatches, and unexplained nonzero exits after dependencies and credentials were already preflighted.

Record the smoke failure signature, smoke log path, and why it is not a dependency/credential blocker in the final report.

## Full Dev16-Deformables Run

Prefer the checked-in Dev32 runner with explicit Dev16-deformables overrides for the full experiment. Launch it after the smoke attempt unless dependencies or credentials are missing. It validates the manifest, records JSONL ledgers, runs every selected asset through `hag4r.agentic.run_pipeline`, captures per-asset logs, checks final export artifacts, and records GPU binding evidence.

For a 2-GPU Runpod:

```bash
cd "$HAG4R_ROOT"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
RECORD_DIR="run_pipeline/runpod_dev16_deformable_${STAMP}"
mkdir -p "$RECORD_DIR"

xvfb-run -a conda run -p "$HAG4R_ENV" python scripts/run_mvimgnet2_dev32_pipeline.py \
  --input-root "$DEV16_DEFORMABLE_INPUT_ROOT" \
  --manifest outputs/build_mvimgnet_MV2_Dev16Deformables/MV2-Dev16Deformables.json \
  --expected-asset-count 16 \
  --config configs/config_gpt55_for_all.yaml \
  --runs-root outputs/agentic_asset_refinement \
  --record-dir "$RECORD_DIR" \
  --fidelity low \
  --worker-gpu-bindings 0,1 \
  --worker-id-prefix mvimgnet2_dev16_deformable_worker \
  --realsim-root "$REALSIM_ROOT" \
  --continue-on-failure \
  > "$RECORD_DIR/full_stdout.log" 2>&1
```

For a Runpod with more GPUs, launch one shard per two-GPU pair. Example for four GPUs:

```bash
cd "$HAG4R_ROOT"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
RECORD_DIR="run_pipeline/runpod_dev16_deformable_${STAMP}"
mkdir -p "$RECORD_DIR"

pids=()
for item in "0:0,1" "1:2,3"; do
  shard="${item%%:*}"
  binding="${item#*:}"
  xvfb-run -a conda run -p "$HAG4R_ENV" python scripts/run_mvimgnet2_dev32_pipeline.py \
    --input-root "$DEV16_DEFORMABLE_INPUT_ROOT" \
    --manifest outputs/build_mvimgnet_MV2_Dev16Deformables/MV2-Dev16Deformables.json \
    --expected-asset-count 16 \
    --config configs/config_gpt55_for_all.yaml \
    --runs-root outputs/agentic_asset_refinement \
    --record-dir "$RECORD_DIR" \
    --fidelity low \
    --shard-count 2 \
    --shard-index "$shard" \
    --worker-gpu-bindings "$binding" \
    --worker-id-prefix mvimgnet2_dev16_deformable_worker \
    --realsim-root "$REALSIM_ROOT" \
    --continue-on-failure \
    > "$RECORD_DIR/shard_${shard}.stdout.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
```

Use `--force` only when the user explicitly requests a fresh rerun of assets that already have successful final export artifacts. Omit `--force` for normal resume behavior.

If the checked-in runner is unavailable or the user asks to reproduce the prior Runpod session exactly, use a direct sequential `find ... | sort` loop with `xvfb-run -a conda run -p "$HAG4R_ENV" python -m hag4r.agentic.run_pipeline`, `--config configs/config_gpt55_for_all.yaml`, `--fidelity low`, `--output_tag "$run_id"`, and `--realsim-root "$REALSIM_ROOT"`. Keep a TSV status file with index, asset folder, image path, run id, exit code, status, timestamps, and compact error summary.

## Monitoring

Poll without interrupting active asset processes:

```bash
tail -20 "$RECORD_DIR"/*.stdout.log 2>/dev/null || true
tail -20 "$RECORD_DIR"/logs/*.log 2>/dev/null || true
find outputs/agentic_asset_refinement -maxdepth 3 -type f -name state.json -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort | tail -20
ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | rg -i 'run_mvimgnet2_dev32|run_pipeline|xvfb-run|RealSimLiveServer|omnipart|sam3' | rg -v 'rg '
```

Do not investigate individual asset failures deeply unless the user asks. For this Runpod deployment, record which asset failed and the compact error signature, continue the batch when `--continue-on-failure` is active, and stop only for missing dependencies or missing credentials that make further launches impossible.

## Archive And Report

At the end, summarize:

- HAG4R commit and RealSim commit after pull.
- RealSim rebuild command and result.
- Config, Dev16-deformables input root, HAG4R env, RealSim env, GPU bindings, and cache settings.
- Smoke run id, smoke image, log path, artifact-check result, and whether any smoke failure was classified as non-blocking or dependency/credential-blocking.
- Full run record dir, JSONL ledgers, summary JSON files, shard stdout logs, and per-asset logs.
- Counts of successful, failed, timed-out, skipped, and running assets.
- Failed asset folders with log paths and compact error summaries.
- Diagnostic recommendation fields for successful assets when available.

Create archives only for canonical Dev16-deformables run outputs that belong to this deployment, excluding smoke-only folders unless the user asks to keep them together:

```bash
COMMIT="$(git -C "$HAG4R_ROOT" rev-parse --short HEAD)"
DATE="$(date -u +%F)"
zip -r "outputs/agentic_asset_refinement/runpod_dev16_deformable_${DATE}_${COMMIT}_full.zip" outputs/agentic_asset_refinement/<run-id-prefixes-or-list>
zip -r "outputs/run_pipeline/runpod_dev16_deformable_${DATE}_${COMMIT}_full.zip" outputs/run_pipeline/<output-tag-prefixes-or-list>
```
