---
name: deploy-runpod-nus-dev4-exp
description: Deploy the HAG4R NUS Dev4 experiment on this Runpod machine. Use when an agent must run or resume the local Runpod NUS Dev4 pipeline from `/workspace/HAG4R-runpod` with RealSim diagnostics, including mandatory pre-run updates of HAG4R branch `guanxiong/runpod` and RealSim branch `Hetero_Asset_Gen` under `/workspace/RealSim-Hetero_Asset_Gen`, rebuilding `RealSimLiveServer`, validating local conda/data/credential paths, launching a smoke asset, running the four repo-local images under `assets/real_world_objects`, monitoring progress, and archiving or reporting results without using Slurm or the MVImgNet Dev32 manifest runner.
---

# Deploy Runpod NUS Dev4 Exp

Use this skill for the Runpod deployment only. The core invariant is ordering: pull latest HAG4R `guanxiong/runpod`, pull latest RealSim-Hetero `Hetero_Asset_Gen`, rebuild `RealSimLiveServer`, then run the experiment. Do not start a smoke or full NUS Dev4 run until those three update/build steps have succeeded in the current session. Treat smoke as an early signal, not as a hard gate: proceed to the full NUS Dev4 batch after smoke unless smoke proves missing dependencies or missing credentials.

NUS Dev4 is not an MVImgNet manifest dataset. Do not use `scripts/run_mvimgnet2_dev32_pipeline.py` for this experiment. Run the four explicit repo-relative images directly through `python -m hag4r.agentic.run_pipeline`.

## Runpod Defaults

Resolve these paths before running commands, then use them unless the user explicitly overrides them:

- HAG4R repo: `/workspace/HAG4R-runpod`
- RealSim repo: `/workspace/RealSim-Hetero_Asset_Gen`
- HAG4R env: `/workspace/HAG4R-runpod/.conda/hag4r`
- RealSim env: `/workspace/RealSim-Hetero_Asset_Gen/.conda/realsim_py`
- Default config: `configs/config_gpt55_for_all.yaml`
- NUS Dev4 image root: `assets/real_world_objects`
- Runs root: `outputs/agentic_asset_refinement`
- Records root: `run_pipeline`
- RealSim live binary: `/workspace/RealSim-Hetero_Asset_Gen/build/bin/RealSimLiveServer`

The NUS Dev4 dataset is exactly these four images, relative to `HAG4R_ROOT`:

```text
assets/real_world_objects/baseball_cap_full_cotton/baseball_cap_full_cotton.jpeg
assets/real_world_objects/benjamin_clawhauser/benjamin_clawhauser.jpeg
assets/real_world_objects/medical_bag_nus/medical_bag_nus_3118.png
assets/real_world_objects/kitchen_tongs/kitchen_tongs.jpg
```

Use `assets/real_world_objects/kitchen_tongs/kitchen_tongs.jpg` as the default smoke asset unless the user asks for a random or specific smoke asset. Do not substitute `medical_bag_nus_3118_gpt-processed.png` for `medical_bag_nus_3118.png`; the dataset definition names the original PNG.

Run HAG4R Python commands through `conda run -p "$HAG4R_ENV"`. Run RealSim build/test/server commands through `conda run -p "$REALSIM_ENV"` or through the HAG4R launcher that internally uses the RealSim env. Use `xvfb-run -a` around pipeline launches on this headless Runpod host.

Keep model/cache writes repo-local unless the user explicitly authorizes another cache:

```bash
export HAG4R_ROOT=/workspace/HAG4R-runpod
export REALSIM_ROOT=/workspace/RealSim-Hetero_Asset_Gen
export HAG4R_ENV="$HAG4R_ROOT/.conda/hag4r"
export REALSIM_ENV="$REALSIM_ROOT/.conda/realsim_py"
export HF_HOME="$HAG4R_ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$HAG4R_ROOT/.cache"
export HF_HUB_ENABLE_HF_TRANSFER=0
unset HF_ENDPOINT
cd "$HAG4R_ROOT"
```

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

Fail loudly on missing environment, images, credentials, or GPU resources. Do not guess substitute paths for NUS Dev4 input images or RealSim.

```bash
cd "$HAG4R_ROOT"
test -d "$HAG4R_ENV/conda-meta"
test -d "$REALSIM_ENV/conda-meta"
test -f configs/config_gpt55_for_all.yaml
test -x "$REALSIM_ROOT/build/bin/RealSimLiveServer"
test -f assets/real_world_objects/baseball_cap_full_cotton/baseball_cap_full_cotton.jpeg
test -f assets/real_world_objects/benjamin_clawhauser/benjamin_clawhauser.jpeg
test -f assets/real_world_objects/medical_bag_nus/medical_bag_nus_3118.png
test -f assets/real_world_objects/kitchen_tongs/kitchen_tongs.jpg
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

For NUS Dev4, run one direct `hag4r.agentic.run_pipeline` process at a time. If two or more GPUs are visible, set `CUDA_VISIBLE_DEVICES=0,1` for the run commands. If exactly one GPU is visible, set `CUDA_VISIBLE_DEVICES=0` and proceed slower. If no GPU is visible, treat it as a missing dependency and stop before full NUS Dev4.

## Smoke Run

Run a smoke asset after the mandatory update/build/preflight. Use a fresh run id and capture logs.

```bash
cd "$HAG4R_ROOT"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
SMOKE_IMAGE="assets/real_world_objects/kitchen_tongs/kitchen_tongs.jpg"
SMOKE_RUN_ID="nus_dev4_runpod_smoke_${STAMP}_kitchen_tongs"
mkdir -p .agents/logs

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" xvfb-run -a conda run -p "$HAG4R_ENV" python -m hag4r.agentic.run_pipeline \
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

- Stop before full NUS Dev4 only when the smoke failure proves missing dependencies or missing credentials, such as absent conda envs, unavailable required Python packages, missing model/checkpoint assets, missing `OPENAI_API_HAG4R_KEY`, missing RealSim binary/build output, or an unavailable required data/config path.
- Proceed to the full NUS Dev4 batch for all other smoke failures. This includes asset-specific failures, model/tool mistakes, timeouts, segmentation or OmniPart failures, RealSim diagnostics failures, final artifact mismatches, and unexplained nonzero exits after dependencies and credentials were already preflighted.

Record the smoke failure signature, smoke log path, and why it is not a dependency/credential blocker in the final report.

## Full NUS Dev4 Run

Launch the full run after the smoke attempt unless dependencies or credentials are missing. Use a direct sequential loop over the four explicit images. Continue to the next image after nonzero exits, but record the failure signature.

```bash
cd "$HAG4R_ROOT"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
RECORD_DIR="run_pipeline/runpod_nus_dev4_${STAMP}"
STATUS_TSV="$RECORD_DIR/nus_dev4_status.tsv"
mkdir -p "$RECORD_DIR/logs"
printf 'index\tslug\timage_path\trun_id\texit_code\tstatus\tstarted_at\tended_at\tlog_path\n' > "$STATUS_TSV"

NUS_DEV4_IMAGES=(
  "1|baseball_cap_full_cotton|assets/real_world_objects/baseball_cap_full_cotton/baseball_cap_full_cotton.jpeg"
  "2|benjamin_clawhauser|assets/real_world_objects/benjamin_clawhauser/benjamin_clawhauser.jpeg"
  "3|medical_bag_nus|assets/real_world_objects/medical_bag_nus/medical_bag_nus_3118.png"
  "4|kitchen_tongs|assets/real_world_objects/kitchen_tongs/kitchen_tongs.jpg"
)

overall_status=0
for item in "${NUS_DEV4_IMAGES[@]}"; do
  IFS='|' read -r index slug image_path <<< "$item"
  test -f "$image_path"
  run_id="nus_dev4_${STAMP}_${index}_${slug}"
  log_path="$RECORD_DIR/logs/${run_id}.log"
  started_at="$(date -u +%FT%TZ)"
  set +e
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" xvfb-run -a conda run -p "$HAG4R_ENV" python -m hag4r.agentic.run_pipeline \
    --source_image "$image_path" \
    --run_id "$run_id" \
    --config configs/config_gpt55_for_all.yaml \
    --fidelity low \
    --output_tag "$run_id" \
    --realsim-root "$REALSIM_ROOT" \
    > "$log_path" 2>&1
  exit_code="$?"
  set -e
  ended_at="$(date -u +%FT%TZ)"
  if [ "$exit_code" -eq 0 ]; then
    status=success
  else
    status=failed
    overall_status=1
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$index" "$slug" "$image_path" "$run_id" "$exit_code" "$status" "$started_at" "$ended_at" "$log_path" >> "$STATUS_TSV"
done
exit "$overall_status"
```

Use fresh run ids by default. Reuse or overwrite existing run ids only when the user explicitly requests a rerun over prior NUS Dev4 outputs.

For each successful asset, verify the final export and diagnostic artifacts listed in the smoke section under `outputs/run_pipeline/<run_id>/`. Also verify every `exported_artifacts` entry in `final_export_manifest.json` exists.

## Monitoring

Poll without interrupting active asset processes:

```bash
tail -20 "$RECORD_DIR"/logs/*.log 2>/dev/null || true
test -f "$STATUS_TSV" && column -t -s $'\t' "$STATUS_TSV" || true
find outputs/agentic_asset_refinement -maxdepth 3 -type f -name state.json -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort | tail -20
ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | rg -i 'run_pipeline|xvfb-run|RealSimLiveServer|omnipart|sam3' | rg -v 'rg '
```

Do not investigate individual asset failures deeply unless the user asks. For this Runpod deployment, record which asset failed and the compact error signature, continue the batch loop when possible, and stop only for missing dependencies or missing credentials that make further launches impossible.

## Archive And Report

At the end, summarize:

- HAG4R commit and RealSim commit after pull.
- RealSim rebuild command and result.
- Config, NUS Dev4 image list, HAG4R env, RealSim env, GPU visibility, selected `CUDA_VISIBLE_DEVICES`, and cache settings.
- Smoke run id, smoke image, log path, artifact-check result, and whether any smoke failure was classified as non-blocking or dependency/credential-blocking.
- Full run record dir, `nus_dev4_status.tsv`, and per-asset logs.
- Counts of successful, failed, timed-out if externally tracked, skipped, and running assets.
- Failed asset slugs with log paths and compact error summaries.
- Diagnostic recommendation fields for successful assets when available.

Create archives only for canonical NUS Dev4 run outputs that belong to this deployment, excluding smoke-only folders unless the user asks to keep them together:

```bash
COMMIT="$(git -C "$HAG4R_ROOT" rev-parse --short HEAD)"
DATE="$(date -u +%F)"
zip -r "outputs/agentic_asset_refinement/runpod_nus_dev4_${DATE}_${COMMIT}_full.zip" outputs/agentic_asset_refinement/<run-id-prefixes-or-list>
zip -r "outputs/run_pipeline/runpod_nus_dev4_${DATE}_${COMMIT}_full.zip" outputs/run_pipeline/<output-tag-prefixes-or-list>
```
