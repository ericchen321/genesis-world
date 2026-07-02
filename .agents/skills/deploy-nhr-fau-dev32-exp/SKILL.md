---
name: deploy-nhr-fau-dev32-exp
description: "Deploy the HAG4R MV2-Dev32 experiment on NHR@FAU's Alex cluster with high autonomy: resolve Alex-local paths, run a one-asset smoke test through RealSim diagnostics, use squeeze-a-bug by default to fix smoke-test failures, then immediately launch the full 32-asset batch on 8 RTX Pro 6000 / Pro 6K GPUs once the smoke gate passes. Use when another agent must execute the real Dev32 Alex deployment, make deployment decisions without looping the human user into low-level choices, repair propagatable smoke failures, commit and push required tracked fixes, and archive final or partial outputs under outputs/agentic_asset_refinement and outputs/run_pipeline."
---

# Deploy NHR@FAU Dev32 Experiment

Use this skill for the real Alex-cluster Dev32 deployment. The operating posture is high autonomy: discover, run, inspect, fix, rerun, and continue toward a completed full-batch experiment. Escalate to the human user only for external blockers that the agent cannot solve from the repo, cluster, logs, artifacts, or available skills.

## Autonomy Contract

- Drive the deployment to completion. Make routine deployment decisions locally from evidence instead of asking the human user to choose between obvious next steps.
- Treat a smoke-test failure as a debuggable bug by default. Invoke `$squeeze-a-bug` for any smoke failure that has enough command/log/artifact evidence to reproduce or reason about the failure.
- Keep using `$squeeze-a-bug` across hypothesis rounds until the smoke test passes, or until the blocker is genuinely outside the agent's reachable control.
- Use `$squeeze-a-bug` even when this skill was not invoked through `/goal`; this deployment skill explicitly authorizes that nested bug-solving loop for smoke-test failures.
- After `$squeeze-a-bug` produces a fix, rerun the smoke test from raw image with a fresh smoke `run_id`. Repeat fix -> rerun until the shared smoke artifact contract passes.
- Once the smoke gate passes, immediately launch the full Dev32 batch. Avoid extra approval checkpoints between a passing smoke gate and the full 32-asset deployment.
- Preserve evidence while moving forward: logs, partial artifacts, diagnosis notes, fix commits, ledgers, and zips are deployment outputs, not reasons to stall.

### External Blockers That Merit A Human

Loop in the human user only when progress requires information or authority that cannot be obtained from the repo, Alex environment, logs, artifacts, or available skills:

- Missing Dev32 asset inputs or an ambiguous Dev32 input root after a reasonable Alex-local search.
- Missing RealSim SIF, an ambiguous SIF choice after inspecting known Alex workspace locations and deployment scripts, or evidence that the RealSim SIF must be recompiled.
- Missing required API key or API credential that is absent from the environment and documented shell/discovery locations.
- A shared service, Slurm job owned by a human, account permission, quota, or policy barrier must be changed and the agent lacks authority to do it.
- Eight consecutive `$squeeze-a-bug` hypotheses cannot produce a determined verdict, or three consecutive hypotheses are refuted with no progress, matching the escalation rules of `$squeeze-a-bug`.

Everything else should be investigated, fixed, worked around, staged, or rerun by the agent. Missing Python packages, stale caches, launcher bugs, bad environment variables, broken path propagation, diagnostics failures, batch-runner issues, and artifact-contract failures are normal deployment work unless they reduce to one of the external blockers above.

## Fixed Deployment Contract

- Run heavy work inside a real Alex GPU allocation acquired through Slurm (`salloc`, `srun`, or `sbatch`).
- Request one node with all `8` Pro 6K / RTX Pro 6000 GPUs using Alex's actual typed GRES, for example `--gres=gpu:<type>:8`.
- Keep the full Dev32 resource contract at exactly `4` concurrent workers across those `8` GPUs. Each worker owns exactly `2` GPUs, and worker GPU sets are mutually exclusive.
- Use `conda run -p <resolved_env>` for every Python command.
- Assume the current workspace is the Alex HAG4R repo root. Align it to branch `bole_feat/local-model-backends` and fast-forward from `origin/bole_feat/local-model-backends` before runtime inspection or execution.
- Discover machine-local paths on Alex. Treat prior paths as search hints, then verify every active path before using it.
- Use the checked-out deployment config explicitly: `configs/config_kimi_k26_hunyuan_cleanup.yaml`.
- Use the public pipeline entrypoint for smoke: `python -m hag4r.agentic.run_pipeline`.
- Use the Dev32 batch entrypoint for the full run: `scripts/run_mvimgnet2_dev32_pipeline.py`.
- Preserve self-hosted model servers. Diagnose around them; request human intervention only if forward progress truly requires stopping or modifying a human-owned/shared service.
- Commit and push tracked fixes that are required to clear the smoke gate before launching the full batch. Mention that the code was co-authored by the invoking agent and the human user of the machine or account that invoked it. If valid Git co-author identities are configured or known, use trailers such as:

```text
Co-authored-by: <human-user-name> <configured-or-approved-human-user-email>
Co-authored-by: <invoking-agent-name> <configured-or-approved-invoking-agent-email>
```

## Repo-Grounded Anchors

- RealSim preflight helper: `hag4r.tools.realsim.live_client.verify_realsim_live_runtime(realsim_root=..., sif_image_path=...)`
- Dev32 manifest: `outputs/build_mvimgnet_MV2_Dev32/MV2-Dev32.json`
- Manifest contents: 32 selected instances, each with a chosen image filename.
- Manifest `image_path` values are source-machine-local paths. Rebuild Alex-local image paths from the discovered Dev32 input root plus the manifest's instance-directory basename and image filename.
- `scripts/run_mvimgnet2_dev32_pipeline.py` loads the Dev32 manifest, rebuilds Alex-local paths from `--input-root`, uses Dev32 run-id/ledger/worker naming, records per-asset logs, checks critical final-export and diagnostics artifacts, and enforces two-GPU worker bindings.
- `configs/config_kimi_k26_hunyuan_cleanup.yaml` routes Kimi through OpenRouter and uses Hunyuan image cleanup. Inspect the checked-out config and code at runtime for image-cleanup backend requirements.
- For this config, pipeline credentials come from `OPENROUTER_API_HAG4R_KEY` plus the Hunyuan image-cleanup variables required by checked-out code.
- Keep real upstream keys out of logs, tracked files, discovery files, and batch ledgers.

## Alex Search Hints

Treat these as hints, then verify:

- Prior Alex repo root: `/anvme/workspace/ihpc125h-real2sim/HAG4R`
- Prior Alex Dev32 copied-input root: `/anvme/workspace/ihpc125h-real2sim/MVImgNet/mvi2_00_mv2-dev32`
- Prior Alex RealSim host root: `/anvme/workspace/ihpc125h-real2sim/realsim_host_root`
- Prior Alex RealSim live SIF: `/anvme/workspace/ihpc125h-real2sim/realsim_apptainer_images/realsim-alex.sif`
- Prior Alex HAG4R env prefix: `/anvme/workspace/ihpc125h-real2sim/.conda/envs/hag4r`

If multiple plausible candidates remain after search and inspection, document the candidates and loop in the human user because choosing the wrong asset root or SIF would invalidate the run.

## Workflow

### 1. Align Workspace And Resolve Runtime Inputs

Confirm the current directory is a Git worktree, then align the branch:

```bash
module load python
git rev-parse --show-toplevel
git fetch origin
git checkout bole_feat/local-model-backends
git pull --ff-only origin bole_feat/local-model-backends
git rev-parse --short HEAD
```

If the checkout is absent, dirty in a way that blocks the branch move, or unable to fast-forward, inspect the concrete conflict/state. Resolve routine local build artifacts or ignored files yourself; escalate only when preserving unknown human edits or resolving a non-fast-forward branch decision requires the human user.

Resolve these before smoke:

- Current HAG4R repo root on Alex, from `git rev-parse --show-toplevel`
- HAG4R env prefix, preferring `./.conda/hag4r` when it exists and has `conda-meta`
- Dev32 input root on Alex
- RealSim host root on Alex
- RealSim live SIF on Alex
- `OPENROUTER_API_HAG4R_KEY`
- `NHR_FAU_API_KEY` only if the checked-out config/code still references it after loading the required config
- Hunyuan/image-cleanup values required by checked-out code, such as `HAG4R_HUNYUAN_ROOT`, `HAG4R_HUNYUAN_SIF`, `HAG4R_HUNYUAN_MODEL_PATH`, `HAG4R_HUNYUAN_PYUSERBASE`, and `HAG4R_HUNYUAN_VARIANT`

Resolve the env with the local prefix first:

```bash
if [ -d ./.conda/hag4r/conda-meta ]; then
  HAG4R_ENV="$(realpath ./.conda/hag4r)"
else
  mapfile -t HAG4R_ENV_CANDIDATES < <(
    {
      conda env list | awk 'NF >= 2 {print $NF}' | grep -E '/hag4r([^/]*$|/)' || true
      find /anvme/workspace /home "$PWD" -maxdepth 5 -type d -path '*/conda-meta' 2>/dev/null \
        | sed 's#/conda-meta$##' \
        | grep -E '/hag4r([^/]*$|/)'
    } | awk '!seen[$0]++'
  )
  if [ "${#HAG4R_ENV_CANDIDATES[@]}" -ne 1 ]; then
    printf 'Could not resolve exactly one HAG4R env. Candidates:\n'
    printf '  %s\n' "${HAG4R_ENV_CANDIDATES[@]}"
    exit 1
  fi
  HAG4R_ENV="${HAG4R_ENV_CANDIDATES[0]}"
fi
conda run -p "$HAG4R_ENV" python -c 'import sys; print(sys.executable)'
```

Resolve `OPENROUTER_API_HAG4R_KEY` from the environment or shell profile:

```bash
if [ -z "${OPENROUTER_API_HAG4R_KEY:-}" ] \
  && [ -f "$HOME/.bashrc" ] \
  && grep -Eq '^(export[[:space:]]+)?OPENROUTER_API_HAG4R_KEY=' "$HOME/.bashrc"; then
  set -a
  # shellcheck disable=SC1090
  . "$HOME/.bashrc"
  set +a
fi
if [ -z "${OPENROUTER_API_HAG4R_KEY:-}" ]; then
  printf 'Missing required credential OPENROUTER_API_HAG4R_KEY.\n' >&2
  exit 1
fi
```

Probe the configured model endpoint from the same environment that will launch the pipeline:

```bash
conda run -p "$HAG4R_ENV" python -c 'import os; from openai import OpenAI; token=os.environ.get("OPENROUTER_API_HAG4R_KEY"); assert token, "OPENROUTER_API_HAG4R_KEY must be set"; models=[m.id for m in OpenAI(api_key=token, base_url="https://openrouter.ai/api/v1").models.list()]; assert any(m == "moonshotai/kimi-k2.6" or "kimi" in m.lower() for m in models), "OpenRouter did not advertise a Kimi model"; print("OpenRouter Kimi preflight OK")'
```

Preflight config and RealSim runtime:

```bash
conda run -p "$HAG4R_ENV" python - <<'PY'
from pathlib import Path
from hag4r.agentic.run_config import load_run_config
from hag4r.tools.realsim.live_client import verify_realsim_live_runtime

repo = Path.cwd()
cfg = load_run_config(Path("configs/config_kimi_k26_hunyuan_cleanup.yaml"), repo_root=repo)
assert cfg.realsim_diagnostics.enable, "diagnostics must be enabled"
provider = cfg.providers["OpenRouter"]
assert provider.api_key_env == "OPENROUTER_API_HAG4R_KEY", provider.api_key_env
assert provider.base_url == "https://openrouter.ai/api/v1", provider.base_url
verify_realsim_live_runtime(
    realsim_root=Path("<ALEX_REALSIM_ROOT>"),
    sif_image_path=Path("<ALEX_REALSIM_SIF>"),
)
print("run config + RealSim preflight OK")
PY
```

### 2. Confirm 8x Pro 6K Allocation

Inspect the live allocation:

```bash
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"
echo "SLURM_GPUS=${SLURM_GPUS:-}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo "SLURM_JOB_GRES=${SLURM_JOB_GRES:-}"
echo "SLURM_STEP_GRES=${SLURM_STEP_GRES:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L
```

Confirm exactly `8` `Pro 6K / RTX Pro 6000` GPUs. If no allocation exists or the wrong hardware was assigned, acquire the correct allocation and continue. Use one-GPU or wrong-GPU smoke tests only for cheap diagnosis when they cannot be mistaken for a passing deployment gate.

For full batch, prefer four mutually exclusive Slurm job steps inside the 8-GPU allocation:

- Use `srun --exclusive --gres=gpu:<type>:2 ...` with Alex's real typed GRES.
- Treat each step's `CUDA_VISIBLE_DEVICES` as that worker's GPU binding.
- Log `worker_id`, current asset, `CUDA_VISIBLE_DEVICES`, `SLURM_JOB_GPUS`, `SLURM_STEP_GPUS`, `SLURM_GPUS`, `SLURM_GPUS_ON_NODE`, `SLURM_JOB_GRES`, and `SLURM_STEP_GRES`.
- If per-worker Slurm isolation is unreliable, create a repo-local helper under `.agents/tmp/` that assigns four disjoint two-GPU pairs, for example `0,1`, `2,3`, `4,5`, and `6,7`, or derives equivalent pairs from the visible allocation.
- If the checked-out batch helper only accepts single GPU ids, create the `.agents/tmp/` paired-GPU launcher and keep moving.

### 3. Build The Dev32 Asset Table

Read `outputs/build_mvimgnet_MV2_Dev32/MV2-Dev32.json` and construct a 32-row table. For each manifest entry:

- Keep the instance key from `subset_instances`.
- Keep the selected image filename from the basename of `image_path`.
- Keep the instance directory basename from `source_instance_dir`.
- Rebuild the Alex-local image path as `<ALEX_DEV32_INPUT_ROOT>/<instance_dir_basename>/<image_filename>`.
- Verify every rebuilt image path exists.

Create:

- A canonical full-run `run_id`.
- A smoke-test `run_id` with a distinct suffix such as `__smoke_<timestamp>`.
- A smoke-test output tag. With direct `python -m hag4r.agentic.run_pipeline` and no explicit `--output_tag`, the expected final-export tag is `<SMOKE_RUN_ID>_volumetric_agentic`.
- Full-batch output tags from `scripts/run_mvimgnet2_dev32_pipeline.py`; use shard ledgers' `final_export_dir` and `output_tag` fields as source of truth.

The full batch must rerun all 32 assets, including the smoke-tested asset.

### 4. Run One Random Smoke Asset

Choose one manifest asset uniformly at random. Record the seed, chosen asset id, instance directory, and image path in the smoke log.

Run the full path:

- raw image
- object description
- image cleanup
- segmentation
- OmniPart
- material inference
- mesh processing
- RealSim diagnostics
- diagnostic-triggered revision path when triggered
- final export

Use:

```bash
conda run -p "$HAG4R_ENV" python -m hag4r.agentic.run_pipeline \
  --source_image "<ALEX_SMOKE_IMAGE>" \
  --run_id "<SMOKE_RUN_ID>" \
  --run_root outputs/agentic_asset_refinement \
  --config configs/config_kimi_k26_hunyuan_cleanup.yaml \
  --representation volumetric \
  --fidelity medium \
  --realsim-root "<ALEX_REALSIM_ROOT>" \
  --sim-diagnostics-live-sif-image-path "<ALEX_REALSIM_SIF>" \
  --sim-diagnostics-max-runs 2 \
  --sim-diagnostics-max-episodes 4 \
  --sim-diagnostics-max-actions-per-episode 5
```

Capture stdout/stderr into `.agents/logs/<date>_dev32_smoke_<run_id>.log`.

If the smoke run exceeds 2 hours, inspect the active stage, logs, process tree, GPU state, latest artifacts, and server logs. Classify it as a smoke failure with evidence and feed it to `$squeeze-a-bug` unless it reduces to a listed external blocker.

### 5. Decide Whether Smoke Passed

The smoke gate passes when the shared artifact contract is intact. Check artifacts, not return code alone.

Verify these exist and are non-empty:

- `outputs/run_pipeline/<SMOKE_OUTPUT_TAG>/final_export_manifest.json`
- `outputs/run_pipeline/<SMOKE_OUTPUT_TAG>/final_mesh.mesh`
- `outputs/run_pipeline/<SMOKE_OUTPUT_TAG>/heterogeneous_params.npz`
- `outputs/run_pipeline/<SMOKE_OUTPUT_TAG>/homogeneous_params.npz`
- `outputs/run_pipeline/<SMOKE_OUTPUT_TAG>/inferred_params.json`
- `outputs/run_pipeline/<SMOKE_OUTPUT_TAG>/cleaned_image.png`
- `outputs/run_pipeline/<SMOKE_OUTPUT_TAG>/vlm_image_cleanup.json`
- `outputs/run_pipeline/<SMOKE_OUTPUT_TAG>/asset_refinement/state.json`
- `outputs/run_pipeline/<SMOKE_OUTPUT_TAG>/asset_refinement/sim_diagnostics/diagnostic_summary.json`

Also verify that every `final_export_manifest.json` `exported_artifacts` entry exists.

Extract from `asset_refinement/sim_diagnostics/diagnostic_summary.json`:

- `recommended_route`
- `recommended_ready`
- recommendation reasons
- episode count
- video paths when present
- route adjudication fields when present

An asset-specific physical recommendation such as `revise` can still pass smoke if the pipeline reached the expected artifact boundary and did not reveal a shared deployment/code failure.

### 6. Autonomous Smoke-Failure Debug Loop

When smoke fails before the artifact contract is met, immediately preserve evidence, then invoke `$squeeze-a-bug`.

First capture the evidence package:

- smoke command, run id, output tag, branch, commit, host, Slurm job id, GPU fields, env prefix
- smoke stdout/stderr log path
- latest stage and traceback/error signature
- generated directories under `outputs/agentic_asset_refinement/<SMOKE_RUN_ID>*` and `outputs/run_pipeline/<SMOKE_OUTPUT_TAG>*`
- relevant `state.json`, scratch files, diagnostics summaries, live server logs, provider logs, and per-stage artifacts
- partial rescue zips when useful, especially before changing code or rerunning with a fresh id

Then call `$squeeze-a-bug` with:

- Bug description: "Dev32 Alex smoke test failed before the shared artifact contract passed."
- Expected behavior: smoke reaches final export and diagnostics artifacts listed in this skill.
- Reproduction steps: the exact smoke command and required env/path variables.
- Error output: log excerpts, artifact paths, traceback, failed artifact checks, and runtime evidence.
- Suspected scope: infer from the failure signature; include launcher, Hunyuan cleanup, segmentation, OmniPart, material inference, mesh processing, RealSim live/SIF diagnostics, or final export as applicable.
- Prior attempts: every smoke run and fix attempted in this deployment.
- Branch knowledge: read `.agents/knowledge/<branch>.md` if present.

Let `$squeeze-a-bug` drive hypotheses and invoke `build-a-feature` for implementation. The deploy agent may still collect runtime evidence, rerun smoke, archive outputs, commit completed fixes, and maintain deployment logs around that loop.

Classify the failure after evidence exists:

- Propagatable shared failure: likely to affect many assets, such as launcher wiring, SIF propagation, provider auth config, backend paths, diagnostics thresholds, batch orchestration, missing shared resources, worker isolation, or final-export contract bugs.
- Asset-specific failure: one mesh/content/geometry case that still reached the artifact boundary or clearly does not threaten the dataset run.

Use this classification to choose fix scope, not to avoid debugging. If classification is uncertain, treat it as propagatable until a squeeze-a-bug round proves otherwise.

For tracked fixes produced by the debug loop:

- Keep the diff focused on the confirmed root cause.
- Write `.agents/logs/<date>_dev32_smoke_fix.md`.
- Include hostname, Slurm job id, asset id, image path, original failure signature, why the issue was shared or blocking, files changed, commands rerun, and final smoke result.
- Commit the tracked fix and smoke-fix log on `bole_feat/local-model-backends`.
- Push with `git push origin HEAD:bole_feat/local-model-backends`.
- Mention co-authorship by the invoking agent and the human user of the machine or account that invoked it. Use valid `Co-authored-by:` trailers when configured or approved identities are available.

Example commit:

```text
Fix Dev32 Alex diagnostics SIF path propagation

Motivation: smoke test failed before diagnostics because the shared launcher did not pass the resolved Alex SIF path into RealSim.

Co-authored-by: <human-user-name> <configured-or-approved-human-user-email>
Co-authored-by: <invoking-agent-name> <configured-or-approved-invoking-agent-email>
```

If no tracked code change is needed, record the operational fix in `.agents/logs/` and continue.

After each fix or operational repair, rerun the smoke test from raw image with a fresh smoke `run_id`. Continue the loop until the smoke artifact contract passes or one of the listed external blockers is reached.

### 7. Launch Full Dev32 Immediately After Smoke Passes

After the smoke gate passes, proceed directly to the full 32-asset run. Verify the current commit and pushed state only when tracked fixes were made after the smoke run.

Launch four shards from the same 8-GPU allocation, one two-GPU job step per shard:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p run_pipeline/logs

shard_pids=()
for shard_index in 0 1 2 3; do
  srun --exclusive --gres=gpu:<type>:2 \
    conda run -p "$HAG4R_ENV" python scripts/run_mvimgnet2_dev32_pipeline.py \
      --input-root "$ALEX_DEV32_INPUT_ROOT" \
      --manifest outputs/build_mvimgnet_MV2_Dev32/MV2-Dev32.json \
      --config configs/config_kimi_k26_hunyuan_cleanup.yaml \
      --runs-root outputs/agentic_asset_refinement \
      --record-dir run_pipeline \
      --shard-count 4 \
      --shard-index "$shard_index" \
      --realsim-root "$ALEX_REALSIM_ROOT" \
      --sim-diagnostics-live-sif-image-path "$ALEX_REALSIM_SIF" \
      --continue-on-failure \
      > "run_pipeline/logs/mv2_dev32_shard_${shard_index}.srun.log" 2>&1 &
  shard_pids+=("$!")
done

batch_status=0
for pid in "${shard_pids[@]}"; do
  if ! wait "$pid"; then
    batch_status=1
  fi
done
if [ "$batch_status" -ne 0 ]; then
  echo "At least one Dev32 shard reported failure; inspect ledgers/logs and preserve partial evidence."
fi
```

Requirements:

- Run exactly 4 concurrent workers.
- Give each worker exactly 2 GPUs.
- Log one per-asset file plus four shard stdout logs.
- Keep shard JSONL ledgers and summary JSON files from `scripts/run_mvimgnet2_dev32_pipeline.py`.
- Record input image, run id, output tag, worker slot, GPU binding, Slurm GPU fields, start/end time, return code, wallclock seconds, final export manifest path, diagnostic summary path, log path, and success/failure classification.
- Keep the per-asset 2-hour timeout. If an asset times out, inspect whether the cause is shared. Shared causes trigger evidence preservation and `$squeeze-a-bug`; asset-local causes are recorded and the batch continues.
- Sample at least two running assets every 15 minutes when attached interactively, reporting completed, failed, and running counts.

### 8. Classify Full-Batch Results

An asset is successful only when critical artifacts exist and are non-empty. Use the same artifact checks as smoke for each full-run output tag, then extract each asset's diagnostic recommendation fields into the ledger or final report.

If most assets fail with the same signature, pause the batch, preserve evidence, and invoke `$squeeze-a-bug` with the repeated full-batch failure as the reproduction scenario. After the fix, resume or relaunch in the least wasteful way supported by the runner and artifacts.

### 9. Archive Outputs

For a completed full run, create exactly:

- `outputs/agentic_asset_refinement/mv2_dev32_<YYYY-MM-DD>_<commit>_full.zip`
- `outputs/run_pipeline/mv2_dev32_<YYYY-MM-DD>_<commit>_full.zip`

Include only canonical full-run 32 asset folders. Exclude smoke-only folders. Record zip paths and binary sizes.

If the deployment ends with partial data because of an external blocker or unsolved repeated failure, still create partial rescue archives from generated Dev32 run directories:

- `outputs/agentic_asset_refinement/mv2_dev32_<YYYY-MM-DD>_<commit>_partial.zip`
- `outputs/run_pipeline/mv2_dev32_<YYYY-MM-DD>_<commit>_partial.zip`

For each zip, summarize:

- included asset run directories
- number of assets with content
- whether it contains final exports, diagnostics evidence, logs only, or mixed artifacts
- important files present, such as `state.json`, `diagnostic_summary.json`, `final_export_manifest.json`, `final_mesh.mesh`, and per-asset logs

### 10. Final Report

Write a concise final report with:

- branch and commit
- Alex-local paths discovered
- env prefix used
- GPU topology and concurrency rule used
- smoke-test asset, seed, attempts, failures, fixes, and final outcome
- `$squeeze-a-bug` report paths and hypothesis count when used
- tracked-code changes, fix-log path, commit hash, and push result when applicable
- full-batch totals: success, failure, timeout
- failed asset list with root causes and log paths
- per-asset route recommendations
- shard JSONL ledger paths, shard summary JSON paths, shard stdout logs, and per-asset log paths
- zip paths and binary sizes
- brief summary of each zip's contents
- remaining caveats that materially affect trust in the run
