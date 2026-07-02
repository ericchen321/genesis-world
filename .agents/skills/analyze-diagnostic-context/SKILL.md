---
name: analyze-diagnostic-context
description: Analyze HAG4R RealSim diagnostics traces, state files, gateway logs, or serve logs to identify context-growth pressure from reasoning traces, model-visible image blocks, pause_and_observe/get_sim_frames frame evidence, tool-result JSON, and CUDA OOM evidence. Use when debugging local Kimi/Claude/GPT diagnostics OOMs, long-context failures, stalled diagnostic ReAct loops, or deciding whether reasoning text or visual frame evidence dominates a diagnostics run.
---

# Analyze Diagnostic Context

Use this skill for offline diagnosis only. Do not edit the HAG4R agentic pipeline while using it.

## Inputs To Gather

Collect as many of these artifacts as are available:

- `state.json` for the diagnostics run.
- `asset_refinement/sim_diagnostics/diagnostic_summary.json`.
- `asset_refinement/sim_diagnostics/observations_digest.json`.
- DeepAgents or gateway request/response traces as `.json` or `.jsonl`.
- Kimi/vLLM/TRT-LLM serve stdout/stderr logs, especially around CUDA OOM.
- Any raw run log that includes the diagnostics ReAct loop or tool-call probes.

Missing artifacts are acceptable. The analyzer reports which measures are exact and which are only proxies.

## Run The Analyzer

From the HAG4R repo root:

```bash
conda run -p ./.conda/hag4r python .agents/skills/analyze-diagnostic-context/scripts/analyze_diagnostic_context.py \
  --input outputs/agentic_asset_refinement/<run_id>/revisions/<revision_id>/state.json \
  --input outputs/run_pipeline/<tag>/asset_refinement/sim_diagnostics/diagnostic_summary.json \
  --input /path/to/kimi_serve.log \
  --output-dir .agents/reports/diagnostic_context_analysis/<run_id>
```

If an input is a directory, the script scans likely text artifacts under it (`.json`, `.jsonl`, `.log`, `.txt`, `.md`, `.out`, `.err`). Keep very large binary/media directories out of the input list.

Use `--repo-root <path>` when running from outside the repo or when virtual `/outputs/...` paths need to be resolved back into the checkout.

## Interpret The Report

The script writes:

- `diagnostic_context_report.json`: machine-readable metrics.
- `diagnostic_context_report.md`: human-readable summary for handoff.

Read these fields first:

- `image_data_url_count` and `image_data_url_payload_bytes`: exact model-visible image payload only when the captured trace includes `image_url` data URLs.
- `frame_path_counts` and `existing_frame_file_bytes`: persisted frame-reference pressure. This is a proxy when the actual model request trace is missing.
- `reasoning_token_counts` and `reasoning_content_chars`: exact only when provider metadata or reasoning text was captured.
- `tool_result_json_chars`: persisted JSON pressure from diagnostic tool results.
- `cuda_oom_events`: serve-side OOM evidence extracted from raw logs.
- `largest_items`: biggest string or JSON payload carriers to inspect next.

For the Kimi diagnostics OOM question, compare:

1. Exact data URL image payload bytes if request traces exist.
2. Otherwise, `rgb_png_paths` count and existing RGB PNG bytes as a lower-fidelity proxy for visual evidence pressure.
3. Reasoning token counts/content chars if gateway metadata exists.
4. Tool-result JSON and raw message text growth.

State clearly whether the conclusion is exact or proxy-based. Do not claim video/image pressure dominates from PNG path counts alone when the actual request trace is unavailable.
