#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import heapq
import json
import re
from pathlib import Path
from typing import Any


TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".md", ".out", ".err"}
FRAME_PATH_KEYS = ("rgb_png_paths", "depth_png_paths", "von_mises_png_paths")
TOKEN_KEYS = {
    "input_tokens",
    "prompt_tokens",
    "output_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning_tokens",
}
TERMINAL_TOOLS = {"submit_diagnostic_recommendation", "halt_diagnostics"}
OOM_PATTERNS = (
    re.compile(r"CUDA out of memory|CUDA OOM|out of memory", re.IGNORECASE),
    re.compile(r"tried to allocate .*? free", re.IGNORECASE),
)
PNG_PATH_RE = re.compile(r"[\w./:-]*frame_\d+\.png")
DATA_URL_RE = re.compile(r"data:image/[^;,\s]+;base64,([A-Za-z0-9+/=\n\r]+)")


def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _payload_bytes_from_data_url(url: str) -> int:
    if not url.startswith("data:image/") or ";base64," not in url:
        return 0
    payload = url.split(",", 1)[1].strip()
    padding = payload.count("=")
    return max(0, (len(payload) * 3) // 4 - padding)


class TopItems:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._heap: list[tuple[int, str, str, str]] = []

    def add(self, *, size: int, source: str, path: str, kind: str) -> None:
        if size <= 0:
            return
        item = (size, source, path, kind)
        if len(self._heap) < self.limit:
            heapq.heappush(self._heap, item)
        elif size > self._heap[0][0]:
            heapq.heapreplace(self._heap, item)

    def items(self) -> list[dict[str, Any]]:
        return [
            {"size": size, "source": source, "path": path, "kind": kind}
            for size, source, path, kind in sorted(self._heap, reverse=True)
        ]


class Analyzer:
    def __init__(self, *, repo_root: Path, largest_limit: int) -> None:
        self.repo_root = repo_root.expanduser().resolve()
        self.files: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self.counters: collections.Counter[str] = collections.Counter()
        self.tool_names: collections.Counter[str] = collections.Counter()
        self.terminal_tools: collections.Counter[str] = collections.Counter()
        self.tokens: collections.Counter[str] = collections.Counter()
        self.frame_path_counts: collections.Counter[str] = collections.Counter()
        self.existing_frame_file_bytes: collections.Counter[str] = collections.Counter()
        self.missing_frame_paths: collections.Counter[str] = collections.Counter()
        self.get_sim_frames: list[dict[str, Any]] = []
        self.cuda_oom_events: list[dict[str, Any]] = []
        self.largest_items = TopItems(largest_limit)

    def scan_file(self, path: Path) -> None:
        data = path.read_bytes()
        source = str(path)
        self.files.append(
            {
                "path": source,
                "bytes": len(data),
                "sha256": _sha256(data),
            }
        )
        self.counters["files_scanned"] += 1
        self.counters["input_file_bytes"] += len(data)
        text = data.decode("utf-8", errors="replace")
        parses_structured = path.suffix.lower() in {".json", ".jsonl"}
        self._scan_text(text, source=source, count_data_urls=not parses_structured)

        if path.suffix.lower() == ".jsonl":
            parsed_any = False
            for line_no, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                parsed_any = True
                self.counters["jsonl_records"] += 1
                self._scan_json(value, source=source, path=f"$[{line_no}]")
            if not parsed_any:
                self.warnings.append(f"No JSONL records parsed from {source}")
            return

        if path.suffix.lower() == ".json":
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                self.warnings.append(f"Could not parse JSON {source}: {exc}")
                return
            self.counters["json_documents"] += 1
            self._scan_json(value, source=source, path="$")

    def _scan_text(self, text: str, *, source: str, count_data_urls: bool) -> None:
        self.counters["raw_text_chars"] += len(text)
        if count_data_urls:
            for match in DATA_URL_RE.finditer(text):
                payload = match.group(1)
                self.counters["image_data_url_count"] += 1
                self.counters["image_data_url_payload_bytes"] += _payload_bytes_from_data_url(
                    "data:image/png;base64," + payload
                )
        for line_no, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in OOM_PATTERNS):
                self.cuda_oom_events.append({"source": source, "line": line_no, "text": line.strip()[:1000]})
        for path_text in PNG_PATH_RE.findall(text):
            if "frame_" in path_text:
                self.counters["raw_frame_png_path_mentions"] += 1

    def _scan_json(self, value: Any, *, source: str, path: str) -> None:
        if isinstance(value, dict):
            self._scan_dict(value, source=source, path=path)
            for key, child in value.items():
                self._scan_json(child, source=source, path=f"{path}.{key}")
            return
        if isinstance(value, list):
            self.largest_items.add(size=len(_json_dumps(value)), source=source, path=path, kind="json_list")
            for index, child in enumerate(value):
                self._scan_json(child, source=source, path=f"{path}[{index}]")
            return
        if isinstance(value, str):
            self._scan_string(value, source=source, path=path)
            return
        if isinstance(value, int | float) and path.rsplit(".", 1)[-1] in TOKEN_KEYS:
            self.tokens[path.rsplit(".", 1)[-1]] += int(value)

    def _scan_dict(self, value: dict[str, Any], *, source: str, path: str) -> None:
        self.largest_items.add(size=len(_json_dumps(value)), source=source, path=path, kind="json_object")
        if self._looks_like_message(value):
            self.counters["message_like_objects"] += 1
        content = value.get("content")
        if isinstance(content, str):
            self.counters["message_content_chars"] += len(content)
        elif isinstance(content, list):
            self.counters["message_content_blocks"] += len(content)
        if "tool" in value and "result" in value:
            self.counters["tool_result_json_chars"] += len(_json_dumps(value))

        for key in ("name", "tool_name", "terminal_tool_name", "tool"):
            name = value.get(key)
            if isinstance(name, str):
                self.tool_names[name] += 1
                if name in TERMINAL_TOOLS:
                    self.terminal_tools[name] += 1

        sequence = value.get("tool_call_sequence")
        if isinstance(sequence, list):
            for name in sequence:
                if isinstance(name, str):
                    self.tool_names[name] += 1
                    if name in TERMINAL_TOOLS:
                        self.terminal_tools[name] += 1

        for call in value.get("tool_calls", []) if isinstance(value.get("tool_calls"), list) else []:
            if isinstance(call, dict) and isinstance(call.get("name"), str):
                name = call["name"]
                self.tool_names[name] += 1
                if name in TERMINAL_TOOLS:
                    self.terminal_tools[name] += 1

        self._scan_image_block(value)
        self._scan_frame_path_keys(value, source=source, path=path)

        if value.get("tool") in {"get_sim_frames", "pause_and_observe"} or value.get("name") in {
            "get_sim_frames",
            "pause_and_observe",
        }:
            self._record_get_sim_frames(value, source=source, path=path)

        if isinstance(value.get("reasoning_content"), str):
            self.counters["reasoning_content_chars"] += len(value["reasoning_content"])
            self.counters["reasoning_content_blocks"] += 1
        elif isinstance(value.get("reasoning_content"), list):
            for item in value["reasoning_content"]:
                if isinstance(item, str):
                    self.counters["reasoning_content_chars"] += len(item)
                    self.counters["reasoning_content_blocks"] += 1

    def _scan_string(self, value: str, *, source: str, path: str) -> None:
        self.counters["json_string_chars"] += len(value)
        self.largest_items.add(size=len(value), source=source, path=path, kind="string")
        if value.startswith("data:image/"):
            self.counters["image_data_url_count"] += 1
            self.counters["image_data_url_payload_bytes"] += _payload_bytes_from_data_url(value)

    def _scan_image_block(self, value: dict[str, Any]) -> None:
        if value.get("type") != "image_url":
            return
        image_url = value.get("image_url")
        if not isinstance(image_url, dict):
            return
        url = image_url.get("url")
        if not isinstance(url, str):
            return
        self.counters["image_url_blocks"] += 1

    def _scan_frame_path_keys(self, value: dict[str, Any], *, source: str, path: str) -> None:
        for key in FRAME_PATH_KEYS:
            paths = value.get(key)
            if not isinstance(paths, list):
                continue
            self.frame_path_counts[key] += len([item for item in paths if isinstance(item, str)])
            for item in paths:
                if not isinstance(item, str):
                    continue
                resolved = self._resolve_path(item)
                if resolved.is_file():
                    self.existing_frame_file_bytes[key] += resolved.stat().st_size
                else:
                    self.missing_frame_paths[key] += 1
            self.largest_items.add(
                size=len(_json_dumps(paths)),
                source=source,
                path=f"{path}.{key}",
                kind="frame_path_list",
            )

    def _record_get_sim_frames(self, value: dict[str, Any], *, source: str, path: str) -> None:
        result = value.get("result")
        if isinstance(result, dict) and isinstance(result.get("result"), dict):
            result = result["result"]
        if not isinstance(result, dict):
            result = value.get("visual_evidence") if isinstance(value.get("visual_evidence"), dict) else {}
        summary = {
            "source": source,
            "path": path,
            "status": value.get("status"),
            "count": result.get("count") if isinstance(result, dict) else None,
            "frame_ids": result.get("frame_ids", []) if isinstance(result, dict) else [],
            "rgb_png_paths": len(result.get("rgb_png_paths", [])) if isinstance(result.get("rgb_png_paths"), list) else 0,
            "depth_png_paths": len(result.get("depth_png_paths", []))
            if isinstance(result.get("depth_png_paths"), list)
            else 0,
            "von_mises_png_paths": len(result.get("von_mises_png_paths", []))
            if isinstance(result.get("von_mises_png_paths"), list)
            else 0,
        }
        self.get_sim_frames.append(summary)

    def _looks_like_message(self, value: dict[str, Any]) -> bool:
        return any(key in value for key in ("role", "content", "tool_calls", "additional_kwargs"))

    def _resolve_path(self, value: str) -> Path:
        raw = Path(value).expanduser()
        if raw.is_absolute():
            if raw.is_file():
                return raw
            try:
                relative = raw.relative_to("/outputs")
            except ValueError:
                return raw
            return self.repo_root / "outputs" / relative
        return self.repo_root / raw

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": "hag4r-diagnostic-context-analysis-v1",
            "generated_at": _now_iso(),
            "repo_root": str(self.repo_root),
            "files": self.files,
            "totals": dict(self.counters),
            "tool_name_counts": dict(self.tool_names),
            "terminal_tool_counts": dict(self.terminal_tools),
            "reasoning_token_counts": dict(self.tokens),
            "frame_path_counts": dict(self.frame_path_counts),
            "existing_frame_file_bytes": dict(self.existing_frame_file_bytes),
            "missing_frame_paths": dict(self.missing_frame_paths),
            "get_sim_frames": self.get_sim_frames,
            "cuda_oom_events": self.cuda_oom_events,
            "largest_items": self.largest_items.items(),
            "warnings": self.warnings,
            "caveats": [
                "image_data_url_payload_bytes is exact only when request traces contain image_url data URLs.",
                "frame_path_counts and existing_frame_file_bytes are proxies when actual model request traces are missing.",
                "reasoning token counts are exact only when provider or gateway metadata was captured.",
            ],
        }


def _iter_input_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved.is_dir():
            files.extend(
                file
                for file in sorted(resolved.rglob("*"))
                if file.is_file() and file.suffix.lower() in TEXT_SUFFIXES
            )
        elif resolved.is_file():
            files.append(resolved)
        else:
            raise FileNotFoundError(f"input does not exist: {path}")
    return files


def _markdown_table(rows: list[tuple[str, Any]]) -> str:
    lines = ["| Metric | Value |", "|---|---:|"]
    for key, value in rows:
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    totals = report["totals"]
    reasoning = report["reasoning_token_counts"]
    rows = [
        ("files scanned", totals.get("files_scanned", 0)),
        ("input file bytes", totals.get("input_file_bytes", 0)),
        ("JSON documents", totals.get("json_documents", 0)),
        ("JSONL records", totals.get("jsonl_records", 0)),
        ("message-like objects", totals.get("message_like_objects", 0)),
        ("message content chars", totals.get("message_content_chars", 0)),
        ("tool result JSON chars", totals.get("tool_result_json_chars", 0)),
        ("JSON string chars", totals.get("json_string_chars", 0)),
        ("image URL blocks", totals.get("image_url_blocks", 0)),
        ("image data URL count", totals.get("image_data_url_count", 0)),
        ("image data URL payload bytes", totals.get("image_data_url_payload_bytes", 0)),
        ("reasoning tokens", reasoning.get("reasoning_tokens", 0)),
        ("reasoning content chars", totals.get("reasoning_content_chars", 0)),
        ("CUDA OOM lines", len(report["cuda_oom_events"])),
    ]
    lines = [
        "# HAG4R Diagnostic Context Analysis",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Repo root: `{report['repo_root']}`",
        "",
        "## Summary",
        "",
        _markdown_table(rows),
        "",
        "## Frame Evidence",
        "",
        "Frame path counts:",
        "",
        "```json",
        json.dumps(report["frame_path_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "Existing frame file bytes:",
        "",
        "```json",
        json.dumps(report["existing_frame_file_bytes"], indent=2, sort_keys=True),
        "```",
        "",
        "## Tool Calls",
        "",
        "```json",
        json.dumps(report["tool_name_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "Terminal tools:",
        "",
        "```json",
        json.dumps(report["terminal_tool_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "## Reasoning Metadata",
        "",
        "```json",
        json.dumps(report["reasoning_token_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "## CUDA OOM Evidence",
        "",
    ]
    if report["cuda_oom_events"]:
        for event in report["cuda_oom_events"][:20]:
            lines.append(f"- `{event['source']}:{event['line']}` {event['text']}")
    else:
        lines.append("No CUDA OOM lines found in scanned text.")
    lines.extend(
        [
            "",
            "## Largest Items",
            "",
            "| Size | Kind | Source | JSON Path |",
            "|---:|---|---|---|",
        ]
    )
    for item in report["largest_items"]:
        lines.append(f"| {item['size']} | {item['kind']} | `{item['source']}` | `{item['path']}` |")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            *[f"- {caveat}" for caveat in report["caveats"]],
        ]
    )
    if report["warnings"]:
        lines.extend(["", "## Warnings", "", *[f"- {warning}" for warning in report["warnings"]]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze HAG4R diagnostics context-growth artifacts.")
    parser.add_argument("--input", action="append", type=Path, required=True, help="Input file or directory. Repeatable.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for JSON and Markdown reports.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="HAG4R repo root for resolving /outputs paths.")
    parser.add_argument("--largest-limit", type=int, default=20, help="Number of largest JSON/string carriers to report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analyzer = Analyzer(repo_root=args.repo_root, largest_limit=args.largest_limit)
    for path in _iter_input_files(args.input):
        analyzer.scan_file(path)
    report = analyzer.report()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "diagnostic_context_report.json"
    md_path = args.output_dir / "diagnostic_context_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
