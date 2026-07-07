from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import genesis as gs

from .protocol import CAPABILITIES, PROTOCOL


def ready_payload(
    *,
    pid: int,
    host: str,
    port: int,
    scene_config_path: str | None,
    start_paused: bool,
    heartbeat_interval_s: float,
    status: dict[str, Any],
    session_token: str | None = None,
) -> dict[str, Any]:
    payload = {
        "protocol": PROTOCOL,
        "server_pid": int(pid),
        "host": host,
        "port": int(port),
        "session_token": session_token,
        "start_paused": bool(start_paused),
        "scene_config_path": scene_config_path,
        "capabilities": list(CAPABILITIES),
        "heartbeat_interval_s": float(heartbeat_interval_s),
        "ready_time": time.time(),
        "genesis_version": gs.__version__,
        "status": status,
    }
    return payload


def write_ready_file(path: str | os.PathLike, payload: dict[str, Any]) -> None:
    ready_path = Path(path)
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{ready_path.name}.", suffix=".tmp", dir=ready_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_name, ready_path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
