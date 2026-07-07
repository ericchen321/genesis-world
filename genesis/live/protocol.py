from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass
from typing import Any


PROTOCOL = "genesis-live-v1"
HEADER_STRUCT = struct.Struct(">I")
MAX_MESSAGE_BYTES = 64 * 1024 * 1024

CAPABILITIES = (
    "volumetric_mesh_import",
    "heterogeneous_fem_material_arrays",
    "static_box_anchors",
    "live_box_controller_actions",
    "pause_resume_reset",
    "rgb_triptych_telemetry",
    "frame_metadata",
    "geometry_context",
    "fused_observation",
)


@dataclass
class GenesisLiveError(Exception):
    code: str
    message: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details is not None:
            payload["details"] = self.details
        return payload


def encode_frame(payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(body) > MAX_MESSAGE_BYTES:
        raise GenesisLiveError("message_too_large", f"message is {len(body)} bytes")
    return HEADER_STRUCT.pack(len(body)) + body


def _recv_exact(sock: socket.socket, n_bytes: int) -> bytes:
    chunks = []
    remaining = n_bytes
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_json(sock: socket.socket) -> dict[str, Any]:
    header = _recv_exact(sock, HEADER_STRUCT.size)
    (length,) = HEADER_STRUCT.unpack(header)
    if length > MAX_MESSAGE_BYTES:
        raise GenesisLiveError("message_too_large", f"message is {length} bytes")
    try:
        payload = json.loads(_recv_exact(sock, length).decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise GenesisLiveError("invalid_json", str(exc)) from exc
    if not isinstance(payload, dict):
        raise GenesisLiveError("invalid_message", "framed JSON message must be an object")
    return payload


def send_json(sock: socket.socket, payload: dict[str, Any]) -> None:
    sock.sendall(encode_frame(payload))


def ok_response(request_id: str | int | None, data: dict[str, Any]) -> dict[str, Any]:
    return {"request_id": request_id, "status": "ok", "protocol": PROTOCOL, "data": data}


def error_response(request_id: str | int | None, error: GenesisLiveError) -> dict[str, Any]:
    return {"request_id": request_id, "status": "error", "protocol": PROTOCOL, "error": error.to_dict()}
