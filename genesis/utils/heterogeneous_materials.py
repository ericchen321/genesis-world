from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np

import genesis as gs


@dataclass(frozen=True)
class HeterogeneousMaterial:
    file: str | PathLike
    e_nu_key: str = "tet_E_nu"
    density_key: str = "tet_density"
    labels_key: str = "tet_part_labels"


@dataclass(frozen=True)
class HeterogeneousMaterialData:
    youngs_modulus: np.ndarray
    poisson_ratio: np.ndarray
    density: np.ndarray
    mu: np.ndarray
    lam: np.ndarray
    labels: np.ndarray | None
    metadata: dict[str, object]


def _float_dtype():
    return gs.np_float if gs._initialized else np.float32


def _int_dtype():
    return gs.np_int if gs._initialized else np.int32


def _raise(path, message):
    gs.raise_exception(f"Invalid heterogeneous material '{path}': {message}")


def _stats(array: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _require_key(npz, key: str, path: str):
    if key not in npz.files:
        _raise(path, f"missing required key '{key}'")
    return npz[key]


def validate_heterogeneous_material_arrays(
    e_nu: np.ndarray,
    density: np.ndarray,
    *,
    tet_count: int,
    labels: np.ndarray | None = None,
    path: str | PathLike = "<array>",
    e_nu_key: str = "tet_E_nu",
    density_key: str = "tet_density",
    labels_key: str = "tet_part_labels",
) -> HeterogeneousMaterialData:
    path = str(path)
    e_nu = np.asarray(e_nu)
    density = np.asarray(density)
    labels = None if labels is None else np.asarray(labels)

    if tet_count <= 0:
        _raise(path, f"tet_count must be positive, got {tet_count}")
    if e_nu.shape != (tet_count, 2):
        _raise(path, f"'{e_nu_key}' must have shape ({tet_count}, 2), got {e_nu.shape}")
    if density.shape == (tet_count, 1):
        density = density.reshape((tet_count,))
    elif density.shape != (tet_count,):
        _raise(path, f"'{density_key}' must have shape ({tet_count},) or ({tet_count}, 1), got {density.shape}")
    if labels is not None and labels.shape != (tet_count,):
        _raise(path, f"'{labels_key}' must have shape ({tet_count},), got {labels.shape}")

    if not np.all(np.isfinite(e_nu)):
        _raise(path, f"'{e_nu_key}' contains non-finite values")
    if not np.all(np.isfinite(density)):
        _raise(path, f"'{density_key}' contains non-finite values")

    youngs_modulus = e_nu[:, 0]
    poisson_ratio = e_nu[:, 1]

    if np.any(youngs_modulus <= 0.0):
        _raise(path, f"'{e_nu_key}' Young's modulus values must be positive")
    if np.any((poisson_ratio <= -1.0) | (poisson_ratio >= 0.5)):
        _raise(path, f"'{e_nu_key}' Poisson ratio values must satisfy -1 < nu < 0.5")
    if np.any(density <= 0.0):
        _raise(path, f"'{density_key}' values must be positive")

    youngs_modulus = youngs_modulus.astype(_float_dtype(), copy=False)
    poisson_ratio = poisson_ratio.astype(_float_dtype(), copy=False)
    density = density.astype(_float_dtype(), copy=False)
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    lam = youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    labels = None if labels is None else labels.astype(_int_dtype(), copy=False)

    metadata = {
        "material_path": path,
        "row_count": int(tet_count),
        "e_nu_key": e_nu_key,
        "density_key": density_key,
        "labels_key": labels_key if labels is not None else None,
        "has_labels": labels is not None,
        "youngs_modulus": _stats(youngs_modulus),
        "density": _stats(density),
        "mu": _stats(mu),
        "lambda": _stats(lam),
    }
    if labels is not None:
        metadata["labels"] = {
            "min": int(np.min(labels)),
            "max": int(np.max(labels)),
            "count": int(np.unique(labels).size),
        }

    return HeterogeneousMaterialData(
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio,
        density=density,
        mu=mu.astype(_float_dtype(), copy=False),
        lam=lam.astype(_float_dtype(), copy=False),
        labels=labels,
        metadata=metadata,
    )


def load_heterogeneous_material(spec: HeterogeneousMaterial, tet_count: int) -> HeterogeneousMaterialData:
    if not isinstance(spec.file, (str, PathLike)):
        gs.raise_exception("Heterogeneous material path must be a filesystem path.")

    material_path = Path(spec.file)
    try:
        with np.load(material_path, allow_pickle=False) as npz:
            e_nu = _require_key(npz, spec.e_nu_key, str(material_path))
            density = _require_key(npz, spec.density_key, str(material_path))
            labels = npz[spec.labels_key] if spec.labels_key in npz.files else None
            return validate_heterogeneous_material_arrays(
                e_nu,
                density,
                tet_count=tet_count,
                labels=labels,
                path=material_path,
                e_nu_key=spec.e_nu_key,
                density_key=spec.density_key,
                labels_key=spec.labels_key,
            )
    except Exception as exc:
        if isinstance(exc, gs.GenesisException):
            raise
        gs.raise_exception_from(f"Failed to load heterogeneous material file '{material_path}'.", exc)
