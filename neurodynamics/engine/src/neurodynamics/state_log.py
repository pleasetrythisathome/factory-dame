"""Snapshot logger: writes time-stamped oscillator state to parquet.

Schema:
    t            float64   seconds from audio start
    layer        string    "rhythm" | "pitch"
    z_real       list<f4>  per-oscillator real part
    z_imag       list<f4>  per-oscillator imaginary part
    amp          list<f4>  |z|
    phase        list<f4>  arg(z)
    phantom      list<bool> phantom mask
    drive        list<f4>  input magnitude
    residual     list<f4>  prediction residual (surprise signal)

One row per snapshot per layer. snapshot_hz controls rate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class StateLog:
    def __init__(self, path: Path, layers: dict[str, dict]):
        """layers: {name: {"f": ndarray of natural freqs}}, stored as metadata."""
        self.path = Path(path)
        self._rows: list[dict] = []
        self.layer_meta = {
            name: {"f": meta["f"].astype(np.float32).tolist()}
            for name, meta in layers.items()
        }

    def snapshot(self, t: float, layer: str, z: np.ndarray,
                 phantom: np.ndarray, drive: np.ndarray,
                 residual: np.ndarray) -> None:
        amp = np.abs(z).astype(np.float32)
        phase = np.angle(z).astype(np.float32)
        self._rows.append({
            "t": float(t),
            "layer": layer,
            "z_real": z.real.astype(np.float32).tolist(),
            "z_imag": z.imag.astype(np.float32).tolist(),
            "amp": amp.tolist(),
            "phase": phase.tolist(),
            "phantom": phantom.astype(bool).tolist(),
            "drive": drive.astype(np.float32).tolist(),
            "residual": residual.astype(np.float32).tolist(),
        })

    def flush(self) -> None:
        if not self._rows:
            return
        table = pa.Table.from_pylist(self._rows)
        meta = {
            f"layer.{name}.f": ",".join(f"{x:.6f}" for x in m["f"])
            for name, m in self.layer_meta.items()
        }
        existing_meta = table.schema.metadata or {}
        merged = {**{k.encode(): v.encode() for k, v in meta.items()},
                  **existing_meta}
        table = table.replace_schema_metadata(merged)
        pq.write_table(table, self.path)
        self._rows.clear()
