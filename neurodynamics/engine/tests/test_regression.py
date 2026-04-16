"""Regression harness: record summary statistics from a reference run and
assert future runs match within tolerance.

We don't compare full parquets byte-for-byte — the RK4 integrator can shift
least-significant bits when numpy/scipy change versions. Instead we compare
a set of stable derived statistics (peak oscillator index, amplitude percentiles,
phantom coverage) that would only meaningfully change if the engine's behavior
itself changed.

Baselines live at tests/baselines/*.json and are regenerated with:
    uv run python -m tests.generate_baselines

Each baseline records the *exact* parameters it was computed under. A run
must reproduce them exactly (the pipeline is deterministic) — any drift
signals a real behavioral change that needs review.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from neurodynamics.run import run

BASELINE_DIR = Path(__file__).parent / "baselines"


def _summarize(state_path: Path) -> dict:
    """Compute the stable statistics we track for regression.

    These capture the meaningful *shape* of the engine's output without
    being sensitive to noise-floor LSB wiggles that come and go with
    dependency updates.
    """
    t = pq.read_table(str(state_path))
    layer = np.array(t.column("layer").to_pylist())
    amp_col = t.column("amp").to_pylist()
    ph_col = t.column("phantom").to_pylist()
    out: dict = {}
    for which in ("rhythm", "pitch"):
        mask = layer == which
        amp = np.stack([np.asarray(a) for a, m in zip(amp_col, mask) if m])
        ph = np.stack([np.asarray(a) for a, m in zip(ph_col, mask) if m])
        f_key = f"layer.{which}.f".encode()
        freqs = np.array(
            [float(x) for x in t.schema.metadata[f_key].decode().split(",")]
        )
        per_osc_max = amp.max(axis=0)
        out[which] = {
            "n_snapshots": int(amp.shape[0]),
            "n_oscillators": int(amp.shape[1]),
            "amp_mean": float(amp.mean()),
            "amp_max": float(amp.max()),
            "amp_p50": float(np.percentile(amp, 50)),
            "amp_p95": float(np.percentile(amp, 95)),
            "phantom_fraction": float(ph.mean()),
            # Rank-order of most active oscillators — invariant to overall
            # amplitude scaling. Top 5 only, by index.
            "top5_osc_idx": [int(i) for i in np.argsort(per_osc_max)[-5:][::-1]],
            "top5_osc_freqs": [
                float(freqs[i]) for i in np.argsort(per_osc_max)[-5:][::-1]
            ],
        }
    return out


def _compare_summary(actual: dict, expected: dict, tol: dict) -> list[str]:
    """Return a list of human-readable mismatch messages (empty if all pass)."""
    problems = []
    for layer in ("rhythm", "pitch"):
        a = actual[layer]
        e = expected[layer]
        # Exact-match fields.
        for k in ("n_snapshots", "n_oscillators", "top5_osc_idx"):
            if a[k] != e[k]:
                problems.append(f"{layer}.{k}: expected {e[k]} got {a[k]}")
        # Tolerance-match fields.
        for k, absol in tol.items():
            if abs(a[k] - e[k]) > absol:
                problems.append(
                    f"{layer}.{k}: expected {e[k]:.6f} ± {absol}, "
                    f"got {a[k]:.6f} (diff {a[k]-e[k]:+.6f})"
                )
    return problems


@pytest.mark.regression
@pytest.mark.slow
class TestRegression:
    """Compare engine output against committed baseline stats."""

    def test_pulse_train_120bpm(
        self, pulse_train_120bpm, temp_audio_file, minimal_config, tmp_path,
    ):
        audio, sr = pulse_train_120bpm
        audio_file = temp_audio_file(audio, sr, "pulse.wav")
        cfg = minimal_config(str(audio_file.name))
        out = tmp_path / "state.parquet"
        run(cfg, audio_override=audio_file, output_override=out)

        actual = _summarize(out)
        baseline_path = BASELINE_DIR / "pulse_train_120bpm.json"
        if not baseline_path.exists():
            pytest.skip(
                f"No baseline at {baseline_path}. "
                "Run `uv run python -m tests.generate_baselines` to create it."
            )
        expected = json.loads(baseline_path.read_text())
        problems = _compare_summary(
            actual, expected,
            tol={"amp_mean": 1e-4, "amp_max": 1e-4,
                 "amp_p50": 1e-4, "amp_p95": 1e-4,
                 "phantom_fraction": 1e-3},
        )
        assert not problems, "\n".join(problems)

    def test_sine_440hz(
        self, sine_440hz, temp_audio_file, minimal_config, tmp_path,
    ):
        audio, sr = sine_440hz
        audio_file = temp_audio_file(audio, sr, "a4.wav")
        cfg = minimal_config(str(audio_file.name))
        out = tmp_path / "state.parquet"
        run(cfg, audio_override=audio_file, output_override=out)

        actual = _summarize(out)
        baseline_path = BASELINE_DIR / "sine_440hz.json"
        if not baseline_path.exists():
            pytest.skip(
                f"No baseline at {baseline_path}. "
                "Run `uv run python -m tests.generate_baselines` to create it."
            )
        expected = json.loads(baseline_path.read_text())
        problems = _compare_summary(
            actual, expected,
            tol={"amp_mean": 1e-4, "amp_max": 1e-4,
                 "amp_p50": 1e-4, "amp_p95": 1e-4,
                 "phantom_fraction": 1e-3},
        )
        assert not problems, "\n".join(problems)
