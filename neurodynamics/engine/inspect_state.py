"""Quick read-out of the state.parquet file to verify it looks sensible."""
import numpy as np
import pyarrow.parquet as pq

t = pq.read_table("state.parquet")
print(f"rows: {t.num_rows}")
print(f"cols: {t.column_names}")

layer = np.array(t.column("layer").to_pylist())
times = np.array(t.column("t").to_pylist())
print(f"time range: {times.min():.2f} – {times.max():.2f} s")
print(f"rhythm rows: {(layer == 'rhythm').sum()}")
print(f"pitch rows:  {(layer == 'pitch').sum()}")


def summarize(which: str):
    mask = layer == which
    amp = np.stack([np.asarray(a) for a, m in zip(t.column("amp").to_pylist(), mask) if m])
    phantom = np.stack([np.asarray(a) for a, m in zip(t.column("phantom").to_pylist(), mask) if m])
    print(f"\n--- {which} ---")
    print(f"shape: {amp.shape}  (snapshots x oscillators)")
    print(f"amp: min {amp.min():.4f}  mean {amp.mean():.4f}  max {amp.max():.4f}")
    # Per-oscillator activity: how active was each oscillator across the run?
    per_osc_max = amp.max(axis=0)
    top = np.argsort(per_osc_max)[-5:][::-1]
    f_key = f"layer.{which}.f".encode()
    freqs = np.array([float(x) for x in t.schema.metadata[f_key].decode().split(",")])
    print("top-5 most active oscillators:")
    for i in top:
        print(f"  osc[{i:3d}] f={freqs[i]:8.2f} Hz  max|z|={per_osc_max[i]:.3f}")
    phantom_fraction = phantom.mean()
    print(f"phantom-active cells: {phantom_fraction:.2%} of (time × osc)")


summarize("rhythm")
summarize("pitch")
