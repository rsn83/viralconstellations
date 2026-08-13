"""
Run Check C (graph persistence) on real data.

Usage:
  python scripts/12_run_check_C_real.py --config configs/colab_2022_test.yaml
"""
import sys, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
args = parser.parse_args()

import yaml, numpy as np
from checkA_and_C_graph_tests import graph_persistence_check_residualized

cfg = yaml.safe_load(open(ROOT / args.config))
matrix_dir = ROOT / cfg["paths"]["matrix_dir"]

months = sorted(p.stem for p in matrix_dir.glob("*.npy") if "_posfreq" not in p.stem)
print(f"Loading {len(months)} months: {months[0]} -> {months[-1]}")
all_mats = {m: np.load(matrix_dir / f"{m}.npy") for m in months}

result_df = graph_persistence_check_residualized(all_mats, horizons=[1, 3, 6])

out_dir = ROOT / "outputs" / "checks"
out_dir.mkdir(parents=True, exist_ok=True)
result_df.to_csv(out_dir / "checkC_graph_persistence_residualized.csv", index=False)
print(f"\nWrote: {out_dir / 'checkC_graph_persistence_residualized.csv'}")
