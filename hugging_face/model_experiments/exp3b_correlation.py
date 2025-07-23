"""
Experiment 3B – model-vs-human correlation on object-change detection.

* Takes BEFORE / AFTER image pairs (same file-name stem + '_init', '_out').
* Runs ChangeDetectionExperiment once to get the best mask for each image.
* Computes |area_after – area_before| / area_before for every pair.
* Sweeps %-change thresholds from 0.10 … 0.90 (default) and classifies
  each pair as “different” (1) or “same” (0).
* Correlates those binary model decisions with mean human judgements.
* Saves JSON results, .txt summary, and three PNG plots.

Usage
-----
python exp3b_correlation.py \
    --images_dir  /path/to/exp3b_imgs \
    --human_csv   /path/to/human_data.csv \
    --output_dir  /tmp/exp3b_out \
    --model_name  nvidia/segformer-b0-finetuned-ade-512-512 \
    --thresholds  0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90 \
    --resume
"""

## remember to activate the virtual environment before running the script
## source venv_exp3b/bin/activate
## remember to change the paths in the script to the correct ones


from __future__ import annotations
import argparse, json, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import torch  # only to enforce no-grad

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from exp3Change import ChangeDetectionExperiment
from segformer.segformer_interface import SegFormerInterface


# helper functions
# ----------------------------------------------------------------------
def load_human_data(csv_path: Path) -> Dict[str, float]:
    """Return {fullShapeName: mean_diff_response (0‒1)}."""
    df = (pd.read_csv(csv_path)
            .query("~shape.str.contains('catch_shape', na=False)", engine="python")
            .query("response in ['same','different']"))
    df["binary"] = (df["response"] == "different").astype(float)
    means = df.groupby("fullShapeName")["binary"].mean()
    return {shape_name: mean for shape_name, mean in means.items()}


def collect_area_ratios(cde: ChangeDetectionExperiment) -> Dict[str, float]:
    """
    Read CDE's `threshold_results_dir/*/per_image_detailed.json` and return
    {base_stem: area_change_ratio}.
    We look only at the *first* results file because the ratio is
    threshold-independent.
    """
    res_files = list(Path(cde.threshold_results_dir).rglob("per_image_detailed.json"))
    if not res_files:
        raise FileNotFoundError(
            f"No per_image_detailed.json found under {cde.threshold_results_dir}. "
            "Did exp3Change run successfully?")
    data = json.loads(Path(res_files[0]).read_text())
    return {d["base"]: d["area_change"] for d in data}


def decide_different(area_ratio: float, thr: float) -> int:
    """Return 1 if |Δarea|/area_before > thr else 0."""
    return int(area_ratio > thr)


def correlations(model: List[int], human: List[float]) -> Tuple[float, float]:
    """Pearson r, Spearman ρ. Return (np.nan, np.nan) if <2 samples."""
    if len(model) < 2 or len(set(model)) == 1:
        return float("nan"), float("nan")
    return pearsonr(model, human)[0], spearmanr(model, human)[0]


# main function
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=Path, default="/Users/andreaprocopio/Desktop/object_reps_neural/hugging_face/model_experiments/Exp3b_Images")
    parser.add_argument("--human_csv", type=Path, default="/Users/andreaprocopio/Desktop/object_reps_neural/detr/EXP_3_CHANGE/Data_processed/Data/exp3b_data.csv")
    parser.add_argument("--output_dir", type=Path, default="/Users/andreaprocopio/Desktop/object_reps_neural/hugging_face/model_experiments/exp3b_results")
    parser.add_argument("--model_name", default=None,
                        help="Hugging-Face checkpoint for SegFormer")
    parser.add_argument("--thresholds", default="0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36,0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70,0.72,0.74,0.76,0.78,0.80,0.82,0.84,0.86,0.88,0.90, 0.92, 0.94, 0.96, 0.98")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    ## ensures that when we don't overwrite results when we run the script multiple times
    ## with different models, in the output we'll get a new folder named after each model
    model_tag = (args.model_name or "default_model").replace("/", "_")
    args.output_dir = args.output_dir / model_tag

    thr_values = [round(float(x), 3) for x in args.thresholds.split(",")]

    # run segmentation once (or resume)
    torch.set_grad_enabled(False)
    model_if = SegFormerInterface(model_name=args.model_name)
    cde = ChangeDetectionExperiment(model_interface=model_if,
                                    output_dir=str(args.output_dir / "cde"))
    cde.run_full_experiment(images_dir=str(args.images_dir), resume=args.resume)

    # area-ratio for each pair
    area_ratio = collect_area_ratios(cde)                 # {base: ratio}

    # human averages
    human_mean = load_human_data(args.human_csv)          # {shape_type: mean}

    # threshold sweep & correlation
    corr_table = []
    for thr in thr_values:
        model_dec = []
        human_dec = []
        for base, ratio in area_ratio.items():
            # base name should match exactly with human data
            if base not in human_mean:
                continue
            model_dec.append(decide_different(ratio, thr))
            human_dec.append(human_mean[base])
        r_p, r_s = correlations(model_dec, human_dec)
        corr_table.append(dict(threshold=thr,
                               pearson=r_p, spearman=r_s,
                               n=len(model_dec)))
        print(f"thr={thr:.2f} : r_P={r_p:.3f} r_S={r_s:.3f}  (n={len(model_dec)})")

    # save JSON
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "correlation_results.json").write_text(json.dumps(corr_table, indent=2))

    # TXT summary
    best_row = max(corr_table, key=lambda d: d["pearson"])
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    txt = (f"Exp 3B summary ({ts})\n"
           f"max Pearson = {best_row['pearson']:.3f} at thr {best_row['threshold']:.2f} "
           f"(Spearman {best_row['spearman']:.3f}, n={best_row['n']})\n")
    (out / "summary.txt").write_text(txt)
    print(txt)

    ## plot correlation vs threshold
    thr_arr = [d["threshold"] for d in corr_table]
    pearson_arr = [d["pearson"] for d in corr_table]
    spearman_arr = [d["spearman"] for d in corr_table]

    plt.figure(figsize=(9, 6))
    plt.plot(thr_arr, pearson_arr, "o-", label="Pearson r")
    plt.plot(thr_arr, spearman_arr, "s--", label="Spearman ρ")
    best_thr = best_row["threshold"]
    plt.axvline(best_thr, ls=":", color="k",
                label=f"max r at {best_thr:.3f}")
    plt.xlabel("Δ area threshold")
    plt.ylabel("Correlation with humans")
    plt.title("Model-human correlation vs. threshold")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "correlation_vs_threshold.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
