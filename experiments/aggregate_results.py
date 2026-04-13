"""将各 method 的 per-method 结果聚合为实验1/实验2 的汇总文件.

在所有 per-method 运行完成后执行一次:
    python -m experiments.aggregate_results

输出:
    results/data/exp1_episodes.csv
    results/data/exp1_steps.csv
    results/data/exp1_main_table.csv
    results/data/exp2_episodes.csv
    results/data/exp2_steps.csv
    results/data/exp2_comparison_table.csv
"""

from pathlib import Path

import pandas as pd

from experiments.runner import build_summary_table, load_method_results


EXP1_SET = ["M1-Fixed", "M2-Threshold", "M3-PPO",
            "M4-SingleLLM", "M5-PA", "M5-PC"]
EXP2_SET = ["M5-PA", "M5-PC", "M5-NoNeg"]

OUT_DIR = Path("results/data")


def _aggregate(method_set: list[str], out_prefix: str,
               summary_filename: str) -> None:
    eps_frames, step_frames, missing = [], [], []
    for m in method_set:
        try:
            df_ep, df_st = load_method_results(m)
        except FileNotFoundError:
            missing.append(m)
            continue
        eps_frames.append(df_ep)
        step_frames.append(df_st)

    if missing:
        print(f"  [WARN] {out_prefix}: missing per-method data for: {missing}")

    if not eps_frames:
        print(f"  [SKIP] {out_prefix}: no methods had per-method data, "
              f"nothing to aggregate.")
        return

    df_ep = pd.concat(eps_frames, ignore_index=True)
    df_st = pd.concat(step_frames, ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_ep.to_csv(OUT_DIR / f"{out_prefix}_episodes.csv", index=False)
    df_st.to_csv(OUT_DIR / f"{out_prefix}_steps.csv", index=False)
    build_summary_table(df_ep).to_csv(OUT_DIR / summary_filename)

    methods_present = sorted(df_ep["method"].unique().tolist())
    print(f"  [OK] {out_prefix}: aggregated {len(df_ep)} episodes "
          f"across {len(methods_present)} methods: {methods_present}")


def aggregate_all() -> None:
    print("=" * 60)
    print("Aggregating per-method results")
    print("=" * 60)
    _aggregate(EXP1_SET, "exp1", "exp1_main_table.csv")
    _aggregate(EXP2_SET, "exp2", "exp2_comparison_table.csv")
    print("\nAggregation complete. Files written to results/data/")


if __name__ == "__main__":
    aggregate_all()
