"""Reproduce the repeated Regular vs Stratified 5-fold CV benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr, ttest_rel, wilcoxon
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATASETS = {
    "Blood Transfusion": 1464,
    "Adult": 1590,
    "QSAR Biodegradation": 1494,
    "Spambase": 44,
    "Iris": 61,
    "German Credit": 31,
    "KC1": 1067,
    "PC1": 1068,
    "ILPD": 1480,
    "Diabetes": 37,
    "Credit Approval": 29,
    "Mammography": 310,
    "Phoneme": 1489,
}
BASE_SEED = 235975
N_REPEATS = 10
N_SPLITS = 5


def preprocessing() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, make_column_selector(dtype_include=np.number)),
        ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
    ])


def models(seed: int) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("preprocess", preprocessing()),
            ("model", LogisticRegression(solver="lbfgs", max_iter=2000)),
        ]),
        "Decision Tree": Pipeline([
            ("preprocess", preprocessing()),
            ("model", DecisionTreeClassifier(random_state=seed)),
        ]),
    }


def run_task(name: str, data_id: int, X: pd.DataFrame, y: pd.Series,
             repeat: int, protocol: str) -> tuple[list[dict], dict]:
    seed = BASE_SEED + repeat
    splitter = (
        KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        if protocol == "Regular"
        else StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    )
    rows, fold_shares = [], []
    for fold, (train, test) in enumerate(splitter.split(X, y)):
        counts = y.iloc[test].value_counts(normalize=True)
        fold_shares.append(float(counts.min()))
        for algorithm, estimator in models(seed).items():
            started = time.perf_counter()
            estimator.fit(X.iloc[train], y.iloc[train])
            pred = estimator.predict(X.iloc[test])
            elapsed = time.perf_counter() - started
            rows.append({
                "dataset": name, "openml_id": data_id, "repeat": repeat,
                "seed": seed, "protocol": protocol, "fold": fold,
                "algorithm": algorithm,
                "macro_f1": f1_score(y.iloc[test], pred, average="macro", zero_division=0),
                "fit_score_seconds": elapsed,
            })
    composition = {
        "dataset": name, "repeat": repeat, "protocol": protocol,
        "minority_share_sd_across_folds": float(np.std(fold_shares, ddof=1)),
    }
    return rows, composition


def paired_test(a: pd.Series, b: pd.Series) -> dict[str, float]:
    diff = b - a
    w = wilcoxon(diff, alternative="two-sided")
    t = ttest_rel(b, a)
    return {
        "mean_effect": float(diff.mean()), "median_effect": float(diff.median()),
        "wilcoxon_statistic": float(w.statistic), "wilcoxon_p": float(w.pvalue),
        "paired_t_statistic": float(t.statistic), "paired_t_p": float(t.pvalue),
    }


def main(output: Path, jobs: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    loaded, metadata = {}, []
    for name, data_id in DATASETS.items():
        bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
        X, y = bunch.data, bunch.target.astype(str)
        loaded[name] = (X, y)
        shares = y.value_counts(normalize=True)
        metadata.append({
            "dataset": name, "openml_id": data_id, "n_samples": len(X),
            "n_features": X.shape[1], "n_classes": y.nunique(),
            "minority_share": float(shares.min()),
        })

    tasks = [
        (name, DATASETS[name], *loaded[name], repeat, protocol)
        for name in DATASETS for repeat in range(N_REPEATS)
        for protocol in ("Regular", "Stratified")
    ]
    wall_start = time.perf_counter()
    completed = Parallel(n_jobs=jobs, verbose=10)(delayed(run_task)(*task) for task in tasks)
    wall_seconds = time.perf_counter() - wall_start
    rows = [row for task_rows, _ in completed for row in task_rows]
    composition = [item for _, item in completed]
    folds = pd.DataFrame(rows)
    comp = pd.DataFrame(composition)
    meta = pd.DataFrame(metadata)

    aggregate = (folds.groupby(["dataset", "protocol", "algorithm"], as_index=False)
                 .agg(mean_macro_f1=("macro_f1", "mean"), sd_fold_macro_f1=("macro_f1", "std")))
    repeat_means = (folds.groupby(["dataset", "protocol", "algorithm", "repeat"], as_index=False)
                    .agg(mean_macro_f1=("macro_f1", "mean")))
    repeat_sd = (repeat_means.groupby(["dataset", "protocol", "algorithm"], as_index=False)
                 .agg(repeat_mean_sd=("mean_macro_f1", "std")))

    wide = aggregate.pivot(index=["dataset", "algorithm"], columns="protocol", values="mean_macro_f1")
    tests = {}
    for algorithm in ("Logistic Regression", "Decision Tree"):
        sub = wide.xs(algorithm, level="algorithm")
        tests[f"score_{algorithm}"] = paired_test(sub["Regular"], sub["Stratified"])
    var_wide = repeat_sd.pivot(index=["dataset", "algorithm"], columns="protocol", values="repeat_mean_sd")
    tests["repeat_mean_sd"] = paired_test(var_wide["Regular"], var_wide["Stratified"])

    gap = aggregate.pivot(index="dataset", columns=["protocol", "algorithm"], values="mean_macro_f1")
    regular_gap = gap[("Regular", "Logistic Regression")] - gap[("Regular", "Decision Tree")]
    stratified_gap = gap[("Stratified", "Logistic Regression")] - gap[("Stratified", "Decision Tree")]
    gap_table = pd.DataFrame({"regular_gap": regular_gap, "stratified_gap": stratified_gap})
    gap_table["delta_gap"] = gap_table["stratified_gap"] - gap_table["regular_gap"]
    gap_table["ranking_flip"] = np.sign(gap_table["regular_gap"]) != np.sign(gap_table["stratified_gap"])
    tests["algorithm_gap"] = paired_test(gap_table["regular_gap"], gap_table["stratified_gap"])

    score_effect = wide["Stratified"] - wide["Regular"]
    avg_effect = score_effect.groupby(level="dataset").mean()
    minority = meta.set_index("dataset")["minority_share"].reindex(avg_effect.index)
    rho = spearmanr(minority, avg_effect)
    tests["imbalance_spearman"] = {"rho": float(rho.statistic), "p": float(rho.pvalue)}

    folds.to_csv(output / "fold_results.csv", index=False)
    comp.to_csv(output / "fold_composition.csv", index=False)
    meta.to_csv(output / "dataset_metadata.csv", index=False)
    aggregate.to_csv(output / "aggregate_scores.csv", index=False)
    repeat_sd.to_csv(output / "repeat_score_sd.csv", index=False)
    gap_table.to_csv(output / "algorithm_gaps.csv")
    (output / "statistical_tests.json").write_text(json.dumps(tests, indent=2), encoding="utf-8")
    compute = {
        "wall_seconds": wall_seconds,
        "summed_fit_score_seconds": float(folds["fit_score_seconds"].sum()),
        "parallel_workers": jobs,
        "logical_cpus": int(os.cpu_count() or 1),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "scikit_learn": sklearn_version,
        "total_fits": len(folds),
    }
    (output / "compute_summary.json").write_text(json.dumps(compute, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    main(args.output, args.jobs)
