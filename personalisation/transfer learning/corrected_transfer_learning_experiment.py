import argparse
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

LABELS = [
    "Highly Disengage",
    "Disengage",
    "Engage",
    "Highly Engage",
]
FEATURE_COLS = [f"emb_{i}" for i in range(1024)]


def parse_group_from_clip_id(clip_id: str):
    match = re.match(r"^(video[^_]+)_(\d+)_((?:person).+)$", str(clip_id))
    if not match:
        return None
    return f"{match.group(1)}_{match.group(3)}"


def load_clip_dataframe(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for clip_id, meta in data.items():
        if not isinstance(meta, dict):
            continue
        label = meta.get("label")
        if label not in LABELS:
            continue
        embeds = meta.get("embeds")
        if not isinstance(embeds, list) or len(embeds) != 1024:
            continue
        group = parse_group_from_clip_id(clip_id)
        if group is None:
            continue
        arr = np.asarray(embeds, dtype=float)
        if not np.all(np.isfinite(arr)):
            continue

        clip_index_match = re.search(r"_(\d+)_", str(clip_id))
        clip_index = int(clip_index_match.group(1)) if clip_index_match else 0

        records.append(
            {
                "clip_id": str(clip_id),
                "group": group,
                "label": str(label),
                "clip_index": int(clip_index),
                "embedding": arr,
            }
        )

    if not records:
        raise ValueError(f"No valid clips were loaded from {json_path}")

    df = pd.DataFrame(records)
    emb_matrix = np.vstack(df["embedding"].to_numpy())
    emb_df = pd.DataFrame(emb_matrix, columns=FEATURE_COLS, index=df.index)
    df = pd.concat([df.drop(columns=["embedding"]), emb_df], axis=1)
    return df


def load_group_fold_map(fold_csv_path: str):
    fold_df = pd.read_csv(fold_csv_path)
    if "group" not in fold_df.columns or "fold" not in fold_df.columns:
        raise ValueError(f"Fold CSV must contain 'group' and 'fold' columns: {fold_csv_path}")

    group_fold = fold_df[["group", "fold"]].drop_duplicates()
    dupes = group_fold.groupby("group")["fold"].nunique()
    dupes = dupes[dupes != 1]
    if len(dupes) > 0:
        raise ValueError(f"Some groups map to multiple folds: {dupes.index.tolist()}")

    return dict(zip(group_fold["group"], group_fold["fold"].astype(int)))


def evaluate_predictions(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
    }
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )
    metrics["per_class_precision"] = {label: float(p) for label, p in zip(LABELS, precision)}
    metrics["per_class_recall"] = {label: float(r) for label, r in zip(LABELS, recall)}
    metrics["per_class_f1"] = {label: float(f) for label, f in zip(LABELS, f1)}
    return metrics


def fit_general_logreg(train_df: pd.DataFrame, seed: int):
    scaler = StandardScaler()
    X = train_df[FEATURE_COLS].to_numpy(dtype=float)
    y = train_df["label"].to_numpy(dtype=str)
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(X_scaled, y)
    return scaler, clf


def fit_general_mlp(train_df: pd.DataFrame, seed: int):
    scaler = StandardScaler()
    X = train_df[FEATURE_COLS].to_numpy(dtype=float)
    y = train_df["label"].to_numpy(dtype=str)
    X_scaled = scaler.fit_transform(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=False,
        random_state=seed,
    )
    clf.fit(X_scaled, y_encoded)
    return scaler, clf, label_encoder


def majority_label(train_df: pd.DataFrame):
    if train_df.empty:
        return None
    majority = train_df["label"].value_counts().idxmax()
    return majority


def sample_support_indices(student_df: pd.DataFrame, k_shot: int, seed: int, stratified: bool = False):
    student_df = student_df.sort_values("clip_index").copy()
    n = len(student_df)
    if k_shot <= 0:
        return [], list(student_df.index)
    if n <= k_shot:
        return list(student_df.index), []

    rng = np.random.default_rng(seed)
    labels = student_df["label"].unique().tolist()
    support_idx = []

    if stratified and len(labels) > 1 and k_shot >= len(labels):
        per_label = student_df["label"].value_counts().to_dict()
        for label in labels:
            label_idx = student_df.index[student_df["label"] == label].tolist()
            if len(label_idx) == 0:
                continue
            sampled = rng.choice(label_idx, size=min(1, len(label_idx)), replace=False).tolist()
            support_idx.extend(sampled)

        remaining = max(0, k_shot - len(support_idx))
        if remaining > 0:
            all_remaining = [idx for idx in student_df.index if idx not in support_idx]
            sampled = rng.choice(all_remaining, size=min(remaining, len(all_remaining)), replace=False).tolist()
            support_idx.extend(sampled)

        support_idx = list(dict.fromkeys(support_idx))
    else:
        all_idx = student_df.index.tolist()
        support_idx = rng.choice(all_idx, size=min(k_shot, len(all_idx)), replace=False).tolist()

    query_idx = [idx for idx in student_df.index.tolist() if idx not in support_idx]
    return list(support_idx), list(query_idx)


def make_result_row(method: str, outer_fold: int, student_group: str, k_shot: int, support_size: int, query_size: int, y_true, y_pred):
    metrics = evaluate_predictions(y_true, y_pred)
    row = {
        "method": method,
        "outer_fold": int(outer_fold),
        "student_group": student_group,
        "k_shot": int(k_shot),
        "support_size": int(support_size),
        "query_size": int(query_size),
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "confusion_matrix": str(metrics["confusion_matrix"]),
    }
    for label in LABELS:
        row[f"precision_{label.replace(' ', '_')}"] = metrics["per_class_precision"].get(label, 0.0)
        row[f"recall_{label.replace(' ', '_')}"] = metrics["per_class_recall"].get(label, 0.0)
        row[f"f1_{label.replace(' ', '_')}"] = metrics["per_class_f1"].get(label, 0.0)
    return row


def run_majority_baseline(clip_df: pd.DataFrame, group_fold: dict, seed: int):
    rows = []
    for fold_id in sorted(set(group_fold.values())):
        test_groups = [g for g, f in group_fold.items() if f == fold_id]
        train_groups = [g for g, f in group_fold.items() if f != fold_id]
        train_df = clip_df[clip_df["group"].isin(train_groups)].copy()
        majority = majority_label(train_df)
        if majority is None:
            continue

        for group in test_groups:
            student_df = clip_df[clip_df["group"] == group].copy()
            if student_df.empty:
                continue
            support_size = 0
            support_df = student_df.iloc[0:0].copy()
            query_df = student_df.copy()
            y_true = query_df["label"].to_numpy(dtype=str)
            y_pred = np.repeat(majority, len(query_df))
            rows.append(
                make_result_row(
                    method="majority",
                    outer_fold=fold_id,
                    student_group=group,
                    k_shot=0,
                    support_size=support_size,
                    query_size=len(query_df),
                    y_true=y_true,
                    y_pred=y_pred,
                )
            )
    return pd.DataFrame(rows)


def run_zero_shot(clip_df: pd.DataFrame, group_fold: dict, seed: int):
    rows = []
    for fold_id in sorted(set(group_fold.values())):
        test_groups = [g for g, f in group_fold.items() if f == fold_id]
        train_groups = [g for g, f in group_fold.items() if f != fold_id]
        train_df = clip_df[clip_df["group"].isin(train_groups)].copy()
        if train_df.empty:
            raise ValueError(f"No training clips for fold {fold_id}")

        scaler, clf = fit_general_logreg(train_df, seed=seed + fold_id)

        for group in test_groups:
            student_df = clip_df[clip_df["group"] == group].copy()
            if student_df.empty:
                continue
            X = scaler.transform(student_df[FEATURE_COLS].to_numpy(dtype=float))
            y_true = student_df["label"].to_numpy(dtype=str)
            y_pred = clf.predict(X)
            rows.append(
                make_result_row(
                    method="zero_shot",
                    outer_fold=fold_id,
                    student_group=group,
                    k_shot=0,
                    support_size=0,
                    query_size=len(student_df),
                    y_true=y_true,
                    y_pred=y_pred,
                )
            )
    return pd.DataFrame(rows)


def run_few_shot(clip_df: pd.DataFrame, group_fold: dict, k_values, seed: int, stratified: bool = False):
    rows = []
    for fold_id in sorted(set(group_fold.values())):
        test_groups = [g for g, f in group_fold.items() if f == fold_id]
        train_groups = [g for g, f in group_fold.items() if f != fold_id]
        train_df = clip_df[clip_df["group"].isin(train_groups)].copy()
        scaler, clf = fit_general_logreg(train_df, seed=seed + fold_id)

        for group in test_groups:
            student_df = clip_df[clip_df["group"] == group].copy()
            if student_df.empty:
                continue
            for k in k_values:
                support_idx, query_idx = sample_support_indices(student_df, k, seed=seed + fold_id + sum(ord(ch) for ch in group), stratified=stratified)
                if len(query_idx) == 0:
                    continue
                support_df = student_df.loc[support_idx].copy()
                query_df = student_df.loc[query_idx].copy()
                if support_df.empty or query_df.empty:
                    continue

                if len(np.unique(support_df["label"])) < 2 and k == 1:
                    # Prototype adaptation on a single labeled support point.
                    support_label = support_df["label"].iloc[0]
                    support_vector = scaler.transform(support_df[FEATURE_COLS].iloc[[0]].to_numpy(dtype=float))[0]
                    train_features = scaler.transform(train_df[FEATURE_COLS].to_numpy(dtype=float))
                    class_centroids = {}
                    for label in LABELS:
                        label_mask = train_df["label"] == label
                        if np.any(label_mask):
                            class_centroids[label] = train_features[label_mask].mean(axis=0)
                        else:
                            class_centroids[label] = np.zeros(train_features.shape[1], dtype=float)

                    adapted_centroids = dict(class_centroids)
                    adapted_centroids[support_label] = 0.5 * class_centroids.get(support_label, support_vector) + 0.5 * support_vector
                    query_X = scaler.transform(query_df[FEATURE_COLS].to_numpy(dtype=float))
                    distances = np.stack([
                        np.linalg.norm(query_X - adapted_centroids[label], axis=1)
                        for label in LABELS
                    ], axis=1)
                    y_pred = np.array([LABELS[int(np.argmin(d))] for d in distances])
                else:
                    X_support = scaler.transform(support_df[FEATURE_COLS].to_numpy(dtype=float))
                    y_support = support_df["label"].to_numpy(dtype=str)
                    if len(np.unique(y_support)) < 2:
                        continue
                    adapted_model = LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=seed + fold_id,
                    )
                    adapted_model.fit(X_support, y_support)
                    X_query = scaler.transform(query_df[FEATURE_COLS].to_numpy(dtype=float))
                    y_pred = adapted_model.predict(X_query)

                y_true = query_df["label"].to_numpy(dtype=str)
                rows.append(
                    make_result_row(
                        method=f"few_shot_random_k{k}" if not stratified else f"few_shot_stratified_k{k}",
                        outer_fold=fold_id,
                        student_group=group,
                        k_shot=k,
                        support_size=len(support_df),
                        query_size=len(query_df),
                        y_true=y_true,
                        y_pred=y_pred,
                    )
                )
    return pd.DataFrame(rows)


def extract_hidden_features(clf: MLPClassifier, X: np.ndarray):
    if len(clf.coefs_) == 0:
        return X
    hidden = X
    for i in range(len(clf.coefs_) - 1):
        hidden = hidden @ clf.coefs_[i] + clf.intercepts_[i]
        hidden = np.maximum(hidden, 0.0)
    return hidden


def run_fine_tuning(clip_df: pd.DataFrame, group_fold: dict, k_values, seed: int):
    rows = []
    for fold_id in sorted(set(group_fold.values())):
        test_groups = [g for g, f in group_fold.items() if f == fold_id]
        train_groups = [g for g, f in group_fold.items() if f != fold_id]
        train_df = clip_df[clip_df["group"].isin(train_groups)].copy()
        scaler, clf, label_encoder = fit_general_mlp(train_df, seed=seed + fold_id)

        for group in test_groups:
            student_df = clip_df[clip_df["group"] == group].copy()
            if student_df.empty:
                continue
            for k in k_values:
                support_idx, query_idx = sample_support_indices(student_df, k, seed=seed + fold_id + sum(ord(ch) for ch in group), stratified=False)
                if len(query_idx) == 0:
                    continue
                support_df = student_df.loc[support_idx].copy()
                query_df = student_df.loc[query_idx].copy()
                if support_df.empty or query_df.empty:
                    continue

                X_support = scaler.transform(support_df[FEATURE_COLS].to_numpy(dtype=float))
                y_support = support_df["label"].to_numpy(dtype=str)
                if len(np.unique(y_support)) < 2:
                    continue

                feature_extractor = clf
                support_hidden = extract_hidden_features(feature_extractor, X_support)
                support_y_encoded = label_encoder.transform(y_support)
                head = LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=seed + fold_id,
                )
                head.fit(support_hidden, support_y_encoded)

                X_query = scaler.transform(query_df[FEATURE_COLS].to_numpy(dtype=float))
                query_hidden = extract_hidden_features(feature_extractor, X_query)
                y_pred = label_encoder.inverse_transform(head.predict(query_hidden))
                y_true = query_df["label"].to_numpy(dtype=str)
                rows.append(
                    make_result_row(
                        method=f"fine_tuned_mlp_k{k}",
                        outer_fold=fold_id,
                        student_group=group,
                        k_shot=k,
                        support_size=len(support_df),
                        query_size=len(query_df),
                        y_true=y_true,
                        y_pred=y_pred,
                    )
                )
    return pd.DataFrame(rows)


def build_summary(results_df: pd.DataFrame):
    if results_df.empty:
        return pd.DataFrame(columns=["method", "n_students", "mean_accuracy", "std_accuracy", "mean_balanced_accuracy", "std_balanced_accuracy", "mean_macro_f1", "std_macro_f1", "total_query_clips"])

    summary = (
        results_df.groupby("method", as_index=False)
        .agg(
            n_students=("student_group", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            total_query_clips=("query_size", "sum"),
        )
        .sort_values("method")
        .reset_index(drop=True)
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Corrected unseen-student transfer-learning evaluation for CMOSE engagement prediction.")
    parser.add_argument("--json", type=str, default=os.path.join(os.environ.get("CMOSE_DATA_DIR", "."), "final_data_1.json"))
    parser.add_argument("--folds", type=str, default=os.path.join(os.environ.get("CMOSE_DATA_DIR", "."), "cmose_h1_h2_fold_protocol.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-values", type=str, default="1,5,10,20")
    parser.add_argument("--output-dir", type=str, default=os.environ.get("CMOSE_OUTPUT_DIR", "."))
    args = parser.parse_args()

    clip_df = load_clip_dataframe(args.json)
    group_fold = load_group_fold_map(args.folds)
    k_values = [int(v.strip()) for v in args.k_values.split(",") if v.strip()]

    majority_df = run_majority_baseline(clip_df, group_fold, args.seed)
    zero_shot_df = run_zero_shot(clip_df, group_fold, args.seed)
    few_shot_df = run_few_shot(clip_df, group_fold, k_values, args.seed, stratified=False)
    fine_tune_df = run_fine_tuning(clip_df, group_fold, k_values, args.seed)

    all_results = pd.concat([majority_df, zero_shot_df, few_shot_df, fine_tune_df], ignore_index=True, sort=False)
    summary_df = build_summary(all_results)

    majority_path = f"{args.output_dir}\\transfer_learning_majority_results.csv"
    zero_shot_path = f"{args.output_dir}\\transfer_learning_zero_shot_results.csv"
    few_shot_path = f"{args.output_dir}\\transfer_learning_few_shot_results.csv"
    fine_tune_path = f"{args.output_dir}\\transfer_learning_fine_tuning_results.csv"
    all_path = f"{args.output_dir}\\transfer_learning_all_methods_results.csv"
    summary_path = f"{args.output_dir}\\transfer_learning_summary.csv"

    majority_df.to_csv(majority_path, index=False)
    zero_shot_df.to_csv(zero_shot_path, index=False)
    few_shot_df.to_csv(few_shot_path, index=False)
    fine_tune_df.to_csv(fine_tune_path, index=False)
    all_results.to_csv(all_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Majority results: {len(majority_df)} rows -> {majority_path}")
    print(f"Zero-shot results: {len(zero_shot_df)} rows -> {zero_shot_path}")
    print(f"Few-shot results: {len(few_shot_df)} rows -> {few_shot_path}")
    print(f"Fine-tuning results: {len(fine_tune_df)} rows -> {fine_tune_path}")
    print(f"Aggregate summary: {len(summary_df)} rows -> {summary_path}")

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
