import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

LABELS = [
    "Highly Disengage",
    "Disengage",
    "Engage",
    "Highly Engage",
]


def parse_group_from_clip_id(clip_id: str):
    match = re.match(r"^(video[^_]+)_(\d+)_((?:person).+)$", clip_id)
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

        group = parse_group_from_clip_id(str(clip_id))
        if group is None:
            continue

        arr = np.asarray(embeds, dtype=float)
        if not np.all(np.isfinite(arr)):
            continue

        records.append({
            "clip_id": str(clip_id),
            "group": group,
            "label": str(label),
            "clip_index": int(re.search(r"_(\d+)_", str(clip_id)).group(1)),
            "embedding": arr,
        })

    if not records:
        raise ValueError(f"No valid clips were loaded from {json_path}")

    df = pd.DataFrame(records)
    emb_matrix = np.vstack(df["embedding"].to_numpy())
    emb_cols = [f"emb_{i}" for i in range(1024)]
    emb_df = pd.DataFrame(emb_matrix, columns=emb_cols, index=df.index)
    df = pd.concat([df.drop(columns=["embedding"]), emb_df], axis=1)
    return df


def load_fold_map(fold_csv_path: str):
    fold_df = pd.read_csv(fold_csv_path)
    required = {"group", "fold"}
    missing = required - set(fold_df.columns)
    if missing:
        raise ValueError(f"Fold CSV missing columns: {sorted(missing)}")

    group_fold = fold_df[["group", "fold"]].drop_duplicates()
    bad = group_fold.groupby("group")["fold"].nunique()
    bad = bad[bad != 1]
    if len(bad) > 0:
        raise ValueError(f"Some groups mapped to multiple folds: {bad.index.tolist()}")

    return dict(zip(group_fold["group"], group_fold["fold"].astype(int)))


def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
    }


def train_seen_student_model(train_df: pd.DataFrame):
    feature_cols = [f"emb_{i}" for i in range(1024)]
    X = train_df[feature_cols].to_numpy(dtype=float)
    y = train_df["label"].to_numpy(dtype=str)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        multi_class="auto",
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_scaled, y)
    return scaler, clf


def sample_support_query(student_df: pd.DataFrame, k_shot: int):
    student_df = student_df.sort_values("clip_index").copy()
    total_n = len(student_df)
    if total_n <= k_shot:
        return student_df.copy(), pd.DataFrame(columns=student_df.columns)

    labels = student_df["label"].unique().tolist()
    if len(labels) < 2:
        return None, student_df.copy()

    label_counts = student_df["label"].value_counts().to_dict()
    support_counts = {label: 0 for label in labels}
    remaining = k_shot

    for label in labels:
        if remaining <= 0:
            break
        count = min(1, label_counts[label])
        support_counts[label] = count
        remaining -= count

    # Distribute remaining shots as evenly as possible across labels with available rows.
    while remaining > 0:
        for label in labels:
            if remaining <= 0:
                break
            if support_counts[label] < label_counts[label]:
                support_counts[label] += 1
                remaining -= 1

    support_idx = []
    for label in labels:
        label_rows = student_df[student_df["label"] == label].index.tolist()
        selected = label_rows[:support_counts[label]]
        support_idx.extend(selected)

    support = student_df.loc[support_idx].copy()
    query = student_df.loc[~student_df.index.isin(support_idx)].copy()
    if support.empty or len(support["label"].unique()) < 2:
        return None, query
    return support, query


def evaluate_student(student_df: pd.DataFrame, scaler: StandardScaler, pretrained_model, k_shot: int):
    support, query = sample_support_query(student_df, k_shot)
    if support is None or query.empty:
        return None

    feature_cols = [f"emb_{i}" for i in range(1024)]

    X_query = scaler.transform(query[feature_cols].to_numpy(dtype=float))
    y_query = query["label"].to_numpy(dtype=str)

    zero_shot_pred = pretrained_model.predict(X_query)
    zero_shot = evaluate_predictions(y_query, zero_shot_pred)

    X_support = scaler.transform(support[feature_cols].to_numpy(dtype=float))
    y_support = support["label"].to_numpy(dtype=str)

    if len(np.unique(y_support)) < 2:
        return None

    adapted_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        multi_class="auto",
        solver="lbfgs",
        random_state=42,
    )
    adapted_model.fit(X_support, y_support)

    y_pred = adapted_model.predict(X_query)
    adapted = evaluate_predictions(y_query, y_pred)

    adapted["zero_shot_accuracy"] = zero_shot["accuracy"]
    adapted["zero_shot_balanced_accuracy"] = zero_shot["balanced_accuracy"]
    adapted["zero_shot_macro_f1"] = zero_shot["macro_f1"]
    adapted["student_group"] = student_df["group"].iloc[0]
    adapted["support_size"] = len(support)
    adapted["query_size"] = len(query)
    return adapted


def summarize_results(results):
    if len(results) == 0:
        return {}

    df = pd.DataFrame(results)
    return {
        "n_students": int(len(df)),
        "mean_accuracy": float(df["accuracy"].mean()),
        "mean_balanced_accuracy": float(df["balanced_accuracy"].mean()),
        "mean_macro_f1": float(df["macro_f1"].mean()),
        "mean_zero_shot_accuracy": float(df["zero_shot_accuracy"].mean()),
        "mean_zero_shot_balanced_accuracy": float(df["zero_shot_balanced_accuracy"].mean()),
        "mean_zero_shot_macro_f1": float(df["zero_shot_macro_f1"].mean()),
        "std_accuracy": float(df["accuracy"].std(ddof=1)) if len(df) > 1 else 0.0,
        "std_macro_f1": float(df["macro_f1"].std(ddof=1)) if len(df) > 1 else 0.0,
    }


def run_experiment(json_path: str, fold_csv_path: str, k_shot: int, seed: int = 42):
    clip_df = load_clip_dataframe(json_path)
    group_fold = load_fold_map(fold_csv_path)

    rows = []
    for fold_id in sorted(set(group_fold.values())):
        test_groups = [g for g, f in group_fold.items() if f == fold_id]
        train_groups = [g for g, f in group_fold.items() if f != fold_id]

        train_df = clip_df[clip_df["group"].isin(train_groups)].copy()
        if train_df.empty:
            raise ValueError(f"No training clips for fold {fold_id}")

        scaler, pretrained_model = train_seen_student_model(train_df)

        for group in test_groups:
            student_df = clip_df[clip_df["group"] == group].copy()
            if student_df.empty:
                continue

            result = evaluate_student(student_df, scaler, pretrained_model, k_shot)
            if result is None:
                continue

            result["fold"] = int(fold_id)
            result["k_shot"] = int(k_shot)
            rows.append(result)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Transfer-learning baseline for unseen-student CMOSE engagement prediction.")
    parser.add_argument("--json", type=str, default=os.path.join(os.environ.get("CMOSE_DATA_DIR", "."), "final_data_1.json"))
    parser.add_argument("--folds", type=str, default=os.path.join(os.environ.get("CMOSE_DATA_DIR", "."), "cmose_h1_h2_fold_protocol.csv"))
    parser.add_argument("--k-shot", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=os.path.join(os.environ.get("CMOSE_OUTPUT_DIR", "."), "transfer_learning_baseline_results.csv"))
    args = parser.parse_args()

    result_df = run_experiment(args.json, args.folds, args.k_shot, args.seed)
    if result_df.empty:
        raise ValueError("No evaluation results were produced for the unseen-student transfer baseline.")

    print(summarize_results(result_df.to_dict("records")))
    result_df.to_csv(args.output, index=False)
    print(f"Saved transfer baseline results to {args.output}")


if __name__ == "__main__":
    main()
