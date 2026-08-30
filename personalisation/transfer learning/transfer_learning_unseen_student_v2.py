import argparse
import copy
import json
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

LABELS = [
    "Highly Disengage",
    "Disengage",
    "Engage",
    "Highly Engage",
]
LABEL_TO_INT = {label: idx for idx, label in enumerate(LABELS)}
INT_TO_LABEL = {idx: label for label, idx in LABEL_TO_INT.items()}
FEATURE_COLS = [f"emb_{i}" for i in range(1024)]


def parse_group_from_clip_id(clip_id: str) -> Optional[str]:
    match = re.match(r"^(video[^_]+)_(\d+)_((?:person).+)$", str(clip_id))
    if not match:
        return None
    return f"{match.group(1)}_{match.group(3)}"


def load_clip_dataframe(json_path: str) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
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
        emb = np.asarray(embeds, dtype=float)
        if not np.all(np.isfinite(emb)):
            continue
        clip_index_match = re.search(r"_(\d+)_", str(clip_id))
        clip_index = int(clip_index_match.group(1)) if clip_index_match else 0
        rows.append(
            {
                "clip_id": str(clip_id),
                "group": group,
                "label": str(label),
                "clip_index": clip_index,
                "split": meta.get("split"),
                "embedding": emb,
            }
        )

    if not rows:
        raise ValueError(f"No valid labeled clips found in {json_path}")

    df = pd.DataFrame(rows)
    emb_matrix = np.vstack(df["embedding"].to_numpy())
    emb_df = pd.DataFrame(emb_matrix, columns=FEATURE_COLS, index=df.index)
    return pd.concat([df.drop(columns=["embedding"]), emb_df], axis=1)


def load_group_fold_map(fold_csv_path: str) -> Dict[str, int]:
    fold_df = pd.read_csv(fold_csv_path)
    required = {"group", "fold"}
    missing = required - set(fold_df.columns)
    if missing:
        raise ValueError(f"Fold CSV missing required columns: {sorted(missing)}")

    group_fold = fold_df[["group", "fold"]].drop_duplicates()
    fold_counts = group_fold.groupby("group")["fold"].nunique()
    bad = fold_counts[fold_counts != 1]
    if len(bad) > 0:
        raise ValueError(f"Some groups map to multiple folds: {bad.index.tolist()}")
    return dict(zip(group_fold["group"], group_fold["fold"].astype(int)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
        "per_class_precision": {label: float(v) for label, v in zip(LABELS, precision)},
        "per_class_recall": {label: float(v) for label, v in zip(LABELS, recall)},
        "per_class_f1": {label: float(v) for label, v in zip(LABELS, f1)},
        "per_class_support": {label: int(v) for label, v in zip(LABELS, support)},
    }


def make_per_student_row(
    *,
    method: str,
    prediction_type: str,
    outer_fold: int,
    student_group: str,
    k_shot: int,
    support_protocol: str,
    support_size: int,
    query_size: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, object]:
    metrics = compute_metrics(y_true, y_pred)
    row = {
        "method": method,
        "prediction_type": prediction_type,
        "outer_fold": int(outer_fold),
        "student_group": student_group,
        "k_shot": int(k_shot),
        "support_protocol": support_protocol,
        "support_size": int(support_size),
        "query_size": int(query_size),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "confusion_matrix": str(metrics["confusion_matrix"]),
    }
    for label in LABELS:
        safe = label.replace(" ", "_")
        row[f"precision_{safe}"] = metrics["per_class_precision"][label]
        row[f"recall_{safe}"] = metrics["per_class_recall"][label]
        row[f"f1_{safe}"] = metrics["per_class_f1"][label]
        row[f"support_{safe}"] = metrics["per_class_support"][label]
    return row


def sample_support_query(
    student_df: pd.DataFrame,
    k_shot: int,
    protocol: str,
    rng: np.random.Generator,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    student_df = student_df.sort_values("clip_index").copy()
    n = len(student_df)
    if k_shot <= 0:
        return student_df.iloc[0:0].copy(), student_df.copy(), None
    if n <= k_shot:
        return None, None, f"insufficient_clips_n={n}_k={k_shot}"

    if protocol == "early":
        support_df = student_df.iloc[:k_shot].copy()
    elif protocol == "random":
        indices = student_df.index.to_numpy()
        sampled = rng.choice(indices, size=k_shot, replace=False)
        support_df = student_df.loc[sampled].copy()
    else:
        return None, None, f"unknown_protocol_{protocol}"

    query_df = student_df.loc[~student_df.index.isin(support_df.index)].copy()
    if len(query_df) == 0:
        return None, None, f"empty_query_n={n}_k={k_shot}"
    return support_df, query_df, None


def pretrain_zero_shot_models(
    train_df: pd.DataFrame,
    seed: int,
):
    X_train = train_df[FEATURE_COLS].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=str)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    majority = train_df["label"].value_counts().idxmax()

    logreg = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs", random_state=seed)
    logreg.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    et = ExtraTreesClassifier(
        n_estimators=600,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    et.fit(X_train, y_train)

    sgd = SGDClassifier(loss="log_loss", random_state=seed, max_iter=2000, tol=1e-3)
    sgd.fit(X_train_scaled, y_train)

    return scaler, majority, logreg, rf, et, sgd


def pretrain_mlp_backbone_with_head(
    train_df: pd.DataFrame,
    scaler: StandardScaler,
    seed: int,
    pretrain_epochs: int,
):
    X_train_scaled = scaler.transform(train_df[FEATURE_COLS].to_numpy(dtype=float))
    y_train_labels = train_df["label"].to_numpy(dtype=str)
    y_train_int = np.asarray([LABEL_TO_INT[y] for y in y_train_labels], dtype=int)

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=1,
        warm_start=True,
        shuffle=True,
        random_state=seed,
    )
    classes = np.arange(len(LABELS), dtype=int)
    for epoch in range(pretrain_epochs):
        if epoch == 0:
            mlp.partial_fit(X_train_scaled, y_train_int, classes=classes)
        else:
            mlp.partial_fit(X_train_scaled, y_train_int)

    train_hidden = extract_hidden_features(mlp, X_train_scaled)
    head = SGDClassifier(loss="log_loss", random_state=seed, max_iter=2000, tol=1e-3)
    head.fit(train_hidden, y_train_labels)
    return mlp, head


def extract_hidden_features(mlp: MLPClassifier, X: np.ndarray) -> np.ndarray:
    hidden = X
    if len(mlp.coefs_) <= 1:
        return hidden
    for i in range(len(mlp.coefs_) - 1):
        hidden = hidden @ mlp.coefs_[i] + mlp.intercepts_[i]
        hidden = np.maximum(hidden, 0.0)
    return hidden


def one_shot_prototype_predict(
    train_df: pd.DataFrame,
    scaler: StandardScaler,
    support_df: pd.DataFrame,
    query_df: pd.DataFrame,
) -> np.ndarray:
    X_train = scaler.transform(train_df[FEATURE_COLS].to_numpy(dtype=float))
    y_train = train_df["label"].to_numpy(dtype=str)
    class_centroids = {}
    for label in LABELS:
        mask = y_train == label
        if np.any(mask):
            class_centroids[label] = X_train[mask].mean(axis=0)
        else:
            class_centroids[label] = np.zeros(X_train.shape[1], dtype=float)

    support_label = support_df["label"].iloc[0]
    support_vector = scaler.transform(support_df[FEATURE_COLS].iloc[[0]].to_numpy(dtype=float))[0]
    class_centroids[support_label] = 0.5 * class_centroids[support_label] + 0.5 * support_vector

    X_query = scaler.transform(query_df[FEATURE_COLS].to_numpy(dtype=float))
    distances = np.stack(
        [np.linalg.norm(X_query - class_centroids[label], axis=1) for label in LABELS],
        axis=1,
    )
    pred_idx = np.argmin(distances, axis=1)
    return np.asarray([LABELS[int(i)] for i in pred_idx], dtype=object)


def build_clip_level_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["method", "prediction_type", "k_shot", "support_protocol"]
    for key_vals, grp in pred_df.groupby(keys):
        y_true = grp["y_true"].to_numpy(dtype=str)
        y_pred = grp["y_pred"].to_numpy(dtype=str)
        metrics = compute_metrics(y_true, y_pred)
        row = {
            "method": key_vals[0],
            "prediction_type": key_vals[1],
            "k_shot": int(key_vals[2]),
            "support_protocol": key_vals[3],
            "n_students": int(grp["student_group"].nunique()),
            "n_query_clips": int(len(grp)),
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "confusion_matrix": str(metrics["confusion_matrix"]),
        }
        for label in LABELS:
            safe = label.replace(" ", "_")
            row[f"precision_{safe}"] = metrics["per_class_precision"][label]
            row[f"recall_{safe}"] = metrics["per_class_recall"][label]
            row[f"f1_{safe}"] = metrics["per_class_f1"][label]
            row[f"support_{safe}"] = metrics["per_class_support"][label]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def build_student_level_summary(per_student_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "prediction_type", "k_shot", "support_protocol"]
    summary = (
        per_student_df.groupby(keys, as_index=False)
        .agg(
            n_students=("student_group", "nunique"),
            total_query_clips=("query_size", "sum"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_weighted_f1=("weighted_f1", "mean"),
            std_weighted_f1=("weighted_f1", "std"),
        )
        .sort_values(keys)
        .reset_index(drop=True)
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="CMOSE unseen-student clip-level transfer learning evaluation (v2).")
    parser.add_argument("--json", type=str, default=r"C:\COLLEGE\capstone\latest\final_data_1.json")
    parser.add_argument("--folds", type=str, default=r"C:\COLLEGE\capstone\latest\cmose_h1_h2_fold_protocol.csv")
    parser.add_argument("--k-values", type=str, default="1,5,10,20")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlp-pretrain-epochs", type=int, default=30)
    parser.add_argument("--mlp-adapt-epochs", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default=r"C:\COLLEGE\capstone\latest")
    args = parser.parse_args()

    rng_global = np.random.default_rng(args.seed)
    k_values = [int(v.strip()) for v in args.k_values.split(",") if v.strip()]

    clip_df = load_clip_dataframe(args.json)
    group_fold = load_group_fold_map(args.folds)

    eligible_groups = set(group_fold.keys())
    missing_fold_df = clip_df.loc[~clip_df["group"].isin(eligible_groups)].copy()
    eval_df = clip_df.loc[clip_df["group"].isin(eligible_groups)].copy()
    if eval_df.empty:
        raise ValueError("No eligible clips after applying fold protocol group map.")

    per_student_rows: List[Dict[str, object]] = []
    clip_pred_rows: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, object]] = []

    for fold_id in sorted(set(group_fold.values())):
        test_groups = [g for g, f in group_fold.items() if f == fold_id]
        train_groups = [g for g, f in group_fold.items() if f != fold_id]

        train_df = eval_df.loc[eval_df["group"].isin(train_groups)].copy()
        test_df = eval_df.loc[eval_df["group"].isin(test_groups)].copy()
        if train_df.empty or test_df.empty:
            raise ValueError(f"Fold {fold_id} has empty train or test split.")

        scaler, majority_label, logreg, rf, et, sgd = pretrain_zero_shot_models(train_df, seed=args.seed + fold_id)
        mlp_backbone, mlp_head = pretrain_mlp_backbone_with_head(
            train_df,
            scaler=scaler,
            seed=args.seed + 1000 + fold_id,
            pretrain_epochs=args.mlp_pretrain_epochs,
        )

        # Step 2/3: clean zero-shot baselines on all held-out test clips.
        for group in sorted(test_groups):
            student_df = test_df.loc[test_df["group"] == group].copy()
            if student_df.empty:
                continue
            y_true = student_df["label"].to_numpy(dtype=str)
            X_raw = student_df[FEATURE_COLS].to_numpy(dtype=float)
            X_scaled = scaler.transform(X_raw)
            hidden = extract_hidden_features(mlp_backbone, X_scaled)

            zero_methods = {
                "majority_class": np.repeat(majority_label, len(student_df)),
                "logistic_regression": logreg.predict(X_scaled),
                "random_forest": rf.predict(X_raw),
                "extra_trees": et.predict(X_raw),
                "transfer_sgd_zero_shot": sgd.predict(X_scaled),
                "transfer_mlp_head_zero_shot": mlp_head.predict(hidden),
            }

            for method, y_pred in zero_methods.items():
                row = make_per_student_row(
                    method=method,
                    prediction_type="zero_shot",
                    outer_fold=fold_id,
                    student_group=group,
                    k_shot=0,
                    support_protocol="none",
                    support_size=0,
                    query_size=len(student_df),
                    y_true=y_true,
                    y_pred=np.asarray(y_pred, dtype=object),
                )
                per_student_rows.append(row)
                for clip_id, yt, yp in zip(student_df["clip_id"], y_true, y_pred):
                    clip_pred_rows.append(
                        {
                            "method": method,
                            "prediction_type": "zero_shot",
                            "outer_fold": fold_id,
                            "student_group": group,
                            "clip_id": clip_id,
                            "k_shot": 0,
                            "support_protocol": "none",
                            "y_true": yt,
                            "y_pred": yp,
                        }
                    )

        # Step 4/5: few-shot personalization / transfer with random and early support.
        for group in sorted(test_groups):
            student_df = test_df.loc[test_df["group"] == group].copy()
            if student_df.empty:
                continue

            for support_protocol in ["random", "early"]:
                for k in k_values:
                    local_seed = args.seed + fold_id * 100000 + k * 1000 + sum(ord(ch) for ch in group) + (17 if support_protocol == "early" else 0)
                    rng = np.random.default_rng(local_seed)
                    support_df, query_df, skip_reason = sample_support_query(student_df, k, support_protocol, rng)
                    if skip_reason is not None:
                        skipped_rows.append(
                            {
                                "outer_fold": fold_id,
                                "student_group": group,
                                "k_shot": k,
                                "support_protocol": support_protocol,
                                "reason": skip_reason,
                            }
                        )
                        continue

                    y_query = query_df["label"].to_numpy(dtype=str)
                    Xq_raw = query_df[FEATURE_COLS].to_numpy(dtype=float)
                    Xq_scaled = scaler.transform(Xq_raw)
                    Xs_raw = support_df[FEATURE_COLS].to_numpy(dtype=float)
                    Xs_scaled = scaler.transform(Xs_raw)
                    y_support = support_df["label"].to_numpy(dtype=str)

                    # Query-matched zero-shot reference for transfer comparisons.
                    y_zero = logreg.predict(Xq_scaled)
                    row_zero = make_per_student_row(
                        method="logistic_regression_query_matched",
                        prediction_type="zero_shot_query_matched",
                        outer_fold=fold_id,
                        student_group=group,
                        k_shot=k,
                        support_protocol=support_protocol,
                        support_size=len(support_df),
                        query_size=len(query_df),
                        y_true=y_query,
                        y_pred=y_zero,
                    )
                    per_student_rows.append(row_zero)
                    for clip_id, yt, yp in zip(query_df["clip_id"], y_query, y_zero):
                        clip_pred_rows.append(
                            {
                                "method": "logistic_regression_query_matched",
                                "prediction_type": "zero_shot_query_matched",
                                "outer_fold": fold_id,
                                "student_group": group,
                                "clip_id": clip_id,
                                "k_shot": k,
                                "support_protocol": support_protocol,
                                "y_true": yt,
                                "y_pred": yp,
                            }
                        )

                    # Support-only adaptation baseline (not transfer): train fresh on support.
                    if len(np.unique(y_support)) >= 2:
                        support_logreg = LogisticRegression(
                            max_iter=4000,
                            class_weight="balanced",
                            solver="lbfgs",
                            random_state=args.seed + fold_id,
                        )
                        support_logreg.fit(Xs_scaled, y_support)
                        y_support_adapt = support_logreg.predict(Xq_scaled)
                        method_name = "support_only_logreg_adaptation"
                    elif k == 1:
                        y_support_adapt = one_shot_prototype_predict(train_df, scaler, support_df, query_df)
                        method_name = "support_only_one_shot_prototype"
                    else:
                        skipped_rows.append(
                            {
                                "outer_fold": fold_id,
                                "student_group": group,
                                "k_shot": k,
                                "support_protocol": support_protocol,
                                "reason": "support_has_single_class_for_support_only_logreg",
                            }
                        )
                        y_support_adapt = None
                        method_name = ""

                    if y_support_adapt is not None:
                        row_support = make_per_student_row(
                            method=method_name,
                            prediction_type="few_shot_adapted",
                            outer_fold=fold_id,
                            student_group=group,
                            k_shot=k,
                            support_protocol=support_protocol,
                            support_size=len(support_df),
                            query_size=len(query_df),
                            y_true=y_query,
                            y_pred=np.asarray(y_support_adapt, dtype=object),
                        )
                        per_student_rows.append(row_support)
                        for clip_id, yt, yp in zip(query_df["clip_id"], y_query, y_support_adapt):
                            clip_pred_rows.append(
                                {
                                    "method": method_name,
                                    "prediction_type": "few_shot_adapted",
                                    "outer_fold": fold_id,
                                    "student_group": group,
                                    "clip_id": clip_id,
                                    "k_shot": k,
                                    "support_protocol": support_protocol,
                                    "y_true": yt,
                                    "y_pred": yp,
                                }
                            )

                    # Proper transfer adaptation 1: pretrained SGD -> partial_fit on support.
                    sgd_adapt = copy.deepcopy(sgd)
                    sgd_adapt.partial_fit(Xs_scaled, y_support)
                    y_sgd = sgd_adapt.predict(Xq_scaled)
                    row_sgd = make_per_student_row(
                        method="transfer_sgd_finetune",
                        prediction_type="few_shot_transfer",
                        outer_fold=fold_id,
                        student_group=group,
                        k_shot=k,
                        support_protocol=support_protocol,
                        support_size=len(support_df),
                        query_size=len(query_df),
                        y_true=y_query,
                        y_pred=np.asarray(y_sgd, dtype=object),
                    )
                    per_student_rows.append(row_sgd)
                    for clip_id, yt, yp in zip(query_df["clip_id"], y_query, y_sgd):
                        clip_pred_rows.append(
                            {
                                "method": "transfer_sgd_finetune",
                                "prediction_type": "few_shot_transfer",
                                "outer_fold": fold_id,
                                "student_group": group,
                                "clip_id": clip_id,
                                "k_shot": k,
                                "support_protocol": support_protocol,
                                "y_true": yt,
                                "y_pred": yp,
                            }
                        )

                    # Proper transfer adaptation 2: pretrained MLP backbone + pretrained head, then head fine-tune.
                    mlp_head_adapt = copy.deepcopy(mlp_head)
                    support_hidden = extract_hidden_features(mlp_backbone, Xs_scaled)
                    query_hidden = extract_hidden_features(mlp_backbone, Xq_scaled)
                    for _ in range(args.mlp_adapt_epochs):
                        mlp_head_adapt.partial_fit(support_hidden, y_support)
                    y_mlp = mlp_head_adapt.predict(query_hidden)
                    row_mlp = make_per_student_row(
                        method="transfer_mlp_head_finetune",
                        prediction_type="few_shot_transfer",
                        outer_fold=fold_id,
                        student_group=group,
                        k_shot=k,
                        support_protocol=support_protocol,
                        support_size=len(support_df),
                        query_size=len(query_df),
                        y_true=y_query,
                        y_pred=np.asarray(y_mlp, dtype=object),
                    )
                    per_student_rows.append(row_mlp)
                    for clip_id, yt, yp in zip(query_df["clip_id"], y_query, y_mlp):
                        clip_pred_rows.append(
                            {
                                "method": "transfer_mlp_head_finetune",
                                "prediction_type": "few_shot_transfer",
                                "outer_fold": fold_id,
                                "student_group": group,
                                "clip_id": clip_id,
                                "k_shot": k,
                                "support_protocol": support_protocol,
                                "y_true": yt,
                                "y_pred": yp,
                            }
                        )

    per_student_df = pd.DataFrame(per_student_rows)
    clip_pred_df = pd.DataFrame(clip_pred_rows)
    skipped_df = pd.DataFrame(skipped_rows)

    clip_level_summary_df = build_clip_level_summary(clip_pred_df)
    student_level_summary_df = build_student_level_summary(per_student_df)

    data_audit_rows = []
    label_counts = eval_df["label"].value_counts().reindex(LABELS, fill_value=0)
    for label, count in label_counts.items():
        data_audit_rows.append({"section": "eligible_clip_label_distribution", "key": label, "value": int(count)})
    for split_name, count in eval_df["split"].value_counts(dropna=False).items():
        data_audit_rows.append({"section": "eligible_clip_split_distribution", "key": str(split_name), "value": int(count)})
    data_audit_rows.append({"section": "coverage", "key": "eligible_groups_in_fold_protocol", "value": int(eval_df["group"].nunique())})
    data_audit_rows.append({"section": "coverage", "key": "excluded_groups_missing_fold_assignment", "value": int(missing_fold_df["group"].nunique())})
    data_audit_rows.append({"section": "coverage", "key": "excluded_clips_missing_fold_assignment", "value": int(len(missing_fold_df))})
    data_audit_df = pd.DataFrame(data_audit_rows)

    out_dir = args.output_dir
    per_student_path = f"{out_dir}\\transfer_v2_per_student_results.csv"
    per_clip_pred_path = f"{out_dir}\\transfer_v2_per_clip_predictions.csv"
    clip_summary_path = f"{out_dir}\\transfer_v2_clip_level_summary.csv"
    student_summary_path = f"{out_dir}\\transfer_v2_student_level_summary.csv"
    skipped_path = f"{out_dir}\\transfer_v2_skipped_cases.csv"
    audit_path = f"{out_dir}\\transfer_v2_data_audit.csv"

    per_student_df.to_csv(per_student_path, index=False)
    clip_pred_df.to_csv(per_clip_pred_path, index=False)
    clip_level_summary_df.to_csv(clip_summary_path, index=False)
    student_level_summary_df.to_csv(student_summary_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)
    data_audit_df.to_csv(audit_path, index=False)

    print(f"Saved per-student results: {per_student_path} ({len(per_student_df)} rows)")
    print(f"Saved per-clip predictions: {per_clip_pred_path} ({len(clip_pred_df)} rows)")
    print(f"Saved clip-level summary: {clip_summary_path} ({len(clip_level_summary_df)} rows)")
    print(f"Saved student-level summary: {student_summary_path} ({len(student_level_summary_df)} rows)")
    print(f"Saved skipped cases: {skipped_path} ({len(skipped_df)} rows)")
    print(f"Saved data audit: {audit_path} ({len(data_audit_df)} rows)")


if __name__ == "__main__":
    main()
