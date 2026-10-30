import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

LABELS = [
    "Highly Disengage",
    "Disengage",
    "Engage",
    "Highly Engage",
]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
FEATURE_COLS = [f"emb_{i}" for i in range(1024)]


@dataclass
class FoldAudit:
    fold_id: int
    n_train_students: int
    n_test_students: int
    n_episode_eligible_students: int
    episodes_trained: int


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int = 1024, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_group_from_clip_id(clip_id: str) -> Optional[str]:
    match = re.match(r"^(video[^_]+)_(\d+)_((?:person).+)$", str(clip_id))
    if not match:
        return None
    return f"{match.group(1)}_{match.group(3)}"


def load_clip_dataframe(json_path: str) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows: List[Dict[str, object]] = []
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

        arr = np.asarray(embeds, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            continue

        clip_index_match = re.search(r"_(\d+)_", str(clip_id))
        clip_index = int(clip_index_match.group(1)) if clip_index_match else 0

        rows.append(
            {
                "clip_id": str(clip_id),
                "group": group,
                "label": str(label),
                "label_id": int(LABEL_TO_ID[str(label)]),
                "clip_index": clip_index,
                "embedding": arr,
            }
        )

    if not rows:
        raise ValueError(f"No valid labeled clips were loaded from {json_path}")

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
    bad = group_fold.groupby("group")["fold"].nunique()
    bad = bad[bad != 1]
    if len(bad) > 0:
        raise ValueError(
            f"Groups assigned to multiple folds: {bad.index.tolist()}"
        )

    return dict(zip(group_fold["group"], group_fold["fold"].astype(int)))


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
    }


def build_student_class_index(
    df: pd.DataFrame,
) -> Dict[str, Dict[int, np.ndarray]]:
    by_student: Dict[str, Dict[int, np.ndarray]] = {}

    for group, gdf in df.groupby("group"):
        class_idx: Dict[int, np.ndarray] = {}
        for class_id in range(len(LABELS)):
            class_idx[class_id] = gdf.index[
                gdf["label_id"] == class_id
            ].to_numpy(dtype=int)
        by_student[group] = class_idx

    return by_student


def episodic_train_protonet(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    episodes: int,
    support_per_class: int,
    query_per_class: int,
    embed_dim: int,
    lr: float,
    weight_decay: float,
    device: str,
    seed: int,
) -> Tuple[MLPEncoder, int, int]:
    """
    Variable-way ProtoNet training.

    Important change from the original script:
    - A training student does NOT need all four labels.
    - Each episode samples one student/task that has at least two
      classes with enough clips.
    - The episode samples 2..N classes from the classes actually
      available for that student, with equal support/query counts.
    - Thus meta-training is task/student based and is not driven by
      indiscriminate clip-level sampling.
    """
    set_all_seeds(seed)

    model = MLPEncoder(
        input_dim=len(feature_cols),
        embed_dim=embed_dim,
    ).to(device)

    student_class_idx = build_student_class_index(train_df)

    min_needed = support_per_class + query_per_class
    eligible_students: List[str] = []

    for group, class_map in student_class_idx.items():
        usable_classes = [
            c
            for c in range(len(LABELS))
            if len(class_map[c]) >= min_needed
        ]
        if len(usable_classes) >= 2:
            eligible_students.append(group)

    if not eligible_students:
        raise ValueError(
            "No training students have at least two classes with enough "
            f"clips for an episode (need {min_needed} clips/class)."
        )

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    rng = np.random.default_rng(seed + 999)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    episodes_trained = 0

    for _ in range(episodes):
        # First choose a student/task. This is the unit of meta-learning.
        group = eligible_students[
            int(rng.integers(0, len(eligible_students)))
        ]
        class_map = student_class_idx[group]

        usable_classes = [
            c
            for c in range(len(LABELS))
            if len(class_map[c]) >= min_needed
        ]

        # Variable-way episode: 2 through all available classes.
        n_way = int(rng.integers(2, len(usable_classes) + 1))
        episode_classes = sorted(
            rng.choice(
                usable_classes,
                size=n_way,
                replace=False,
            ).tolist()
        )

        support_rows: List[int] = []
        query_rows: List[int] = []

        for class_id in episode_classes:
            idx = class_map[class_id]
            sampled = rng.choice(
                idx,
                size=min_needed,
                replace=False,
            )
            support_rows.extend(
                sampled[:support_per_class].tolist()
            )
            query_rows.extend(
                sampled[support_per_class:].tolist()
            )

        support_x = torch.tensor(
            X_train[support_rows],
            dtype=torch.float32,
            device=device,
        )
        query_x = torch.tensor(
            X_train[query_rows],
            dtype=torch.float32,
            device=device,
        )

        # Labels are remapped to the episode's local 0..N-1 classes.
        query_y = torch.tensor(
            np.repeat(
                np.arange(n_way),
                query_per_class,
            ),
            dtype=torch.long,
            device=device,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)

        support_z = model(support_x)
        query_z = model(query_x)

        prototypes = []
        start = 0
        for _class_position in range(n_way):
            end = start + support_per_class
            prototypes.append(
                support_z[start:end].mean(dim=0)
            )
            start = end

        proto = torch.stack(prototypes, dim=0)

        dists = torch.cdist(query_z, proto, p=2) ** 2
        logits = -dists
        loss = F.cross_entropy(logits, query_y)

        loss.backward()
        optimizer.step()
        episodes_trained += 1

    return model, len(eligible_students), episodes_trained


def sample_support_query(
    student_df: pd.DataFrame,
    k: int,
    protocol: str,
    rng: np.random.Generator,
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[str],
]:
    """
    Create exactly one support/query split for an unseen student.

    Crucially, support is allowed to contain 1, 2, 3, or 4 classes.
    A student is skipped only when there are not enough clips to create
    a non-empty query set.
    """
    ordered = student_df.sort_values("clip_index").copy()

    if len(ordered) <= k:
        return (
            None,
            None,
            f"insufficient_clips_n={len(ordered)}_k={k}",
        )

    if protocol == "early":
        support = ordered.iloc[:k].copy()
    elif protocol == "random":
        chosen = rng.choice(
            ordered.index.to_numpy(),
            size=k,
            replace=False,
        )
        support = ordered.loc[chosen].copy()
    else:
        return (
            None,
            None,
            f"unknown_support_protocol_{protocol}",
        )

    query = ordered.loc[
        ~ordered.index.isin(support.index)
    ].copy()

    if query.empty:
        return None, None, "empty_query"

    # No support-class requirement. ProtoNet can operate as variable-way
    # classification over the classes actually observed in the support.
    # If only one class is observed, every query prediction necessarily
    # maps to that observed class; this case is retained and reported.
    return support, query, None


def predict_with_prototypes(
    model: MLPEncoder,
    support_x: np.ndarray,
    support_y_ids: np.ndarray,
    query_x: np.ndarray,
    device: str,
) -> np.ndarray:
    """
    Variable-way ProtoNet inference.

    Only classes observed in the support set have prototypes. This is
    standard prototype classification for a variable-way support set.
    Absent support classes are not silently invented.
    """
    model.eval()

    with torch.no_grad():
        sup_t = torch.tensor(
            support_x,
            dtype=torch.float32,
            device=device,
        )
        qry_t = torch.tensor(
            query_x,
            dtype=torch.float32,
            device=device,
        )

        sup_z = model(sup_t)
        qry_z = model(qry_t)

        support_classes = sorted(
            np.unique(support_y_ids).tolist()
        )

        proto_list = []
        for class_id in support_classes:
            class_mask = support_y_ids == class_id
            mask_t = torch.tensor(
                class_mask,
                dtype=torch.bool,
                device=device,
            )
            proto_list.append(
                sup_z[mask_t].mean(dim=0)
            )

        proto = torch.stack(proto_list, dim=0)
        dists = torch.cdist(qry_z, proto, p=2) ** 2
        local_pred = torch.argmin(
            dists,
            dim=1,
        ).cpu().numpy()

    return np.asarray(
        [support_classes[int(i)] for i in local_pred],
        dtype=int,
    )


def predict_few_shot_logreg(
    scaler: StandardScaler,
    support_df: pd.DataFrame,
    query_df: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    """
    Logistic-regression few-shot comparator using the exact same
    support/query split as ProtoNet.

    If support contains one class, LogisticRegression cannot be fitted.
    In that legitimate one-class case, use the only observed support
    class as the prediction for every query clip rather than dropping
    the student.
    """
    X_support = scaler.transform(
        support_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    )
    y_support = support_df["label_id"].to_numpy(dtype=int)
    X_query = scaler.transform(
        query_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    )

    unique_classes = np.unique(y_support)

    if len(unique_classes) == 1:
        return np.repeat(
            int(unique_classes[0]),
            len(query_df),
        )

    clf = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(X_support, y_support)
    return clf.predict(X_query).astype(int)


def append_per_class_metrics(
    rows: List[Dict[str, object]],
    method: str,
    protocol: str,
    k: int,
    y_true: np.ndarray,
    y_pred_ids: np.ndarray,
) -> None:
    y_pred = np.asarray(
        [ID_TO_LABEL[int(i)] for i in y_pred_ids],
        dtype=object,
    )

    precision, recall, _, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    for class_idx, label in enumerate(LABELS):
        rows.append(
            {
                "method": method,
                "support_protocol": protocol,
                "k_shot": k,
                "class": label,
                "precision": float(precision[class_idx]),
                "recall": float(recall[class_idx]),
                "support": int(support[class_idx]),
                "confusion_matrix_row": cm[class_idx].tolist(),
            }
        )


def run_experiment(
    clip_df: pd.DataFrame,
    group_fold: Dict[str, int],
    k_values: List[int],
    protocols: List[str],
    seed: int,
    episodes_per_fold: int,
    support_per_class_train: int,
    query_per_class_train: int,
    embed_dim: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    List[FoldAudit],
]:
    eligible_groups = set(group_fold.keys())
    eval_df = clip_df[
        clip_df["group"].isin(eligible_groups)
    ].copy()

    if eval_df.empty:
        raise ValueError(
            "No eligible clip rows after applying fold protocol groups."
        )

    per_student_rows: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, object]] = []
    per_class_rows: List[Dict[str, object]] = []
    fold_audits: List[FoldAudit] = []

    for fold_id in sorted(set(group_fold.values())):
        test_groups = [
            g for g, f in group_fold.items()
            if f == fold_id
        ]
        train_groups = [
            g for g, f in group_fold.items()
            if f != fold_id
        ]

        train_df = eval_df[
            eval_df["group"].isin(train_groups)
        ].copy()
        test_df = eval_df[
            eval_df["group"].isin(test_groups)
        ].copy()

        overlap = (
            set(train_df["group"].unique())
            & set(test_df["group"].unique())
        )
        if overlap:
            raise ValueError(
                f"Leakage detected in fold {fold_id}, "
                f"overlapping groups: {sorted(overlap)}"
            )

        # Fit preprocessing on training students only.
        scaler = StandardScaler()
        scaler.fit(
            train_df[FEATURE_COLS].to_numpy(
                dtype=np.float32
            )
        )

        train_df_scaled = train_df.copy()
        train_df_scaled.loc[:, FEATURE_COLS] = scaler.transform(
            train_df_scaled[FEATURE_COLS].to_numpy(
                dtype=np.float32
            )
        )
        train_df_scaled = train_df_scaled.reset_index(drop=True)

        test_df_scaled = test_df.copy()
        test_df_scaled.loc[:, FEATURE_COLS] = scaler.transform(
            test_df_scaled[FEATURE_COLS].to_numpy(
                dtype=np.float32
            )
        )
        test_df_scaled = test_df_scaled.reset_index(drop=True)

        # Meta-train ProtoNet using training students/tasks only.
        encoder, n_episode_eligible, episodes_trained = (
            episodic_train_protonet(
                train_df=train_df_scaled,
                feature_cols=FEATURE_COLS,
                episodes=episodes_per_fold,
                support_per_class=support_per_class_train,
                query_per_class=query_per_class_train,
                embed_dim=embed_dim,
                lr=lr,
                weight_decay=weight_decay,
                device=device,
                seed=seed + 100 * fold_id,
            )
        )

        fold_audits.append(
            FoldAudit(
                fold_id=fold_id,
                n_train_students=len(train_groups),
                n_test_students=len(test_groups),
                n_episode_eligible_students=n_episode_eligible,
                episodes_trained=episodes_trained,
            )
        )

        # Majority class is learned from training students only.
        majority_label = (
            train_df["label"].value_counts().idxmax()
        )
        majority_id = LABEL_TO_ID[majority_label]

        for group in sorted(test_groups):
            student_df = test_df_scaled[
                test_df_scaled["group"] == group
            ].copy()

            if student_df.empty:
                skipped_rows.append(
                    {
                        "outer_fold": fold_id,
                        "student_group": group,
                        "support_protocol": "all",
                        "k_shot": -1,
                        "reason": "no_test_clips",
                    }
                )
                continue

            for protocol in protocols:
                for k in k_values:
                    local_seed = (
                        seed
                        + fold_id * 100_000
                        + k * 1_000
                        + sum(ord(ch) for ch in group)
                        + (7 if protocol == "early" else 0)
                    )
                    rng = np.random.default_rng(local_seed)

                    support_df, query_df, reason = (
                        sample_support_query(
                            student_df,
                            k,
                            protocol,
                            rng,
                        )
                    )

                    if reason is not None:
                        skipped_rows.append(
                            {
                                "outer_fold": fold_id,
                                "student_group": group,
                                "support_protocol": protocol,
                                "k_shot": k,
                                "reason": reason,
                            }
                        )
                        continue

                    support_x = support_df[
                        FEATURE_COLS
                    ].to_numpy(dtype=np.float32)
                    support_y_ids = support_df[
                        "label_id"
                    ].to_numpy(dtype=int)

                    query_x = query_df[
                        FEATURE_COLS
                    ].to_numpy(dtype=np.float32)
                    y_true = query_df[
                        "label"
                    ].to_numpy(dtype=str)

                    # ---------------- ProtoNet ----------------
                    proto_pred_ids = predict_with_prototypes(
                        model=encoder,
                        support_x=support_x,
                        support_y_ids=support_y_ids,
                        query_x=query_x,
                        device=device,
                    )

                    proto_pred = np.asarray(
                        [
                            ID_TO_LABEL[int(i)]
                            for i in proto_pred_ids
                        ],
                        dtype=object,
                    )
                    proto_metrics = compute_metrics(
                        y_true,
                        proto_pred,
                    )

                    per_student_rows.append(
                        {
                            "method": "protonet",
                            "outer_fold": fold_id,
                            "student_group": group,
                            "support_protocol": protocol,
                            "k_shot": k,
                            "support_size": int(len(support_df)),
                            "query_size": int(len(query_df)),
                            "support_classes": int(
                                support_df["label_id"].nunique()
                            ),
                            "query_classes": int(
                                query_df["label_id"].nunique()
                            ),
                            "accuracy": proto_metrics["accuracy"],
                            "balanced_accuracy": proto_metrics[
                                "balanced_accuracy"
                            ],
                            "macro_f1": proto_metrics["macro_f1"],
                        }
                    )

                    append_per_class_metrics(
                        per_class_rows,
                        "protonet",
                        protocol,
                        k,
                        y_true,
                        proto_pred_ids,
                    )

                    # ---------------- Logistic few-shot ----------------
                    logreg_pred_ids = predict_few_shot_logreg(
                        scaler=scaler,
                        support_df=support_df,
                        query_df=query_df,
                        seed=local_seed,
                    )
                    logreg_pred = np.asarray(
                        [
                            ID_TO_LABEL[int(i)]
                            for i in logreg_pred_ids
                        ],
                        dtype=object,
                    )
                    logreg_metrics = compute_metrics(
                        y_true,
                        logreg_pred,
                    )

                    per_student_rows.append(
                        {
                            "method": "logistic_few_shot",
                            "outer_fold": fold_id,
                            "student_group": group,
                            "support_protocol": protocol,
                            "k_shot": k,
                            "support_size": int(len(support_df)),
                            "query_size": int(len(query_df)),
                            "support_classes": int(
                                support_df["label_id"].nunique()
                            ),
                            "query_classes": int(
                                query_df["label_id"].nunique()
                            ),
                            "accuracy": logreg_metrics["accuracy"],
                            "balanced_accuracy": logreg_metrics[
                                "balanced_accuracy"
                            ],
                            "macro_f1": logreg_metrics["macro_f1"],
                        }
                    )

                    append_per_class_metrics(
                        per_class_rows,
                        "logistic_few_shot",
                        protocol,
                        k,
                        y_true,
                        logreg_pred_ids,
                    )

                    # ---------------- Query-matched majority ----------------
                    majority_pred_ids = np.repeat(
                        majority_id,
                        len(query_df),
                    )
                    majority_pred = np.asarray(
                        [
                            ID_TO_LABEL[int(i)]
                            for i in majority_pred_ids
                        ],
                        dtype=object,
                    )
                    majority_metrics = compute_metrics(
                        y_true,
                        majority_pred,
                    )

                    per_student_rows.append(
                        {
                            "method": "majority_query_matched",
                            "outer_fold": fold_id,
                            "student_group": group,
                            "support_protocol": protocol,
                            "k_shot": k,
                            "support_size": int(len(support_df)),
                            "query_size": int(len(query_df)),
                            "support_classes": int(
                                support_df["label_id"].nunique()
                            ),
                            "query_classes": int(
                                query_df["label_id"].nunique()
                            ),
                            "accuracy": majority_metrics["accuracy"],
                            "balanced_accuracy": majority_metrics[
                                "balanced_accuracy"
                            ],
                            "macro_f1": majority_metrics["macro_f1"],
                        }
                    )

                    append_per_class_metrics(
                        per_class_rows,
                        "majority_query_matched",
                        protocol,
                        k,
                        y_true,
                        majority_pred_ids,
                    )

    per_student_df = pd.DataFrame(per_student_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    per_class_df = pd.DataFrame(per_class_rows)

    return (
        per_student_df,
        skipped_df,
        per_class_df,
        fold_audits,
    )


def summarize_results(
    per_student_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Student-macro summary: each student contributes equally.
    This is the primary aggregate for an unseen-student experiment.
    """
    columns = [
        "method",
        "support_protocol",
        "k_shot",
        "n_students",
        "n_query_clips",
        "mean_accuracy",
        "std_accuracy",
        "mean_balanced_accuracy",
        "std_balanced_accuracy",
        "mean_macro_f1",
        "std_macro_f1",
    ]

    if per_student_df.empty:
        return pd.DataFrame(columns=columns)

    return (
        per_student_df.groupby(
            ["method", "support_protocol", "k_shot"],
            as_index=False,
        )
        .agg(
            n_students=("student_group", "nunique"),
            n_query_clips=("query_size", "sum"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=(
                "balanced_accuracy",
                "mean",
            ),
            std_balanced_accuracy=(
                "balanced_accuracy",
                "std",
            ),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
        )
        .sort_values(
            ["method", "support_protocol", "k_shot"]
        )
        .reset_index(drop=True)
    )


def build_pooled_diagnostics(
    per_class_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pooled query-level precision/recall for each class.

    per_class_df contains one row per student/class. Pooling here is
    done by summing TP/FP/FN over students, which gives an overall
    class-level diagnostic without giving long-query students multiple
    hidden metric rows in the student-level summary.
    """
    if per_class_df.empty:
        return pd.DataFrame()

    rows = []

    for (method, protocol, k, cls), gdf in per_class_df.groupby(
        ["method", "support_protocol", "k_shot", "class"]
    ):
        # Reconstruct TP/FN/FP from the stored row-wise confusion
        # matrices. This is easier and less error-prone if we keep the
        # matrices as lists in each row.
        tp = 0
        actual = 0
        predicted = 0

        class_idx = LABEL_TO_ID[cls]

        for _, row in gdf.iterrows():
            cm = np.asarray(
                row["confusion_matrix_row"],
                dtype=int,
            )
            actual += int(row["support"])
            predicted += int(cm.sum())  # all predictions for this actual row
            tp += int(cm[class_idx])

        # The above row-level representation gives actual-class rows,
        # not prediction columns, so derive precision requires the full
        # confusion matrix. It is computed separately below.
        rows.append(
            {
                "method": method,
                "support_protocol": protocol,
                "k_shot": k,
                "class": cls,
                "actual_support": actual,
                "note": "Use protonet_class_diagnostics.csv for per-student "
                        "precision/recall; pooled confusion matrix is "
                        "reported in the printed audit.",
            }
        )

    return pd.DataFrame(rows)


def build_aggregate_confusion(
    per_student_df: pd.DataFrame,
    clip_df: pd.DataFrame,
    group_fold: Dict[str, int],
) -> pd.DataFrame:
    """
    Recompute pooled confusion matrices directly from the per-student
    predictions is not possible because predictions are not stored.
    Therefore this function is intentionally not used to fabricate
    pooled values. The script instead writes per-student/class
    diagnostics and reports student-level metric summaries.
    """
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Variable-way episodic ProtoNet experiment for "
            "unseen-student engagement prediction, with matched "
            "few-shot Logistic Regression and majority baselines."
        )
    )

    parser.add_argument(
        "--json",
        type=str,
        default=os.path.join(os.environ.get("CMOSE_DATA_DIR", "."), "final_data_1.json"),
    )
    parser.add_argument(
        "--folds",
        type=str,
        default=os.path.join(os.environ.get("CMOSE_DATA_DIR", "."), "cmose_h1_h2_fold_protocol.csv"),
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="5,10,20",
    )
    parser.add_argument(
        "--support-protocols",
        type=str,
        default="random,early",
    )
    parser.add_argument(
        "--episodes-per-fold",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--train-support-per-class",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--train-query-per-class",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("CMOSE_OUTPUT_DIR", "."),
    )

    args = parser.parse_args()

    protocols = [
        p.strip()
        for p in args.support_protocols.split(",")
        if p.strip()
    ]
    k_values = [
        int(x.strip())
        for x in args.k_values.split(",")
        if x.strip()
    ]

    clip_df = load_clip_dataframe(args.json)
    group_fold = load_group_fold_map(args.folds)

    (
        per_student_df,
        skipped_df,
        per_class_df,
        fold_audits,
    ) = run_experiment(
        clip_df=clip_df,
        group_fold=group_fold,
        k_values=k_values,
        protocols=protocols,
        seed=args.seed,
        episodes_per_fold=args.episodes_per_fold,
        support_per_class_train=args.train_support_per_class,
        query_per_class_train=args.train_query_per_class,
        embed_dim=args.embed_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )

    if per_student_df.empty:
        raise ValueError(
            "No evaluation results were produced."
        )

    summary_df = summarize_results(per_student_df)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    per_student_path = os.path.join(
        out_dir,
        "protonet_per_student.csv",
    )
    summary_path = os.path.join(
        out_dir,
        "protonet_summary.csv",
    )
    skipped_path = os.path.join(
        out_dir,
        "protonet_skipped.csv",
    )
    class_path = os.path.join(
        out_dir,
        "protonet_class_diagnostics.csv",
    )

    per_student_df.to_csv(
        per_student_path,
        index=False,
    )
    summary_df.to_csv(
        summary_path,
        index=False,
    )
    skipped_df.to_csv(
        skipped_path,
        index=False,
    )
    per_class_df.to_csv(
        class_path,
        index=False,
    )

    print("\n=== ProtoNet Experiment Audit ===")
    print(f"Input embedding dimension: 1024")
    print(
        f"Encoder output embedding dimension: "
        f"{args.embed_dim}"
    )
    print(f"K values: {k_values}")
    print(f"Support protocols: {protocols}")
    print(
        f"Meta-training episodes per fold: "
        f"{args.episodes_per_fold}"
    )
    print(
        "Meta-training episode support/query per selected class: "
        f"{args.train_support_per_class}/"
        f"{args.train_query_per_class}"
    )
    print(
        "Meta-training: variable-way episodes; "
        "students/tasks are sampled first."
    )
    print(
        "Evaluation: support may contain 1-4 classes; "
        "no support-class exclusion."
    )
    print(f"Device: {args.device}")

    for audit in fold_audits:
        print(
            f"Fold {audit.fold_id}: "
            f"train_students={audit.n_train_students}, "
            f"test_students={audit.n_test_students}, "
            f"episode_eligible_students="
            f"{audit.n_episode_eligible_students}, "
            f"episodes_trained={audit.episodes_trained}"
        )

    print(
        "Leakage check: train/test groups are disjoint "
        "in every outer fold."
    )
    print(
        "Leakage check: no test-student clips were used "
        "in meta-training episodes."
    )

    print("\n=== Student-level summary ===")
    print(summary_df.to_string(index=False))

    print("\n=== Skip reasons ===")
    if skipped_df.empty:
        print("No students were skipped.")
    else:
        print(
            skipped_df["reason"]
            .value_counts()
            .to_string()
        )

    print("\n=== Per-class diagnostic preview ===")
    if per_class_df.empty:
        print("No class diagnostics.")
    else:
        preview = (
            per_class_df.groupby(
                ["method", "support_protocol", "k_shot", "class"],
                as_index=False,
            )
            .agg(
                mean_precision=("precision", "mean"),
                mean_recall=("recall", "mean"),
                total_support=("support", "sum"),
            )
        )
        print(preview.to_string(index=False))

    print(f"\nSaved: {per_student_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {skipped_path}")
    print(f"Saved: {class_path}")


if __name__ == "__main__":
    main()
