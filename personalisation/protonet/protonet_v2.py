"""
ProtoNet V2: Unseen-Student Engagement Prediction

Research question:
    Given a small number K of labelled clips from a completely unseen
    student, can we predict engagement on that student's remaining clips?

Task definition:
    - Each student/group is a meta-learning task.
    - Outer evaluation is strict 5-fold group-disjoint.
    - Test students are NEVER used during meta-training.
    - Each clip is represented by its 1024-D I3D embedding.

Support protocols:
    1. random:
       Class-diverse K-shot sampling. The sampler tries to cover as many
       classes as are actually present for the student, then fills the
       remaining support slots from the student's other clips.

    2. early:
       Strictly the first K chronological clips.
       No future clips are used to construct support.

ProtoNet:
    - Shared neural encoder: 1024 -> 256 -> 128
    - Student-specific prototypes are computed from the support set.
    - Variable-way classification is allowed.
    - A class only receives a prototype if it actually occurs in support.

Important:
    We do NOT require all four engagement classes to occur in support.
    If a true query class has no support prototype, the query is still
    evaluated using the available prototypes and that case is reported
    separately.

Outputs:
    - protonet_v2_per_student.csv
    - protonet_v2_summary.csv
    - protonet_v2_confusion_matrices.csv
    - protonet_v2_skipped.csv
    - protonet_v2_missing_prototype.csv
"""

import argparse
import json
import os
import re
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


LABELS = [
    "Highly Disengage",
    "Disengage",
    "Engage",
    "Highly Engage",
]

LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABELS)}

FEATURE_COLS = [f"emb_{i}" for i in range(1024)]


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# DATA LOADING
# ============================================================================

def parse_group_from_clip_id(clip_id):
    """
    videoX_Y_personZ -> videoX_personZ
    """
    match = re.match(
        r"^(video[^_]+)_(\d+)_((?:person).+)$",
        str(clip_id),
    )

    if not match:
        return None

    return f"{match.group(1)}_{match.group(3)}"


def load_clip_dataframe(json_path):
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

        if not isinstance(embeds, list):
            continue

        if len(embeds) != 1024:
            continue

        group = parse_group_from_clip_id(clip_id)

        if group is None:
            continue

        arr = np.asarray(embeds, dtype=np.float32)

        if not np.all(np.isfinite(arr)):
            continue

        match = re.search(r"_(\d+)_", str(clip_id))

        if match:
            clip_index = int(match.group(1))
        else:
            clip_index = 0

        records.append(
            {
                "clip_id": str(clip_id),
                "group": group,
                "label": str(label),
                "clip_index": clip_index,
                "embedding": arr,
            }
        )

    if not records:
        raise ValueError(
            f"No valid labelled clips found in {json_path}"
        )

    df = pd.DataFrame(records)

    emb_matrix = np.vstack(
        df["embedding"].to_numpy()
    )

    emb_df = pd.DataFrame(
        emb_matrix,
        columns=FEATURE_COLS,
        index=df.index,
    )

    df = pd.concat(
        [
            df.drop(columns=["embedding"]),
            emb_df,
        ],
        axis=1,
    )

    return df


def load_group_fold_map(fold_csv_path):

    fold_df = pd.read_csv(fold_csv_path)

    required = {"group", "fold"}

    missing = required - set(fold_df.columns)

    if missing:
        raise ValueError(
            f"Fold CSV missing required columns: {sorted(missing)}"
        )

    group_fold = (
        fold_df[["group", "fold"]]
        .drop_duplicates()
    )

    check = (
        group_fold
        .groupby("group")["fold"]
        .nunique()
    )

    bad = check[check != 1]

    if len(bad) > 0:
        raise ValueError(
            "Groups assigned to multiple folds: "
            f"{bad.index.tolist()}"
        )

    return dict(
        zip(
            group_fold["group"],
            group_fold["fold"].astype(int),
        )
    )


# ============================================================================
# ENCODER
# ============================================================================

class ProtoNetEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(256, 128),
        )

    def forward(self, x):

        z = self.network(x)

        # Normalize embeddings before distance calculation.
        z = F.normalize(z, p=2, dim=-1)

        return z


# ============================================================================
# PROTOTYPE CLASSIFICATION
# ============================================================================

def compute_prototypes(
    support_embeddings,
    support_labels,
):
    """
    Create one prototype per class actually represented
    in the support set.
    """

    prototypes = {}
    unique_classes = torch.unique(
        support_labels
    ).tolist()

    for class_id in unique_classes:

        mask = support_labels == class_id

        prototype = support_embeddings[mask].mean(
            dim=0,
            keepdim=True,
        )

        prototype = F.normalize(
            prototype,
            p=2,
            dim=-1,
        )

        prototypes[int(class_id)] = prototype.squeeze(0)

    return prototypes


def predict_from_prototypes(
    query_embeddings,
    prototypes,
):
    """
    Variable-way prediction.

    Only classes represented in support receive
    a prototype.
    """

    if len(prototypes) == 0:
        raise ValueError(
            "Cannot predict without prototypes."
        )

    class_ids = sorted(prototypes.keys())

    prototype_matrix = torch.stack(
        [
            prototypes[c]
            for c in class_ids
        ],
        dim=0,
    )

    prototype_matrix = F.normalize(
        prototype_matrix,
        p=2,
        dim=-1,
    )

    query_embeddings = F.normalize(
        query_embeddings,
        p=2,
        dim=-1,
    )

    # Cosine similarity.
    similarities = (
        query_embeddings
        @ prototype_matrix.T
    )

    best_indices = torch.argmax(
        similarities,
        dim=1,
    )

    predicted = torch.tensor(
        [
            class_ids[int(i)]
            for i in best_indices.cpu().numpy()
        ],
        dtype=torch.long,
        device=query_embeddings.device,
    )

    return predicted


# ============================================================================
# EPISODE SAMPLING
# ============================================================================

def sample_class_diverse_support(
    student_df,
    k_shot,
    rng,
):
    """
    Random support protocol.

    First tries to cover as many classes as possible.
    Then fills remaining slots from remaining clips.

    This does NOT require all four classes.

    Importantly, it only samples classes that actually occur
    in the student's available data.
    """

    ordered = student_df.sort_values(
        "clip_index"
    ).copy()

    if len(ordered) <= k_shot:
        return None, None, "insufficient_clips"

    labels_present = (
        ordered["label"]
        .value_counts()
        .index
        .tolist()
    )

    # Shuffle class order so that no class has systematic priority.
    labels_present = list(labels_present)
    rng.shuffle(labels_present)

    selected_indices = []

    # ------------------------------------------------------------
    # Phase 1: one example from each available class
    # ------------------------------------------------------------

    for label in labels_present:

        if len(selected_indices) >= k_shot:
            break

        rows = ordered[
            ordered["label"] == label
        ]

        if len(rows) == 0:
            continue

        chosen = rng.choice(
            rows.index.to_numpy()
        )

        selected_indices.append(chosen)

    # ------------------------------------------------------------
    # Phase 2: fill remaining slots randomly
    # ------------------------------------------------------------

    remaining_indices = [
        idx
        for idx in ordered.index
        if idx not in selected_indices
    ]

    remaining_slots = (
        k_shot - len(selected_indices)
    )

    if remaining_slots > 0:

        chosen_extra = rng.choice(
            np.asarray(remaining_indices),
            size=min(
                remaining_slots,
                len(remaining_indices),
            ),
            replace=False,
        )

        selected_indices.extend(
            chosen_extra.tolist()
        )

    support = ordered.loc[
        selected_indices
    ].copy()

    query = ordered.loc[
        ~ordered.index.isin(selected_indices)
    ].copy()

    if query.empty:
        return None, None, "empty_query"

    return support, query, None


def sample_early_support(
    student_df,
    k_shot,
):
    """
    Strict chronological support.

    The first K clips are support.
    Everything after them is query.

    This must NOT be class-balanced because doing so
    would use future information.
    """

    ordered = student_df.sort_values(
        "clip_index"
    ).copy()

    if len(ordered) <= k_shot:
        return None, None, "insufficient_clips"

    support = ordered.iloc[
        :k_shot
    ].copy()

    query = ordered.iloc[
        k_shot:
    ].copy()

    if query.empty:
        return None, None, "empty_query"

    return support, query, None


# ============================================================================
# META-TRAINING EPISODES
# ============================================================================

def build_training_episode(
    train_df,
    train_groups,
    rng,
    min_classes=2,
    max_support_per_class=5,
    max_query_per_class=5,
):
    """
    Build ONE ProtoNet meta-training episode.

    A student/group is selected first.

    This is important:
        We sample TASKS (students), not arbitrary clips.

    The episode contains classes available for that student.
    """

    if len(train_groups) == 0:
        return None

    # Try several students until we get a usable episode.
    for _ in range(50):

        group = rng.choice(train_groups)

        student_df = train_df[
            train_df["group"] == group
        ].sort_values("clip_index").copy()

        if len(student_df) < 4:
            continue

        class_groups = {}

        for label, rows in student_df.groupby("label"):

            if len(rows) >= 2:
                class_groups[label] = rows

        if len(class_groups) < min_classes:
            continue

        selected_labels = list(
            class_groups.keys()
        )

        # Randomly choose 2-4 classes when possible.
        rng.shuffle(selected_labels)

        n_way = min(
            len(selected_labels),
            4,
        )

        selected_labels = selected_labels[
            :n_way
        ]

        support_rows = []
        query_rows = []

        valid = True

        for label in selected_labels:

            rows = class_groups[label]

            rows_idx = rows.index.to_numpy()
            rng.shuffle(rows_idx)

            support_n = min(
                max_support_per_class,
                max(1, len(rows_idx) // 2),
            )

            remaining_n = (
                len(rows_idx) - support_n
            )

            query_n = min(
                max_query_per_class,
                remaining_n,
            )

            if query_n < 1:
                valid = False
                break

            support_idx = rows_idx[
                :support_n
            ]

            query_idx = rows_idx[
                support_n:
                support_n + query_n
            ]

            support_rows.extend(
                student_df.loc[
                    support_idx
                ].to_dict("records")
            )

            query_rows.extend(
                student_df.loc[
                    query_idx
                ].to_dict("records")
            )

        if not valid:
            continue

        if len(support_rows) == 0:
            continue

        if len(query_rows) == 0:
            continue

        support = pd.DataFrame(
            support_rows
        )

        query = pd.DataFrame(
            query_rows
        )

        return support, query, group

    return None


def episodic_train_protonet(
    train_df,
    train_groups,
    device,
    seed,
    episodes=1000,
    learning_rate=1e-3,
):
    """
    Train ProtoNet encoder episodically.

    Each episode corresponds to one student/task.
    """

    set_seed(seed)

    rng = np.random.default_rng(seed)

    encoder = ProtoNetEncoder().to(device)

    optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    encoder.train()

    completed = 0

    for episode_id in range(episodes):

        episode = build_training_episode(
            train_df=train_df,
            train_groups=train_groups,
            rng=rng,
        )

        if episode is None:
            continue

        support_df, query_df, _ = episode

        X_support = torch.tensor(
            support_df[FEATURE_COLS].to_numpy(
                dtype=np.float32
            ),
            dtype=torch.float32,
            device=device,
        )

        y_support_original = [
            LABEL_TO_ID[x]
            for x in support_df["label"]
        ]

        y_query_original = [
            LABEL_TO_ID[x]
            for x in query_df["label"]
        ]

        X_query = torch.tensor(
            query_df[FEATURE_COLS].to_numpy(
                dtype=np.float32
            ),
            dtype=torch.float32,
            device=device,
        )

        y_support = torch.tensor(
            y_support_original,
            dtype=torch.long,
            device=device,
        )

        y_query = torch.tensor(
            y_query_original,
            dtype=torch.long,
            device=device,
        )

        # Encode.
        z_support = encoder(X_support)
        z_query = encoder(X_query)

        # Prototypes.
        prototypes = compute_prototypes(
            z_support,
            y_support,
        )

        class_ids = sorted(
            prototypes.keys()
        )

        prototype_matrix = torch.stack(
            [
                prototypes[c]
                for c in class_ids
            ]
        )

        # Similarity logits.
        logits = (
            z_query
            @ prototype_matrix.T
        )

        # Convert original class IDs into
        # episode-local class IDs.
        class_to_episode_id = {
            class_id: i
            for i, class_id
            in enumerate(class_ids)
        }

        local_targets = torch.tensor(
            [
                class_to_episode_id[int(y)]
                for y in y_query
            ],
            dtype=torch.long,
            device=device,
        )

        loss = F.cross_entropy(
            logits,
            local_targets,
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            encoder.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        completed += 1

    return encoder, completed


# ============================================================================
# METRICS
# ============================================================================

def evaluate_predictions(
    y_true,
    y_pred,
):
    """
    Explicitly specify all four classes so that
    absent predicted/true classes do not produce
    misleading sklearn warnings.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    label_ids = np.arange(
        len(LABELS)
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=label_ids,
        average="macro",
        zero_division=0,
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=label_ids,
            average=None,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=label_ids,
    )

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "macro_f1": float(macro_f1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "confusion_matrix": cm,
    }


# ============================================================================
# MAJORITY BASELINE
# ============================================================================

def query_matched_majority(
    support_df,
    query_df,
):
    """
    Predict the most common class in the support set
    for every query clip.

    This is the fair student-specific majority baseline.
    """

    support_counts = (
        support_df["label"]
        .value_counts()
    )

    majority_label = (
        support_counts
        .index[0]
    )

    return np.array(
        [
            LABEL_TO_ID[majority_label]
            for _ in range(len(query_df))
        ]
    )


# ============================================================================
# ONE STUDENT EVALUATION
# ============================================================================

def evaluate_student(
    student_df,
    encoder,
    device,
    k_shot,
    protocol,
):
    """
    Evaluate one unseen student.

    Returns:
        result dictionary
        OR None + skip reason
    """

    if protocol == "random":

        # Deterministic per-student seed.
        seed_string = (
            str(student_df["group"].iloc[0])
            + str(k_shot)
        )

        local_seed = sum(
            ord(c)
            for c in seed_string
        )

        rng = np.random.default_rng(
            local_seed
        )

        support_df, query_df, reason = (
            sample_class_diverse_support(
                student_df,
                k_shot,
                rng,
            )
        )

    elif protocol == "early":

        support_df, query_df, reason = (
            sample_early_support(
                student_df,
                k_shot,
            )
        )

    else:
        return None, "unknown_protocol"

    if reason is not None:
        return None, reason

    X_support = torch.tensor(
        support_df[FEATURE_COLS].to_numpy(
            dtype=np.float32
        ),
        dtype=torch.float32,
        device=device,
    )

    X_query = torch.tensor(
        query_df[FEATURE_COLS].to_numpy(
            dtype=np.float32
        ),
        dtype=torch.float32,
        device=device,
    )

    y_support = torch.tensor(
        [
            LABEL_TO_ID[x]
            for x in support_df["label"]
        ],
        dtype=torch.long,
        device=device,
    )

    y_query = np.array(
        [
            LABEL_TO_ID[x]
            for x in query_df["label"]
        ],
        dtype=int,
    )

    encoder.eval()

    with torch.no_grad():

        z_support = encoder(
            X_support
        )

        z_query = encoder(
            X_query
        )

        prototypes = compute_prototypes(
            z_support,
            y_support,
        )

        y_pred_tensor = (
            predict_from_prototypes(
                z_query,
                prototypes,
            )
        )

        y_pred = (
            y_pred_tensor
            .cpu()
            .numpy()
        )

    # ------------------------------------------------------------
    # ProtoNet metrics
    # ------------------------------------------------------------

    metrics = evaluate_predictions(
        y_query,
        y_pred,
    )

    # ------------------------------------------------------------
    # Majority baseline
    # ------------------------------------------------------------

    majority_pred = query_matched_majority(
        support_df,
        query_df,
    )

    majority_metrics = evaluate_predictions(
        y_query,
        majority_pred,
    )

    # ------------------------------------------------------------
    # Missing-prototype diagnostics
    # ------------------------------------------------------------

    support_classes = sorted(
        set(
            support_df["label"]
        )
    )

    support_class_ids = {
        LABEL_TO_ID[x]
        for x in support_classes
    }

    missing_mask = np.array(
        [
            int(y) not in support_class_ids
            for y in y_query
        ],
        dtype=bool,
    )

    missing_count = int(
        missing_mask.sum()
    )

    query_count = len(
        query_df
    )

    # Per-class metrics.
    result = {
        "method": "ProtoNet_v2",
        "student_group": student_df["group"].iloc[0],
        "k_shot": int(k_shot),
        "support_protocol": protocol,
        "support_size": int(len(support_df)),
        "query_size": int(len(query_df)),
        "support_classes": int(
            len(support_classes)
        ),
        "support_class_names": "|".join(
            support_classes
        ),

        # ProtoNet
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics[
            "balanced_accuracy"
        ],
        "macro_f1": metrics["macro_f1"],

        # Majority
        "majority_accuracy": (
            majority_metrics["accuracy"]
        ),
        "majority_balanced_accuracy": (
            majority_metrics[
                "balanced_accuracy"
            ]
        ),
        "majority_macro_f1": (
            majority_metrics["macro_f1"]
        ),

        # Missing prototype
        "missing_prototype_query_clips": (
            missing_count
        ),
        "missing_prototype_query_rate": (
            missing_count / query_count
            if query_count > 0
            else 0.0
        ),
    }

    # Per-class precision / recall / F1.
    for i, label in enumerate(LABELS):

        result[
            f"precision_{label}"
        ] = float(
            metrics["precision"][i]
        )

        result[
            f"recall_{label}"
        ] = float(
            metrics["recall"][i]
        )

        result[
            f"f1_{label}"
        ] = float(
            metrics["f1"][i]
        )

        result[
            f"support_{label}"
        ] = int(
            metrics["support"][i]
        )

    # Confusion matrix stored separately later.
    result["_confusion_matrix"] = (
        metrics["confusion_matrix"]
    )

    return result, None


# ============================================================================
# EXPERIMENT
# ============================================================================

def run_experiment(
    clip_df,
    group_fold,
    k_values,
    seed,
    episodes,
    device,
):
    per_student_rows = []
    skipped_rows = []
    confusion_rows = []

    all_folds = sorted(
        set(group_fold.values())
    )

    for fold_id in all_folds:

        print(
            f"\n{'=' * 70}"
        )
        print(
            f"OUTER FOLD {fold_id}"
        )
        print(
            f"{'=' * 70}"
        )

        test_groups = [
            g
            for g, f in group_fold.items()
            if f == fold_id
        ]

        train_groups = [
            g
            for g, f in group_fold.items()
            if f != fold_id
        ]

        train_df = clip_df[
            clip_df["group"].isin(
                train_groups
            )
        ].copy()

        test_df = clip_df[
            clip_df["group"].isin(
                test_groups
            )
        ].copy()

        # Explicit leakage check.
        overlap = (
            set(train_groups)
            & set(test_groups)
        )

        if overlap:
            raise RuntimeError(
                f"LEAKAGE: train/test overlap "
                f"in fold {fold_id}: {overlap}"
            )

        print(
            f"Train students: {len(train_groups)}"
        )

        print(
            f"Test students:  {len(test_groups)}"
        )

        print(
            f"Training ProtoNet for "
            f"{episodes} episodes..."
        )

        encoder, completed = (
            episodic_train_protonet(
                train_df=train_df,
                train_groups=train_groups,
                device=device,
                seed=seed + fold_id,
                episodes=episodes,
            )
        )

        print(
            f"Completed episodes: {completed}"
        )

        # ------------------------------------------------------------
        # Evaluate each unseen student.
        # ------------------------------------------------------------

        for group in test_groups:

            student_df = test_df[
                test_df["group"] == group
            ].copy()

            if student_df.empty:
                continue

            # Verify that this student's data never entered training.
            if group in train_groups:
                raise RuntimeError(
                    f"Test student {group} "
                    f"appeared in training."
                )

            for k in k_values:

                for protocol in [
                    "random",
                    "early",
                ]:

                    result, reason = (
                        evaluate_student(
                            student_df=student_df,
                            encoder=encoder,
                            device=device,
                            k_shot=k,
                            protocol=protocol,
                        )
                    )

                    if reason is not None:

                        skipped_rows.append(
                            {
                                "outer_fold": fold_id,
                                "student_group": group,
                                "k_shot": k,
                                "support_protocol": protocol,
                                "reason": reason,
                            }
                        )

                        continue

                    result[
                        "outer_fold"
                    ] = int(fold_id)

                    per_student_rows.append(
                        result
                    )

                    # ------------------------------------------------
                    # Confusion matrix
                    # ------------------------------------------------

                    cm = result[
                        "_confusion_matrix"
                    ]

                    for true_id in range(
                        len(LABELS)
                    ):

                        for pred_id in range(
                            len(LABELS)
                        ):

                            confusion_rows.append(
                                {
                                    "method": "ProtoNet_v2",
                                    "outer_fold": fold_id,
                                    "student_group": group,
                                    "k_shot": k,
                                    "support_protocol": protocol,
                                    "true_class": LABELS[
                                        true_id
                                    ],
                                    "predicted_class": LABELS[
                                        pred_id
                                    ],
                                    "count": int(
                                        cm[
                                            true_id,
                                            pred_id
                                        ]
                                    ),
                                }
                            )

    per_student_df = pd.DataFrame(
        per_student_rows
    )

    skipped_df = pd.DataFrame(
        skipped_rows
    )

    confusion_df = pd.DataFrame(
        confusion_rows
    )

    if per_student_df.empty:
        raise RuntimeError(
            "No ProtoNet evaluation results were produced."
        )

    # Remove internal matrix before saving.
    if "_confusion_matrix" in per_student_df.columns:
        per_student_df = per_student_df.drop(
            columns=["_confusion_matrix"]
        )

    # ------------------------------------------------------------
    # Aggregate summary.
    #
    # IMPORTANT:
    # Mean is taken across students, so one student
    # does not dominate simply because they have many clips.
    # ------------------------------------------------------------

    summary_df = (
        per_student_df
        .groupby(
            [
                "support_protocol",
                "k_shot",
            ],
            as_index=False,
        )
        .agg(
            n_students=(
                "student_group",
                "nunique",
            ),
            total_query_clips=(
                "query_size",
                "sum",
            ),

            mean_accuracy=(
                "accuracy",
                "mean",
            ),

            sd_accuracy=(
                "accuracy",
                "std",
            ),

            mean_balanced_accuracy=(
                "balanced_accuracy",
                "mean",
            ),

            sd_balanced_accuracy=(
                "balanced_accuracy",
                "std",
            ),

            mean_macro_f1=(
                "macro_f1",
                "mean",
            ),

            sd_macro_f1=(
                "macro_f1",
                "std",
            ),

            mean_majority_accuracy=(
                "majority_accuracy",
                "mean",
            ),

            mean_majority_balanced_accuracy=(
                "majority_balanced_accuracy",
                "mean",
            ),

            mean_majority_macro_f1=(
                "majority_macro_f1",
                "mean",
            ),

            mean_missing_prototype_rate=(
                "missing_prototype_query_rate",
                "mean",
            ),
        )
        .sort_values(
            [
                "support_protocol",
                "k_shot",
            ]
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # Per-class aggregate metrics.
    # ------------------------------------------------------------

    class_rows = []

    for (
        protocol,
        k,
    ), subset in per_student_df.groupby(
        [
            "support_protocol",
            "k_shot",
        ]
    ):

        for label in LABELS:

            class_rows.append(
                {
                    "support_protocol": protocol,
                    "k_shot": k,
                    "class": label,

                    "mean_precision": subset[
                        f"precision_{label}"
                    ].mean(),

                    "mean_recall": subset[
                        f"recall_{label}"
                    ].mean(),

                    "mean_f1": subset[
                        f"f1_{label}"
                    ].mean(),

                    "total_support": subset[
                        f"support_{label}"
                    ].sum(),
                }
            )

    class_metrics_df = pd.DataFrame(
        class_rows
    )

    return (
        per_student_df,
        summary_df,
        class_metrics_df,
        skipped_df,
        confusion_df,
    )


# ============================================================================
# AUDIT
# ============================================================================

def print_audit(
    clip_df,
    group_fold,
    per_student_df,
    skipped_df,
    summary_df,
):
    print("\n")
    print("=" * 80)
    print("PROTONET V2 FINAL AUDIT")
    print("=" * 80)

    all_groups = set(
        clip_df["group"].unique()
    )

    fold_groups = set(
        group_fold.keys()
    )

    print(
        f"Total labelled clips: "
        f"{len(clip_df)}"
    )

    print(
        f"Total students/groups in data: "
        f"{len(all_groups)}"
    )

    print(
        f"Students in fold protocol: "
        f"{len(fold_groups)}"
    )

    missing_from_fold = (
        all_groups - fold_groups
    )

    if missing_from_fold:
        print(
            f"WARNING: {len(missing_from_fold)} "
            "students are not in fold protocol."
        )
    else:
        print(
            "PASS: every loaded student "
            "is represented in fold protocol."
        )

    # Check whether any student was evaluated in multiple outer folds.
    if not per_student_df.empty:

        student_fold_counts = (
            per_student_df
            .groupby("student_group")[
                "outer_fold"
            ]
            .nunique()
        )

        bad = student_fold_counts[
            student_fold_counts > 1
        ]

        if len(bad) > 0:
            print(
                "FAIL: students evaluated "
                "in multiple outer folds."
            )
        else:
            print(
                "PASS: every student belongs "
                "to one outer test fold."
            )

    print(
        "\nEvaluated student counts:"
    )

    if not summary_df.empty:
        print(
            summary_df[
                [
                    "support_protocol",
                    "k_shot",
                    "n_students",
                    "total_query_clips",
                    "mean_accuracy",
                    "mean_balanced_accuracy",
                    "mean_macro_f1",
                    "mean_missing_prototype_rate",
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "\nSkipped cases:"
    )

    if skipped_df.empty:
        print(
            "None."
        )
    else:
        print(
            skipped_df[
                "reason"
            ].value_counts().to_string()
        )

    print(
        "\nClass distribution:"
    )

    print(
        clip_df["label"]
        .value_counts()
        .to_string()
    )

    print(
        "\n" + "=" * 80
    )
    print(
        "AUDIT COMPLETE"
    )
    print(
        "=" * 80
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "ProtoNet V2 unseen-student "
            "engagement prediction."
        )
    )

    parser.add_argument(
        "--json",
        type=str,
        default=(
            r"C:\COLLEGE\capstone\latest"
            r"\final_data_1.json"
        ),
    )

    parser.add_argument(
        "--folds",
        type=str,
        default=(
            r"C:\COLLEGE\capstone\latest"
            r"\cmose_h1_h2_fold_protocol.csv"
        ),
    )

    parser.add_argument(
        "--k-values",
        type=str,
        default="5,10,20",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help=(
            "Meta-training episodes per outer fold."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            r"C:\COLLEGE\capstone\latest"
        ),
    )

    args = parser.parse_args()

    set_seed(args.seed)

    k_values = [
        int(v.strip())
        for v in args.k_values.split(",")
        if v.strip()
    ]

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------

    print(
        "\nLoading dataset..."
    )

    clip_df = load_clip_dataframe(
        args.json
    )

    print(
        f"Loaded {len(clip_df)} labelled clips "
        f"from {clip_df['group'].nunique()} students."
    )

    group_fold = load_group_fold_map(
        args.folds
    )

    # ------------------------------------------------------------
    # Run
    # ------------------------------------------------------------

    (
        per_student_df,
        summary_df,
        class_metrics_df,
        skipped_df,
        confusion_df,
    ) = run_experiment(
        clip_df=clip_df,
        group_fold=group_fold,
        k_values=k_values,
        seed=args.seed,
        episodes=args.episodes,
        device=device,
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    per_student_path = os.path.join(
        args.output_dir,
        "protonet_v2_per_student.csv",
    )

    summary_path = os.path.join(
        args.output_dir,
        "protonet_v2_summary.csv",
    )

    class_metrics_path = os.path.join(
        args.output_dir,
        "protonet_v2_class_metrics.csv",
    )

    skipped_path = os.path.join(
        args.output_dir,
        "protonet_v2_skipped.csv",
    )

    confusion_path = os.path.join(
        args.output_dir,
        "protonet_v2_confusion_matrices.csv",
    )

    per_student_df.to_csv(
        per_student_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    class_metrics_df.to_csv(
        class_metrics_path,
        index=False,
    )

    skipped_df.to_csv(
        skipped_path,
        index=False,
    )

    confusion_df.to_csv(
        confusion_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("PROTONET V2 RESULTS")
    print("=" * 80)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 80)
    print("PER-CLASS RESULTS")
    print("=" * 80)

    print(
        class_metrics_df.to_string(
            index=False
        )
    )

    print_audit(
        clip_df=clip_df,
        group_fold=group_fold,
        per_student_df=per_student_df,
        skipped_df=skipped_df,
        summary_df=summary_df,
    )

    print("\nSaved:")
    print(
        per_student_path
    )
    print(
        summary_path
    )
    print(
        class_metrics_path
    )
    print(
        skipped_path
    )
    print(
        confusion_path
    )


if __name__ == "__main__":
    main()