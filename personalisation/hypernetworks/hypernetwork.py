import argparse
import json
import os
import re
import random
from collections import Counter

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
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

LABELS = [
    "Highly Disengage",
    "Disengage",
    "Engage",
    "Highly Engage",
]

LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}

FEATURE_COLS = [f"emb_{i}" for i in range(1024)]

NUM_CLASSES = 4
INPUT_DIM = 1024

EMBED_DIM = 128
CONTEXT_DIM = 128
HYPER_HIDDEN = 128

HEAD_PARAM_DIM = NUM_CLASSES * EMBED_DIM + NUM_CLASSES

DEFAULT_EPISODES = 1000
DEFAULT_K_VALUES = [5, 10, 20]
DEFAULT_SEED = 42

FOCAL_GAMMA = 2.0
ORDINAL_WEIGHT = 0.25

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Number of examples per class in H5 meta-training query batches.
H5_QUERY_PER_CLASS = 16


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA LOADING
# ============================================================

def parse_group_from_clip_id(clip_id: str):
    match = re.match(
        r"^(video[^_]+)_(\d+)_((?:person).+)$",
        str(clip_id)
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

        if len(embeds) != INPUT_DIM:
            continue

        group = parse_group_from_clip_id(clip_id)

        if group is None:
            continue

        arr = np.asarray(embeds, dtype=np.float32)

        if not np.all(np.isfinite(arr)):
            continue

        clip_index_match = re.search(
            r"_(\d+)_",
            str(clip_id)
        )

        clip_index = (
            int(clip_index_match.group(1))
            if clip_index_match
            else 0
        )

        records.append(
            {
                "clip_id": str(clip_id),
                "group": group,
                "label": str(label),
                "label_id": LABEL_TO_ID[str(label)],
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
            f"Fold CSV missing required columns: "
            f"{sorted(missing)}"
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


# ============================================================
# SUPPORT / QUERY SAMPLING
# ============================================================

def sample_support_query(
    student_df,
    k_shot,
    protocol,
    rng,
):
    ordered = (
        student_df
        .sort_values("clip_index")
        .copy()
    )

    if len(ordered) <= k_shot:
        return (
            None,
            None,
            f"insufficient_clips_n={len(ordered)}_k={k_shot}",
        )

    if protocol == "early":

        support = ordered.iloc[:k_shot].copy()

    elif protocol == "random":

        sampled_idx = rng.choice(
            ordered.index.to_numpy(),
            size=k_shot,
            replace=False,
        )

        support = ordered.loc[
            sampled_idx
        ].copy()

    else:
        return (
            None,
            None,
            f"unknown_protocol_{protocol}",
        )

    query = ordered.loc[
        ~ordered.index.isin(support.index)
    ].copy()

    if query.empty:
        return (
            None,
            None,
            "empty_query",
        )

    return support, query, None


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_class_weights(train_df):

    counts = (
        train_df["label_id"]
        .value_counts()
        .reindex(
            range(NUM_CLASSES),
            fill_value=0,
        )
        .to_numpy(dtype=np.float32)
    )

    counts = np.maximum(counts, 1.0)

    weights = 1.0 / counts
    weights = weights / weights.mean()

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# CONTEXT ENCODER
# ============================================================

class ContextEncoder(nn.Module):

    def __init__(
        self,
        input_dim=INPUT_DIM,
        context_dim=CONTEXT_DIM,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),

            nn.Linear(256, context_dim),
            nn.ReLU(),
            nn.LayerNorm(context_dim),
        )

    def forward(self, support_x):

        encoded = self.net(support_x)

        context = encoded.mean(dim=0)

        return context


# ============================================================
# HYPERNETWORK
# ============================================================

class HyperNetwork(nn.Module):

    def __init__(
        self,
        context_dim=CONTEXT_DIM,
        hidden_dim=HYPER_HIDDEN,
        embed_dim=EMBED_DIM,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.output = nn.Linear(
            hidden_dim,
            HEAD_PARAM_DIM,
        )

    def forward(self, context):

        params = self.output(
            self.net(context)
        )

        weight_end = (
            NUM_CLASSES * self.embed_dim
        )

        weight = params[
            :weight_end
        ].view(
            NUM_CLASSES,
            self.embed_dim,
        )

        bias = params[
            weight_end:
        ]

        return weight, bias


# ============================================================
# QUERY ENCODER
# ============================================================

class QueryEncoder(nn.Module):

    def __init__(
        self,
        input_dim=INPUT_DIM,
        embed_dim=EMBED_DIM,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),

            nn.Linear(256, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# FULL MODEL
# ============================================================

class HyperNetworkModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.context_encoder = ContextEncoder()
        self.query_encoder = QueryEncoder()
        self.hypernetwork = HyperNetwork()

    def generate_student_head(
        self,
        support_x,
    ):

        context = self.context_encoder(
            support_x
        )

        weight, bias = self.hypernetwork(
            context
        )

        return weight, bias

    def predict(
        self,
        support_x,
        query_x,
    ):

        weight, bias = (
            self.generate_student_head(
                support_x
            )
        )

        query_emb = self.query_encoder(
            query_x
        )

        logits = F.linear(
            query_emb,
            weight,
            bias,
        )

        return logits


# ============================================================
# LOSSES
# ============================================================

def standard_cross_entropy(
    logits,
    targets,
):
    return F.cross_entropy(
        logits,
        targets,
    )


def class_balanced_focal_loss(
    logits,
    targets,
    class_weights,
    gamma=FOCAL_GAMMA,
):

    log_probs = F.log_softmax(
        logits,
        dim=1,
    )

    probs = torch.exp(log_probs)

    target_log_probs = log_probs.gather(
        1,
        targets.unsqueeze(1),
    ).squeeze(1)

    target_probs = probs.gather(
        1,
        targets.unsqueeze(1),
    ).squeeze(1)

    focal_factor = (
        1.0 - target_probs
    ).pow(gamma)

    weights = class_weights[
        targets
    ]

    loss = (
        -weights
        * focal_factor
        * target_log_probs
    )

    return loss.mean()


def ordinal_loss(
    logits,
    targets,
):

    probs = F.softmax(
        logits,
        dim=1,
    )

    cumulative_probs = torch.stack(
        [
            probs[:, 1:].sum(dim=1),
            probs[:, 2:].sum(dim=1),
            probs[:, 3:].sum(dim=1),
        ],
        dim=1,
    )

    thresholds = torch.stack(
        [
            (targets > 0).float(),
            (targets > 1).float(),
            (targets > 2).float(),
        ],
        dim=1,
    )

    cumulative_probs = torch.clamp(
        cumulative_probs,
        1e-6,
        1.0 - 1e-6,
    )

    return F.binary_cross_entropy(
        cumulative_probs,
        thresholds,
    )


# ============================================================
# H5 BALANCED QUERY SAMPLING
# ============================================================

def sample_balanced_query(
    query_df,
    per_class,
    rng,
):
    """
    H5 only.

    Attempts to sample an equal number of query examples
    from every class.

    If a class has fewer than per_class examples, the
    available examples are used.

    This changes ONLY the meta-training query sampling.
    Test evaluation remains completely unchanged.
    """

    pieces = []

    for class_id in range(NUM_CLASSES):

        class_df = query_df[
            query_df["label_id"] == class_id
        ]

        if class_df.empty:
            continue

        n = min(
            per_class,
            len(class_df)
        )

        selected = rng.choice(
            class_df.index.to_numpy(),
            size=n,
            replace=False,
        )

        pieces.append(
            class_df.loc[selected]
        )

    if not pieces:
        return query_df.iloc[0:0].copy()

    balanced = pd.concat(
        pieces,
        axis=0,
    )

    shuffled_idx = rng.permutation(
        balanced.index.to_numpy()
    )

    return balanced.loc[
        shuffled_idx
    ].copy()


# ============================================================
# EPISODIC TRAINING
# ============================================================

def episodic_train(
    model,
    train_df,
    scaler,
    episodes,
    seed,
    device,
    class_weights,
    experiment,
):
    """
    H1-H5 meta-training.

    H1:
        CE only.

    H2:
        Class-balanced focal loss.

    H3:
        CE + ordinal objective.

    H4:
        Class-balanced focal + ordinal.

    H5:
        Class-balanced focal + ordinal +
        class-balanced episodic query sampling.
    """

    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    rng = np.random.default_rng(seed)

    groups = sorted(
        train_df["group"].unique()
    )

    if len(groups) == 0:
        raise ValueError(
            "No training students available."
        )

    for episode in range(episodes):

        group = rng.choice(groups)

        student_df = train_df[
            train_df["group"] == group
        ].copy()

        k = int(
            rng.choice([5, 10, 20])
        )

        support_df, query_df, reason = (
            sample_support_query(
                student_df,
                k,
                "random",
                rng,
            )
        )

        if reason is not None:
            continue

        # ----------------------------------------------------
        # H5: class-balanced query sampling
        # ----------------------------------------------------

        if experiment == "H5":

            balanced_query = (
                sample_balanced_query(
                    query_df,
                    H5_QUERY_PER_CLASS,
                    rng,
                )
            )

            if len(balanced_query) > 0:
                query_df = balanced_query

        # ----------------------------------------------------
        # H1-H4: ordinary bounded query sampling
        # ----------------------------------------------------

        else:

            query_max = min(
                len(query_df),
                64,
            )

            if len(query_df) > query_max:

                selected = rng.choice(
                    query_df.index.to_numpy(),
                    size=query_max,
                    replace=False,
                )

                query_df = query_df.loc[
                    selected
                ]

        support_x = scaler.transform(
            support_df[
                FEATURE_COLS
            ].to_numpy(dtype=np.float32)
        )

        query_x = scaler.transform(
            query_df[
                FEATURE_COLS
            ].to_numpy(dtype=np.float32)
        )

        query_y = query_df[
            "label_id"
        ].to_numpy(dtype=np.int64)

        support_x = torch.tensor(
            support_x,
            dtype=torch.float32,
            device=device,
        )

        query_x = torch.tensor(
            query_x,
            dtype=torch.float32,
            device=device,
        )

        query_y = torch.tensor(
            query_y,
            dtype=torch.long,
            device=device,
        )

        logits = model.predict(
            support_x,
            query_x,
        )

        # ----------------------------------------------------
        # LOSS SELECTION
        # ----------------------------------------------------

        if experiment == "H1":

            loss = standard_cross_entropy(
                logits,
                query_y,
            )

        elif experiment == "H2":

            loss = class_balanced_focal_loss(
                logits,
                query_y,
                class_weights,
            )

        elif experiment == "H3":

            ce = standard_cross_entropy(
                logits,
                query_y,
            )

            ord_loss = ordinal_loss(
                logits,
                query_y,
            )

            loss = (
                ce
                + ORDINAL_WEIGHT * ord_loss
            )

        elif experiment == "H4":

            focal = class_balanced_focal_loss(
                logits,
                query_y,
                class_weights,
            )

            ord_loss = ordinal_loss(
                logits,
                query_y,
            )

            loss = (
                focal
                + ORDINAL_WEIGHT * ord_loss
            )

        elif experiment == "H5":

            focal = class_balanced_focal_loss(
                logits,
                query_y,
                class_weights,
            )

            ord_loss = ordinal_loss(
                logits,
                query_y,
            )

            loss = (
                focal
                + ORDINAL_WEIGHT * ord_loss
            )

        else:
            raise ValueError(
                f"Unknown experiment: {experiment}"
            )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        if (
            (episode + 1) % 100 == 0
        ):
            print(
                f"    {experiment} | Episode "
                f"{episode + 1}/{episodes} "
                f"loss={loss.item():.4f}"
            )

    return model


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
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


# ============================================================
# MAJORITY BASELINE
# ============================================================

def majority_prediction(y_query):

    counts = Counter(
        y_query.tolist()
    )

    majority_class = max(
        counts,
        key=counts.get,
    )

    return np.full(
        len(y_query),
        majority_class,
        dtype=np.int64,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_student(
    model,
    support_df,
    query_df,
    scaler,
    device,
):

    support_x = scaler.transform(
        support_df[
            FEATURE_COLS
        ].to_numpy(dtype=np.float32)
    )

    query_x = scaler.transform(
        query_df[
            FEATURE_COLS
        ].to_numpy(dtype=np.float32)
    )

    y_query = query_df[
        "label_id"
    ].to_numpy(dtype=np.int64)

    support_x = torch.tensor(
        support_x,
        dtype=torch.float32,
        device=device,
    )

    query_x = torch.tensor(
        query_x,
        dtype=torch.float32,
        device=device,
    )

    model.eval()

    with torch.no_grad():

        logits = model.predict(
            support_x,
            query_x,
        )

        y_pred = (
            torch.argmax(
                logits,
                dim=1,
            )
            .cpu()
            .numpy()
        )

    metrics = calculate_metrics(
        y_query,
        y_pred,
    )

    majority_pred = majority_prediction(
        y_query
    )

    majority_metrics = calculate_metrics(
        y_query,
        majority_pred,
    )

    return (
        y_query,
        y_pred,
        metrics,
        majority_metrics,
    )


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    clip_df,
    group_fold,
    k_values,
    seed,
    episodes,
    device,
    output_dir,
    experiment,
):

    eligible_groups = set(
        group_fold.keys()
    )

    eval_df = clip_df[
        clip_df["group"].isin(
            eligible_groups
        )
    ].copy()

    summary_rows = []
    per_student_rows = []
    per_class_rows = []
    confusion_rows = []
    skipped_rows = []

    fold_ids = sorted(
        set(group_fold.values())
    )

    for fold_id in fold_ids:

        print()
        print("=" * 80)
        print(
            f"{experiment} | OUTER FOLD {fold_id}"
        )
        print("=" * 80)

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

        train_df = eval_df[
            eval_df["group"].isin(
                train_groups
            )
        ].copy()

        test_df = eval_df[
            eval_df["group"].isin(
                test_groups
            )
        ].copy()

        overlap = (
            set(train_df["group"])
            &
            set(test_df["group"])
        )

        if overlap:
            raise RuntimeError(
                f"DATA LEAKAGE in fold "
                f"{fold_id}: {overlap}"
            )

        print(
            f"Training students: "
            f"{train_df['group'].nunique()}"
        )

        print(
            f"Test students: "
            f"{test_df['group'].nunique()}"
        )

        scaler = StandardScaler()

        X_train = train_df[
            FEATURE_COLS
        ].to_numpy(
            dtype=np.float32
        )

        scaler.fit(X_train)

        class_weights = (
            compute_class_weights(
                train_df
            )
            .to(device)
        )

        print(
            "Class weights:",
            class_weights.detach()
            .cpu()
            .numpy()
        )

        set_seed(
            seed
            + fold_id
            + sum(
                ord(c)
                for c in experiment
            )
        )

        model = HyperNetworkModel()

        model.to(device)

        print(
            f"{experiment}: Meta-training "
            f"for {episodes} episodes..."
        )

        episodic_train(
            model=model,
            train_df=train_df,
            scaler=scaler,
            episodes=episodes,
            seed=(
                seed
                + fold_id * 1000
                + sum(
                    ord(c)
                    for c in experiment
                )
            ),
            device=device,
            class_weights=class_weights,
            experiment=experiment,
        )

        # ----------------------------------------------------
        # Test students
        # ----------------------------------------------------

        for group in test_groups:

            student_df = test_df[
                test_df["group"] == group
            ].copy()

            if student_df.empty:
                continue

            for k in k_values:

                for protocol in [
                    "random",
                    "early",
                ]:

                    local_seed = (
                        seed
                        + fold_id * 10000
                        + k * 100
                        + sum(
                            ord(ch)
                            for ch in group
                        )
                        + (
                            1
                            if protocol == "early"
                            else 0
                        )
                    )

                    rng = np.random.default_rng(
                        local_seed
                    )

                    (
                        support_df,
                        query_df,
                        reason,
                    ) = sample_support_query(
                        student_df,
                        k,
                        protocol,
                        rng,
                    )

                    if reason is not None:

                        skipped_rows.append(
                            {
                                "experiment": experiment,
                                "outer_fold": fold_id,
                                "student_group": group,
                                "k_shot": k,
                                "support_protocol": protocol,
                                "reason": reason,
                            }
                        )

                        continue

                    (
                        y_true,
                        y_pred,
                        metrics,
                        majority_metrics,
                    ) = evaluate_student(
                        model,
                        support_df,
                        query_df,
                        scaler,
                        device,
                    )

                    per_student_rows.append(
                        {
                            "experiment": experiment,
                            "method": "hypernetwork",
                            "outer_fold": fold_id,
                            "student_group": group,
                            "k_shot": k,
                            "support_protocol": protocol,
                            "support_size": len(
                                support_df
                            ),
                            "query_size": len(
                                query_df
                            ),
                            "accuracy": metrics[
                                "accuracy"
                            ],
                            "balanced_accuracy":
                                metrics[
                                    "balanced_accuracy"
                                ],
                            "macro_f1": metrics[
                                "macro_f1"
                            ],
                            "majority_accuracy":
                                majority_metrics[
                                    "accuracy"
                                ],
                            "majority_balanced_accuracy":
                                majority_metrics[
                                    "balanced_accuracy"
                                ],
                            "majority_macro_f1":
                                majority_metrics[
                                    "macro_f1"
                                ],
                        }
                    )

                    for class_id, label in enumerate(
                        LABELS
                    ):

                        per_class_rows.append(
                            {
                                "experiment": experiment,
                                "method": "hypernetwork",
                                "outer_fold": fold_id,
                                "student_group": group,
                                "k_shot": k,
                                "support_protocol": protocol,
                                "class": label,
                                "precision": float(
                                    metrics[
                                        "precision"
                                    ][class_id]
                                ),
                                "recall": float(
                                    metrics[
                                        "recall"
                                    ][class_id]
                                ),
                                "f1": float(
                                    metrics[
                                        "f1"
                                    ][class_id]
                                ),
                                "support": int(
                                    metrics[
                                        "support"
                                    ][class_id]
                                ),
                            }
                        )

                    cm = metrics[
                        "confusion_matrix"
                    ]

                    for true_id in range(
                        NUM_CLASSES
                    ):

                        for pred_id in range(
                            NUM_CLASSES
                        ):

                            confusion_rows.append(
                                {
                                    "experiment": experiment,
                                    "method": "hypernetwork",
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

    # ========================================================
    # DATAFRAMES
    # ========================================================

    per_student_df = pd.DataFrame(
        per_student_rows
    )

    per_class_df = pd.DataFrame(
        per_class_rows
    )

    confusion_df = pd.DataFrame(
        confusion_rows
    )

    skipped_df = pd.DataFrame(
        skipped_rows
    )

    if per_student_df.empty:
        raise RuntimeError(
            f"No evaluations produced for {experiment}."
        )

    summary_df = (
        per_student_df
        .groupby(
            [
                "experiment",
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
        )
        .sort_values(
            [
                "support_protocol",
                "k_shot",
            ]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    experiment_dir = os.path.join(
        output_dir,
        experiment.lower(),
    )

    os.makedirs(
        experiment_dir,
        exist_ok=True,
    )

    per_student_path = os.path.join(
        experiment_dir,
        f"{experiment.lower()}_per_student.csv",
    )

    summary_path = os.path.join(
        experiment_dir,
        f"{experiment.lower()}_summary.csv",
    )

    per_class_path = os.path.join(
        experiment_dir,
        f"{experiment.lower()}_per_class.csv",
    )

    confusion_path = os.path.join(
        experiment_dir,
        f"{experiment.lower()}_confusion_matrix.csv",
    )

    skipped_path = os.path.join(
        experiment_dir,
        f"{experiment.lower()}_skipped.csv",
    )

    per_student_df.to_csv(
        per_student_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    per_class_df.to_csv(
        per_class_path,
        index=False,
    )

    confusion_df.to_csv(
        confusion_path,
        index=False,
    )

    skipped_df.to_csv(
        skipped_path,
        index=False,
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 80)
    print(f"{experiment} RESULTS")
    print("=" * 80)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print(f"{experiment} PER-CLASS RESULTS")
    print("=" * 80)

    class_summary = (
        per_class_df
        .groupby(
            [
                "experiment",
                "support_protocol",
                "k_shot",
                "class",
            ],
            as_index=False,
        )
        .agg(
            mean_precision=(
                "precision",
                "mean",
            ),
            mean_recall=(
                "recall",
                "mean",
            ),
            mean_f1=(
                "f1",
                "mean",
            ),
            total_support=(
                "support",
                "sum",
            ),
        )
    )

    print(
        class_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print(f"{experiment} SKIPPED CASES")
    print("=" * 80)

    if skipped_df.empty:
        print("None.")
    else:
        print(
            skipped_df[
                "reason"
            ].value_counts()
        )

    print()
    print("Saved:")
    print(per_student_path)
    print(summary_path)
    print(per_class_path)
    print(confusion_path)
    print(skipped_path)

    return (
        per_student_df,
        summary_df,
        per_class_df,
        confusion_df,
        skipped_df,
    )


# ============================================================
# RUN ALL H1-H5
# ============================================================

def run_all_experiments(
    clip_df,
    group_fold,
    k_values,
    seed,
    episodes,
    device,
    output_dir,
):

    all_summary = []
    all_per_student = []
    all_per_class = []
    all_confusion = []
    all_skipped = []

    experiments = [
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
    ]

    for experiment in experiments:

        print()
        print("#" * 80)
        print(
            f"STARTING {experiment}"
        )
        print("#" * 80)

        (
            per_student_df,
            summary_df,
            per_class_df,
            confusion_df,
            skipped_df,
        ) = run_experiment(
            clip_df=clip_df,
            group_fold=group_fold,
            k_values=k_values,
            seed=seed,
            episodes=episodes,
            device=device,
            output_dir=output_dir,
            experiment=experiment,
        )

        all_summary.append(
            summary_df
        )

        all_per_student.append(
            per_student_df
        )

        all_per_class.append(
            per_class_df
        )

        all_confusion.append(
            confusion_df
        )

        if not skipped_df.empty:
            all_skipped.append(
                skipped_df
            )

    # ========================================================
    # COMBINED OUTPUTS
    # ========================================================

    combined_summary = pd.concat(
        all_summary,
        ignore_index=True,
    )

    combined_per_student = pd.concat(
        all_per_student,
        ignore_index=True,
    )

    combined_per_class = pd.concat(
        all_per_class,
        ignore_index=True,
    )

    combined_confusion = pd.concat(
        all_confusion,
        ignore_index=True,
    )

    if all_skipped:
        combined_skipped = pd.concat(
            all_skipped,
            ignore_index=True,
        )
    else:
        combined_skipped = pd.DataFrame()

    combined_summary.to_csv(
        os.path.join(
            output_dir,
            "hypernetwork_H1_H5_summary.csv",
        ),
        index=False,
    )

    combined_per_student.to_csv(
        os.path.join(
            output_dir,
            "hypernetwork_H1_H5_per_student.csv",
        ),
        index=False,
    )

    combined_per_class.to_csv(
        os.path.join(
            output_dir,
            "hypernetwork_H1_H5_per_class.csv",
        ),
        index=False,
    )

    combined_confusion.to_csv(
        os.path.join(
            output_dir,
            "hypernetwork_H1_H5_confusion_matrix.csv",
        ),
        index=False,
    )

    if not combined_skipped.empty:

        combined_skipped.to_csv(
            os.path.join(
                output_dir,
                "hypernetwork_H1_H5_skipped.csv",
            ),
            index=False,
        )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print()
    print("=" * 100)
    print("H1-H5 FINAL COMPARISON")
    print("=" * 100)

    comparison = (
        combined_summary[
            [
                "experiment",
                "support_protocol",
                "k_shot",
                "mean_accuracy",
                "mean_balanced_accuracy",
                "mean_macro_f1",
            ]
        ]
        .sort_values(
            [
                "support_protocol",
                "k_shot",
                "experiment",
            ]
        )
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("BEST CONFIGURATION BY BALANCED ACCURACY")
    print("=" * 100)

    best_balanced = (
        combined_summary
        .sort_values(
            "mean_balanced_accuracy",
            ascending=False,
        )
        .head(10)
    )

    print(
        best_balanced[
            [
                "experiment",
                "support_protocol",
                "k_shot",
                "mean_balanced_accuracy",
                "mean_macro_f1",
                "mean_accuracy",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 100)
    print("BEST CONFIGURATION BY MACRO-F1")
    print("=" * 100)

    best_f1 = (
        combined_summary
        .sort_values(
            "mean_macro_f1",
            ascending=False,
        )
        .head(10)
    )

    print(
        best_f1[
            [
                "experiment",
                "support_protocol",
                "k_shot",
                "mean_macro_f1",
                "mean_balanced_accuracy",
                "mean_accuracy",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "Combined H1-H5 files saved to:"
    )
    print(
        output_dir
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "H1-H5 HyperNetwork ablation for "
            "few-shot unseen-student engagement "
            "prediction."
        )
    )

    parser.add_argument(
        "--json",
        type=str,
        default=(
            os.path.join(os.environ.get("CMOSE_DATA_DIR", "."), "final_data_1.json")
        ),
    )

    parser.add_argument(
        "--folds",
        type=str,
        default=(
            os.path.join(os.environ.get("CMOSE_DATA_DIR", "."), "cmose_h1_h2_fold_protocol.csv")
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
            "Meta-training episodes per outer fold "
            "for EACH H1-H5 experiment."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            os.environ.get("CMOSE_OUTPUT_DIR", ".")
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if args.device == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but no CUDA "
                "device is available."
            )

        device = torch.device(
            "cuda"
        )

    elif args.device == "cpu":

        device = torch.device(
            "cpu"
        )

    else:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print(
        f"Using device: {device}"
    )

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(args.seed)

    # --------------------------------------------------------
    # K values
    # --------------------------------------------------------

    k_values = [
        int(v.strip())
        for v in args.k_values.split(",")
        if v.strip()
    ]

    if not k_values:
        raise ValueError(
            "No K values supplied."
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print(
        "Loading clip data..."
    )

    clip_df = load_clip_dataframe(
        args.json
    )

    print(
        f"Loaded "
        f"{len(clip_df)} labelled clips "
        f"from "
        f"{clip_df['group'].nunique()} "
        f"students."
    )

    print()
    print(
        clip_df[
            "label"
        ].value_counts()
    )

    print(
        "Loading fold protocol..."
    )

    group_fold = load_group_fold_map(
        args.folds
    )

    print(
        f"Fold protocol contains "
        f"{len(group_fold)} students."
    )

    # --------------------------------------------------------
    # Run H1-H5
    # --------------------------------------------------------

    run_all_experiments(
        clip_df=clip_df,
        group_fold=group_fold,
        k_values=k_values,
        seed=args.seed,
        episodes=args.episodes,
        device=device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
