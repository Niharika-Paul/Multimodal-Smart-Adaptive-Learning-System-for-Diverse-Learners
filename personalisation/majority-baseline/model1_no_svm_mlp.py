# ============================================================
# CMOSE H1 / H2 MAIN MULTI-MODEL EXPERIMENT
# ============================================================
#
# PURPOSE
# -------
# Clean main experiment for evaluating whether increasing
# participant-specific adaptation information improves
# prediction of future engagement.
#
# THIS SCRIPT RUNS ONLY:
#
#   1. Majority baseline
#   2. Logistic Regression
#   3. Random Forest
#   4. Extra Trees
#
# ADAPTATION BUDGETS:
#
#   0, 30, 60, 120, 180, 300 seconds
#
# EVALUATION:
#
#   5 group-disjoint outer folds
#
# IMPORTANT
# ---------
# This is a GROUP-LEVEL experiment.
#
# Each row in the representation table represents:
#
#       group × adaptation budget
#
# Therefore the unit of evaluation is the group/session,
# NOT individual future clips.
#
# IMPORTANT
# ---------
# This script DOES NOT run:
#
#   - Linear SVM
#   - MLP
#   - future clip-level prediction
#   - future distribution prediction
#   - CMOSE
#   - meta-learning
#
# Those will be handled separately.
#
# ============================================================

import os
import re
import warnings
import traceback

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = os.environ.get("CMOSE_DATA_DIR", ".")

REPRESENTATION_FILE = os.path.join(
    BASE_DIR,
    "cmose_modeling_group_budget_representations.csv"
)

FUTURE_CLIP_FILE = os.path.join(
    BASE_DIR,
    "cmose_modeling_future_clip_table.csv"
)

FOLD_PROTOCOL_FILE = os.path.join(
    BASE_DIR,
    "cmose_h1_h2_fold_protocol.csv"
)

OUTPUT_FOLD_RESULTS = os.path.join(
    BASE_DIR,
    "cmose_h1_h2_main_model_fold_results.csv"
)

OUTPUT_SUMMARY = os.path.join(
    BASE_DIR,
    "cmose_h1_h2_main_model_summary.csv"
)

OUTPUT_AUDIT = os.path.join(
    BASE_DIR,
    "cmose_h1_h2_main_model_run_audit.csv"
)

BUDGETS = [0, 30, 60, 120, 180, 300]

FOLDS = [1, 2, 3, 4, 5]

EXPECTED_EMBEDDING_DIM = 1024

RANDOM_STATE = 42

LABELS = [
    "Highly Disengage",
    "Disengage",
    "Engage",
    "Highly Engage",
]


# ============================================================
# 2. UTILITIES
# ============================================================

def section(title):
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


def ok(message):
    print(f"[PASS] {message}")


def warning(message):
    print(f"[WARNING] {message}")


def fail(message):
    raise ValueError(message)


def require_file(path):
    if not os.path.exists(path):
        fail(
            "Required file does not exist:\n"
            + path
        )


# ============================================================
# 3. START
# ============================================================

section(
    "CMOSE H1 / H2 CLEAN MAIN MULTI-MODEL EXPERIMENT"
)

print()
print("THIS IS THE CLEAN MAIN RUN.")
print()
print("Models:")
print("  - Majority")
print("  - LogisticRegression")
print("  - RandomForest")
print("  - ExtraTrees")
print()
print("NOT included:")
print("  - LinearSVM")
print("  - MLP")
print("  - Future clip-level prediction")
print("  - Distribution prediction")
print("  - CMOSE")
print("  - Meta-learning")
print()
print("Budgets:")
print(" ", BUDGETS)
print()
print("Outer folds:")
print(" ", FOLDS)
print()
print("Evaluation unit:")
print("  group × adaptation budget")
print()
print(
    "The script will terminate after the main results are saved."
)


# ============================================================
# 4. INPUT FILE CHECK
# ============================================================

section("1. CHECKING INPUT FILES")

require_file(REPRESENTATION_FILE)
require_file(FUTURE_CLIP_FILE)
require_file(FOLD_PROTOCOL_FILE)

print("Representation file:")
print(f"  {REPRESENTATION_FILE}")

print("Future clip file:")
print(f"  {FUTURE_CLIP_FILE}")

print("Fold protocol:")
print(f"  {FOLD_PROTOCOL_FILE}")

ok("All required input files exist.")


# ============================================================
# 5. LOAD REPRESENTATION TABLE
# ============================================================

section("2. LOADING GROUP × BUDGET REPRESENTATIONS")

representations = pd.read_csv(
    REPRESENTATION_FILE
)

print(
    f"Rows loaded: {len(representations)}"
)

print(
    f"Columns loaded: {len(representations.columns)}"
)

required_rep_columns = {
    "group",
    "fold",
    "adaptation_budget_sec",
    "adaptation_representation_available",
    "adaptation_valid_embeddings",
    "adaptation_invalid_embeddings",
    "adaptation_embedding_coverage",
}

missing = (
    required_rep_columns
    - set(representations.columns)
)

if missing:
    fail(
        "Representation table is missing columns:\n"
        + "\n".join(sorted(missing))
    )

ok("Required representation columns present.")


# ============================================================
# 6. VERIFY BUDGETS
# ============================================================

section("3. VERIFYING ADAPTATION BUDGETS")

found_budgets = sorted(
    pd.to_numeric(
        representations["adaptation_budget_sec"],
        errors="coerce"
    )
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

print(
    "Found budgets:"
)

print(found_budgets)

print()
print(
    "Expected budgets:"
)

print(BUDGETS)

if found_budgets != BUDGETS:
    fail(
        "Budget mismatch.\n"
        f"Found: {found_budgets}\n"
        f"Expected: {BUDGETS}"
    )

ok("All expected adaptation budgets are present.")


# ============================================================
# 7. IDENTIFY 1024-D ADAPTATION REPRESENTATION
# ============================================================

section(
    "4. IDENTIFYING 1024-D ADAPTATION REPRESENTATION"
)

pattern = re.compile(
    r"^adapt_emb_(\d+)$"
)

adapt_pairs = []

for column in representations.columns:

    match = pattern.match(
        str(column)
    )

    if match:

        index = int(
            match.group(1)
        )

        adapt_pairs.append(
            (index, column)
        )

adapt_pairs = sorted(
    adapt_pairs,
    key=lambda x: x[0]
)

adapt_indices = [
    x[0]
    for x in adapt_pairs
]

adapt_columns = [
    x[1]
    for x in adapt_pairs
]

print(
    f"Adaptation embedding columns: "
    f"{len(adapt_columns)}"
)

if len(adapt_columns) != EXPECTED_EMBEDDING_DIM:
    fail(
        f"Expected exactly "
        f"{EXPECTED_EMBEDDING_DIM} embedding columns, "
        f"found {len(adapt_columns)}."
    )

expected_indices = list(
    range(EXPECTED_EMBEDDING_DIM)
)

if adapt_indices != expected_indices:
    fail(
        "Embedding columns are not exactly "
        "adapt_emb_0 ... adapt_emb_1023."
    )

ok(
    "Exactly 1024 adaptation dimensions found."
)


# ============================================================
# 8. LOAD FUTURE CLIP TABLE
# ============================================================

section("5. LOADING FUTURE CLIP TABLE")

future_clips = pd.read_csv(
    FUTURE_CLIP_FILE
)

print(
    f"Future clip rows: "
    f"{len(future_clips)}"
)

print(
    f"Future clip columns: "
    f"{len(future_clips.columns)}"
)

required_future_columns = {
    "group",
    "label",
}

missing = (
    required_future_columns
    - set(future_clips.columns)
)

if missing:
    fail(
        "Future clip table is missing:\n"
        + "\n".join(sorted(missing))
    )

unknown_labels = sorted(
    set(
        future_clips["label"]
        .dropna()
        .unique()
    )
    - set(LABELS)
)

if unknown_labels:
    fail(
        "Unexpected labels found:\n"
        + "\n".join(
            map(str, unknown_labels)
        )
    )

ok("Future clip table verified.")


# ============================================================
# 9. LOAD FOLD PROTOCOL
# ============================================================

section("6. LOADING OUTER FOLD PROTOCOL")

fold_protocol = pd.read_csv(
    FOLD_PROTOCOL_FILE
)

print(
    f"Fold protocol rows: "
    f"{len(fold_protocol)}"
)

required_fold_columns = {
    "group",
    "fold",
}

missing = (
    required_fold_columns
    - set(fold_protocol.columns)
)

if missing:
    fail(
        "Fold protocol is missing:\n"
        + "\n".join(sorted(missing))
    )

fold_values = sorted(
    pd.to_numeric(
        fold_protocol["fold"],
        errors="coerce"
    )
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

print(
    "Fold values:"
)

print(fold_values)

if fold_values != FOLDS:
    fail(
        f"Expected folds {FOLDS}, "
        f"found {fold_values}"
    )

ok("Five outer folds verified.")


# ============================================================
# 10. BUILD GROUP → FOLD MAP
# ============================================================

section(
    "7. VERIFYING GROUP-DISJOINT OUTER FOLDS"
)

group_fold = (
    fold_protocol[
        ["group", "fold"]
    ]
    .drop_duplicates()
)

group_fold_counts = (
    group_fold
    .groupby("group")["fold"]
    .nunique()
)

bad_groups = group_fold_counts[
    group_fold_counts != 1
]

if len(bad_groups) > 0:

    fail(
        "Some groups have multiple fold assignments:\n"
        + str(bad_groups)
    )

GROUP_TO_FOLD = dict(
    zip(
        group_fold["group"],
        group_fold["fold"].astype(int)
    )
)

print(
    f"Unique groups: "
    f"{len(GROUP_TO_FOLD)}"
)

print()
print("Groups per fold:")

for fold in FOLDS:

    n = (
        group_fold[
            group_fold["fold"] == fold
        ]["group"]
        .nunique()
    )

    print(
        f"  Fold {fold}: {n}"
    )

ok(
    "Every group belongs to exactly one outer fold."
)


# ============================================================
# 11. VERIFY REPRESENTATION GROUPS
# ============================================================

section(
    "8. VERIFYING REPRESENTATION GROUP POPULATION"
)

representation_groups = set(
    representations["group"]
    .dropna()
    .unique()
)

fold_groups = set(
    GROUP_TO_FOLD.keys()
)

print(
    f"Representation groups: "
    f"{len(representation_groups)}"
)

print(
    f"Fold protocol groups:  "
    f"{len(fold_groups)}"
)

missing_groups = (
    representation_groups
    - fold_groups
)

if missing_groups:
    fail(
        "Representation groups without fold assignments:\n"
        + str(sorted(missing_groups))
    )

ok(
    "All representation groups have outer-fold assignments."
)


# ============================================================
# 12. VERIFY GROUP × BUDGET UNIQUENESS
# ============================================================

section(
    "9. VERIFYING GROUP × BUDGET STRUCTURE"
)

representations["adaptation_budget_sec"] = (
    pd.to_numeric(
        representations[
            "adaptation_budget_sec"
        ],
        errors="coerce"
    )
    .astype(int)
)

duplicate_pairs = representations.duplicated(
    subset=[
        "group",
        "adaptation_budget_sec"
    ],
    keep=False
)

if duplicate_pairs.any():

    print(
        representations[
            duplicate_pairs
        ][
            [
                "group",
                "adaptation_budget_sec"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    fail(
        "Duplicate group × budget representation rows detected."
    )

n_groups = len(
    representation_groups
)

expected_rows = (
    n_groups
    * len(BUDGETS)
)

print(
    f"Groups:          {n_groups}"
)

print(
    f"Budgets:         {len(BUDGETS)}"
)

print(
    f"Expected rows:   {expected_rows}"
)

print(
    f"Actual rows:     {len(representations)}"
)

if len(representations) != expected_rows:
    fail(
        "Representation table does not contain exactly "
        "one row for every group × budget combination."
    )

ok(
    "Complete group × budget representation structure verified."
)


# ============================================================
# 13. ATTACH OUTER FOLD
# ============================================================

section(
    "10. ATTACHING VERIFIED OUTER FOLDS"
)

representations["outer_fold"] = (
    representations["group"]
    .map(GROUP_TO_FOLD)
)

if representations["outer_fold"].isna().any():

    fail(
        "Some representation rows have no outer fold."
    )

representations["outer_fold"] = (
    representations["outer_fold"]
    .astype(int)
)

ok(
    "Outer fold attached to all representation rows."
)


# ============================================================
# 14. CONSTRUCT GROUP-LEVEL FUTURE TARGET
# ============================================================
#
# This is important.
#
# The main experiment is GROUP LEVEL.
#
# Therefore we aggregate future clips within each group
# to construct the future engagement target.
#
# The model does NOT train on 65,762 future clips.
#
# It trains on the group × budget representation rows.
#
# For each group, the future target is:
#
#       dominant future engagement class
#
# based on that group's future clips.
#
# ============================================================

section(
    "11. CONSTRUCTING GROUP-LEVEL FUTURE TARGET"
)

future_group_counts = (
    future_clips
    .groupby(
        ["group", "label"]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

future_group_counts = (
    future_group_counts
    .reindex(
        columns=LABELS,
        fill_value=0
    )
)

future_group_total = (
    future_group_counts
    .sum(axis=1)
)

future_group_target = (
    future_group_counts
    .idxmax(axis=1)
)

future_targets = pd.DataFrame({
    "group": future_group_counts.index,
    "future_dominant_class": (
        future_group_target.values
    ),
    "future_total_clips": (
        future_group_total.values
    ),
})

print(
    f"Groups with future targets: "
    f"{len(future_targets)}"
)

print()
print("Future dominant-class distribution:")

print(
    future_targets[
        "future_dominant_class"
    ]
    .value_counts()
    .reindex(
        LABELS,
        fill_value=0
    )
)

missing_target_groups = (
    representation_groups
    - set(future_targets["group"])
)

if missing_target_groups:

    fail(
        "Some representation groups have no future target:\n"
        + str(
            sorted(
                missing_target_groups
            )
        )
    )

ok(
    "Every modeling group has a future target."
)


# ============================================================
# 15. MERGE TARGET INTO REPRESENTATIONS
# ============================================================

section(
    "12. BUILDING FINAL GROUP-LEVEL EXPERIMENT TABLE"
)

data = representations.merge(
    future_targets,
    on="group",
    how="left",
    validate="many_to_one"
)

print(
    f"Final experiment rows: "
    f"{len(data)}"
)

if len(data) != len(representations):
    fail(
        "Target merge changed representation row count."
    )

if data[
    "future_dominant_class"
].isna().any():

    fail(
        "Some experiment rows have no future target."
    )

ok(
    "Final group-level experiment table constructed."
)


# ============================================================
# 16. CHECK FEATURES
# ============================================================

section(
    "13. CHECKING ADAPTATION FEATURES"
)

X_all = data[
    adapt_columns
].apply(
    pd.to_numeric,
    errors="coerce"
)

print(
    f"Feature matrix: "
    f"{X_all.shape}"
)

X_numpy = X_all.to_numpy(
    dtype=float
)

nan_count = np.isnan(
    X_numpy
).sum()

inf_count = np.isinf(
    X_numpy
).sum()

print(
    f"NaN values: {nan_count}"
)

print(
    f"Infinite values: {inf_count}"
)

ok(
    "Feature matrix checked."
)

print()
print(
    "Missing feature values, if any, will be handled "
    "inside the training-fold pipeline."
)


# ============================================================
# 17. MODEL DEFINITIONS
# ============================================================

section(
    "14. DEFINING MAIN MODELS"
)

MODELS = {

    "LogisticRegression": Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1
            )
        ),
    ]),

    "RandomForest": Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        ),
    ]),

    "ExtraTrees": Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "model",
            ExtraTreesClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        ),
    ]),
}

print()
print(
    "Main ML models:"
)

for name in MODELS:
    print(
        f"  - {name}"
    )

print()
print(
    "Plus:"
)

print(
    "  - MajorityClass"
)

print()
print(
    "Total models:"
)

print(
    len(MODELS) + 1
)


# ============================================================
# 18. METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_prob=None
):

    metrics = {}

    metrics[
        "accuracy"
    ] = accuracy_score(
        y_true,
        y_pred
    )

    metrics[
        "balanced_accuracy"
    ] = balanced_accuracy_score(
        y_true,
        y_pred
    )

    metrics[
        "macro_f1"
    ] = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    metrics[
        "weighted_f1"
    ] = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="weighted",
        zero_division=0
    )

    if y_prob is not None:

        try:

            metrics[
                "log_loss"
            ] = log_loss(
                y_true,
                y_prob,
                labels=LABELS
            )

        except Exception:

            metrics[
                "log_loss"
            ] = np.nan

    else:

        metrics[
            "log_loss"
        ] = np.nan

    return metrics


# ============================================================
# 19. MAJORITY BASELINE
# ============================================================

def majority_predictions(
    y_train,
    n_test
):

    counts = (
        pd.Series(y_train)
        .value_counts()
        .reindex(
            LABELS,
            fill_value=0
        )
        .astype(float)
    )

    probabilities = (
        counts.values
        / counts.sum()
    )

    dominant_index = int(
        np.argmax(
            probabilities
        )
    )

    dominant_class = LABELS[
        dominant_index
    ]

    predictions = np.repeat(
        dominant_class,
        n_test
    )

    probabilities = np.tile(
        probabilities,
        (
            n_test,
            1
        )
    )

    return (
        predictions,
        probabilities
    )


# ============================================================
# 20. MAIN EXPERIMENT
# ============================================================

section(
    "15. STARTING MAIN OUTER-FOLD EXPERIMENT"
)

print()
print(
    "Total conditions:"
)

print(
    f"  {len(BUDGETS)} budgets × "
    f"{len(FOLDS)} folds × "
    f"{len(MODELS) + 1} models"
)

print(
    f"  = "
    f"{len(BUDGETS) * len(FOLDS) * (len(MODELS) + 1)}"
)

print()
print(
    "There will be NO clip-level analysis after this."
)

print(
    "There will be NO SVM after this."
)

print(
    "There will be NO MLP after this."
)


classification_results = []

audit_results = []

progress_counter = 0

total_conditions = (
    len(BUDGETS)
    * len(FOLDS)
    * (len(MODELS) + 1)
)


# ============================================================
# LOOP OVER BUDGETS
# ============================================================

for budget in BUDGETS:

    section(
        f"BUDGET: {budget} SEC"
    )

    budget_data = data[
        data[
            "adaptation_budget_sec"
        ] == budget
    ].copy()

    print(
        f"Group rows: "
        f"{len(budget_data)}"
    )

    if len(budget_data) != n_groups:

        fail(
            f"Budget {budget}: expected "
            f"{n_groups} groups, found "
            f"{len(budget_data)}."
        )

    # --------------------------------------------------------
    # BUDGET 0
    # --------------------------------------------------------

    if budget == 0:

        print()
        print(
            "0 sec has no adaptation representation."
        )

        print(
            "Only MajorityClass baseline will be evaluated."
        )

    # --------------------------------------------------------
    # FOLDS
    # --------------------------------------------------------

    for fold in FOLDS:

        print()
        print(
            "-" * 60
        )

        print(
            f"BUDGET {budget} | "
            f"FOLD {fold}"
        )

        print(
            "-" * 60
        )

        train = budget_data[
            budget_data[
                "outer_fold"
            ] != fold
        ].copy()

        test = budget_data[
            budget_data[
                "outer_fold"
            ] == fold
        ].copy()

        train_groups = set(
            train["group"]
        )

        test_groups = set(
            test["group"]
        )

        print(
            f"Training groups: "
            f"{len(train_groups)}"
        )

        print(
            f"Test groups:     "
            f"{len(test_groups)}"
        )

        # ----------------------------------------------------
        # LEAKAGE CHECK
        # ----------------------------------------------------

        overlap = (
            train_groups
            & test_groups
        )

        print(
            f"Group overlap:   "
            f"{len(overlap)}"
        )

        if overlap:

            fail(
                f"GROUP LEAKAGE DETECTED "
                f"at budget {budget}, fold {fold}:\n"
                + str(
                    sorted(overlap)
                )
            )

        ok(
            "Training/test groups are disjoint."
        )

        y_train = (
            train[
                "future_dominant_class"
            ]
            .astype(str)
            .values
        )

        y_test = (
            test[
                "future_dominant_class"
            ]
            .astype(str)
            .values
        )

        print()
        print(
            "Training target distribution:"
        )

        print(
            pd.Series(
                y_train
            )
            .value_counts()
            .reindex(
                LABELS,
                fill_value=0
            )
            .to_string()
        )

        print()
        print(
            "Test target distribution:"
        )

        print(
            pd.Series(
                y_test
            )
            .value_counts()
            .reindex(
                LABELS,
                fill_value=0
            )
            .to_string()
        )

        # ====================================================
        # MAJORITY BASELINE
        # ====================================================

        progress_counter += 1

        print()
        print(
            f"[{progress_counter}/"
            f"{total_conditions}] "
            f"MajorityClass"
        )

        try:

            y_pred, y_prob = (
                majority_predictions(
                    y_train,
                    len(y_test)
                )
            )

            metrics = calculate_metrics(
                y_test,
                y_pred,
                y_prob
            )

            classification_results.append({

                "budget_sec": budget,

                "fold": fold,

                "model": "MajorityClass",

                "n_train_groups":
                    len(train_groups),

                "n_test_groups":
                    len(test_groups),

                "n_train_rows":
                    len(train),

                "n_test_rows":
                    len(test),

                **metrics

            })

            audit_results.append({

                "budget_sec": budget,

                "fold": fold,

                "model": "MajorityClass",

                "status":
                    "completed",

                "n_train_groups":
                    len(train_groups),

                "n_test_groups":
                    len(test_groups),

            })

            print(
                f"  Accuracy: "
                f"{metrics['accuracy']:.4f}"
            )

            print(
                f"  Balanced accuracy: "
                f"{metrics['balanced_accuracy']:.4f}"
            )

            print(
                f"  Macro F1: "
                f"{metrics['macro_f1']:.4f}"
            )

        except Exception as exc:

            print(
                "[ERROR] Majority baseline failed:"
            )

            print(
                repr(exc)
            )

            audit_results.append({

                "budget_sec": budget,

                "fold": fold,

                "model": "MajorityClass",

                "status":
                    "error",

                "error":
                    str(exc)

            })

        # ====================================================
        # BUDGET 0 STOPS HERE
        # ====================================================

        if budget == 0:

            continue

        # ====================================================
        # EMBEDDING AVAILABILITY
        # ====================================================

        train_available = (
            train[
                "adaptation_representation_available"
            ]
            .astype(bool)
        )

        test_available = (
            test[
                "adaptation_representation_available"
            ]
            .astype(bool)
        )

        train_model = train[
            train_available
        ].copy()

        test_model = test[
            test_available
        ].copy()

        print()
        print(
            "Representation availability:"
        )

        print(
            f"  Training: "
            f"{len(train_model)} / "
            f"{len(train)}"
        )

        print(
            f"  Test:     "
            f"{len(test_model)} / "
            f"{len(test)}"
        )

        if len(train_model) == 0:

            warning(
                "No training representations. "
                "Skipping ML models."
            )

            continue

        if len(test_model) == 0:

            warning(
                "No test representations. "
                "Skipping ML models."
            )

            continue

        X_train = train_model[
            adapt_columns
        ].apply(
            pd.to_numeric,
            errors="coerce"
        )

        X_test = test_model[
            adapt_columns
        ].apply(
            pd.to_numeric,
            errors="coerce"
        )

        y_train_model = (
            train_model[
                "future_dominant_class"
            ]
            .astype(str)
            .values
        )

        y_test_model = (
            test_model[
                "future_dominant_class"
            ]
            .astype(str)
            .values
        )

        training_classes = sorted(
            set(
                y_train_model
            )
        )

        print()
        print(
            "Training classes:"
        )

        print(
            training_classes
        )

        if len(training_classes) < 2:

            warning(
                "Only one training class exists. "
                "Skipping ML models."
            )

            for model_name in MODELS:

                progress_counter += 1

                audit_results.append({

                    "budget_sec":
                        budget,

                    "fold":
                        fold,

                    "model":
                        model_name,

                    "status":
                        "skipped_single_class",

                    "n_train_groups":
                        len(train_groups),

                    "n_test_groups":
                        len(test_groups),

                })

            continue

        # ====================================================
        # RUN THREE ML MODELS
        # ====================================================

        for model_name, model_template in MODELS.items():

            progress_counter += 1

            print()
            print(
                "=" * 60
            )

            print(
                f"[{progress_counter}/"
                f"{total_conditions}] "
                f"STARTING {model_name}"
            )

            print(
                f"Budget: {budget} sec"
            )

            print(
                f"Fold:   {fold}"
            )

            print(
                f"Train:  {X_train.shape}"
            )

            print(
                f"Test:   {X_test.shape}"
            )

            print(
                "=" * 60
            )

            model = clone(
                model_template
            )

            try:

                model.fit(
                    X_train,
                    y_train_model
                )

                print(
                    f"[PASS] "
                    f"{model_name} fit complete."
                )

                y_pred = model.predict(
                    X_test
                )

                print(
                    f"[PASS] "
                    f"{model_name} prediction complete."
                )

                y_prob = None

                if hasattr(
                    model,
                    "predict_proba"
                ):

                    raw_prob = (
                        model.predict_proba(
                            X_test
                        )
                    )

                    y_prob = np.zeros(
                        (
                            len(X_test),
                            len(LABELS)
                        ),
                        dtype=float
                    )

                    model_classes = list(
                        model.classes_
                    )

                    for j, class_name in enumerate(
                        model_classes
                    ):

                        if class_name in LABELS:

                            target_index = (
                                LABELS.index(
                                    class_name
                                )
                            )

                            y_prob[
                                :,
                                target_index
                            ] = raw_prob[
                                :,
                                j
                            ]

                    row_sums = (
                        y_prob.sum(
                            axis=1
                        )
                    )

                    valid_rows = (
                        row_sums > 0
                    )

                    y_prob[
                        valid_rows
                    ] = (
                        y_prob[
                            valid_rows
                        ]
                        /
                        row_sums[
                            valid_rows,
                            None
                        ]
                    )

                metrics = calculate_metrics(
                    y_test_model,
                    y_pred,
                    y_prob
                )

                classification_results.append({

                    "budget_sec":
                        budget,

                    "fold":
                        fold,

                    "model":
                        model_name,

                    "n_train_groups":
                        len(train_groups),

                    "n_test_groups":
                        len(test_groups),

                    "n_train_rows":
                        len(train_model),

                    "n_test_rows":
                        len(test_model),

                    "excluded_train_rows":
                        len(train)
                        - len(train_model),

                    "excluded_test_rows":
                        len(test)
                        - len(test_model),

                    **metrics

                })

                audit_results.append({

                    "budget_sec":
                        budget,

                    "fold":
                        fold,

                    "model":
                        model_name,

                    "status":
                        "completed",

                    "n_train_groups":
                        len(train_groups),

                    "n_test_groups":
                        len(test_groups),

                    "n_train_rows":
                        len(train_model),

                    "n_test_rows":
                        len(test_model),

                })

                print()
                print(
                    f"{model_name} RESULTS"
                )

                print(
                    f"  Accuracy: "
                    f"{metrics['accuracy']:.4f}"
                )

                print(
                    f"  Balanced accuracy: "
                    f"{metrics['balanced_accuracy']:.4f}"
                )

                print(
                    f"  Macro F1: "
                    f"{metrics['macro_f1']:.4f}"
                )

                print(
                    f"  Weighted F1: "
                    f"{metrics['weighted_f1']:.4f}"
                )

                if not np.isnan(
                    metrics["log_loss"]
                ):

                    print(
                        f"  Log loss: "
                        f"{metrics['log_loss']:.4f}"
                    )

            except Exception as exc:

                print()
                print(
                    f"[ERROR] "
                    f"{model_name} failed."
                )

                print(
                    f"Error type: "
                    f"{type(exc).__name__}"
                )

                print(
                    f"Error: "
                    f"{exc}"
                )

                print()
                traceback.print_exc()

                audit_results.append({

                    "budget_sec":
                        budget,

                    "fold":
                        fold,

                    "model":
                        model_name,

                    "status":
                        "error",

                    "error":
                        str(exc),

                    "n_train_groups":
                        len(train_groups),

                    "n_test_groups":
                        len(test_groups),

                })


# ============================================================
# 21. SAVE FOLD RESULTS
# ============================================================

section(
    "16. SAVING MAIN MODEL RESULTS"
)

classification_df = pd.DataFrame(
    classification_results
)

audit_df = pd.DataFrame(
    audit_results
)

print(
    f"Classification result rows: "
    f"{len(classification_df)}"
)

print(
    f"Audit rows: "
    f"{len(audit_df)}"
)

if len(classification_df) == 0:

    fail(
        "No classification results were produced."
    )

classification_df.to_csv(
    OUTPUT_FOLD_RESULTS,
    index=False
)

audit_df.to_csv(
    OUTPUT_AUDIT,
    index=False
)

print()
print(
    "Fold results saved:"
)

print(
    OUTPUT_FOLD_RESULTS
)

print()
print(
    "Run audit saved:"
)

print(
    OUTPUT_AUDIT
)


# ============================================================
# 22. AGGREGATE RESULTS
# ============================================================

section(
    "17. AGGREGATING FIVE-FOLD RESULTS"
)

summary = (
    classification_df
    .groupby(
        [
            "model",
            "budget_sec"
        ],
        as_index=False
    )
    .agg(

        folds_evaluated=(
            "fold",
            "nunique"
        ),

        mean_accuracy=(
            "accuracy",
            "mean"
        ),

        sd_accuracy=(
            "accuracy",
            "std"
        ),

        mean_balanced_accuracy=(
            "balanced_accuracy",
            "mean"
        ),

        sd_balanced_accuracy=(
            "balanced_accuracy",
            "std"
        ),

        mean_macro_f1=(
            "macro_f1",
            "mean"
        ),

        sd_macro_f1=(
            "macro_f1",
            "std"
        ),

        mean_weighted_f1=(
            "weighted_f1",
            "mean"
        ),

        sd_weighted_f1=(
            "weighted_f1",
            "std"
        ),

        mean_log_loss=(
            "log_loss",
            "mean"
        ),

        sd_log_loss=(
            "log_loss",
            "std"
        ),
    )
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False
)

print(
    "Summary saved:"
)

print(
    OUTPUT_SUMMARY
)


# ============================================================
# 23. PRINT RESULTS
# ============================================================

section(
    "18. MAIN FIVE-FOLD RESULTS"
)

display_columns = [
    "model",
    "budget_sec",
    "folds_evaluated",
    "mean_accuracy",
    "sd_accuracy",
    "mean_balanced_accuracy",
    "sd_balanced_accuracy",
    "mean_macro_f1",
    "sd_macro_f1",
    "mean_weighted_f1",
    "mean_log_loss",
]

display_columns = [
    column
    for column in display_columns
    if column in summary.columns
]

ordered_summary = (
    summary[
        display_columns
    ]
    .sort_values(
        [
            "budget_sec",
            "mean_macro_f1"
        ],
        ascending=[
            True,
            False
        ]
    )
)

print(
    ordered_summary.to_string(
        index=False
    )
)


# ============================================================
# 24. COMPLETION AUDIT
# ============================================================

section(
    "19. FINAL RUN AUDIT"
)

expected_per_budget = (
    len(FOLDS)
    * (len(MODELS) + 1)
)

expected_total = (
    len(BUDGETS)
    * expected_per_budget
)

completed_rows = len(
    classification_df
)

print(
    f"Expected maximum result rows: "
    f"{expected_total}"
)

print(
    f"Actual result rows:            "
    f"{completed_rows}"
)

print()
print(
    "Models actually represented:"
)

print(
    sorted(
        classification_df[
            "model"
        ].unique()
    )
)

print()
print(
    "Budgets actually represented:"
)

print(
    sorted(
        classification_df[
            "budget_sec"
        ].unique()
    )
)

print()
print(
    "Folds actually represented:"
)

print(
    sorted(
        classification_df[
            "fold"
        ].unique()
    )
)


# ============================================================
# 25. EXPLICIT SAFETY CHECKS
# ============================================================

section(
    "20. EXPERIMENT SAFETY CHECKS"
)

actual_models = set(
    classification_df[
        "model"
    ].unique()
)

forbidden_models = {
    "LinearSVM",
    "MLP",
}

found_forbidden = (
    actual_models
    & forbidden_models
)

if found_forbidden:

    fail(
        "FORBIDDEN MODEL DETECTED:\n"
        + str(
            found_forbidden
        )
    )

ok(
    "No SVM or MLP results were produced."
)


if len(
    classification_df[
        classification_df[
            "model"
        ] == "MajorityClass"
    ]
) == 0:

    fail(
        "Majority baseline is missing."
    )

ok(
    "Majority baseline present."
)


expected_ml_models = {
    "LogisticRegression",
    "RandomForest",
    "ExtraTrees",
}

actual_ml_models = (
    actual_models
    - {"MajorityClass"}
)

unexpected_models = (
    actual_ml_models
    - expected_ml_models
)

if unexpected_models:

    fail(
        "Unexpected models found:\n"
        + str(
            unexpected_models
        )
    )

ok(
    "Only the requested main ML models were run."
)


# ============================================================
# 26. FINAL MESSAGE
# ============================================================

section(
    "21. MAIN EXPERIMENT COMPLETE"
)

print()
print(
    "The clean main experiment is finished."
)

print()
print(
    "RUNNED:"
)

print(
    "  ✓ MajorityClass"
)

print(
    "  ✓ LogisticRegression"
)

print(
    "  ✓ RandomForest"
)

print(
    "  ✓ ExtraTrees"
)

print()
print(
    "BUDGETS:"
)

print(
    "  ✓ 0 sec"
)

print(
    "  ✓ 30 sec"
)

print(
    "  ✓ 60 sec"
)

print(
    "  ✓ 120 sec"
)

print(
    "  ✓ 180 sec"
)

print(
    "  ✓ 300 sec"
)

print()
print(
    "OUTER EVALUATION:"
)

print(
    "  ✓ 5 group-disjoint folds"
)

print()
print(
    "NOT RUN:"
)

print(
    "  ✗ Linear SVM"
)

print(
    "  ✗ MLP"
)

print(
    "  ✗ Future clip-level prediction"
)

print(
    "  ✗ Distribution prediction"
)

print(
    "  ✗ CMOSE"
)

print(
    "  ✗ Meta-learning"
)

print()
print(
    "OUTPUT FILES:"
)

print(
    f"  Fold results:"
)

print(
    f"    {OUTPUT_FOLD_RESULTS}"
)

print()
print(
    f"  Summary:"
)

print(
    f"    {OUTPUT_SUMMARY}"
)

print()
print(
    f"  Audit:"
)

print(
    f"    {OUTPUT_AUDIT}"
)

print()
print(
    "=" * 76
)

print(
    "SCRIPT TERMINATED AFTER MAIN EXPERIMENT."
)

print(
    "NO ADDITIONAL ANALYSIS WILL RUN."
)

print(
    "=" * 76
)
