import pandas as pd
from ortools.sat.python import cp_model

# ============================================================
# FIXED RESEARCH DESIGN
# ============================================================

N_FOLDS = 5
MIN_TOTAL_SEC = 10 * 60

LABELS = [
    "Highly Disengage",
    "Disengage",
    "Engage",
    "Highly Engage"
]

# ============================================================
# LOAD DATA
# ============================================================

print("\n[1/8] Loading data...", flush=True)

duration_df = pd.read_csv("cmose_clip_durations.csv")
audit_df = pd.read_csv("cmose_audit.csv")

duration_df["group"] = duration_df["participant"]
audit_df["group"] = audit_df["participant"]

duration_df = duration_df.merge(
    audit_df[["clip", "label"]],
    on="clip",
    how="left"
)

print(
    f"      Loaded {len(duration_df)} clips.",
    flush=True
)

# ============================================================
# ELIGIBILITY
# ============================================================

print("\n[2/8] Applying eligibility criterion...", flush=True)

group_duration = (
    duration_df
    .groupby("group")["duration_sec"]
    .sum()
)

eligible_groups = group_duration[
    group_duration >= MIN_TOTAL_SEC
].index.tolist()

df = duration_df[
    duration_df["group"].isin(eligible_groups)
].copy()

print(
    f"      Total groups: {len(group_duration)}",
    flush=True
)

print(
    f"      Eligible groups (>=10 min): "
    f"{len(eligible_groups)}",
    flush=True
)

print(
    f"      Excluded groups: "
    f"{len(group_duration) - len(eligible_groups)}",
    flush=True
)

# ============================================================
# GROUP × CLASS COUNTS
# ============================================================

print("\n[3/8] Building group-level label profiles...", flush=True)

profiles = (
    df.groupby(["group", "label"])
      .size()
      .unstack(fill_value=0)
)

for label in LABELS:

    if label not in profiles.columns:
        profiles[label] = 0

profiles = profiles[LABELS]

profiles["total_clips"] = profiles.sum(axis=1)

groups = list(profiles.index)
n_groups = len(groups)

print(
    f"      Groups represented: {n_groups}",
    flush=True
)

# ============================================================
# FOLD SIZES
# ============================================================

print("\n[4/8] Setting fold-size constraints...", flush=True)

base = n_groups // N_FOLDS
remainder = n_groups % N_FOLDS

fold_sizes = [
    base + (1 if f < remainder else 0)
    for f in range(N_FOLDS)
]

print(
    f"      Required fold sizes: {fold_sizes}",
    flush=True
)

# ============================================================
# GLOBAL CLASS COUNTS
# ============================================================

global_counts = profiles[LABELS].sum()
total_clips = int(global_counts.sum())

global_props = (
    global_counts / total_clips
)

print("\n[5/8] Computing eligible-population targets...", flush=True)

print("\n      Global class proportions:")

for label in LABELS:

    print(
        f"        {label}: "
        f"{global_props[label]:.4f}",
        flush=True
    )

# ============================================================
# BUILD CP-SAT MODEL
# ============================================================

print(
    "\n[6/8] Building constrained optimization model...",
    flush=True
)

model = cp_model.CpModel()

# x[g,f] = 1 if group g belongs to fold f
x = {}

for g_idx, group in enumerate(groups):

    for f in range(N_FOLDS):

        x[g_idx, f] = model.NewBoolVar(
            f"x_{g_idx}_{f}"
        )

print(
    f"      Created {n_groups * N_FOLDS} "
    f"group-fold decision variables.",
    flush=True
)

# ============================================================
# EVERY GROUP EXACTLY ONCE
# ============================================================

print(
    "      Adding group-independence constraints...",
    flush=True
)

for g_idx in range(n_groups):

    model.Add(
        sum(
            x[g_idx, f]
            for f in range(N_FOLDS)
        ) == 1
    )

# ============================================================
# EXACT FOLD SIZES
# ============================================================

print(
    "      Adding fold-size constraints...",
    flush=True
)

for f in range(N_FOLDS):

    model.Add(
        sum(
            x[g_idx, f]
            for g_idx in range(n_groups)
        ) == fold_sizes[f]
    )

# ============================================================
# FOLD STATISTICS
# ============================================================

print(
    "      Creating fold class-count variables...",
    flush=True
)

fold_class_counts = {}
fold_clip_counts = {}

for f in range(N_FOLDS):

    fold_clip_counts[f] = model.NewIntVar(
        0,
        total_clips,
        f"clips_{f}"
    )

    model.Add(
        fold_clip_counts[f]
        ==
        sum(
            int(
                profiles.iloc[g_idx]["total_clips"]
            ) * x[g_idx, f]

            for g_idx in range(n_groups)
        )
    )

    for label in LABELS:

        max_count = int(
            global_counts[label]
        )

        fold_class_counts[f, label] = (
            model.NewIntVar(
                0,
                max_count,
                f"class_{f}_{label}"
            )
        )

        model.Add(
            fold_class_counts[f, label]
            ==
            sum(
                int(
                    profiles.iloc[g_idx][label]
                ) * x[g_idx, f]

                for g_idx in range(n_groups)
            )
        )

# ============================================================
# PRIMARY OBJECTIVE:
# CLASS-DISTRIBUTION BALANCE
# ============================================================

print(
    "      Building class-distribution objective...",
    flush=True
)

deviation_vars = []

for f in range(N_FOLDS):

    for label in LABELS:

        deviation = model.NewIntVar(
            0,
            total_clips * total_clips,
            f"dev_{f}_{label}"
        )

        # Compare:
        #
        # fold_class / fold_total
        #
        # against:
        #
        # global_class / global_total
        #
        # Cross multiplication avoids floating point.

        expression = (
            fold_class_counts[f, label]
            * total_clips
            -
            fold_clip_counts[f]
            * int(global_counts[label])
        )

        model.AddAbsEquality(
            deviation,
            expression
        )

        deviation_vars.append(
            deviation
        )

# ============================================================
# SECONDARY OBJECTIVE:
# CLIP-COUNT BALANCE
# ============================================================

print(
    "      Building clip-count balance objective...",
    flush=True
)

clip_deviation_vars = []

for f in range(N_FOLDS):

    deviation = model.NewIntVar(
        0,
        total_clips * N_FOLDS,
        f"clip_dev_{f}"
    )

    model.AddAbsEquality(
        deviation,
        fold_clip_counts[f] * N_FOLDS
        - total_clips
    )

    clip_deviation_vars.append(
        deviation
    )

# ============================================================
# FINAL OBJECTIVE
# ============================================================

print(
    "      Combining objectives...",
    flush=True
)

model.Minimize(
    sum(deviation_vars)
    +
    sum(clip_deviation_vars)
)

print(
    "      Optimization model complete.",
    flush=True
)

# ============================================================
# SOLVE
# ============================================================

print(
    "\n[7/8] SOLVING...",
    flush=True
)

print(
    "      CPU workers: 12",
    flush=True
)

print(
    "      Maximum solve time: 600 seconds",
    flush=True
)

print(
    "      Solver progress will appear below.",
    flush=True
)

solver = cp_model.CpSolver()

# Up to 10 minutes
solver.parameters.max_time_in_seconds = 600

# Your computer reports 12 logical CPU threads
solver.parameters.num_search_workers = 12

# Show OR-Tools search progress
solver.parameters.log_search_progress = True

status = solver.Solve(model)

print(
    "\n      Solver finished.",
    flush=True
)

if status == cp_model.OPTIMAL:

    print(
        "      STATUS: OPTIMAL",
        flush=True
    )

elif status == cp_model.FEASIBLE:

    print(
        "      STATUS: FEASIBLE",
        flush=True
    )

    print(
        "      IMPORTANT: optimality was NOT proven.",
        flush=True
    )

else:

    print(
        "      STATUS: NO SOLUTION",
        flush=True
    )

    raise SystemExit

print(
    f"      Objective value: "
    f"{solver.ObjectiveValue()}",
    flush=True
)

# ============================================================
# EXTRACT SOLUTION
# ============================================================

print(
    "\n[8/8] Extracting and validating solution...",
    flush=True
)

fold_assignments = []

for f in range(N_FOLDS):

    fold_groups = []

    for g_idx, group in enumerate(groups):

        if solver.Value(
            x[g_idx, f]
        ) == 1:

            fold_groups.append(group)

    fold_assignments.append(
        fold_groups
    )

# ============================================================
# FINAL DIAGNOSTICS
# ============================================================

print(
    "\n=========================================="
)

print(
    "FINAL FOLD DIAGNOSTICS"
)

print(
    "=========================================="
)

for f, fold_groups in enumerate(
    fold_assignments,
    start=1
):

    subset = profiles.loc[fold_groups]

    counts = subset[LABELS].sum()
    total = counts.sum()

    print(
        f"\n========== FOLD {f} ==========",
        flush=True
    )

    print(
        f"Groups: {len(fold_groups)}",
        flush=True
    )

    print(
        f"Clips: {int(total)}",
        flush=True
    )

    print(
        "\nClass proportions:",
        flush=True
    )

    for label in LABELS:

        proportion = (
            counts[label] / total
        )

        difference = (
            proportion
            - global_props[label]
        )

        print(
            f"  {label}: "
            f"{proportion:.4f} "
            f"(difference {difference:+.4f})",
            flush=True
        )

# ============================================================
# GROUP INDEPENDENCE CHECK
# ============================================================

all_assigned = []

for fold in fold_assignments:

    all_assigned.extend(fold)

print(
    "\n=========================================="
)

print(
    "GROUP INDEPENDENCE CHECK"
)

print(
    "=========================================="
)

print(
    f"Assigned groups: {len(all_assigned)}"
)

print(
    f"Unique groups: {len(set(all_assigned))}"
)

if (
    len(all_assigned) == n_groups
    and len(set(all_assigned)) == n_groups
):

    print(
        "PASS: every eligible group appears "
        "exactly once."
    )

else:

    print(
        "FAIL: group assignment problem!"
    )

    raise SystemExit

# ============================================================
# SAVE
# ============================================================

rows = []

for f, fold_groups in enumerate(
    fold_assignments,
    start=1
):

    for group in fold_groups:

        row = {
            "fold": f,
            "group": group,
            "total_clips": int(
                profiles.loc[group, "total_clips"]
            ),
            "duration_sec": float(
                group_duration[group]
            )
        }

        for label in LABELS:

            row[label] = int(
                profiles.loc[group, label]
            )

        rows.append(row)

result = pd.DataFrame(rows)

result.to_csv(
    "cmose_exact_5folds.csv",
    index=False
)

print(
    "\n=========================================="
)

print(
    "SAVED: cmose_exact_5folds.csv"
)

print(
    "=========================================="
)

print("\nDONE.")