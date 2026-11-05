import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = [
    "Highly Disengage",
    "Disengage",
    "Engage",
    "Highly Engage",
]


def parse_group_from_clip_id(clip_id):
    """
    videoX_Y_personZ -> videoX_personZ
    """
    match = re.match(r"^(video[^\_]+)\_(\d+)\_((?:person).+)$", str(clip_id))
    if not match:
        return None
    return f"{match.group(1)}_{match.group(3)}"


def load_data(json_path):
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

        group = parse_group_from_clip_id(clip_id)
        if group is None:
            continue

        match = re.search(r"\_(\d+)\_", str(clip_id))
        clip_index = int(match.group(1)) if match else 0

        rows.append(
            {
                "clip_id": str(clip_id),
                "group": group,
                "label": label,
                "clip_index": clip_index,
            }
        )

    if not rows:
        raise ValueError("No valid labeled clips found.")

    return pd.DataFrame(rows)


def load_folds(fold_path):
    df = pd.read_csv(fold_path)

    required = {"group", "fold"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Fold CSV missing columns: {sorted(missing)}")

    df = df[["group", "fold"]].drop_duplicates()

    check = df.groupby("group")["fold"].nunique()
    bad = check[check != 1]

    if len(bad):
        raise ValueError(
            f"Some groups occur in multiple folds: {bad.index.tolist()}"
        )

    return dict(zip(df["group"], df["fold"].astype(int)))


def dataset_diagnostics(df):
    print("\n" + "=" * 80)
    print("1. DATASET CLASS DISTRIBUTION")
    print("=" * 80)

    class_counts = (
        df["label"]
        .value_counts()
        .reindex(LABELS, fill_value=0)
    )

    print("\nTotal clips:")
    print(class_counts.to_string())

    student_class = (
        df.groupby(["group", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=LABELS, fill_value=0)
    )

    print("\nNumber of students containing each class:")
    print((student_class > 0).sum().to_string())

    print("\nStudents with >= 3 clips of each class:")
    print((student_class >= 3).sum().to_string())

    print("\nStudents with >= 5 clips of each class:")
    print((student_class >= 5).sum().to_string())

    print("\nPer-student clip-count statistics:")
    stats = (
        student_class.describe()
        .loc[["count", "mean", "min", "25%", "50%", "75%", "max"]]
    )
    print(stats.to_string())


def training_episode_eligibility(df, fold_map):
    print("\n" + "=" * 80)
    print("2. META-TRAINING EPISODE ELIGIBILITY")
    print("=" * 80)

    rows = []

    for fold in sorted(set(fold_map.values())):
        train_groups = [
            g for g, f in fold_map.items()
            if f != fold
        ]

        train = df[df["group"].isin(train_groups)]

        counts = (
            train.groupby(["group", "label"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=LABELS, fill_value=0)
        )

        eligible_3_5 = (
            (counts >= 3).all(axis=1)
            & (counts >= 8).all(axis=1)
        )

        eligible_3_only = (counts >= 3).all(axis=1)
        eligible_5_only = (counts >= 5).all(axis=1)

        rows.append(
            {
                "fold": fold,
                "train_students": len(counts),
                "students_ge3_all_classes": int(eligible_3_only.sum()),
                "students_ge5_all_classes": int(eligible_5_only.sum()),
                "students_ge3_support_ge5_query_all_classes": int(
                    eligible_3_5.sum()
                ),
            }
        )

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))

    print(
        "\nInterpretation:"
        "\n- >=3 all classes means a student can provide 3 support examples"
        " for every class."
        "\n- >=5 all classes means a student can provide 5 query examples"
        " for every class."
        "\n- >=8 all classes means a student can provide 3 support + 5 query"
        " for every class."
    )

    return result


def support_distribution(df, fold_map, k, protocol, seed=42):
    rng = np.random.default_rng(seed)

    rows = []

    for fold in sorted(set(fold_map.values())):
        test_groups = [
            g for g, f in fold_map.items()
            if f == fold
        ]

        for group in test_groups:
            student = (
                df[df["group"] == group]
                .sort_values("clip_index")
                .copy()
            )

            if len(student) <= k:
                continue

            if protocol == "early":
                support = student.iloc[:k].copy()
            else:
                idx = rng.choice(
                    student.index.to_numpy(),
                    size=k,
                    replace=False,
                )
                support = student.loc[idx].copy()

            query = student[
                ~student.index.isin(support.index)
            ].copy()

            support_classes = set(support["label"])
            query_classes = set(query["label"])

            missing_query_classes = query_classes - support_classes

            # Count query clips whose true class has no prototype.
            missing_proto_query_clips = query[
                ~query["label"].isin(support_classes)
            ]

            rows.append(
                {
                    "fold": fold,
                    "group": group,
                    "k": k,
                    "protocol": protocol,
                    "support_classes": len(support_classes),
                    "query_classes": len(query_classes),
                    "support_has_all_4": int(
                        len(support_classes) == 4
                    ),
                    "missing_query_classes": len(
                        missing_query_classes
                    ),
                    "query_clips_without_true_class_prototype": len(
                        missing_proto_query_clips
                    ),
                    "query_size": len(query),
                }
            )

    return pd.DataFrame(rows)


def print_support_diagnostics(df, fold_map):
    print("\n" + "=" * 80)
    print("3. TEST SUPPORT-SET DIAGNOSTICS")
    print("=" * 80)

    all_results = []

    for protocol in ["random", "early"]:
        for k in [5, 10, 20]:
            result = support_distribution(
                df,
                fold_map,
                k,
                protocol,
                seed=42,
            )

            if result.empty:
                continue

            all_results.append(result)

            distribution = (
                result["support_classes"]
                .value_counts()
                .reindex([1, 2, 3, 4], fill_value=0)
            )

            total_students = len(result)

            pct = (
                distribution / total_students * 100
            )

            missing_clips = (
                result[
                    "query_clips_without_true_class_prototype"
                ].sum()
            )

            total_query = result["query_size"].sum()

            print(
                f"\n{protocol.upper()}, K={k}"
            )
            print(
                f"Students evaluated: {total_students}"
            )
            print(
                "Support class count:"
            )
            print(
                pd.DataFrame(
                    {
                        "students": distribution,
                        "percent": pct.round(2),
                    }
                ).to_string()
            )

            print(
                f"Students with all 4 support classes: "
                f"{result['support_has_all_4'].sum()} "
                f"({result['support_has_all_4'].mean() * 100:.2f}%)"
            )

            print(
                f"Query clips whose TRUE class has no support prototype: "
                f"{missing_clips}/{total_query} "
                f"({100 * missing_clips / total_query:.2f}%)"
            )

    return pd.concat(all_results, ignore_index=True)


def episode_sampling_diagnostic(df, fold_map):
    print("\n" + "=" * 80)
    print("4. CLASS IMBALANCE AND EPISODE-SAMPLING DIAGNOSTIC")
    print("=" * 80)

    rows = []

    for fold in sorted(set(fold_map.values())):
        train_groups = [
            g for g, f in fold_map.items()
            if f != fold
        ]

        train = df[df["group"].isin(train_groups)]

        class_counts = (
            train["label"]
            .value_counts()
            .reindex(LABELS, fill_value=0)
        )

        total = class_counts.sum()

        for label in LABELS:
            rows.append(
                {
                    "fold": fold,
                    "class": label,
                    "train_clips": int(class_counts[label]),
                    "train_percent": float(
                        class_counts[label] / total * 100
                    ),
                }
            )

    result = pd.DataFrame(rows)

    print(
        result.pivot(
            index="fold",
            columns="class",
            values="train_percent",
        ).round(2).to_string()
    )

    print(
        "\nImportant:"
        "\nThe current ProtoNet code uses variable-way episodes."
        "\nThis diagnostic does NOT assume that every episode must contain"
        " all four classes."
        "\nIt checks whether the data actually provides enough examples"
        " for the current 3-support / 5-query-per-class design."
    )

    return result


def final_verdict(eligibility, support_results):
    print("\n" + "=" * 80)
    print("5. FINAL DIAGNOSTIC VERDICT")
    print("=" * 80)

    max_eligible = eligibility[
        "students_ge3_support_ge5_query_all_classes"
    ].max()

    print(
        f"\nMaximum students per fold capable of supplying "
        f"3 support + 5 query for ALL FOUR classes: {max_eligible}"
    )

    print(
        "\nThe important evaluation question is NOT whether every "
        "student has all four classes in their K-shot support."
    )

    missing_pct = (
        support_results
        .groupby(["protocol", "k"])
        .apply(
            lambda x: 100
            * x[
                "query_clips_without_true_class_prototype"
            ].sum()
            / x["query_size"].sum(),
            include_groups=False,
        )
    )

    print(
        "\nPercentage of query clips whose true class has no "
        "support prototype:"
    )
    print(missing_pct.round(2).to_string())

    worst = missing_pct.max()

    if worst > 20:
        print(
            "\nVERDICT: IMPORTANT STRUCTURAL ISSUE."
            "\nA substantial fraction of query clips have no corresponding "
            "support prototype. This should be addressed before treating "
            "ProtoNet as the final comparison."
        )
    elif worst > 10:
        print(
            "\nVERDICT: MODERATE STRUCTURAL ISSUE."
            "\nThe support/query formulation deserves consideration before "
            "finalizing the ProtoNet comparison."
        )
    else:
        print(
            "\nVERDICT: SUPPORT PROTOTYPE COVERAGE LOOKS REASONABLE."
            "\nNo major structural problem is apparent from this diagnostic."
        )

    print(
        "\nDo NOT change the ProtoNet experiment based only on this "
        "diagnostic. Use the reported numbers to decide whether the "
        "3/5 episode construction needs modification."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic for ProtoNet unseen-student experiment."
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
        "--output-dir",
        type=str,
        default=os.environ.get("CMOSE_OUTPUT_DIR", "."),
    )

    args = parser.parse_args()

    df = load_data(args.json)
    fold_map = load_folds(args.folds)

    print(
        "\nProtoNet support / episode diagnostic"
    )
    print(
        f"Loaded {len(df)} labeled clips "
        f"from {df['group'].nunique()} students."
    )

    dataset_diagnostics(df)

    eligibility = training_episode_eligibility(
        df,
        fold_map,
    )

    episode_sampling_diagnostic(
        df,
        fold_map,
    )

    support_results = print_support_diagnostics(
        df,
        fold_map,
    )

    final_verdict(
        eligibility,
        support_results,
    )

    eligibility.to_csv(
        Path(args.output_dir)
        / "protonet_episode_eligibility_diagnostic.csv",
        index=False,
    )

    support_results.to_csv(
        Path(args.output_dir)
        / "protonet_support_diagnostic.csv",
        index=False,
    )

    print(
        "\nSaved diagnostic CSVs to:"
        f"\n{args.output_dir}"
    )


if __name__ == "__main__":
    main()
