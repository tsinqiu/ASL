from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold


DEFAULT_DATA_ROOT = Path(r"C:\ASL\asl-signs")
REQUIRED_COLUMNS = {"path", "participant_id", "sequence_id", "sign"}
DEFAULT_OUTPUT = Path("outputs") / "train_with_folds.csv"
SUMMARY_OUTPUT = Path("outputs") / "split_summary.txt"


def add(lines: list[str], text: str = "") -> None:
    lines.append(text)


def validate_train_csv(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"train.csv is missing required columns: {sorted(missing)}. "
            f"Actual columns: {list(df.columns)}"
        )


def build_folds(train_df: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    validate_train_csv(train_df)
    participant_count = train_df["participant_id"].nunique()
    if n_splits > participant_count:
        raise ValueError(
            f"n_splits={n_splits} is greater than participant_id count={participant_count}. "
            "Use fewer folds or inspect participant_id coverage."
        )

    result = train_df.copy()
    result["fold"] = -1
    groups = result["participant_id"]
    splitter = GroupKFold(n_splits=n_splits)

    for fold, (_, valid_idx) in enumerate(splitter.split(result, result["sign"], groups)):
        result.loc[valid_idx, "fold"] = fold

    if (result["fold"] < 0).any():
        raise RuntimeError("Some rows were not assigned a fold.")

    result["fold"] = result["fold"].astype(int)
    return result


def summarize_folds(fold_df: pd.DataFrame, n_splits: int) -> str:
    lines: list[str] = []
    add(lines, "Kaggle ISLR GroupKFold Summary")
    add(lines, "=" * 34)
    add(lines, f"total samples: {len(fold_df)}")
    add(lines, f"fold count: {n_splits}")
    add(lines, f"participant_id count: {fold_df['participant_id'].nunique()}")
    add(lines)

    all_participants = set(fold_df["participant_id"].astype(int).unique().tolist())
    for fold in range(n_splits):
        valid_df = fold_df[fold_df["fold"] == fold]
        train_df = fold_df[fold_df["fold"] != fold]
        valid_participants = sorted(valid_df["participant_id"].astype(int).unique().tolist())
        train_participants = sorted(train_df["participant_id"].astype(int).unique().tolist())
        disjoint = set(train_participants).isdisjoint(valid_participants)

        add(lines, f"fold {fold}")
        add(lines, f"  valid samples: {len(valid_df)}")
        add(lines, f"  valid sign classes: {valid_df['sign'].nunique()}")
        add(lines, f"  valid participant_id: {valid_participants}")
        add(lines, f"  train participant count: {len(set(train_participants))}")
        add(lines, f"  train/valid participant disjoint: {disjoint}")
        add(lines, "  first 5 valid paths:")
        for path in valid_df["path"].head(5).astype(str).tolist():
            add(lines, f"    {path}")
        add(lines)

    assigned_participants = set(fold_df["participant_id"].astype(int).unique().tolist())
    add(lines, f"all participants assigned: {assigned_participants == all_participants}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create participant GroupKFold splits for Kaggle ISLR.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_csv = args.data_root / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found: {train_csv}")
    if args.n_splits < 2:
        raise ValueError(f"--n-splits must be at least 2, got {args.n_splits}")

    train_df = pd.read_csv(train_csv)
    fold_df = build_folds(train_df, args.n_splits)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(args.output, index=False)

    summary = summarize_folds(fold_df, args.n_splits)
    SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"saved folds to: {args.output.resolve()}")
    print(f"saved summary to: {SUMMARY_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
