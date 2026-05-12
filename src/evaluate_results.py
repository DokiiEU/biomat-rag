import pandas as pd
from pathlib import Path

from config import RESULTS_DIR

RESULTS_CSV = RESULTS_DIR / "qa_results_shuffled_v3.csv"


def acc(series_true, series_pred) -> float:
    return (series_true.astype(str).str.strip() == series_pred.astype(str).str.strip()).mean()


def main():
    df = pd.read_csv(RESULTS_CSV)

    df["baseline_correct"] = (
        df["gold_answer"].astype(str).str.strip()
        == df["baseline_answer"].astype(str).str.strip()
    )
    df["rag_correct"] = (
        df["gold_answer"].astype(str).str.strip()
        == df["rag_answer"].astype(str).str.strip()
    )

    print("\n=== OVERALL ===")
    print(f"Baseline accuracy: {df['baseline_correct'].mean():.2%}")
    print(f"RAG accuracy:      {df['rag_correct'].mean():.2%}")

    print("\n=== BY SOURCE SPLIT ===")
    split_stats = df.groupby("source_split")[["baseline_correct", "rag_correct"]].mean()
    print(split_stats.map(lambda x: f"{x:.2%}"))

    print("\n=== BY MATERIAL CLASS ===")
    mat_stats = df.groupby("material_class")[["baseline_correct", "rag_correct"]].mean()
    print(split_stats.map(lambda x: f"{x:.2%}"))

    print("\n=== UNKNOWN QUESTIONS (gold = F) ===")
    unknown_df = df[df["gold_answer"].astype(str).str.strip() == "F"]
    if len(unknown_df) > 0:
        print(f"Count: {len(unknown_df)}")
        print(f"Baseline unknown accuracy: {unknown_df['baseline_correct'].mean():.2%}")
        print(f"RAG unknown accuracy:      {unknown_df['rag_correct'].mean():.2%}")
    else:
        print("No Unknown questions found.")

    print("\n=== BASELINE ERRORS ===")
    baseline_errors = df[~df["baseline_correct"]][
        ["qid", "question", "gold_answer", "baseline_answer"]
    ]
    print(baseline_errors.to_string(index=False))

    print("\n=== RAG ERRORS ===")
    rag_errors = df[~df["rag_correct"]][
        ["qid", "question", "gold_answer", "rag_answer", "retrieved_chunk_ids"]
    ]
    print(rag_errors.to_string(index=False))

    print("\n=== IMPROVEMENT CASES (RAG correct, Baseline wrong) ===")
    improved = df[(~df["baseline_correct"]) & (df["rag_correct"])][
        ["qid", "question", "gold_answer", "baseline_answer", "rag_answer"]
    ]
    print(improved.to_string(index=False))

    print("\n=== REGRESSION CASES (Baseline correct, RAG wrong) ===")
    regressions = df[(df["baseline_correct"]) & (~df["rag_correct"])][
        ["qid", "question", "gold_answer", "baseline_answer", "rag_answer"]
    ]
    print(regressions.to_string(index=False))


if __name__ == "__main__":
    main()