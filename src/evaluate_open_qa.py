import pandas as pd

from config import RESULTS_DIR

RESULTS_CSV = RESULTS_DIR / "open_qa_results_v1.csv"


def contains_unknown(text: str) -> bool:
    return "unknown" in str(text).lower()


def answer_length(text: str) -> int:
    return len(str(text).split())


def main():
    df = pd.read_csv(RESULTS_CSV)

    print("\n=== OPEN QA EVALUATION ===")

    print(f"\nTotal questions: {len(df)}")

    # =========================
    # UNKNOWN ANALYSIS
    # =========================

    baseline_unknown = df["baseline_answer"].apply(contains_unknown)
    rag_unknown = df["rag_answer"].apply(contains_unknown)

    print("\n=== UNKNOWN RATES ===")
    print(f"Baseline Unknown rate: {baseline_unknown.mean():.2%}")
    print(f"RAG Unknown rate:      {rag_unknown.mean():.2%}")

    # =========================
    # ANSWER LENGTHS
    # =========================

    baseline_len = df["baseline_answer"].apply(answer_length)
    rag_len = df["rag_answer"].apply(answer_length)

    print("\n=== ANSWER LENGTH ===")
    print(f"Baseline avg length: {baseline_len.mean():.2f} words")
    print(f"RAG avg length:      {rag_len.mean():.2f} words")

    # =========================
    # QUESTION TYPES
    # =========================

    print("\n=== QUESTION TYPES ===")
    qtypes = df["question_type"].value_counts()

    for qtype, count in qtypes.items():
        print(f"{qtype}: {count}")

    # =========================
    # MATERIAL CLASSES
    # =========================

    print("\n=== MATERIAL CLASSES ===")
    mclasses = df["material_class"].value_counts()

    for mclass, count in mclasses.items():
        print(f"{mclass}: {count}")

    # =========================
    # SOURCE SPLIT
    # =========================

    print("\n=== SOURCE SPLIT ===")
    splits = df["source_split"].value_counts()

    for split, count in splits.items():
        print(f"{split}: {count}")

    # =========================
    # DIFFICULTY
    # =========================

    print("\n=== DIFFICULTY ===")
    diff = df["difficulty"].value_counts()

    for d, count in diff.items():
        print(f"{d}: {count}")

    # =========================
    # POSSIBLE HALLUCINATIONS
    # =========================

    print("\n=== POSSIBLE HALLUCINATIONS ===")

    hallucinations = df[
        (~rag_unknown)
        & (
            df["retrieved_chunk_ids"].isna()
            | (df["retrieved_chunk_ids"].astype(str).str.len() < 3)
        )
    ]

    print(f"Potential hallucinations: {len(hallucinations)}")

    if len(hallucinations) > 0:
        print(
            hallucinations[
                [
                    "qid",
                    "question",
                    "rag_answer",
                ]
            ].head(10)
        )

    # =========================
    # LONG ANSWERS
    # =========================

    print("\n=== VERY LONG ANSWERS ===")

    long_answers = df[rag_len > 120]

    print(f"Very long answers: {len(long_answers)}")

    if len(long_answers) > 0:
        print(
            long_answers[
                [
                    "qid",
                    "question",
                ]
            ].head(10)
        )

    # =========================
    # SAVE ENRICHED FILE
    # =========================

    df["baseline_unknown"] = baseline_unknown
    df["rag_unknown"] = rag_unknown
    df["baseline_length"] = baseline_len
    df["rag_length"] = rag_len

    output_path = RESULTS_DIR / "open_qa_results_evaluated.csv"

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved enriched results:")
    print(output_path)


if __name__ == "__main__":
    main()