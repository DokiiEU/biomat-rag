import pandas as pd
from datasets import Dataset, Features, Sequence, Value

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from config import RESULTS_DIR, PROCESSED_DIR

RESULTS_CSV = RESULTS_DIR / "qa_results_shuffled_v3.csv"
CHUNKS_CSV = PROCESSED_DIR / "chunks.csv"


def build_chunk_map():
    chunks_df = pd.read_csv(CHUNKS_CSV)
    return dict(zip(chunks_df["chunk_id"], chunks_df["chunk_text"]))


def parse_chunk_ids(value):
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(";") if x.strip()]


def main():
    results_df = pd.read_csv(RESULTS_CSV)
    chunk_map = build_chunk_map()

    rows = []
    for _, row in results_df.iterrows():
        chunk_ids = parse_chunk_ids(row["retrieved_chunk_ids"])
        contexts = [str(chunk_map[cid]) for cid in chunk_ids if cid in chunk_map]

        if not contexts:
            continue

        rows.append({
            "question": str(row["question"]),
            "answer": str(row["rag_answer"]),
            "contexts": [str(c) for c in contexts],
        })

    features = Features({
        "question": Value("string"),
        "answer": Value("string"),
        "contexts": Sequence(Value("string")),
    })

    dataset = Dataset.from_list(rows, features=features)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
        ]
    )

    print("\n=== RAGAS RESULTS ===")
    print(result)

    result.to_pandas().to_csv(
        RESULTS_DIR / "ragas_results.csv",
        index=False,
        encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main()