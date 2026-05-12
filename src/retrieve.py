import time

import faiss
import numpy as np
import pandas as pd
from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDINGS_META_CSV,
    FAISS_INDEX_PATH,
    TOP_K,
)

client = OpenAI(api_key=OPENAI_API_KEY, timeout=60)


def embed_query(query: str) -> np.ndarray:
    for attempt in range(3):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[query],
            )

            embedding = response.data[0].embedding

            # FAISS treba numpy array oblika (1, dim), dtype float32
            qvec = np.array([embedding], dtype="float32")

            return qvec

        except (APITimeoutError, APIConnectionError, RateLimitError) as e:
            print(f"Embedding error: {e}")
            print(f"Retry {attempt + 1}/3...")
            time.sleep(5 * (attempt + 1))

    raise RuntimeError("Embedding failed after retries.")


def retrieve(query: str, top_k: int = TOP_K) -> pd.DataFrame:
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    meta = pd.read_csv(EMBEDDINGS_META_CSV)

    qvec = embed_query(query)

    scores, indices = index.search(qvec, top_k)

    rows = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        row = meta.iloc[int(idx)].to_dict()
        row["score"] = float(score)
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    test_query = "Why are composites useful in biomedical implants?"
    results = retrieve(test_query, top_k=4)

    print("\nQUERY:", test_query)

    for i, row in results.iterrows():
        print(
            f"\nResult {i + 1} | score={row['score']:.4f} | "
            f"{row['chunk_id']} | {row['title']}"
        )
        print(str(row["chunk_text"])[:500])