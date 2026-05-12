import time
import numpy as np
import pandas as pd
import faiss
from openai import OpenAI
from tqdm import tqdm

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CHUNKS_CSV,
    EMBEDDINGS_NPY,
    EMBEDDINGS_META_CSV,
    FAISS_INDEX_PATH,
)

BATCH_SIZE = 100


def get_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]


def main() -> None:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY missing in .env")

    client = OpenAI(api_key=OPENAI_API_KEY)

    df = pd.read_csv(CHUNKS_CSV)
    texts = df["chunk_text"].fillna("").astype(str).tolist()

    all_embeddings = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding chunks"):
        batch = texts[i:i + BATCH_SIZE]
        embeddings = get_embeddings(client, batch)
        all_embeddings.extend(embeddings)
        time.sleep(0.1)

    emb_array = np.array(all_embeddings, dtype="float32")

    faiss.normalize_L2(emb_array)
    dim = emb_array.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_array)

    np.save(EMBEDDINGS_NPY, emb_array)
    df.to_csv(EMBEDDINGS_META_CSV, index=False, encoding="utf-8-sig")
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    print(f"Saved embeddings: {EMBEDDINGS_NPY}")
    print(f"Saved metadata: {EMBEDDINGS_META_CSV}")
    print(f"Saved FAISS index: {FAISS_INDEX_PATH}")
    print(f"Total vectors indexed: {index.ntotal}")


if __name__ == "__main__":
    main()