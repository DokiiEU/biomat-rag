import json
import re
import csv
from config import CHUNKS_CSV, CHUNK_OVERLAP, CHUNK_SIZE, PROCESSED_DIR

PARSED_DOCS_JSON = PROCESSED_DIR / "parsed_docs.json"


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text_by_words(text: str, chunk_size_words: int, overlap_words: int):
    text = clean_text(text)
    words = text.split()

    start = 0
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if len(chunk) >= 200:
            yield chunk
        if end == len(words):
            break
        start += max(1, chunk_size_words - overlap_words)


def main():
    with open(PARSED_DOCS_JSON, "r", encoding="utf-8") as f:
        docs = json.load(f)

    with open(CHUNKS_CSV, "w", newline="", encoding="utf-8-sig") as csvfile:
        fieldnames = [
            "chunk_id", "doc_id", "filename", "title",
            "material_class", "source_split", "year", "doc_type",
            "chunk_index", "chunk_text", "char_count"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_chunks = 0

        for doc in docs:
            print(f"Processing {doc['doc_id']}...")
            chunk_idx = 1

            for chunk in chunk_text_by_words(
                doc["full_text"],
                CHUNK_SIZE,
                CHUNK_OVERLAP
            ):
                writer.writerow({
                    "chunk_id": f"{doc['doc_id']}_C{chunk_idx:03d}",
                    "doc_id": doc["doc_id"],
                    "filename": doc["filename"],
                    "title": doc["title"],
                    "material_class": doc["material_class"],
                    "source_split": doc["source_split"],
                    "year": doc["year"],
                    "doc_type": doc["doc_type"],
                    "chunk_index": chunk_idx,
                    "chunk_text": chunk,
                    "char_count": len(chunk),
                })
                chunk_idx += 1
                total_chunks += 1

    print(f"Done. Total chunks: {total_chunks}")


if __name__ == "__main__":
    main()