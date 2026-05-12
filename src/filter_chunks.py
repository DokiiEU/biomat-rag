import pandas as pd
from pathlib import Path

from config import PROCESSED_DIR

CHUNKS_PATH = PROCESSED_DIR / "chunks.csv"
OUTPUT_PATH = PROCESSED_DIR / "filtered_chunks.csv"

MIN_WORDS = 80

BAD_PATTERNS = [
    "references",
    "bibliography",
    "copyright",
    "all rights reserved",
    "doi:",
    "received:",
    "accepted:",
    "conflict of interest",
    "funding",
]


def clean_text(text: str) -> str:
    return " ".join(str(text).split())


chunks = pd.read_csv(CHUNKS_PATH)

chunks["text_start"] = chunks["chunk_text"].str[:250]

chunks = chunks.drop_duplicates(subset=["text_start"])

chunks["chunk_text"] = chunks["chunk_text"].astype(str)
chunks["clean_text"] = chunks["chunk_text"].apply(clean_text)
chunks["word_count"] = chunks["clean_text"].apply(lambda x: len(x.split()))

# remove short chunks
chunks = chunks[chunks["word_count"] >= MIN_WORDS]

# remove bad patterns
mask = pd.Series(False, index=chunks.index)

for pattern in BAD_PATTERNS:
    mask |= chunks["clean_text"].str.lower().str.contains(pattern)

chunks = chunks[~mask]



chunks.to_csv(OUTPUT_PATH, index=False)

print(f"Saved filtered chunks: {OUTPUT_PATH}")
print(f"Remaining chunks: {len(chunks)}")