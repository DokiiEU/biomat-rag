import json
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from tqdm import tqdm

from config import DOCUMENTS_CSV, PROCESSED_DIR, RAW_DOCS_DIR

PARSED_DOCS_JSON = PROCESSED_DIR / "parsed_docs.json"
PARSED_PREVIEW_CSV = PROCESSED_DIR / "parsed_docs_preview.csv"


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []

    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        pages.append(page_text)

    return clean_text(" ".join(pages))


def main() -> None:
    docs_df = pd.read_csv(DOCUMENTS_CSV)

    parsed_rows = []

    for _, row in tqdm(docs_df.iterrows(), total=len(docs_df), desc="Parsing PDFs"):
        pdf_path = RAW_DOCS_DIR / row["filename"]

        if not pdf_path.exists():
            print(f"[WARNING] Missing file: {pdf_path}")
            continue

        text = extract_pdf_text(pdf_path)

        parsed_rows.append(
            {
                "doc_id": row["doc_id"],
                "filename": row["filename"],
                "title": row["title"],
                "material_class": row["material_class"],
                "source_split": row["source_split"],
                "year": int(row["year"]),
                "doc_type": row["doc_type"],
                "notes": row["notes"],
                "full_text": text,
                "char_count": len(text),
            }
        )

    with open(PARSED_DOCS_JSON, "w", encoding="utf-8") as f:
        json.dump(parsed_rows, f, ensure_ascii=False, indent=2)

    preview_rows = []
    for row in parsed_rows:
        preview_rows.append(
            {
                "doc_id": row["doc_id"],
                "filename": row["filename"],
                "title": row["title"],
                "material_class": row["material_class"],
                "source_split": row["source_split"],
                "year": row["year"],
                "doc_type": row["doc_type"],
                "char_count": row["char_count"],
                "text_preview": row["full_text"][:500],
            }
        )

    pd.DataFrame(preview_rows).to_csv(PARSED_PREVIEW_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved full parsed docs to: {PARSED_DOCS_JSON}")
    print(f"Saved preview CSV to: {PARSED_PREVIEW_CSV}")


if __name__ == "__main__":
    main()