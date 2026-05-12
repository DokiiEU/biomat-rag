import time
from pathlib import Path

import pandas as pd
from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError

from config import (
    OPENAI_API_KEY,
    GENERATION_MODEL,
    PROCESSED_DIR,
    RESULTS_DIR,
    TOP_K,
)
from retrieve import retrieve

QUESTIONS_CSV = PROCESSED_DIR / "open_qa_dataset1.csv"
RESULTS_CSV = RESULTS_DIR / "open_qa_results_v1.csv"

# Za test: 30 ili 50
# Za sve: None
MAX_QUESTIONS = 30

client = OpenAI(api_key=OPENAI_API_KEY, timeout=60)


def safe_call(fn, retries: int = 3, sleep_seconds: int = 5):
    for attempt in range(retries):
        try:
            return fn()
        except (APITimeoutError, APIConnectionError, RateLimitError) as e:
            print(f"API error: {e}")
            print(f"Retry {attempt + 1}/{retries}...")
            time.sleep(sleep_seconds * (attempt + 1))

    raise RuntimeError("API call failed after retries.")


def ask_baseline(question: str, question_type: str) -> str:
    prompt = f"""
Answer the biomaterials question as accurately as possible.

Rules:
- Answer in 1-3 concise scientific sentences.
- If the question cannot be answered confidently, respond exactly with: Unknown
- Do not invent specific numbers, mechanisms, or study results.

Question type:
{question_type}

Question:
{question}

Answer:
"""

    response = safe_call(
        lambda: client.responses.create(
            model=GENERATION_MODEL,
            input=prompt,
            temperature=0,
        )
    )

    return response.output_text.strip()


def ask_rag(question: str, question_type: str, top_k: int = TOP_K) -> tuple[str, str]:
    retrieved = retrieve(question, top_k=top_k)

    context_blocks = []
    used_chunk_ids = []

    for _, row in retrieved.iterrows():
        used_chunk_ids.append(str(row["chunk_id"]))
        context_blocks.append(
            f"[{row['chunk_id']}]\n{row['chunk_text']}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
Answer the biomaterials question using ONLY the provided context.

Strict rules:
- Use only the context below.
- If the answer is not explicitly supported by the context, respond exactly with: Unknown
- Do not use outside knowledge.
- Do not guess.
- Answer in 1-3 concise scientific sentences.
- For unanswerable questions, respond exactly with: Unknown

Question type:
{question_type}

Question:
{question}

Context:
{context}

Answer:
"""

    response = safe_call(
        lambda: client.responses.create(
            model=GENERATION_MODEL,
            input=prompt,
            temperature=0,
        )
    )

    return response.output_text.strip(), ";".join(used_chunk_ids)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    questions_df = pd.read_csv(QUESTIONS_CSV)

    if MAX_QUESTIONS is not None:
        questions_df = questions_df.head(MAX_QUESTIONS)

    if RESULTS_CSV.exists():
        existing_df = pd.read_csv(RESULTS_CSV)
        rows = existing_df.to_dict("records")
        processed_qids = set(existing_df["qid"].astype(str).tolist())
        print(f"Resuming existing results.")
        print(f"Already processed: {len(processed_qids)}")
    else:
        rows = []
        processed_qids = set()

    for _, row in questions_df.iterrows():
        qid = str(row["qid"])

        if qid in processed_qids:
            print(f"Skipping {qid}...")
            continue

        question = str(row["question"])
        question_type = str(row["question_type"])

        print(f"Processing {qid}...")

        try:
            baseline_answer = ask_baseline(question, question_type)
            rag_answer, retrieved_chunk_ids = ask_rag(
                question=question,
                question_type=question_type,
                top_k=TOP_K,
            )

            rows.append(
                {
                    "qid": qid,
                    "question": question,
                    "question_type": question_type,
                    "material_class": row.get("material_class", ""),
                    "source_split": row.get("source_split", ""),
                    "source_doc": row.get("doc_id", ""),
                    "difficulty": row.get("difficulty", ""),
                    "gold_answer": row.get("answer", ""),
                    "gold_reasoning": row.get("reasoning", ""),
                    "baseline_answer": baseline_answer,
                    "rag_answer": rag_answer,
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                    "supporting_chunk_id": row.get("chunk_id", ""),
                }
            )

            pd.DataFrame(rows).to_csv(
                RESULTS_CSV,
                index=False,
                encoding="utf-8-sig",
            )

            processed_qids.add(qid)
            print(f"Saved progress after {qid}")

            time.sleep(0.5)

        except Exception as e:
            print(f"\nERROR on {qid}: {e}")
            print("Progress saved. Re-run script to continue.")
            pd.DataFrame(rows).to_csv(
                RESULTS_CSV,
                index=False,
                encoding="utf-8-sig",
            )
            raise

    print(f"\nDone.")
    print(f"Saved results to: {RESULTS_CSV}")
    print(f"Total processed: {len(rows)}")


if __name__ == "__main__":
    main()