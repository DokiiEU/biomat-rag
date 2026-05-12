import json
import time
import pandas as pd
from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError

from config import OPENAI_API_KEY, GENERATION_MODEL, RESULTS_DIR

INPUT_CSV = RESULTS_DIR / "open_qa_results_v1.csv"
OUTPUT_CSV = RESULTS_DIR / "open_qa_judged_v1.csv"

# Za test stavi 10 ili 30.
# Za sve stavi None.
MAX_ROWS = 30

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


def extract_json(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        return {
            "baseline_correctness": 0,
            "baseline_groundedness": 0,
            "baseline_completeness": 0,
            "baseline_hallucination": "unknown",
            "rag_correctness": 0,
            "rag_groundedness": 0,
            "rag_completeness": 0,
            "rag_hallucination": "unknown",
            "winner": "unknown",
            "notes": f"JSON parse failed. Raw output: {text[:500]}",
        }


def judge_row(row: pd.Series) -> dict:
    prompt = f"""
You are evaluating answers for a biomaterials QA benchmark.

Evaluate the BASELINE answer and the RAG answer against the GOLD answer.

Use this rubric:
- correctness: 0 to 5
  0 = completely wrong
  3 = partially correct
  5 = fully correct
- groundedness: 0 to 5
  0 = unsupported / hallucinated
  5 = strongly grounded in the provided gold answer and question intent
- completeness: 0 to 5
  0 = missing key information
  5 = complete but concise
- hallucination:
  "yes", "no", or "partial"

Important:
- For unanswerable questions, the best answer is "Unknown" or a clear statement that the context does not provide the requested information.
- Penalize answers that invent specific mechanisms, numbers, study results, or comparisons not present in the gold answer.
- Do not require exact wording.
- Judge semantic correctness.

Return ONLY valid JSON in this format:

{{
  "baseline_correctness": 0,
  "baseline_groundedness": 0,
  "baseline_completeness": 0,
  "baseline_hallucination": "yes/no/partial",
  "rag_correctness": 0,
  "rag_groundedness": 0,
  "rag_completeness": 0,
  "rag_hallucination": "yes/no/partial",
  "winner": "baseline/rag/tie",
  "notes": "short explanation"
}}

Question type:
{row.get("question_type", "")}

Question:
{row.get("question", "")}

Gold answer:
{row.get("gold_answer", "")}

Baseline answer:
{row.get("baseline_answer", "")}

RAG answer:
{row.get("rag_answer", "")}
"""

    response = safe_call(
        lambda: client.responses.create(
            model=GENERATION_MODEL,
            input=prompt,
            temperature=0,
        )
    )

    return extract_json(response.output_text)


def main():
    df = pd.read_csv(INPUT_CSV)

    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS)

    judged_rows = []

    for _, row in df.iterrows():
        qid = row["qid"]
        print(f"Judging {qid}...")

        judgment = judge_row(row)

        combined = row.to_dict()
        combined.update(judgment)
        judged_rows.append(combined)

        pd.DataFrame(judged_rows).to_csv(
            OUTPUT_CSV,
            index=False,
            encoding="utf-8-sig",
        )

        time.sleep(0.5)

    out = pd.DataFrame(judged_rows)

    print("\n=== SUMMARY ===")
    print(f"Total judged: {len(out)}")

    print("\nWinner counts:")
    print(out["winner"].value_counts())

    print("\nAverage scores:")
    score_cols = [
        "baseline_correctness",
        "baseline_groundedness",
        "baseline_completeness",
        "rag_correctness",
        "rag_groundedness",
        "rag_completeness",
    ]
    print(out[score_cols].mean(numeric_only=True))

    print(f"\nSaved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()