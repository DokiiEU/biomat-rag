import pandas as pd
from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    GENERATION_MODEL,
    PROCESSED_DIR,
    RESULTS_DIR,
)
from retrieve import retrieve

QUESTIONS_CSV = PROCESSED_DIR / "questions_reviewed_1.csv"
RESULTS_CSV = RESULTS_DIR / "qa_results_reviewed_1.csv"

client = OpenAI(api_key=OPENAI_API_KEY)


def format_options(row: pd.Series) -> str:
    """Pretvori option_a...option_f u A)...F) format."""
    options = []
    for letter in ["a", "b", "c", "d", "e", "f"]:
        col = f"option_{letter}"
        if col in row and pd.notna(row[col]):
            options.append(f"{letter.upper()}) {row[col]}")
    return "\n".join(options)


def normalize_letter(text: str) -> str:
    """Vrati samo A-F ako postoji, inače vrati original stripped tekst."""
    if not isinstance(text, str):
        return ""
    text = text.strip().upper()

    if text in {"A", "B", "C", "D", "E", "F"}:
        return text

    # ako model napiše npr. "A)" ili "Answer: C"
    for ch in text:
        if ch in {"A", "B", "C", "D", "E", "F"}:
            return ch

    # ako slučajno vrati Unknown, mapiraj na F
    if "UNKNOWN" in text:
        return "F"

    return text


def ask_baseline(question: str, options: str) -> str:
    prompt = f"""
Answer the multiple-choice biomaterials question.

RULES:
- Select ONLY one option (A, B, C, D, E, or F)
- Output ONLY the letter
- If you are not sure, output EXACTLY: F

Question:
{question}

Options:
{options}

Answer:
"""
    response = client.responses.create(
        model=GENERATION_MODEL,
        input=prompt
    )
    return normalize_letter(response.output_text)


def check_answerable(question: str, options: str, context: str) -> str:
    """
    Procjena može li se odgovoriti iz konteksta.
    Vraća ANSWERABLE ili UNANSWERABLE.
    """
    prompt = f"""
You must decide whether the multiple-choice question can be answered ONLY from the provided context.

STRICT RULES:
- Answer ONLY with: ANSWERABLE or UNANSWERABLE
- ANSWERABLE if at least one option is directly and sufficiently supported by the context
- UNANSWERABLE only if no option is sufficiently supported by the context
- Do NOT require perfect wording match
- Do NOT use outside knowledge
- If the context contains enough evidence for one option, answer ANSWERABLE

Question:
{question}

Options:
{options}

Context:
{context}

Answer:
"""
    response = client.responses.create(
        model=GENERATION_MODEL,
        input=prompt
    )
    text = response.output_text.strip().upper()

    if "UNANSWERABLE" in text:
        return "UNANSWERABLE"
    if "ANSWERABLE" in text:
        return "ANSWERABLE"

    # fallback
    return "ANSWERABLE"


def ask_rag(question: str, options: str, top_k: int = 4) -> tuple[str, str]:
    retrieved = retrieve(question, top_k=top_k)

    context_blocks = []
    used_chunk_ids = []

    for _, row in retrieved.iterrows():
        used_chunk_ids.append(row["chunk_id"])
        context_blocks.append(f"[{row['chunk_id']}] {row['chunk_text']}")

    context = "\n\n".join(context_blocks)

    answerability = check_answerable(
        question=question,
        options=options,
        context=context
    )

    if answerability == "UNANSWERABLE":
        return "F", ";".join(used_chunk_ids)

    prompt = f"""
Answer the multiple-choice biomaterials question using ONLY the provided context.

STRICT RULES:
- Select ONLY one option (A, B, C, D, E, or F)
- Output ONLY the letter
- Use ONLY statements directly present in the context
- Do NOT use prior knowledge
- Do NOT guess
- If no option is sufficiently supported by the context, output EXACTLY: F
- If multiple options seem equally plausible from the context, output EXACTLY: F

Question:
{question}

Options:
{options}

Context:
{context}

Answer:
"""
    response = client.responses.create(
        model=GENERATION_MODEL,
        input=prompt
    )

    return normalize_letter(response.output_text), ";".join(used_chunk_ids)


def main() -> None:
    questions_df = pd.read_csv(QUESTIONS_CSV)

    rows = []

    for _, row in questions_df.iterrows():
        qid = row["qid"]
        question = row["question"]
        options = format_options(row)

        print(f"Processing {qid}...")

        baseline_answer = ask_baseline(question, options)
        rag_answer, retrieved_chunk_ids = ask_rag(
            question=question,
            options=options,
            top_k=4
        )

        rows.append({
            "qid": qid,
            "question": question,
            "material_class": row["material_class"],
            "source_split": row["source_split"],
            "gold_answer": row["correct_answer"],
            "baseline_answer": baseline_answer,
            "rag_answer": rag_answer,
            "retrieved_chunk_ids": retrieved_chunk_ids,
        })

    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved results to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()