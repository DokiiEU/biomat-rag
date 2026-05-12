import pandas as pd
from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    GENERATION_MODEL,
    PROCESSED_DIR,
    RESULTS_DIR,
)
from retrieve import retrieve

# koristi shuffled pitanja
QUESTIONS_CSV = PROCESSED_DIR / "questions_shuffled_final.csv"
RESULTS_CSV = RESULTS_DIR / "qa_results_shuffled_v2.csv"

client = OpenAI(api_key=OPENAI_API_KEY)


def format_options(row: pd.Series) -> str:
    """Pretvori option_a ... option_f u format A) ... F) ..."""
    options = []
    for letter in ["a", "b", "c", "d", "e", "f"]:
        col = f"option_{letter}"
        if col in row and pd.notna(row[col]):
            options.append(f"{letter.upper()}) {row[col]}")
    return "\n".join(options)


def ask_baseline(question: str, options: str) -> str:
    prompt = f"""
Answer the multiple-choice biomaterials question.

RULES:
- Select ONLY one option (A, B, C, D, E, or F)
- Output ONLY the letter
- If you are not sure, output exactly: Unknown

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
    return response.output_text.strip()


def check_answerable(question: str, options: str, context: str) -> str:
    """
    Vrati YES ili NO ovisno postoji li odgovor eksplicitno u kontekstu.
    """
    prompt = f"""
You must decide whether the multiple-choice question can be answered ONLY from the provided context.

STRICT RULES:
- Answer ONLY with: YES or NO
- Answer NO if:
  - the answer is missing
  - the evidence is partial
  - the context is ambiguous
  - the question cannot be answered with confidence from the context alone
- Do NOT guess
- Do NOT solve the question
- Do NOT output anything except YES or NO

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

    if "YES" in text:
        return "YES"
    return "NO"


def ask_rag(question: str, options: str, top_k: int = 4) -> tuple[str, str]:
    retrieved = retrieve(question, top_k=top_k)

    context_blocks = []
    used_chunk_ids = []

    for _, row in retrieved.iterrows():
        used_chunk_ids.append(row["chunk_id"])
        context_blocks.append(f"[{row['chunk_id']}] {row['chunk_text']}")

    context = "\n\n".join(context_blocks)

    # 🔥 NOVO: prvo provjeri može li se uopće odgovoriti iz konteksta
    answerable = check_answerable(question=question, options=options, context=context)

    if answerable == "NO":
        return "Unknown", ";".join(used_chunk_ids)

    prompt = f"""
Answer the multiple-choice biomaterials question using ONLY the provided context.

STRICT RULES:
- Select ONLY one option (A, B, C, D, E, or F)
- Output ONLY the letter
- If the answer is NOT explicitly supported by the context, output EXACTLY: Unknown
- Do NOT guess
- Do NOT use prior knowledge
- Do NOT infer beyond the context

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

    return response.output_text.strip(), ";".join(used_chunk_ids)


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