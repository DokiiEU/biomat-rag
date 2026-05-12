import pandas as pd
from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    GENERATION_MODEL,
    PROCESSED_DIR,
    RESULTS_DIR,
)
from retrieve import retrieve

QUESTIONS_CSV = PROCESSED_DIR / "questions.csv"
RESULTS_CSV = RESULTS_DIR / "qa_results.csv"

client = OpenAI(api_key=OPENAI_API_KEY)


def ask_baseline(question: str, question_type: str, options: str) -> str:
    if question_type == "mcq":
        prompt = f"""
Answer the multiple-choice biomaterials question.

RULES:
- Select ONLY one option (A, B, C, or D)
- Output ONLY the letter
- If you are not sure, output exactly: Unknown

Question:
{question}

Options:
{options}

Answer:
"""
    else:
        prompt = f"""
Answer the biomaterials question as clearly as possible.

Question:
{question}

Give a short scientific answer in 2-4 sentences.
"""
    response = client.responses.create(
        model=GENERATION_MODEL,
        input=prompt
    )
    return response.output_text.strip()


def ask_rag(question: str, question_type: str, options: str, top_k: int = 4) -> tuple[str, str]:
    retrieved = retrieve(question, top_k=top_k)

    context_blocks = []
    used_chunk_ids = []

    for _, row in retrieved.iterrows():
        used_chunk_ids.append(row["chunk_id"])
        context_blocks.append(f"[{row['chunk_id']}] {row['chunk_text']}")

    context = "\n\n".join(context_blocks)

    if question_type == "mcq":
        prompt = f"""
Answer the multiple-choice biomaterials question using ONLY the provided context.

STRICT RULES:
- Select ONLY one option (A, B, C, or D)
- Output ONLY the letter
- If the answer is NOT explicitly supported by the context, output EXACTLY: Unknown
- Do NOT guess
- Do NOT use prior knowledge
- Only use information from the context

Question:
{question}

Options:
{options}

Context:
{context}

Answer:
"""
    else:
        prompt = f"""
Answer the biomaterials question using ONLY the provided context.

STRICT RULES:
- If the answer is NOT explicitly supported by the context, output EXACTLY: Unknown
- Do NOT guess
- Do NOT use prior knowledge
- Only use information from the context
- Be concise
- For short_answer, output a short phrase if possible
- For open_ended / reasoning / multi_hop, answer in 2-4 sentences

Question:
{question}

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

    # ako options stupac ne postoji, dodaj prazan
    if "options" not in questions_df.columns:
        questions_df["options"] = ""

    rows = []

    for _, row in questions_df.iterrows():
        qid = row["qid"]
        question = row["question"]
        question_type = row["question_type"]
        options = row["options"] if pd.notna(row["options"]) else ""

        print(f"Processing {qid}...")

        baseline_answer = ask_baseline(question, question_type, options)
        rag_answer, retrieved_chunk_ids = ask_rag(
            question=question,
            question_type=question_type,
            options=options,
            top_k=4
        )

        rows.append({
            "qid": qid,
            "question": question,
            "question_type": question_type,
            "material_class": row["material_class"],
            "source_split": row["source_split"],
            "options": options,
            "gold_answer": row["gold_answer"],
            "baseline_answer": baseline_answer,
            "rag_answer": rag_answer,
            "retrieved_chunk_ids": retrieved_chunk_ids,
        })

    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved results to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()