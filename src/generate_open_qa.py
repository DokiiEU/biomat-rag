import json
import random
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from config import OPENAI_API_KEY, GENERATION_MODEL, PROCESSED_DIR

FILTERED_CHUNKS_PATH = PROCESSED_DIR / "filtered_chunks.csv"
OUTPUT_PATH = PROCESSED_DIR / "open_qa_dataset1.csv"

QUESTIONS_PER_CHUNK = 2
MAX_CHUNKS = 100

client = OpenAI(api_key=OPENAI_API_KEY)


PROMPT_TEMPLATE = """
You are generating a high-quality benchmark dataset for evaluating Retrieval-Augmented Generation (RAG) systems in the biomaterials domain.

Your task is to generate EXACTLY 2 high-quality questions from the provided scientific context.

The benchmark should evaluate:
- retrieval quality
- scientific reasoning
- hallucination resistance
- contextual understanding
- multi-hop reasoning

QUESTION QUALITY REQUIREMENTS:
- Questions must be scientifically meaningful and realistic.
- Avoid trivial definition-style questions.
- Avoid generic textbook questions.
- Prefer reasoning, comparison, application-oriented, synthesis, or analytical questions.
- Questions should require understanding of the provided context.
- Do NOT copy sentences directly from the context.
- Do NOT generate repetitive or semantically similar questions.
- Questions should resemble graduate-level biomaterials questions.

ALLOWED QUESTION TYPES:
- factual
- reasoning
- comparison
- multi-hop
- unanswerable

QUESTION TYPE GUIDELINES:

factual:
- answerable directly from context
- but should still require retrieval

reasoning:
- requires connecting multiple concepts from the context

comparison:
- compare materials, mechanisms, properties, applications, or limitations

multi-hop:
- requires combining multiple pieces of information from the context

unanswerable:
- should sound scientifically plausible
- should appear related to the context
- BUT the answer must genuinely NOT exist in the context
- do NOT make joke or absurd questions
- use realistic missing scientific details

IMPORTANT:
- At least some generated questions should be unanswerable when appropriate.
- Avoid generating questions answerable from a single obvious sentence.
- Questions should test whether retrieval actually helps.

ANSWER REQUIREMENTS:
- Answers must be concise.
- Maximum 1–3 sentences.
- Scientifically precise.
- No unnecessary filler text.

REASONING FIELD:
- Briefly explain what information from the context supports the answer.
- For unanswerable questions, explain why the context is insufficient.

DIFFICULTY GUIDELINES:
- easy = direct retrieval
- medium = requires interpretation
- hard = requires synthesis or multi-step reasoning

Return ONLY valid JSON.

Required format:
[
  {{
    "question": "...",
    "question_type": "...",
    "answer": "...",
    "reasoning": "...",
    "difficulty": "easy/medium/hard"
  }},
  {{
    "question": "...",
    "question_type": "...",
    "answer": "...",
    "reasoning": "...",
    "difficulty": "easy/medium/hard"
  }}
]

Context:
\"\"\"
{context}
\"\"\"
"""


def generate_questions(context: str):
    prompt = PROMPT_TEMPLATE.format(context=context)

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": "You are an expert biomaterials dataset generator."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.choices[0].message.content.strip()

    try:
        data = json.loads(content)
        return data
    except Exception:
        print("Failed to parse JSON:")
        print(content)
        return []


def main():
    chunks = pd.read_csv(FILTERED_CHUNKS_PATH)

    print(f"Loaded chunks: {len(chunks)}")

    chunks = chunks.sample(
        n=min(MAX_CHUNKS, len(chunks)),
        random_state=42
    ).reset_index(drop=True)

    rows = []

    qid_counter = 1

    for _, row in tqdm(chunks.iterrows(), total=len(chunks)):
        context = row["chunk_text"]

        generated = generate_questions(context)

        for item in generated:
            rows.append({
                "qid": f"OQ{qid_counter:04d}",
                "question": item.get("question", ""),
                "question_type": item.get("question_type", ""),
                "answer": item.get("answer", ""),
                "reasoning": item.get("reasoning", ""),
                "difficulty": item.get("difficulty", ""),
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "title": row["title"],
                "material_class": row["material_class"],
                "source_split": row["source_split"],
            })

            qid_counter += 1

        time.sleep(0.5)

    df = pd.DataFrame(rows)

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\nSaved dataset: {OUTPUT_PATH}")
    print(f"Total questions: {len(df)}")


if __name__ == "__main__":
    main()