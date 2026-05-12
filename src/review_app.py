import pandas as pd
import streamlit as st

from config import PROCESSED_DIR, RESULTS_DIR

# =========================================================
# FILES
# =========================================================

# MCQ
MCQ_RESULTS_CSV = RESULTS_DIR / "qa_results_reviewed_1.csv"
MCQ_QUESTIONS_CSV = PROCESSED_DIR / "questions_reviewed_1.csv"

# OPEN QA
OPEN_RESULTS_CSV = RESULTS_DIR / "open_qa_results_v1.csv"
OPEN_QUESTIONS_CSV = PROCESSED_DIR / "open_qa_dataset1.csv"

# CHUNKS
CHUNKS_CSV = PROCESSED_DIR / "chunks.csv"


# =========================================================
# LOADERS
# =========================================================

@st.cache_data
@st.cache_data
def load_mcq_data():
    results = pd.read_csv(MCQ_RESULTS_CSV)
    questions = pd.read_csv(MCQ_QUESTIONS_CSV)
    print("MCQ RESULTS COLUMNS:")
    print(results.columns.tolist())

    print("MCQ QUESTIONS COLUMNS:")
    print(questions.columns.tolist())

    needed_cols = [
        "qid",
        "question",
        "material_class",
        "source_split",
        "gold_answer",
        "source_doc",
        "option_a",
        "option_b",
        "option_c",
        "option_d",
        "option_e",
        "option_f",
    ]

    existing_cols = [c for c in needed_cols if c in questions.columns]

    merge_cols = [c for c in existing_cols if c not in results.columns or c == "qid"]

    df = results.merge(
        questions[merge_cols],
        on="qid",
        how="left",
    )

    if "gold_answer" in df.columns:
        df["baseline_correct"] = (
            df["baseline_answer"].astype(str).str.strip()
            == df["gold_answer"].astype(str).str.strip()
        )

        df["rag_correct"] = (
            df["rag_answer"].astype(str).str.strip()
            == df["gold_answer"].astype(str).str.strip()
        )
    else:
        df["baseline_correct"] = False
        df["rag_correct"] = False

    return df


@st.cache_data
def load_open_data():
    results = pd.read_csv(OPEN_RESULTS_CSV)
    questions = pd.read_csv(OPEN_QUESTIONS_CSV)
    print("OPEN RESULTS COLUMNS:")
    print(results.columns.tolist())

    print("OPEN QUESTIONS COLUMNS:")
    print(questions.columns.tolist())
    needed_cols = [
        "qid",
        "question",
        "question_type",
        "answer",
        "reasoning",
        "difficulty",
        "chunk_id",
        "doc_id",
        "title",
        "material_class",
        "source_split",
    ]

    existing_cols = [c for c in needed_cols if c in questions.columns]

    merge_cols = [c for c in existing_cols if c not in results.columns or c == "qid"]

    df = results.merge(
        questions[merge_cols],
        on="qid",
        how="left",
    )

    return df


@st.cache_data
def load_chunk_map():
    chunks_df = pd.read_csv(CHUNKS_CSV)

    chunk_map = {}

    for _, row in chunks_df.iterrows():
        chunk_map[str(row["chunk_id"]).strip()] = {
            "title": str(row["title"]) if pd.notna(row["title"]) else "",
            "doc_id": str(row["doc_id"]) if pd.notna(row["doc_id"]) else "",
            "filename": str(row["filename"]) if pd.notna(row["filename"]) else "",
            "chunk_text": str(row["chunk_text"]) if pd.notna(row["chunk_text"]) else "",
            "material_class": str(row["material_class"]) if pd.notna(row["material_class"]) else "",
            "source_split": str(row["source_split"]) if pd.notna(row["source_split"]) else "",
        }

    return chunk_map


# =========================================================
# HELPERS
# =========================================================

def badge(text: str, color: str) -> str:
    return (
        f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        f"background:{color};color:white;font-size:0.85rem;font-weight:600;'>"
        f"{text}</span>"
    )


def find_index_by_qid(df: pd.DataFrame, qid: str):
    matches = df.index[df["qid"].astype(str).str.upper() == qid.upper()].tolist()
    return matches[0] if matches else None


def render_chunks(retrieved_chunk_ids, chunk_map):
    st.write("## Retrieved chunks")

    retrieved = str(retrieved_chunk_ids)
    chunk_list = [x.strip() for x in retrieved.split(";") if x.strip()]

    if not chunk_list:
        st.info("No chunk IDs found.")
        return

    for i, cid in enumerate(chunk_list, start=1):
        chunk_info = chunk_map.get(cid)

        if chunk_info is None:
            with st.expander(f"{i}. {cid}"):
                st.warning("Chunk not found in chunks.csv")

        else:
            with st.expander(f"{i}. {cid} | {chunk_info['title']}"):

                st.markdown(
                    f"""
                    **Doc ID:** {chunk_info['doc_id']}  
                    **Material class:** {chunk_info['material_class']}  
                    **Source split:** {chunk_info['source_split']}
                    """
                )

                st.write("### Chunk text")
                st.write(chunk_info["chunk_text"])


# =========================================================
# PAGE
# =========================================================

st.set_page_config(page_title="Biomaterials QA Review", layout="wide")

st.title("Biomaterials QA Review")

mode = st.sidebar.radio(
    "Dataset type",
    ["MCQ", "Open QA"],
)

chunk_map = load_chunk_map()

# =========================================================
# MCQ MODE
# =========================================================

if mode == "MCQ":

    st.caption("MCQ benchmark review")

    df = load_mcq_data()

    # =====================================================
    # FILTERS
    # =====================================================

    st.sidebar.header("Filters")

    search_text = st.sidebar.text_input("Search")

    material_filter = st.sidebar.multiselect(
        "Material class",
        options=sorted(df["material_class"].dropna().unique().tolist()),
        default=sorted(df["material_class"].dropna().unique().tolist()),
    )

    source_filter = st.sidebar.multiselect(
        "Source split",
        options=sorted(df["source_split"].dropna().unique().tolist()),
        default=sorted(df["source_split"].dropna().unique().tolist()),
    )

    show_only_errors = st.sidebar.checkbox("Show only errors")

    filtered = df.copy()

    if search_text.strip():
        q = search_text.strip().lower()

        filtered = filtered[
            filtered["qid"].astype(str).str.lower().str.contains(q)
            | filtered["question"].astype(str).str.lower().str.contains(q)
        ]

    filtered = filtered[
        filtered["material_class"].isin(material_filter)
        & filtered["source_split"].isin(source_filter)
    ]

    if show_only_errors:
        filtered = filtered[
            (~filtered["baseline_correct"])
            | (~filtered["rag_correct"])
        ]

    filtered = filtered.reset_index(drop=True)

    # =====================================================
    # SESSION
    # =====================================================

    if "mcq_idx" not in st.session_state:
        st.session_state.mcq_idx = 0

    st.session_state.mcq_idx = max(
        0,
        min(st.session_state.mcq_idx, len(filtered) - 1),
    )

    # =====================================================
    # STATS
    # =====================================================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Questions", len(filtered))
    c2.metric("Baseline correct", int(filtered["baseline_correct"].sum()))
    c3.metric("RAG correct", int(filtered["rag_correct"].sum()))
    c4.metric(
        "RAG improvement",
        int(filtered["rag_correct"].sum() - filtered["baseline_correct"].sum()),
    )

    # =====================================================
    # NAVIGATION
    # =====================================================

    nav1, nav2, nav3 = st.columns([1, 1, 2])

    with nav1:
        if st.button("⬅ Prev"):
            st.session_state.mcq_idx = max(
                0,
                st.session_state.mcq_idx - 1,
            )

    with nav2:
        if st.button("Next ➡"):
            st.session_state.mcq_idx = min(
                len(filtered) - 1,
                st.session_state.mcq_idx + 1,
            )

    with nav3:
        jump_qid = st.text_input("Go to QID")

        if st.button("Jump"):
            idx = find_index_by_qid(filtered, jump_qid)

            if idx is not None:
                st.session_state.mcq_idx = idx

    current = filtered.iloc[st.session_state.mcq_idx]

    # =====================================================
    # META
    # =====================================================

    st.markdown(
        f"### {current['qid']} · Question {st.session_state.mcq_idx + 1}/{len(filtered)}"
    )

    meta1, meta2, meta3, meta4 = st.columns(4)

    with meta1:
        st.markdown(
            badge(f"material: {current['material_class']}", "#2563eb"),
            unsafe_allow_html=True,
        )

    with meta2:
        st.markdown(
            badge(f"source: {current['source_split']}", "#ea580c"),
            unsafe_allow_html=True,
        )

    with meta3:
        st.markdown(
            badge(f"gold: {current['gold_answer']}", "#7c3aed"),
            unsafe_allow_html=True,
        )

    with meta4:
        st.markdown(
            badge(f"source doc: {current['source_doc']}", "#475569"),
            unsafe_allow_html=True,
        )

    # =====================================================
    # QUESTION
    # =====================================================

    st.write("## Question")
    st.write(current["question"])

    st.write("## Options")

    options = [
        ("A", current.get("option_a", "")),
        ("B", current.get("option_b", "")),
        ("C", current.get("option_c", "")),
        ("D", current.get("option_d", "")),
        ("E", current.get("option_e", "")),
        ("F", current.get("option_f", "")),
    ]

    for letter, value in options:

        value = value if pd.notna(value) else ""

        prefix = "✅ " if current["gold_answer"] == letter else ""

        st.write(f"{prefix}{letter}) {value}")

    # =====================================================
    # ANSWERS
    # =====================================================

    a1, a2, a3 = st.columns(3)

    with a1:
        st.subheader("Gold")
        st.code(current["gold_answer"])

    with a2:
        st.subheader("Baseline")
        st.code(current["baseline_answer"])

    with a3:
        st.subheader("RAG")
        st.code(current["rag_answer"])

    # =====================================================
    # CHUNKS
    # =====================================================

    render_chunks(current["retrieved_chunk_ids"], chunk_map)

    st.divider()

    st.write("## Quick list")

    st.dataframe(
        filtered[
            [
                "qid",
                "material_class",
                "source_split",
                "gold_answer",
                "baseline_answer",
                "rag_answer",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

# =========================================================
# OPEN QA MODE
# =========================================================

else:

    st.caption("Open-ended QA benchmark review")

    df = load_open_data()

    # =====================================================
    # FILTERS
    # =====================================================

    st.sidebar.header("Open QA Filters")

    search_text = st.sidebar.text_input("Search")

    type_filter = st.sidebar.multiselect(
        "Question type",
        options=sorted(df["question_type"].dropna().unique().tolist()),
        default=sorted(df["question_type"].dropna().unique().tolist()),
    )

    material_filter = st.sidebar.multiselect(
        "Material class",
        options=sorted(df["material_class"].dropna().unique().tolist()),
        default=sorted(df["material_class"].dropna().unique().tolist()),
    )

    difficulty_filter = st.sidebar.multiselect(
        "Difficulty",
        options=sorted(df["difficulty"].dropna().unique().tolist()),
        default=sorted(df["difficulty"].dropna().unique().tolist()),
    )

    filtered = df.copy()

    if search_text.strip():
        q = search_text.strip().lower()

        filtered = filtered[
            filtered["qid"].astype(str).str.lower().str.contains(q)
            | filtered["question"].astype(str).str.lower().str.contains(q)
        ]

    filtered = filtered[
        filtered["question_type"].isin(type_filter)
        & filtered["material_class"].isin(material_filter)
        & filtered["difficulty"].isin(difficulty_filter)
    ]

    filtered = filtered.reset_index(drop=True)

    # =====================================================
    # SESSION
    # =====================================================

    if "open_idx" not in st.session_state:
        st.session_state.open_idx = 0

    st.session_state.open_idx = max(
        0,
        min(st.session_state.open_idx, len(filtered) - 1),
    )

    # =====================================================
    # STATS
    # =====================================================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Questions", len(filtered))
    c2.metric("Reasoning", int((filtered["question_type"] == "reasoning").sum()))
    c3.metric("Unanswerable", int((filtered["question_type"] == "unanswerable").sum()))
    c4.metric("Hard", int((filtered["difficulty"] == "hard").sum()))

    # =====================================================
    # NAVIGATION
    # =====================================================

    nav1, nav2, nav3 = st.columns([1, 1, 2])

    with nav1:
        if st.button("⬅ Prev", key="open_prev"):
            st.session_state.open_idx = max(
                0,
                st.session_state.open_idx - 1,
            )

    with nav2:
        if st.button("Next ➡", key="open_next"):
            st.session_state.open_idx = min(
                len(filtered) - 1,
                st.session_state.open_idx + 1,
            )

    with nav3:
        jump_qid = st.text_input("Go to Open QID")

        if st.button("Jump Open"):
            idx = find_index_by_qid(filtered, jump_qid)

            if idx is not None:
                st.session_state.open_idx = idx

    current = filtered.iloc[st.session_state.open_idx]

    # =====================================================
    # META
    # =====================================================

    st.markdown(
        f"### {current['qid']} · Open Question {st.session_state.open_idx + 1}/{len(filtered)}"
    )

    meta1, meta2, meta3, meta4, meta5 = st.columns(5)

    with meta1:
        st.markdown(
            badge(f"type: {current['question_type']}", "#7c3aed"),
            unsafe_allow_html=True,
        )

    with meta2:
        st.markdown(
            badge(f"difficulty: {current['difficulty']}", "#dc2626"),
            unsafe_allow_html=True,
        )

    with meta3:
        st.markdown(
            badge(f"material: {current['material_class']}", "#2563eb"),
            unsafe_allow_html=True,
        )

    with meta4:
        st.markdown(
            badge(f"source: {current['source_split']}", "#ea580c"),
            unsafe_allow_html=True,
        )

    with meta5:
        st.markdown(
            badge(f"doc: {current['doc_id']}", "#475569"),
            unsafe_allow_html=True,
        )

    # =====================================================
    # QUESTION
    # =====================================================

    st.write("## Question")
    st.write(current["question"])

    # =====================================================
    # GOLD ANSWER
    # =====================================================

    st.write("## Gold Answer")
    st.success(current["answer"])

    with st.expander("Gold reasoning"):
        st.write(current["reasoning"])

    # =====================================================
    # LLM ANSWERS
    # =====================================================

    a1, a2 = st.columns(2)

    with a1:
        st.subheader("Baseline Answer")

        if pd.notna(current.get("baseline_answer")):
            st.write(current["baseline_answer"])
        else:
            st.warning("No baseline answer found.")

    with a2:
        st.subheader("RAG Answer")

        if pd.notna(current.get("rag_answer")):
            st.write(current["rag_answer"])
        else:
            st.warning("No RAG answer found.")

    # =====================================================
    # SOURCE CHUNK
    # =====================================================

    st.write("## Original source chunk")

    source_chunk_id = str(current["chunk_id"])

    chunk_info = chunk_map.get(source_chunk_id)

    if chunk_info:

        with st.expander(f"{source_chunk_id} | {chunk_info['title']}"):

            st.markdown(
                f"""
                **Doc ID:** {chunk_info['doc_id']}  
                **Material class:** {chunk_info['material_class']}  
                **Source split:** {chunk_info['source_split']}
                """
            )

            st.write(chunk_info["chunk_text"])

    # =====================================================
    # RETRIEVED CHUNKS
    # =====================================================

    if "retrieved_chunk_ids" in current:
        render_chunks(current["retrieved_chunk_ids"], chunk_map)

    st.divider()

    st.write("## Quick list")

    st.dataframe(
        filtered[
            [
                "qid",
                "question_type",
                "difficulty",
                "material_class",
                "source_split",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )