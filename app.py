"""Streamlit web application for the Hallucination-Aware GenAI Assistant.

Provides an interactive user interface featuring:
- Collection discovery and selection
- Real-time index status inspection and index building/rebuilding
- Document-grounded chat interface calling the core pipeline
- Traceable source citations and transparent inspection of all retrieved passages
- Visual indicators for pipeline outcomes (Answered, Partial, Needs clarification, Not enough evidence, Error)
- Pre-retrieval index guards preventing queries against unindexed collections
"""

import sys
from pathlib import Path

# Ensure project root is available on sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st

from src.config import load_config
from src.indexing import build_index, get_embedding_model, index_chunk_count, index_exists
from src.models import PipelineOutcome, PipelineResult
from src.pipeline import run_pipeline
from src.ui_helpers import (
    discover_collections,
    format_chunk_citation_header,
    get_fix_guidance,
    get_outcome_info,
)

# ---------------------------------------------------------------------------
# Page Configuration & Caching
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Hallucination-Aware GenAI Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

config = load_config()


@st.cache_resource(show_spinner=False)
def get_cached_embedding_model(model_name: str):
    """Load and cache the SentenceTransformer embedding model across UI runs."""
    return get_embedding_model(model_name)


# Warm up or load cached embedding model
embedding_model = get_cached_embedding_model(config.embedding_model)

# ---------------------------------------------------------------------------
# Sidebar: Collection Selection, Index Status, & Model Info
# ---------------------------------------------------------------------------

st.sidebar.title("Collection & Index")

# 1. Collection Selector
available_collections = discover_collections(config.data_dir)
if not available_collections:
    available_collections = [config.default_collection]

default_idx = (
    available_collections.index(config.default_collection)
    if config.default_collection in available_collections
    else 0
)

selected_collection = st.sidebar.selectbox(
    "Select Document Collection",
    options=available_collections,
    index=default_idx,
    help="Target document collection used for grounding answers.",
)

# 2. Index Status Inspection
has_index = index_exists(selected_collection, config=config)
chunk_count = index_chunk_count(selected_collection, config=config)

if has_index:
    st.sidebar.success(f"Index built ({chunk_count} chunks)")
else:
    st.sidebar.warning("Index not built (0 chunks)")

# 3. Build or Rebuild Index Button
if st.sidebar.button("Build or rebuild index", use_container_width=True):
    with st.spinner(f"Building index for '{selected_collection}'..."):
        try:
            build_res = build_index(
                collection_name=selected_collection,
                config=config,
                embedding_model=embedding_model,
            )
            files_loaded = build_res.get("files_loaded", 0)
            files_failed = build_res.get("files_failed", 0)
            chunks_indexed = build_res.get("chunks_indexed", 0)
            duration = build_res.get("duration_seconds", 0.0)

            if chunks_indexed > 0:
                st.sidebar.success(
                    f"Index created: {chunks_indexed} chunks from {files_loaded} files "
                    f"in {duration}s."
                )
            else:
                st.sidebar.warning(
                    f"Build completed: 0 chunks created. (Loaded {files_loaded} files, "
                    f"{files_failed} failures). Please verify documents exist in the collection folder."
                )
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Index build failed: {e}")

# 4. Active Model & Configuration Info
st.sidebar.divider()
st.sidebar.subheader("Active Configuration")
st.sidebar.markdown(f"**LLM Model:** `{config.llm_model}`")
st.sidebar.markdown(f"**Embedding:** `{config.embedding_model}`")
st.sidebar.markdown(
    f"**Thresholds:** Strong: `{config.strong_score:.2f}` | Weak: `{config.weak_score:.2f}`"
)
st.sidebar.markdown(
    f"**Retrieval Limits:** Top-K: `{config.top_k}` | Retry Top-K: `{config.retry_top_k}`"
)

# ---------------------------------------------------------------------------
# Main Area: Chat Interface & Outcome Presentation
# ---------------------------------------------------------------------------

st.title("🛡️ Hallucination-Aware GenAI Assistant")
st.caption(
    "Domain-independent question answering with citation grounding, "
    "strict evidence verification, and deterministic abstention."
)

# Missing Index Guidance Banner
if not has_index:
    st.info(
        f"ℹ️ The index for collection **{selected_collection}** has not been built yet. "
        "Asking questions is disabled until the index is created. "
        "Click **'Build or rebuild index'** in the sidebar to index the collection documents."
    )

# Initialize Session State Messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Reset chat if collection switches
if "active_collection" not in st.session_state:
    st.session_state.active_collection = selected_collection
elif st.session_state.active_collection != selected_collection:
    st.session_state.active_collection = selected_collection
    st.session_state.messages = []


def render_assistant_response(msg: dict):
    """Render structured assistant response including outcome, message, citations, and debug passages."""
    outcome_key = msg.get("outcome", PipelineOutcome.ANSWER.value)
    info = get_outcome_info(outcome_key)
    variant = info.get("variant", "info")
    badge_label = info.get("label", str(outcome_key))
    badge_icon = info.get("icon", "ℹ️")

    # 1. Outcome badge
    badge_md = f"### {badge_icon} Status: {badge_label}"
    if variant == "success":
        st.success(badge_md)
    elif variant == "error":
        st.error(badge_md)
    elif variant == "warning":
        st.warning(badge_md)
    else:
        st.info(badge_md)

    # 2. Main message
    st.markdown(msg.get("content", ""))

    # 3. Actionable fix suggestion for error outcomes
    if outcome_key == PipelineOutcome.ERROR.value:
        fix = get_fix_guidance(msg.get("content", ""))
        if fix:
            st.error(fix)

    # 4. Sources Section (Cited chunks only)
    citations = msg.get("citations", [])
    if citations:
        with st.expander(f"📚 Sources ({len(citations)})", expanded=False):
            for idx, chunk in enumerate(citations, start=1):
                st.markdown(f"**Citation [{idx}]** — {format_chunk_citation_header(chunk)}")
                st.text(chunk.text)
                st.divider()

    # 5. All Retrieved Passages (Full transparency & debugging)
    retrieved = msg.get("retrieved", [])
    if retrieved:
        with st.expander(f"🔍 All retrieved passages ({len(retrieved)})", expanded=False):
            for idx, chunk in enumerate(retrieved, start=1):
                st.markdown(f"**Passage [{idx}]** — {format_chunk_citation_header(chunk)}")
                st.text(chunk.text)
                st.divider()

    # 6. Warnings
    warnings = msg.get("warnings", [])
    if warnings:
        for w in warnings:
            st.warning(f"⚠️ {w}")

    # 7. Metadata footer
    latency = msg.get("latency", 0.0)
    top_score = msg.get("top_score", 0.0)
    retried = msg.get("retried", False)
    st.caption(
        f"⏱️ Latency: **{latency:.3f}s** | Top Similarity: **{top_score:.4f}** | "
        f"Retrieval Retried: **{'Yes' if retried else 'No'}**"
    )


# Render Past Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant_response(msg)

# Chat Input with Index Guard
input_placeholder = (
    f"Ask a question about {selected_collection}..."
    if has_index
    else "Build the collection index in the sidebar first to ask questions..."
)

user_query = st.chat_input(input_placeholder, disabled=not has_index)

if user_query:
    # 1. Render user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Execute pipeline and render assistant response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and evaluating..."):
            pipeline_result: PipelineResult = run_pipeline(
                collection=selected_collection,
                question=user_query,
                history=st.session_state.messages,
                config=config,
            )

        outcome_val = (
            pipeline_result.outcome.value
            if isinstance(pipeline_result.outcome, PipelineOutcome)
            else str(pipeline_result.outcome)
        )

        msg_data = {
            "role": "assistant",
            "outcome": outcome_val,
            "content": pipeline_result.message,
            "citations": pipeline_result.citations,
            "retrieved": pipeline_result.retrieved,
            "warnings": pipeline_result.warnings,
            "top_score": pipeline_result.top_score,
            "retried": pipeline_result.retried,
            "latency": pipeline_result.latency_seconds,
        }

        render_assistant_response(msg_data)
        st.session_state.messages.append(msg_data)
