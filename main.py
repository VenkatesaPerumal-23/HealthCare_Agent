"""
Streamlit UI for HealthGuard AI. No LLM, retrieval, or graph-node logic lives
here — this file takes user input, calls the compiled graph, and renders the
result (answer, sources, agent details, sidebar controls).

"""
import os

import streamlit as st

st.set_page_config(
    page_title="HealthGuard AI",
    page_icon="🩺",
    layout="wide",
)

from config import KNOWLEDGE_BASE_DIR, LANGFUSE_AVAILABLE, LANGFUSE_ENABLED, langfuse_client
from retrieval import (
    create_retriever,
    ingest_pdfs_into_vectordb,
    create_vectorstore,
    _discover_pdfs,
)
from graph import build_graph, run_graph_turn


# ============================================================
# PRESENTATION HELPERS (UI only — no app logic)
# ============================================================

EXAMPLE_QUESTIONS = [
    "How can dengue be prevented?",
    "What are the warning signs of severe malaria?",
    "What joint symptoms does chikungunya cause?",
    "How does Zika affect pregnancy?",
]


def inject_css():
    """One small style pass. Kept conservative so it survives Streamlit upgrades:
    it styles our own badge spans rather than overriding Streamlit internals."""
    st.markdown(
        """
        <style>
          /* Tighten the main column and give content room to breathe */
          .block-container { padding-top: 2.5rem; max-width: 1100px; }

          /* Badge pills used in Agent details */
          .hg-badges { display: flex; flex-wrap: wrap; gap: .4rem; margin: .2rem 0 .6rem 0; }
          .hg-badge {
            display: inline-flex; align-items: center; gap: .35rem;
            padding: .18rem .6rem; border-radius: 999px;
            font-size: .78rem; font-weight: 600; line-height: 1.4;
            border: 1px solid transparent; white-space: nowrap;
          }
          .hg-badge .hg-k { opacity: .65; font-weight: 500; }
          .hg-good    { background: rgba(34,197,94,.14);  color: #4ade80; border-color: rgba(34,197,94,.30); }
          .hg-warn    { background: rgba(245,158,11,.14); color: #fbbf24; border-color: rgba(245,158,11,.30); }
          .hg-bad     { background: rgba(239,68,68,.14);  color: #f87171; border-color: rgba(239,68,68,.30); }
          .hg-info    { background: rgba(59,130,246,.14); color: #60a5fa; border-color: rgba(59,130,246,.30); }
          .hg-neutral { background: rgba(148,163,184,.14);color: #cbd5e1; border-color: rgba(148,163,184,.28); }

          /* Example-question starter buttons: full width, left aligned */
          div[data-testid="stButton"] button { border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _confidence_kind(value: str) -> str:
    return {"high": "hg-good", "medium": "hg-warn", "low": "hg-bad"}.get(
        str(value).lower(), "hg-neutral"
    )


def _evidence_kind(value: str) -> str:
    value = str(value).lower()
    if value in {"supported", "full"}:
        return "hg-good"
    if value in {"partial", "unchecked", "not_checked"}:
        return "hg-warn"
    if value in {"no_evidence", "none", "unsupported"}:
        return "hg-bad"
    return "hg-neutral"


def _badge(key: str, value: str, kind: str) -> str:
    return f'<span class="hg-badge {kind}"><span class="hg-k">{key}</span> {value}</span>'


def render_agent_details(metadata: dict):
    """Colour-coded badges + retrieval metrics, with the raw JSON kept available."""
    intent = metadata.get("intent", "unknown")
    disease = metadata.get("detected_disease", "none")
    confidence = metadata.get("confidence", "unknown")
    evidence = metadata.get("evidence_status", "unknown")
    attempts = metadata.get("generation_attempts", 0)

    badges = [
        _badge("intent", intent, "hg-info"),
        _badge("disease", disease, "hg-info" if disease not in ("none", "", None) else "hg-neutral"),
        _badge("confidence", confidence, _confidence_kind(confidence)),
        _badge("evidence", evidence, _evidence_kind(evidence)),
        _badge("attempts", str(attempts), "hg-warn" if attempts and int(attempts) > 1 else "hg-neutral"),
    ]
    st.markdown('<div class="hg-badges">' + "".join(badges) + "</div>", unsafe_allow_html=True)

    issues = metadata.get("validation_issues") or []
    if issues:
        st.caption("⚠️ Validation notes: " + "; ".join(str(i) for i in issues))

    retrieval = metadata.get("retrieval")
    if retrieval:
        cols = st.columns(3)
        cols[0].metric("Candidates fetched", retrieval.get("candidates_fetched", "—"))
        cols[1].metric("Chunks kept", retrieval.get("chunks_kept", "—"))
        cols[2].metric("Coverage", str(retrieval.get("evidence_coverage", "—")))
        filt = retrieval.get("disease_filter")
        line = f"Disease filter: **{filt}**" if filt else "Disease filter: none (searched all)"
        if retrieval.get("disease_filter_fellback"):
            line += " — fell back to unfiltered (nothing under that disease)"
        line += f"  ·  threshold {retrieval.get('relevance_threshold', '—')}"
        st.caption(line)
        if retrieval.get("coverage_gaps"):
            st.caption("Gaps: " + str(retrieval["coverage_gaps"]))

    trace_url = metadata.get("langfuse_trace_url")
    if trace_url:
        st.caption(f"🔗 [View trace in Langfuse]({trace_url})")

    with st.expander("Raw metadata (JSON)"):
        st.json(metadata)


def render_sources(sources: list):
    st.caption("Citation labels [S1], [S2], etc. follow this source-list order.")
    for source_index, source in enumerate(sources, start=1):
        title = source.get("title") or source.get("source") or "Source"
        url = source.get("source")
        page = source.get("page")
        disease = source.get("disease")
        extra = f" — page {int(page) + 1}" if page is not None else ""
        tag = f" · _{disease}_" if disease and disease not in ("unspecified", None) else ""
        if url and str(url).startswith("https://"):
            st.markdown(f"- **[S{source_index}]** [{title}]({url}){extra}{tag}")
        else:
            st.markdown(f"- **[S{source_index}]** {title}{extra}{tag}")


# ============================================================
# MAIN
# ============================================================

def main():
    inject_css()

    st.title("🩺 HealthGuard AI")
    st.caption(
        "Agentic healthcare information assistant — educational use only, "
        "not a substitute for professional medical care."
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        st.subheader("📚 Medical knowledge base")

        if os.path.isdir(KNOWLEDGE_BASE_DIR):
            # Walk the disease sub-folders (knowledge-base/<disease>/*.pdf), the
            # same way ingestion does, so the count here matches what gets indexed.
            pdf_entries = _discover_pdfs(KNOWLEDGE_BASE_DIR)
            st.info(f"Found {len(pdf_entries)} PDF(s)")
            if pdf_entries:
                by_disease = {}
                for _path, name, disease in pdf_entries:
                    by_disease.setdefault(disease, []).append(name)
                with st.expander("View indexed files"):
                    for disease in sorted(by_disease):
                        st.markdown(f"**{disease}**")
                        for name in sorted(by_disease[disease]):
                            st.caption(f"📄 {name}")
        else:
            st.warning(f"Directory '{KNOWLEDGE_BASE_DIR}' not found.")

        if st.button("🔄 Ingest / update PDFs", use_container_width=True):
            with st.spinner("Indexing PDF chunks into Chroma..."):
                try:
                    ingest_pdfs_into_vectordb.clear()
                    count = ingest_pdfs_into_vectordb()
                    create_vectorstore.clear()
                    create_retriever.clear()
                    if count:
                        st.success(f"Indexed {count} chunks.")
                    else:
                        st.warning("No PDF chunks were indexed.")
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

        st.subheader("💾 Vector database")
        try:
            store = create_vectorstore()
            if store is not None:
                st.success(f"Chroma ready ({store._collection.count()} chunks)")
            else:
                st.warning("No indexed chunks found. Ingest PDFs first.")
        except Exception:
            st.warning("Vector database is not ready.")

        st.divider()
        st.subheader("📈 Observability")
        if LANGFUSE_ENABLED:
            st.success("Langfuse tracing is active.")
        elif LANGFUSE_AVAILABLE:
            st.info("Langfuse installed but LANGFUSE_PUBLIC_KEY is not set in .env.")
        else:
            st.caption("Langfuse not installed — `pip install langfuse` to enable tracing.")

        st.divider()
        with st.expander("🧠 App capabilities"):
            st.markdown(
                "- Safety-first symptom guidance\n"
                "- Top-k PDF retrieval and trusted web research\n"
                "- Evidence-grounded answers with [S1] citation labels\n"
                "- Evidence validation and one bounded retry\n"
                "- Confidence, evidence status, and validation issues\n"
                "- Conversation context for follow-up questions\n"
                "- Per-chat LangGraph checkpointing (in-memory)"
            )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = os.urandom(16).hex()

    # Empty-state welcome with clickable example questions. Clicking one stores it
    # as a pending question (picked up below), so the chat starts with one tap.
    if not st.session_state.messages:
        st.markdown("#### 👋 Ask about a vector-borne disease")
        st.caption("Try one of these, or type your own question below.")
        cols = st.columns(2)
        for i, example in enumerate(EXAMPLE_QUESTIONS):
            if cols[i % 2].button(example, use_container_width=True, key=f"ex_{i}"):
                st.session_state.pending_question = example
                st.rerun()

    # Show earlier messages on every Streamlit rerun.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("metadata"):
                with st.expander("🧠 Agent details"):
                    render_agent_details(message["metadata"])
            if message.get("sources"):
                with st.expander("📚 Sources"):
                    render_sources(message["sources"])

    # Question can come from a clicked example (pending) or the chat box.
    pending_question = st.session_state.pop("pending_question", None)
    typed_question = st.chat_input("Ask a healthcare question...")
    question = pending_question or typed_question

    if question:
        # Include a bounded amount of earlier conversation in the LLM prompts.
        previous_messages = st.session_state.messages[-12:]  # most recent 12 messages
        conversation_history = "\n".join(
            f'{message["role"].upper()}: {message["content"]}'
            for message in previous_messages
        )
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("HealthGuard AI is working..."):
                try:
                    app = build_graph()
                    inputs = {
                        "question": question,
                        # Reset every turn: MemorySaver would otherwise carry the
                        # previous turn's resolved question into this one.
                        "standalone_question": "",
                        "disease": "",
                        "intent": "",
                        "symptoms": [],
                        "duration": "unknown",
                        "severity": "unknown",
                        "documents": [],
                        "sender": "",
                        "answer": "",
                        "urgency": "unknown",
                        "triage_reason": "",
                        "sources": [],
                        "confidence": "unknown",
                        "evidence_status": "unknown",
                        "validation_feedback": "",
                        "validation_issues": [],
                        "generation_attempts": 0,
                        "retry_needed": False,
                        "conversation_history": conversation_history,
                        # Reset every turn so diagnostics never leak from a previous question.
                        "retrieval_diagnostics": {},
                    }
                    result = run_graph_turn(app, inputs, st.session_state.thread_id)
                    if LANGFUSE_ENABLED and langfuse_client is not None:
                        # Streamlit reruns the whole script per interaction and may exit
                        # before the background batching thread sends queued events;
                        # flush synchronously so this turn's trace isn't lost.
                        langfuse_client.flush()
                    answer = result.get("answer", "I couldn't generate an answer.")
                    st.markdown(answer)

                    metadata = {
                        "intent": result.get("intent", "unknown"),
                        "resolved_question": result.get("standalone_question", ""),
                        "detected_disease": result.get("disease", "") or "none",
                        "final_node": result.get("sender", "unknown"),
                        "confidence": result.get("confidence", "unknown"),
                        "evidence_status": result.get("evidence_status", "unknown"),
                        "generation_attempts": result.get("generation_attempts", 0),
                        "validation_issues": result.get("validation_issues", []),
                    }
                    if result.get("retrieval_diagnostics"):
                        metadata["retrieval"] = result["retrieval_diagnostics"]
                    # run_graph_turn captures this while the trace span is still
                    # active; calling get_trace_url() out here (after the span has
                    # closed) returns nothing, which is why the link was missing.
                    if result.get("langfuse_trace_url"):
                        metadata["langfuse_trace_url"] = result["langfuse_trace_url"]
                    if result.get("intent") == "symptom_check":
                        metadata.update({
                            "symptoms": result.get("symptoms", []),
                            "duration": result.get("duration", "unknown"),
                            "stated_severity": result.get("severity", "unknown"),
                            "urgency": result.get("urgency", "unknown"),
                            "triage_reason": result.get("triage_reason", ""),
                        })
                        st.warning(
                            "This is automated general guidance, not a validated "
                            "medical triage assessment. Seek professional care when concerned."
                        )

                    with st.expander("🧠 Agent details"):
                        render_agent_details(metadata)

                    sources = result.get("sources", [])
                    if sources:
                        with st.expander("📚 Sources"):
                            render_sources(sources)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": metadata,
                        "sources": sources,
                    })

                except Exception as exc:
                    error_msg = f"An error occurred: {exc}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })

        # A pending (example-click) question doesn't trigger chat_input's own
        # rerun, so refresh once to move it into history and clear the starters.
        if pending_question:
            st.rerun()

    if st.session_state.messages:
        if st.button("🗑️ Clear chat history"):
            st.session_state.messages = []
            st.session_state.thread_id = os.urandom(16).hex()
            st.rerun()


if __name__ == "__main__":
    main()