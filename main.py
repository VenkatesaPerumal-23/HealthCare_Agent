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
from retrieval import create_retriever, ingest_pdfs_into_vectordb, create_vectorstore
from graph import build_graph, run_graph_turn


def main():
    st.title("🩺 HealthGuard AI")
    st.caption(
        "Agentic healthcare information assistant — educational use only, "
        "not a substitute for professional medical care."
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        st.subheader("📚 Medical knowledge base")

        if os.path.isdir(KNOWLEDGE_BASE_DIR):
            pdf_files = sorted(
                name for name in os.listdir(KNOWLEDGE_BASE_DIR)
                if name.lower().endswith(".pdf")
            )
            st.info(f"Found {len(pdf_files)} PDF(s)")
            for pdf in pdf_files:
                st.caption(f"📄 {pdf}")
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
        st.subheader("🧠 App capabilities")
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

    # Show earlier messages on every Streamlit rerun.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("metadata"):
                with st.expander("🧠 Agent details"):
                    st.json(message["metadata"])
            if message.get("sources"):
                with st.expander("📚 Sources"):
                    for source_index, source in enumerate(message["sources"], start=1):
                        title = source.get("title") or source.get("source") or "Source"
                        url = source.get("source")
                        page = source.get("page")
                        extra = f" — page {int(page) + 1}" if page is not None else ""
                        if url and str(url).startswith("https://"):
                            st.markdown(f"- **[S{source_index}]** [{title}]({url}){extra}")
                        else:
                            st.markdown(f"- **[S{source_index}]** {title}{extra}")

    question = st.chat_input("Ask a healthcare question...")
    if question:
        # Include a bounded amount of earlier conversation in the LLM prompts.
        previous_messages = st.session_state.messages[-12:] #most recent 12 messages
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
                        "final_node": result.get("sender", "unknown"),
                        "confidence": result.get("confidence", "unknown"),
                        "evidence_status": result.get("evidence_status", "unknown"),
                        "generation_attempts": result.get("generation_attempts", 0),
                        "validation_issues": result.get("validation_issues", []),
                    }
                    if result.get("retrieval_diagnostics"):
                        metadata["retrieval"] = result["retrieval_diagnostics"]
                    if LANGFUSE_ENABLED and langfuse_client is not None:
                        trace_url = langfuse_client.get_trace_url()
                        if trace_url:
                            metadata["langfuse_trace_url"] = trace_url
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
                        st.json(metadata)

                    sources = result.get("sources", [])
                    if sources:
                        with st.expander("📚 Sources"):
                            st.caption("Citation labels [S1], [S2], etc. follow this source-list order.")
                            for source_index, source in enumerate(sources, start=1):
                                title = source.get("title") or source.get("source") or "Source"
                                url = source.get("source")
                                page = source.get("page")
                                extra = f" — page {int(page) + 1}" if page is not None else ""
                                if url and str(url).startswith("https://"):
                                    st.markdown(f"- **[S{source_index}]** [{title}]({url}){extra}")
                                else:
                                    st.markdown(f"- **[S{source_index}]** {title}{extra}")

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

    if st.session_state.messages:
        if st.button("🗑️ Clear chat history"):
            st.session_state.messages = []
            st.session_state.thread_id = os.urandom(16).hex()
            st.rerun()

if __name__ == "__main__":
    main()
