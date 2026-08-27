import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import src.config  # noqa: F401
import streamlit as st
from src.graph.pipeline import build_pipeline

st.set_page_config(page_title="AI Research Intelligence System", layout="wide")
st.title("🔬 AI Research Intelligence System")
st.caption("Multi-agent RAG pipeline: web search → ingestion → advanced RAG analysis → report generation")

with st.sidebar:
    st.header("Research Input")
    topic = st.text_input("Research Topic", placeholder="e.g. retrieval augmented generation techniques")
    questions_raw = st.text_area(
        "Research Questions (one per line)",
        placeholder="What is RAG?\nWhat are common RAG challenges?",
        height=150,
    )
    st.divider()
    st.subheader("Advanced RAG Settings")
    use_query_expansion = st.checkbox("Query Expansion", value=True)
    use_reranking = st.checkbox("Cross-Encoder Reranking", value=True)
    use_hierarchical_chunking = st.checkbox("Hierarchical Chunking", value=True)
    run_button = st.button("Run Pipeline", type="primary", use_container_width=True)

if run_button:
    if not topic or not questions_raw.strip():
        st.error("Please provide both a research topic and at least one question.")
    else:
        questions = [q.strip() for q in questions_raw.split("\n") if q.strip()]
        st.session_state["topic"] = topic
        st.session_state["questions"] = questions
        st.session_state["run_requested"] = True
        st.session_state["use_query_expansion"] = use_query_expansion
        st.session_state["use_reranking"] = use_reranking
        st.session_state["use_hierarchical_chunking"] = use_hierarchical_chunking

def run_pipeline_streaming(topic: str, questions: list[str], status_placeholder):
    """Run the pipeline via astream so we can update the UI as each agent finishes."""
    pipeline = build_pipeline()
    initial_state = {
        "research_topic": topic,
        "research_questions": questions,
        "web_documents": [],
        "ingestion_summary": {},
        "answers": [],
        "report_paths": {},
        "current_agent": "",
        "error": None,
        "retry_count": 0,
        "use_query_expansion": st.session_state.get("use_query_expansion", True),
        "use_reranking": st.session_state.get("use_reranking", True),
        "use_hierarchical_chunking": st.session_state.get("use_hierarchical_chunking", True),
    }
    config = {"configurable": {"thread_id": f"streamlit-{hash(topic)}"}}

    agent_labels = {
        "web_intelligence": "🔎 Agent 1: Searching and crawling the web...",
        "ingestion": "📦 Agent 2: Chunking and indexing into Qdrant...",
        "rag_analysis": "🧠 Agent 3: Running advanced RAG analysis...",
        "report_generation": "📄 Agent 4: Generating final report...",
    }

    final_state = None
    async def _run():
        nonlocal final_state
        async for event in pipeline.astream(initial_state, config=config):
            for node_name, node_output in event.items():
                if node_name in agent_labels:
                    status_placeholder.info(agent_labels[node_name])
                final_state = node_output
    asyncio.run(_run())
    return final_state


if st.session_state.get("run_requested"):
    progress_placeholder = st.empty()

    result = run_pipeline_streaming(
        st.session_state["topic"], st.session_state["questions"], progress_placeholder
    )
    progress_placeholder.empty()
    st.session_state["result"] = result
    st.session_state["run_requested"] = False

if st.session_state.get("result"):
    result = st.session_state["result"]

    if result.get("error"):
        st.error(f"Pipeline failed: {result['error']}")
    else:
        st.success("Pipeline completed successfully!")

        tab1, tab2, tab3, tab4 = st.tabs(["🔎 Web Intelligence", "📦 Ingestion", "🧠 RAG Answers", "📄 Report"])

        with tab1:
            docs = result.get("web_documents", [])
            st.write(f"**{len(docs)} documents retrieved**")
            for doc in docs:
                with st.expander(doc.get("url", "unknown")):
                    st.write(f"Format: {doc.get('format')}")
                    st.write(doc.get("content", "")[:500] + "...")

        with tab2:
            st.json(result.get("ingestion_summary", {}))

        with tab3:
            for a in result.get("answers", []):
                st.subheader(a["question"])
                st.write(a["answer"])
                st.caption(f"Sources: {', '.join(a.get('sources', []))}")
                st.divider()

        with tab4:
            paths = result.get("report_paths", {})
            col1, col2 = st.columns(2)
            if paths.get("markdown"):
                with open(paths["markdown"], "r", encoding="utf-8") as f:
                    md_content = f.read()
                with col1:
                    st.download_button("⬇️ Download Markdown", md_content, file_name="research_report.md")
            if paths.get("pdf"):
                with open(paths["pdf"], "rb") as f:
                    pdf_bytes = f.read()
                with col2:
                    st.download_button("⬇️ Download PDF", pdf_bytes, file_name="research_report.pdf")

            st.divider()
            st.markdown(md_content if paths.get("markdown") else "No report available")
            st.markdown(md_content if paths.get("markdown") else "No report available")