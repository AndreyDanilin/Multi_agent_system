"""Streamlit demo console for the Agentic Research Copilot."""

from __future__ import annotations

import streamlit as st

from research_copilot.service import ResearchCopilotService


@st.cache_resource
def get_service() -> ResearchCopilotService:
    service = ResearchCopilotService.create_demo()
    service.ingest_sample()
    return service


def render() -> None:
    st.set_page_config(page_title="Agentic Research Copilot", layout="wide")
    st.title("Agentic Research Copilot")
    st.caption("Multi-agent research workflow with RAG as a tool, hybrid retrieval and citations.")

    service = get_service()
    mode = st.sidebar.selectbox(
        "Retrieval mode",
        ["lexical", "vector", "hybrid", "hybrid_rerank"],
        index=3,
    )

    tab_chat, tab_eval = st.tabs(["Research", "Evaluation"])
    with tab_chat:
        question = st.chat_input("Ask a technical research question")
        if question:
            with st.status("Running agent graph", expanded=True):
                response = service.query(question=question, retrieval_mode=mode)
                for event in response.trace:
                    st.write(f"{event.node}: {event.message}")

            st.chat_message("user").write(question)
            st.chat_message("assistant").write(response.answer)

            if response.citations:
                st.subheader("Citations")
                for citation in response.citations:
                    st.markdown(
                        f"**{citation.title}** · score `{citation.score:.3f}`\n\n"
                        f"> {citation.quote}\n\n"
                        f"`{citation.source}` · `{citation.chunk_id}`"
                    )

    with tab_eval:
        if st.button("Run retrieval evaluation"):
            report = service.run_evaluation()
            rows = [
                {
                    "mode": mode,
                    "hit@k": metrics.hit_at_k,
                    "mrr": metrics.mrr,
                    "citation_coverage": metrics.citation_coverage,
                    "avg_latency_ms": metrics.average_latency_ms,
                }
                for mode, metrics in report.mode_metrics.items()
            ]
            st.dataframe(rows, use_container_width=True)


if __name__ == "__main__":
    render()
