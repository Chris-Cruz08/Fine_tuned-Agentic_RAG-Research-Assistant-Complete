import src.config  # noqa: F401 — loads .env as a side effect
import os
from datetime import datetime, timezone
from langchain_groq import ChatGroq
from src.graph.state import ResearchState
from src.tools.mcp_tools import mcp_client
from src.evaluation.retrieval_metrics import evaluate_retrieval
from fpdf import FPDF
import textwrap

REPORT_SYSTEM_PROMPT = """You are a technical writer producing a research report. Given a research \
topic, a set of question/answer pairs, and evaluation metrics, write a structured markdown report \
with these exact sections:

## Executive Summary
## Key Findings
## Methodology Analysis
## Limitations and Future Directions
## References

Be concise, technical, and grounded only in the provided answers. Do not invent facts."""


def synthesize_report(topic: str, answers: list[dict]) -> str:
    """Use Groq to write the narrative sections of the report from Agent 3's answers."""
    llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.environ["GROQ_API_KEY"], temperature=0.3)

    qa_block = "\n\n".join(
        f"Q: {a['question']}\nA: {a['answer']}\nSources: {', '.join(a['sources'])}"
        for a in answers
    )

    user_prompt = f"Research topic: {topic}\n\nQuestions and answers:\n{qa_block}"

    response = llm.invoke([
        {"role": "system", "content": REPORT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])
    return response.content

def build_metrics_table(answers: list[dict]) -> str:
    """Build a markdown table summarizing retrieval + generation metrics across all questions."""
    rows = []
    for a in answers:
        retrieval = a.get("retrieval_scores", {})
        generation = a.get("generation_scores", {})
        rows.append(
            f"| {a['question'][:50]} "
            f"| {retrieval.get('mrr@10', '-')} "
            f"| {retrieval.get('ndcg@10', '-')} "
            f"| {retrieval.get('precision@5', '-')} "
            f"| {retrieval.get('recall@10', '-')} "
            f"| {generation.get('faithfulness', '-')} "
            f"| {generation.get('context_precision', '-')} "
            f"| {generation.get('answer_relevancy', '-')} |"
        )

    header = (
        "| Question | MRR@10 | nDCG@10 | Precision@5 | Recall@10 "
        "| Faithfulness | Context Precision | Answer Relevancy |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    return "## RAG Pipeline Evaluation Results\n\n" + header + "\n" + "\n".join(rows)


def assemble_full_report(topic: str, answers: list[dict]) -> str:
    """Combine the LLM-written narrative with the evaluation metrics table."""
    narrative = synthesize_report(topic, answers)
    metrics_table = build_metrics_table(answers)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"# Research Report: {topic}\n\n*Generated {timestamp}*\n\n"

    return header + narrative + "\n\n" + metrics_table

def export_to_pdf(markdown_content: str, output_path: str) -> str:
    """Convert the markdown report to a simple PDF (plain text, manually wrapped)."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    for line in markdown_content.split("\n"):
        clean_line = line.replace("#", "").replace("*", "").replace("|", " ").strip()
        if not clean_line:
            pdf.ln(3)
            continue

        safe_line = clean_line.encode("latin-1", "replace").decode("latin-1")
        wrapped_lines = textwrap.wrap(safe_line, width=95) or [""]
        for wrapped in wrapped_lines:
            pdf.cell(0, 6, wrapped, new_x="LMARGIN", new_y="NEXT")

    pdf.output(output_path)
    return output_path


async def report_generation_agent(state: ResearchState) -> ResearchState:
    """Agent 4: synthesize final report from Agent 3's answers, save as MD + PDF via MCP."""
    answers = state.get("answers", [])
    if not answers:
        return {**state, "error": "No answers to report on", "current_agent": "report_generation_agent"}

    topic = state["research_topic"]
    full_report = assemble_full_report(topic, answers)

    tools = await mcp_client.get_tools()
    save_tool = next(t for t in tools if t.name == "save_report")
    md_result = await save_tool.ainvoke({"markdown_content": full_report, "filename_prefix": "research_report"})

    if isinstance(md_result, list):
        md_text = md_result[0].get("text", "") if isinstance(md_result[0], dict) else str(md_result[0])
    else:
        md_text = str(md_result)

    md_path = md_text.split("Report saved to ")[-1].strip()
    pdf_path = md_path.replace(".md", ".pdf")
    export_to_pdf(full_report, pdf_path)

    return {
        **state,
        "report_paths": {"markdown": md_path, "pdf": pdf_path},
        "current_agent": "report_generation_agent",
        "error": None,
    }