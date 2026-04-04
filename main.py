import json
import streamlit as st
import tempfile
import pandas as pd 
from managers.pdf_manager import PDFManager
from managers.llm_manager import LLMManager
from dotenv import load_dotenv

load_dotenv()

st.title("AI Financial Assistant")
st.write(
    "Upload a financial report PDF and ask questions about its contents."
)

def _flatten_section(section: dict) -> dict:
    if not section:
        return {}
    base = {k: v for k, v in section.items() if k != "other_line_items"}
    other = section.get("other_line_items") or {}
    base.update(other)
    return base

def reset_state_for_new_file(file_name: str) -> None:
    """Reset RAG-related state when a new file is uploaded."""
    for key in ["llm_manager", "financial_pages", "normal_pages", "index_ready", "financial_data"]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state["current_file_name"] = file_name


uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    # Reset when user switches to a different file
    if st.session_state.get("current_file_name") != uploaded_file.name:
        reset_state_for_new_file(uploaded_file.name)

    # Build pages, index, and structured data once per file
    if "llm_manager" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        pdf_manager = PDFManager()

        with st.spinner("Reading PDF pages..."):
            pages = pdf_manager.load_pdf_pages(tmp_path)
            financial_pages, normal_pages = pdf_manager.split_pages(pages)

        st.session_state["financial_pages"] = financial_pages
        st.session_state["normal_pages"] = normal_pages

        llm_manager = LLMManager()

        # Build vector index ONLY over normal (non-financial-table) pages
        if normal_pages:
            with st.spinner("Building vector index (RAG) over standard text pages..."):
                llm_manager.build_normal_pages_index(normal_pages)
            st.session_state["index_ready"] = True
        else:
            st.session_state["index_ready"] = False

        # Extract structured financial data from financial pages (not embedded)
        if financial_pages:
            with st.spinner("Extracting structured financial tables with LlamaParse..."):
                md = llm_manager.extract_financial_markdown_tables(tmp_path, financial_pages)
                financial_data = llm_manager.build_structured_financial_data(md)
        else:
            financial_data = None

        st.session_state["financial_data"] = financial_data
        st.session_state["llm_manager"] = llm_manager

    # UI summary
    financial_pages = st.session_state.get("financial_pages", [])
    normal_pages = st.session_state.get("normal_pages", [])
    financial_data = st.session_state.get("financial_data")

    st.subheader("Parsed PDF summary")
    st.write(f"Financial table pages detected: **{len(financial_pages)}**")
    st.write(f"Standard text pages detected: **{len(normal_pages)}**")
    
    if financial_data is not None:
        st.subheader("Structured financial data (preview)")
        st.json(financial_data)

        income = _flatten_section(financial_data.get("income_statement") or {})
        balance = _flatten_section(financial_data.get("balance_sheet") or {})
        cash_flow = _flatten_section(financial_data.get("cash_flow") or {})

        if income:
            st.subheader("Income Statement (structured)")
            st.dataframe(pd.DataFrame([income]))

        if balance:
            st.subheader("Balance Sheet (structured)")
            st.dataframe(pd.DataFrame([balance]))

        if cash_flow:
            st.subheader("Cash Flow Statement (structured)")
            st.dataframe(pd.DataFrame([cash_flow]))

    # Question answering
    st.subheader("Ask questions about this report")
    question = st.text_input(
        "Your question", placeholder="e.g. What was the net income in 2024?"
    )

    if question and st.button("Ask"):
        llm_manager: LLMManager = st.session_state["llm_manager"]

        if not st.session_state.get("index_ready") and st.session_state.get("financial_data") is None:
            st.warning("No vector index or structured financial data available to answer questions.")
        else:
            with st.spinner("Thinking with RAG + structured tables..."):
                answer = llm_manager.answer_question_over_context(
                    question,
                    financial_data=st.session_state.get("financial_data"),
                )

            st.markdown("**Answer:**")
            st.write(answer)
else:
    st.info("Upload a PDF to build the RAG index and start asking questions.")