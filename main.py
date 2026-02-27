import streamlit as st
import tempfile
from managers.pdf_manager import PDFManager

st.title("AI Financial Assistant")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    pdf_manager = PDFManager()

    pages = pdf_manager.load_pdf_pages(tmp_path)
    financial_pages, standard_pages = pdf_manager.split_pages(pages)

    st.write("Financial Table Pages:")
    st.write(financial_pages)

    st.write("Standard Text Pages:")
    st.write(standard_pages)