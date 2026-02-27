from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from typing import List, Tuple
import re

class PDFManager:

    # =================================================================
    # GLOBAL FUNCTIONS
    # =================================================================

    def load_pdf_pages(self, pdf_path: str) -> List[Document]:
        loader = PyMuPDFLoader(pdf_path)
        pages = loader.load()
        return pages


    def split_pages(self, pages: list) -> Tuple[List[int], List[int]]:
        financial_table_pages = []
        standard_text_pages = []

        for i, page in enumerate(pages):
            page_text = page.page_content
            page_number = i + 1

            is_table, page_score = self._detect_financial_table(page_text)

            if is_table:
                financial_table_pages.append(page_number)
            else:
                standard_text_pages.append(page_number)
        
        return financial_table_pages, standard_text_pages


    # =================================================================
    # LOCAL FUNCTIONS
    # =================================================================

    def _detect_financial_table(self, text) -> Tuple[bool, int]:
        score = 0
        text_lower = text.lower()
        threshold = 50
        
        # Kriter 1: Birincil ve İkincil Anahtar Kelimeler
        primary_keywords = [
            "income statement",
            "consolidated statement of income",
            "statement of comprehensive income", 
            "balance sheet",
            "changes in equity",
            "statement of financial position", 
            'cash flow',
            "cash flows",
            'capital structure'
        ]

        secondary_keywords = [
            "assets", 
            "liabilities", 
            "shareholders' equity", 
            'equity and liabilities',
            'total equity',
            "inventories",
            "operating activities", 
            "in millions of", 
            "€ million", 
            "$ in millions",
            "revenue", 
            "gross profit", 
            "net income", 
            "amortization", 
            "depreciation",
            'long term debt',
            'long-term debt',
            'short-term debt',
            'short term debt'
        ]
        
        # Primary keywords pass threshold automatically
        for kw in primary_keywords:
            if kw in text_lower:
                score += threshold
                
        # Add point for every words in secondary keywords
        keyword_hits = sum(1 for kw in secondary_keywords if kw in text_lower)
        score += (keyword_hits * 10)
        
        # Numeric Density of page
        total_chars = len(text.replace(" ", "").replace("\n", "")) # Boşlukları sayma
        if total_chars > 0:
            digit_count = sum(c.isdigit() for c in text)
            numeric_ratio = digit_count / total_chars
            
            # Numeric Ratio is bigger then 10%
            if numeric_ratio > 0.10:
                score += 20
            # Numeric Ratio is bigger then 20%
            if numeric_ratio > 0.20:
                score += 30
                
        # Money Symbols
        currency_count = len(re.findall(r'[\$\€\£]', text))
        if currency_count >= 5:
            score += 15
        
        # Percent Symbol
        percent_count = text.count('%')
        if percent_count >= 5:
            score += 10
            
        # Comparable Date Format (Ex: 2022, 2023)
        year_matches = len(re.findall(r'\b20[1-2][0-9]\b', text))
        if year_matches >= 4:
            score += 30

        is_table = score >= threshold
        
        return is_table, score
    