from typing import List, Literal, Optional, Sequence
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse
import json
from managers.financial_tables import FinancialStatements

class LLMManager:
    def __init__(
        self,
        llm_backend: Literal["local_ollama", "openai"] = "local_ollama",
        model_name: str = "llama3.1:8b-instruct-q4_K_M",
        embedding_model_name: str = "BAAI/bge-m3",
        chroma_persist_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            llm_backend: Which LLM backend to use. For now "local_ollama". Later extend to "openai", etc.
            model_name: LLM model name (e.g. "llama3" in Ollama, or "gpt-4.1" later).
            embedding_model_name: HuggingFace model id for BGE-M3.
            chroma_persist_dir: Optional folder to persist Chroma DB.
        """
        self.llm_backend = llm_backend
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.chroma_persist_dir = chroma_persist_dir

        self._embeddings = self._build_embedding_model()
        self._llm = self._build_llm()

        self._vectorstore: Optional[Chroma] = None
        self._retriever = None

    # ------------------------------------------------------------------
    # LOCAL FUNCTIONS
    # ------------------------------------------------------------------

    def _build_embedding_model(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


    def _build_llm(self):
        if self.llm_backend == "local_ollama":
            return ChatOllama(model=self.model_name,temperature=0.1)

        elif self.llm_backend == "openai":
            raise NotImplementedError("OpenAI backend not wired yet. Implement when needed.")

        else:
            raise ValueError(f"Unsupported llm_backend: {self.llm_backend}")
    
    # ------------------------------------------------------------------
    # GLOBAL FUNCTIONS
    # ------------------------------------------------------------------


    @staticmethod
    def extract_financial_markdown_tables(pdf_path: str, financial_pages: list[Document]) -> str:
        """
        Use LlamaParse to extract markdown tables only from the pages that were detected as financial pages.
        """

        # Collect 0-based page indices from metadata
        page_numbers_0 = sorted({
            p.metadata.get("page")
            for p in financial_pages
            if isinstance(p.metadata, dict) and p.metadata.get("page") is not None
        })

        # If no financial pages detected, do not parse at all
        if not page_numbers_0:
            return ""

        # Convert to 1-based page indices for LlamaParse
        target_pages = ",".join(str(p + 1) for p in page_numbers_0)

        try:
            parser = LlamaParse(
                result_type="markdown",
                target_pages=target_pages,
                verbose=False,
            )
            llama_docs = parser.load_data(pdf_path)
        except Exception:
            return ""

        if not llama_docs:
            return ""

        md = "\n\n".join(getattr(d, "text", "") for d in llama_docs).strip()
        return md





    def build_structured_financial_data(self, markdown: str) -> dict:
        """
        Turn LlamaParse markdown into strict FinancialStatements Pydantic models
        and then into a plain Python dict.
        """
        # Empty markdown => return all-None model
        if not markdown.strip():
            empty = FinancialStatements(
                income_statement={},
                balance_sheet={},
                cash_flow={},
            )
            return empty.model_dump()

        from langchain_core.prompts import ChatPromptTemplate  # local import to avoid cycles

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert accountant. Extract standardized financial statements "
                    "from the markdown tables. If a numeric value is not clearly present, "
                    "set it to null and do NOT guess or infer.",
                ),
                ("human", "Markdown tables:\n{md}"),
            ]
        )

        try:
            chain = prompt | self._llm.with_structured_output(FinancialStatements)
            statements: FinancialStatements = chain.invoke({"md": markdown})
            return statements.model_dump()
        except Exception:
            empty = FinancialStatements(
                income_statement={},
                balance_sheet={},
                cash_flow={},
            )
            return empty.model_dump()







    def build_normal_pages_index(
        self,
        normal_pages: List[Document],
        collection_name: str = "normal_pages",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunks: List[Document] = splitter.split_documents(normal_pages)

        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            collection_name=collection_name,
            persist_directory=self.chroma_persist_dir,
        )

        self._retriever = self._vectorstore.as_retriever()





    def answer_question_over_context(
        self,
        question: str,
        k: int = 4,
        system_instructions: Optional[str] = None,
        financial_data: Optional[dict] = None,
    ) -> str:

        docs: List[Document] = self._retriever.invoke(question)[:k]

        context = "\n\n".join(d.page_content for d in docs)

        base_system = (
            "You are a financial analyst assistant. Only answer questions based solely on the "
            "provided document context and structured JSON data. "
            "Do not use outside knowledge or make assumptions not supported explicitly by the inputs. "
            "If the inputs are insufficient, say you don't know instead of guessing."
        )
        
        financial_json = ""
        if financial_data is not None:
            base_system += (
                "\n\nYou are also given structured financial statement data as JSON. "
                "If the JSON contains the relevant value, prefer it over the text context. "
                "If a JSON field is null or missing, rely on the text context. Never fabricate values."
            )
            financial_json = json.dumps(financial_data, ensure_ascii=False, indent=2)

        if system_instructions:
            base_system += "\n\nAdditional instructions:\n" + system_instructions


        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", base_system),
                (
                    "human",
                    "Structured financial data (JSON, may contain nulls):\n{financial_json}\n\n"
                    "Context:\n{context}\n\n"
                    "Question: {question}\n\n"
                    "Answer in a concise, expert tone.",
                ),
            ]
        )

        chain = prompt | self._llm
        answer = chain.invoke(
            {
                "context": context,
                "question": question,
                "financial_json": financial_json,
            }
        )
        return getattr(answer, "content", str(answer))