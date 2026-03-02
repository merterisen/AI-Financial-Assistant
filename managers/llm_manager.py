from typing import List, Literal, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    ) -> str:

        docs: List[Document] = self._retriever.invoke(question)[:k]

        context = "\n\n".join(d.page_content for d in docs)

        base_system = (
            "You are a financial analyst assistant. Only answer questions based solely on the provided document context. "
            "Do not use outside knowledge or make assumptions not supported explicitly by the context. "
            "If the context is insufficient, say you don't know instead of guessing."
        )
        
        if system_instructions:
            base_system += "\n\nAdditional instructions:\n" + system_instructions

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", base_system),
                (
                    "human",
                    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer in a concise, expert tone.",
                ),
            ]
        )

        chain = prompt | self._llm
        answer = chain.invoke({"context": context, "question": question})
        return getattr(answer, "content", str(answer))