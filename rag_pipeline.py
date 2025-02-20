#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##
# Description:-
#
#       Enterprise RAG Pipeline Orchestrator for Microsoft Azure OpenAI Service.
#       Integrates multi-format file ingestion (PDF, JSON, XML, CSV, TXT, MD),
#       hybrid dense/sparse vector indexing, Microsoft Entra ID (Azure AD) RBAC,
#       cross-encoder re-ranking, and grounded multi-turn synthesis with Azure OpenAI (GPT-4o).
#
##
# Development date    Developed by       Comments
# ----------------    ------------       ---------
# 16/01/2025          Saddam Khan        Initial implementation
# 20/02/2025          Saddam Khan        Added hybrid semantic-BM25 retrieval and Azure OpenAI synthesis
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
from typing import Generator, Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import settings
from parsers.base import DocumentChunk
from parsers.factory import ParserFactory
from indexing.hybrid_index import AzureEnterpriseHybridIndex, SearchResult

class AzureOpenAIRAGPipeline:
    """
    Explanation: Enterprise RAG Orchestrator integrating Azure OpenAI, ADF, and Databricks.
                 Performs hybrid indexing, Entra ID RBAC security filtering, cross-encoder re-ranking,
                 and multi-turn streaming answer generation.
    """

    def __init__(
        self,
        temperature: float = 0.2,
        chat_deployment: Optional[str] = None,
        openai_model: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        """
        Explanation: Initializes Azure OpenAI RAG pipeline with client credentials and hybrid index
        :param  temperature float: Generation temperature
        :param  chat_deployment Optional[str]: Azure OpenAI model deployment name (e.g., gpt-4o)
        :param  openai_model Optional[str]: OpenAI model name fallback
        :param  azure_endpoint Optional[str]: Azure OpenAI Service endpoint URL
        :param  azure_api_key Optional[str]: Azure OpenAI API authentication key
        :param  openai_api_key Optional[str]: Standard OpenAI API key
        :return None: Instantiates AzureOpenAIRAGPipeline object
        """
        self.temperature = temperature
        self.chat_deployment = chat_deployment or settings.azure_chat_deployment
        self.openai_model = openai_model or settings.openai_model
        self.azure_endpoint = azure_endpoint.strip() if azure_endpoint else settings.azure_openai_endpoint
        self.azure_api_key = azure_api_key.strip() if azure_api_key else settings.azure_openai_api_key
        self.openai_api_key = openai_api_key.strip() if openai_api_key else settings.openai_api_key

        self.index = AzureEnterpriseHybridIndex(
            azure_endpoint=self.azure_endpoint,
            azure_api_key=self.azure_api_key,
            openai_api_key=self.openai_api_key,
        )
        self._sessions: Dict[str, InMemoryChatMessageHistory] = {}
        self.last_retrieved_results: List[SearchResult] = []
        self.active_llm_provider = "Uninitialized"
        self.conversational_chain = None

        self._init_llm()
        self._build_chain()

    def _init_llm(self):
        """
        Explanation: Initializes AzureChatOpenAI client or falls back to standard ChatOpenAI based on credentials
        :return None: Configures active LLM instance and provider status string
        """
        endpoint = self.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", settings.azure_openai_endpoint)
        azure_key = self.azure_api_key or os.getenv("AZURE_OPENAI_API_KEY", settings.azure_openai_api_key)
        openai_key = self.openai_api_key or os.getenv("OPENAI_API_KEY", settings.openai_api_key)

        # 1. Try Azure OpenAI Service
        if endpoint and azure_key:
            try:
                from langchain_openai import AzureChatOpenAI
                self.llm = AzureChatOpenAI(
                    azure_deployment=self.chat_deployment,
                    azure_endpoint=endpoint,
                    api_key=azure_key,
                    api_version=settings.azure_openai_api_version,
                    temperature=self.temperature,
                    max_tokens=settings.max_output_tokens,
                )
                self.active_llm_provider = f"Azure OpenAI ({self.chat_deployment})"
                return
            except Exception:
                pass

        # 2. Try Standard OpenAI API
        if openai_key:
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model=self.openai_model,
                    api_key=openai_key,
                    temperature=self.temperature,
                    max_tokens=settings.max_output_tokens,
                )
                self.active_llm_provider = f"OpenAI ({self.openai_model})"
                return
            except Exception:
                pass

        # Fallback: keep llm as None so UI can render and ask user for keys
        self.llm = None
        self.active_llm_provider = "Pending API Credentials"

    def _build_chain(self):
        """
        Explanation: Constructs Azure grounding prompt template and LangChain runnable with multi-turn history
        :return None: Binds prompt, LLM, and session message store
        """
        if self.llm is None:
            self.conversational_chain = None
            return

        system_template = (
            "You are the Enterprise AI Knowledge Intelligence Assistant for our organization on Microsoft Azure.\n"
            "Your objective is to provide accurate, grounded, and concise answers based on the provided enterprise documents "
            "ingested via Azure Data Factory (ADF), processed by Azure Databricks, and stored in ADLS Gen2.\n\n"
            "=== VERIFIED ENTERPRISE KNOWLEDGE CONTEXT ===\n"
            "{context}\n"
            "=============================================\n\n"
            "Strict Enterprise Operational Rules:\n"
            "1. Grounding Rule: Answer strictly using the verified enterprise knowledge context above.\n"
            "2. Citation Rule: Always cite the source for every factual statement using standard markdown tags: "
            "[Source: filename (Section/Page)].\n"
            "3. Gap Identification: If the provided documents do not contain the answer, "
            "clearly state: 'The uploaded enterprise documents do not contain this information.' Do not fabricate policies, specs, or financial figures.\n"
            "4. Formatting: Present complex comparisons and metrics using clean markdown tables."
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        self.chain = self.prompt | self.llm
        self.conversational_chain = RunnableWithMessageHistory(
            self.chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Explanation: Retrieves or instantiates in-memory chat message history for the session
        :param  session_id str: Conversation session identifier
        :return history BaseChatMessageHistory: Stored session message history
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = InMemoryChatMessageHistory()
        return self._sessions[session_id]

    def ingest_bytes(
        self,
        content: bytes,
        filename: str,
        allowed_roles: Optional[List[str]] = None,
        storage_uri: Optional[str] = None,
    ) -> int:
        """
        Explanation: Parses binary file byte payload, attaches ADLS storage URI metadata, and adds to hybrid index
        :param  content bytes: Binary file contents
        :param  filename str: Filename for parser resolution
        :param  allowed_roles Optional[List[str]]: Microsoft Entra ID RBAC security roles
        :param  storage_uri Optional[str]: Cloud ADLS Gen2 or Blob storage URI
        :return count int: Total count of parsed and indexed chunks
        """
        parser = ParserFactory.get_parser(filename)
        kwargs = {}
        if allowed_roles:
            kwargs["allowed_roles"] = allowed_roles
        chunks = parser.parse_bytes(content, filename, **kwargs)
        if storage_uri:
            for c in chunks:
                c.metadata["storage_uri"] = storage_uri
        self.index.add_chunks(chunks)
        return len(chunks)

    def ingest_text(
        self,
        text: str,
        filename: str,
        allowed_roles: Optional[List[str]] = None,
        storage_uri: Optional[str] = None,
    ) -> int:
        """
        Explanation: Parses string text content, extracts structured chunks, and commits to index
        :param  text str: Text string to parse
        :param  filename str: Source filename
        :param  allowed_roles Optional[List[str]]: Microsoft Entra ID authorized roles
        :param  storage_uri Optional[str]: Cloud storage path
        :return count int: Total count of parsed chunks
        """
        parser = ParserFactory.get_parser(filename)
        kwargs = {}
        if allowed_roles:
            kwargs["allowed_roles"] = allowed_roles
        chunks = parser.parse_text(text, filename, **kwargs)
        if storage_uri:
            for c in chunks:
                c.metadata["storage_uri"] = storage_uri
        self.index.add_chunks(chunks)
        return len(chunks)

    def stream_chat(
        self,
        query: str,
        user_role: str = "all",
        session_id: str = "default_session",
    ) -> Generator[str, None, None]:
        """
        Explanation: Performs Entra ID RBAC hybrid retrieval and streams response tokens from Azure OpenAI
        :param  query str: Natural language prompt or query
        :param  user_role str: User security role
        :param  session_id str: Conversation session identifier
        :return token Generator[str, None, None]: Stream of generated answer tokens
        """
        self.last_retrieved_results = self.index.search(
            query=query,
            user_role=user_role,
            top_k=settings.hybrid_top_k,
            rerank_top_k=settings.rerank_top_k,
        )

        if self.last_retrieved_results:
            context_blocks = []
            for res in self.last_retrieved_results:
                c = res.chunk
                loc = f"Page {c.page}" if c.page else f"Section: {c.section}"
                context_blocks.append(
                    f"--- Source: {c.source} ({loc}) [Format: {c.file_format.upper()}] ---\n{c.text}"
                )
            context = "\n\n".join(context_blocks)
        else:
            context = "NO MATCHING DOCUMENTS FOUND UNDER CURRENT USER ACCESS ROLE."

        if self.conversational_chain is None:
            yield (
                "🔑 **API Credentials Required for Synthesis**\n\n"
                "Please enter your **Azure OpenAI Endpoint & API Key** (or standard OpenAI Key) in the sidebar "
                "under **⚙️ Azure OpenAI & Storage Controls** to enable LLM answer generation.\n\n"
                "*(Document parsing, indexing, and hybrid search with Entra ID RBAC are fully active and shown in the diagnostics below)*"
            )
            return

        for chunk in self.conversational_chain.stream(
            {"input": query, "context": context},
            config={"configurable": {"session_id": session_id}},
        ):
            if chunk.content:
                yield chunk.content

    def clear_history(self, session_id: str = "default_session"):
        """
        Explanation: Clears conversational chat memory for a specified session ID
        :param  session_id str: Session identifier to purge
        :return None: Resets session history in place
        """
        if session_id in self._sessions:
            self._sessions[session_id].clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Explanation: Gathers telemetry statistics from underlying Azure hybrid index
        :return stats Dict[str, Any]: Metrics dictionary including chunk volume and backend info
        """
        return self.index.get_stats()
