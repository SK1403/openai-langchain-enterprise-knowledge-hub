#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##
# Description:-
#
#       Enterprise Configuration module for Azure OpenAI and Databricks RAG.
#       Maintains environment variables for Azure OpenAI deployments, standard OpenAI,
#       Azure Blob/ADLS Gen2 storage endpoints, and hybrid search hyperparameters.
#
##
# Development date    Developed by       Comments
# ----------------    ------------       ---------
# 16/01/2025          Saddam Khan        Initial implementation
# 20/02/2025          Saddam Khan        Added Azure OpenAI and Databricks vector index settings
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AzureOpenAISettings:
    """
    Explanation: Dataclass encapsulating Azure OpenAI and storage configuration settings
    :param  platform_mode str: Deployment mode ("azure" or "openai")
    :param  azure_openai_endpoint str: Azure OpenAI resource base URL
    :param  azure_openai_api_key str: API authorization key
    :param  azure_openai_api_version str: REST API version date
    :param  azure_chat_deployment str: Model deployment name (e.g. gpt-4o)
    :param  azure_embedding_deployment str: Azure embedding model deployment
    :param  openai_api_key str: Direct OpenAI API secret key
    :param  openai_model str: Direct OpenAI model identifier
    :param  openai_embedding_model str: OpenAI embedding model identifier
    :param  azure_storage_connection_string str: ADLS Gen2 connection string
    :param  azure_storage_account_name str: Azure storage account name
    :param  azure_storage_container str: Container / blob filesystem name
    :param  temperature float: Generation sampling temperature
    :param  max_output_tokens int: Max output token limit
    :param  chunk_size int: RAG character chunk size
    :param  chunk_overlap int: Overlap character count
    :param  hybrid_top_k int: Initial retrieval candidate count
    :param  rerank_top_k int: Re-ranked candidate count
    :param  dense_weight float: Dense score weighting
    :param  sparse_weight float: Sparse score weighting
    """
    # Mode selection: "azure" or "openai"
    platform_mode: str = os.getenv("PLATFORM_MODE", "azure")

    # Azure OpenAI Configuration
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    azure_chat_deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
    azure_embedding_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

    # Standard OpenAI Configuration (Fallback / Alternative)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Azure Storage / ADLS Gen2 Configuration
    azure_storage_connection_string: str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    azure_storage_account_name: str = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "")
    azure_storage_container: str = os.getenv("AZURE_STORAGE_CONTAINER", "enterprise-knowledge-corpus")

    # Model Hyperparameters
    temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0.2"))
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))

    # RAG Tuning Parameters
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
    hybrid_top_k: int = int(os.getenv("HYBRID_TOP_K", "10"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "4"))
    dense_weight: float = float(os.getenv("DENSE_WEIGHT", "0.65"))
    sparse_weight: float = float(os.getenv("SPARSE_WEIGHT", "0.35"))

settings = AzureOpenAISettings()
