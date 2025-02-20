#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##
# Description:-
#
#       Streamlit interactive web user interface for Azure Enterprise Knowledge Hub.
#       Integrates Microsoft Entra ID RBAC security filtering, Azure OpenAI inference,
#       ADLS Gen2 storage upload synchronization, and multi-format document RAG.
#
##
# Development date    Developed by       Comments
# ----------------    ------------       ---------
# 16/01/2025          Saddam Khan        Initial implementation
# 20/02/2025          Saddam Khan        Enhanced Azure OpenAI RAG UI and Databricks ingestion
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""
Explanation: Streamlit Web UI Application for the Azure & Databricks Enterprise Knowledge Hub.
             Coordinates Microsoft Entra ID RBAC security filtering, ADLS Gen2 storage synchronization,
             hybrid vector retrieval with cross-encoder re-ranking, and streaming Azure OpenAI synthesis.
:param None: Reads execution configuration from settings and runtime user inputs
:return None: Renders interactive enterprise knowledge hub in web browser
"""

import os
import glob
from typing import Dict, Any, List, Optional
import streamlit as st
from config import settings
from rag_pipeline import AzureOpenAIRAGPipeline
from utils.auth import verify_azure_openai_credentials
from utils.azure_storage import upload_to_adls, list_adls_blobs

ROLES: Dict[str, Dict[str, str]] = {
    "all": {"label": "🌐 General Employee (All Access)", "icon": "🌐", "desc": "Standard corporate policies and general documents."},
    "engineering": {"label": "💻 Engineering & Databricks", "icon": "💻", "desc": "Azure VNet topologies, Databricks cluster configs, SRE runbooks."},
    "legal": {"label": "⚖️ Corporate Legal", "icon": "⚖️", "desc": "Enterprise cloud agreements, data ownership clauses, liability caps."},
    "finance": {"label": "💼 Finance & Treasury", "icon": "💼", "desc": "High-value settlements, Databricks commitments, audit ledgers."},
    "executive": {"label": "👔 Executive Leadership", "icon": "👔", "desc": "Unrestricted strategic, financial, and architectural access."},
}

ROLE_SUGGESTIONS: Dict[str, List[str]] = {
    "all": [
        "What is our corporate policy on Azure OpenAI Data Ownership and Training commitments?",
        "Who is our Principal Azure Architect and what office are they based in?",
        "What is our Recovery Time Objective (RTO) for disaster recovery?",
    ],
    "engineering": [
        "What subnets and delegations are configured in vnet-azure-core-prod?",
        "What are the cluster specifications and node types for dbw-enterprise-rag-analytics?",
        "Describe the regional failover procedure from East US 2 to West US 3.",
    ],
    "legal": [
        "What is the limitation of liability cap under our Microsoft Enterprise Cloud Agreement?",
        "What are the Service Level Agreement (SLA) credits in Section 1 if uptime drops below 99.9%?",
        "What are the contract termination notice terms for convenience?",
    ],
    "finance": [
        "Detail Settlement SETTLE-AZ-101: Amount, beneficiary, and purpose in our audit ledger.",
        "How much was allocated to Databricks Inc. for 24/7 Premium Mission-Critical Support?",
        "What compliance standard was verified in our Azure Treasury Audit Report?",
    ],
    "executive": [
        "Summarize all major multi-million dollar cloud commitments and settlements across Databricks and Microsoft.",
        "What are our top architectural redundancy mechanisms and SRE failover protocols?",
        "Give me a consolidated executive briefing of our Contoso Global AI infrastructure.",
    ],
}

def init_page_config() -> None:
    """
    Explanation:
        Sets Streamlit page layout configuration including browser page title,
        corporate hub favicon icon, and expanded wide layout mode.

    :param None: Reads no input parameters.
    :return None: Configures Streamlit page display settings.
    """
    st.set_page_config(
        page_title="Azure Enterprise Knowledge Hub (ADF + Databricks RAG)",
        page_icon="🔷",
        layout="wide",
    )

def init_session_state() -> None:
    """
    Explanation:
        Initializes Streamlit session state stores for message logs, uploaded file sets,
        and enterprise pipeline persistence across reruns.

    :param None: Initializes state dictionary keys.
    :return None: Sets default session state variables.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files_set" not in st.session_state:
        st.session_state.uploaded_files_set = set()
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = AzureOpenAIRAGPipeline()

def render_sidebar() -> Dict[str, Any]:
    """
    Explanation:
        Renders the sidebar controls for Entra ID RBAC roles, sample data corpus preloading,
        file uploads, Azure OpenAI credentials, ADLS Gen2 container settings, and index clearing.

    :param None: Reads interactive selections and parameters from sidebar controls.
    :return config Dict[str, Any]: Dictionary containing active user configuration selections.
    """
    with st.sidebar:
        st.header("🔐 Microsoft Entra ID (Azure AD) RBAC")
        selected_role_key = st.selectbox(
            "Active Employee Role",
            options=list(ROLES.keys()),
            format_func=lambda k: ROLES[k]["label"],
            index=0,
        )
        st.caption(f"**Security Group**: `{ROLES[selected_role_key]['desc']}`")

        st.markdown("---")
        st.header("📥 Multi-Format Ingestion (ADF & ADLS)")

        adls_container = st.text_input(
            "ADLS Gen2 Container",
            value=settings.azure_storage_container,
            placeholder="e.g. enterprise-knowledge-corpus",
        )

        # Pre-load Sample Enterprise Corpus button
        if st.button("⚡ Ingest Sample Corpus (JSON, XML, MD, CSV, TXT)", use_container_width=True):
            sample_dir = os.path.join(os.path.dirname(__file__), "sample_data")
            if os.path.exists(sample_dir) and "pipeline" in st.session_state:
                files = glob.glob(os.path.join(sample_dir, "*.*"))
                loaded_count = 0
                for f in files:
                    fname = os.path.basename(f)
                    if fname not in st.session_state.uploaded_files_set:
                        with open(f, "rb") as fh:
                            raw_bytes = fh.read()

                        # 1. Persist to Azure Data Lake Storage Gen2 (raw/ container)
                        target_container = adls_container or settings.azure_storage_container
                        sim_adls_uri = f"https://{settings.azure_storage_account_name}.blob.core.windows.net/{target_container}/raw/{fname}"
                        upload_to_adls(raw_bytes, fname, target_container, "raw")

                        # 2. Parse and load chunks into Vector Database
                        st.session_state.pipeline.ingest_bytes(raw_bytes, fname, storage_uri=sim_adls_uri)
                        st.session_state.uploaded_files_set.add(fname)
                        loaded_count += 1
                st.success(f"Uploaded {loaded_count} docs to ADLS Gen2 & indexed into Vector DB!")
                st.rerun()

        uploaded_files = st.file_uploader(
            "Ingest Files into Knowledge Base",
            type=["pdf", "json", "jsonl", "xml", "csv", "tsv", "txt", "md"],
            accept_multiple_files=True,
            help="Upload multi-terabyte files in any supported format.",
        )

        st.markdown("---")
        st.header("⚙️ Azure OpenAI & Storage Controls")
        is_valid, cred_status = verify_azure_openai_credentials()
        if is_valid:
            st.success(f"Status: `{cred_status}`")
        else:
            st.warning(f"⚠️ {cred_status}")

        platform_mode = st.radio("Platform Mode", options=["Azure OpenAI", "Standard OpenAI"], index=0)

        azure_endpoint = ""
        azure_key = ""
        openai_key = ""

        if platform_mode == "Azure OpenAI":
            azure_endpoint = st.text_input(
                "Azure OpenAI Endpoint",
                value=settings.azure_openai_endpoint,
                placeholder="https://my-resource.openai.azure.com/",
            )
            azure_key = st.text_input(
                "Azure OpenAI API Key",
                value=settings.azure_openai_api_key,
                type="password",
                placeholder="Leave empty if using Azure Managed Identity",
            )
            chat_deployment = st.selectbox(
                "Azure Chat Deployment",
                options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-35-turbo"],
                index=0,
            )
        else:
            openai_key = st.text_input(
                "OpenAI API Key",
                value=settings.openai_api_key,
                type="password",
            )
            chat_deployment = st.selectbox(
                "OpenAI Model",
                options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                index=0,
            )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                if "pipeline" in st.session_state:
                    st.session_state.pipeline.clear_history("streamlit_session")
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🧹 Clear Index", use_container_width=True):
                if "pipeline" in st.session_state:
                    st.session_state.pipeline.index.clear()
                st.session_state.uploaded_files_set.clear()
                st.rerun()

        return {
            "selected_role_key": selected_role_key,
            "uploaded_files": uploaded_files,
            "adls_container": adls_container,
            "platform_mode": platform_mode,
            "azure_endpoint": azure_endpoint,
            "azure_key": azure_key,
            "openai_key": openai_key,
            "chat_deployment": chat_deployment,
        }

def get_or_create_pipeline(
    platform_mode: str,
    chat_deployment: str,
    adls_container: str,
    azure_endpoint: str,
    azure_key: str,
    openai_key: str,
) -> AzureOpenAIRAGPipeline:
    """
    Explanation: Manages the lifecycle of AzureOpenAIRAGPipeline, dynamically updating LLM
                 credentials while preserving existing indexed vectors.
    :param  platform_mode str: Active deployment platform ("Azure OpenAI" or "Standard OpenAI")
    :param  chat_deployment str: Target deployment or model name
    :param  adls_container str: Target Azure Storage container name
    :param  azure_endpoint str: Azure OpenAI endpoint URL
    :param  azure_key str: Azure OpenAI authorization key
    :param  openai_key str: Direct OpenAI API secret key
    :return pipeline AzureOpenAIRAGPipeline: Initialized RAG pipeline instance
    """
    pipeline_key = f"{platform_mode}_{chat_deployment}_{adls_container}_{azure_endpoint}_{azure_key}_{openai_key}"
    if "current_pipeline_key" not in st.session_state or st.session_state.current_pipeline_key != pipeline_key:
        existing_index = st.session_state.pipeline.index if "pipeline" in st.session_state else None

        st.session_state.pipeline = AzureOpenAIRAGPipeline(
            chat_deployment=chat_deployment,
            openai_model=chat_deployment,
            azure_endpoint=azure_endpoint if platform_mode == "Azure OpenAI" else "",
            azure_api_key=azure_key if platform_mode == "Azure OpenAI" else "",
            openai_api_key=openai_key if platform_mode == "Standard OpenAI" else "",
        )
        if existing_index:
            existing_index.azure_endpoint = azure_endpoint if platform_mode == "Azure OpenAI" else ""
            existing_index.azure_api_key = azure_key if platform_mode == "Azure OpenAI" else ""
            existing_index.openai_api_key = openai_key if platform_mode == "Standard OpenAI" else ""
            existing_index._init_embeddings()
            st.session_state.pipeline.index = existing_index
        st.session_state.current_pipeline_key = pipeline_key
    return st.session_state.pipeline

def handle_document_ingestion(
    pipeline: AzureOpenAIRAGPipeline,
    uploaded_files: list,
    adls_container: str,
) -> None:
    """
    Explanation: Persists user-uploaded documents to Azure Data Lake Storage Gen2 (ADLS Gen2)
                 and indexes chunked representations into the vector database.
    :param  pipeline AzureOpenAIRAGPipeline: Active Azure RAG pipeline instance
    :param  uploaded_files list: List of UploadedFile objects from Streamlit
    :param  adls_container str: Target ADLS Gen2 storage container
    :return None: Modifies vector database and session state
    """
    if not uploaded_files:
        return

    for f in uploaded_files:
        if f.name not in st.session_state.uploaded_files_set:
            bytes_data = f.getvalue()

            # STEP 1: Persist Raw Document to Azure Data Lake Storage Gen2 (raw/ container)
            adls_uri = f"https://{settings.azure_storage_account_name}.blob.core.windows.net/{adls_container}/raw/{f.name}"
            if adls_container and adls_container.strip():
                ok, adls_res = upload_to_adls(
                    file_bytes=bytes_data,
                    filename=f.name,
                    container_name=adls_container,
                    folder="raw",
                )
                if ok:
                    adls_uri = adls_res
                    st.sidebar.success(f"☁️ 1. Stored in ADLS Gen2: `{f.name}`")
                else:
                    st.sidebar.info(f"☁️ 1. Staged in ADLS (Mock Lakehouse): `{f.name}`")

            # STEP 2: Parse, Chunk, Embed & Load into Vector Database with Provenance
            chunks_count = pipeline.ingest_bytes(bytes_data, f.name, storage_uri=adls_uri)
            st.session_state.uploaded_files_set.add(f.name)
            st.sidebar.success(f"⚡ 2. Loaded into Vector DB: `{chunks_count}` chunks")

def render_dashboard_and_chat(
    pipeline: AzureOpenAIRAGPipeline,
    selected_role_key: str,
) -> None:
    """
    Explanation: Renders the Azure & Databricks knowledge base metrics, role-based question cards,
                 message history, grounded ADLS citations, and streams real-time LLM answers.
    :param  pipeline AzureOpenAIRAGPipeline: Initialized RAG pipeline instance
    :param  selected_role_key str: Active employee role key for Entra ID RBAC filtering
    :return None: Renders interactive UI components
    """
    stats = pipeline.get_stats()
    st.title("🔷 Azure & Databricks Enterprise Knowledge Hub")
    st.caption("Distributed Multi-Format RAG (PDF, JSON, XML, CSV, MD) with Entra ID RBAC, ADF & Databricks Delta Lake.")

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Indexed Chunks", stats["total_chunks"])
    mcol2.metric("Documents Ingested", stats["unique_documents"])
    mcol3.metric("Formats Supported", len(stats.get("format_breakdown", {})))
    mcol4.metric("Active Role", selected_role_key.upper())

    if stats["total_chunks"] > 0:
        format_tags = " ".join([f"`{fmt.upper()}: {cnt}`" for fmt, cnt in stats.get("format_breakdown", {}).items()])
        st.caption(f"**Format Distribution**: {format_tags} | **Vector Backend**: `{stats['backend']}`")

    st.markdown("---")

    # Role-specific suggested questions
    if not st.session_state.messages:
        st.markdown(f"##### 💡 Suggested Questions for **{ROLES[selected_role_key]['label']}**:")
        cols = st.columns(3)
        for idx, prompt_text in enumerate(ROLE_SUGGESTIONS.get(selected_role_key, ROLE_SUGGESTIONS["all"])[:3]):
            if cols[idx].button(prompt_text, key=f"q_{idx}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": prompt_text})
                st.rerun()

    # Render Message History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "results" in msg and msg["results"]:
                with st.expander(f"🔍 Grounded Citations & Re-Rank Diagnostics ({len(msg['results'])} Sources)"):
                    for res in msg["results"]:
                        c = res.chunk
                        loc = f"Page {c.page}" if c.page else f"Sec: {c.section}"
                        adls_tag = f" | ☁️ `{c.metadata['storage_uri']}`" if c.metadata.get("storage_uri") else ""
                        st.markdown(f"**• [{c.source} - {loc}]** `Format: {c.file_format.upper()}`{adls_tag}")
                        st.caption(f"Scores: Dense={res.dense_score} | Sparse={res.sparse_score} | RRF={res.rrf_score} | Re-rank={res.final_score}")
                        st.code(c.text[:300] + ("..." if len(c.text) > 300 else ""))

    # Chat Input Box
    prompt_input = st.chat_input("Ask a question across all Azure docs, Databricks tables, XMLs, and configs...")
    pending_query = None

    if prompt_input:
        pending_query = prompt_input
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)
    elif st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        pending_query = st.session_state.messages[-1]["content"]

    if pending_query:
        with st.chat_message("assistant"):
            response_box = st.empty()
            full_text = ""

            try:
                for chunk in pipeline.stream_chat(
                    query=pending_query,
                    user_role=selected_role_key,
                    session_id="streamlit_session",
                ):
                    full_text += chunk
                    response_box.markdown(full_text + "▌")

                if full_text:
                    response_box.markdown(full_text)
                    recorded_results = list(pipeline.last_retrieved_results)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_text,
                        "results": recorded_results,
                    })

                    if recorded_results:
                        with st.expander(f"🔍 Grounded Citations & Re-Rank Diagnostics ({len(recorded_results)} Sources)"):
                            for res in recorded_results:
                                c = res.chunk
                                loc = f"Page {c.page}" if c.page else f"Sec: {c.section}"
                                adls_tag = f" | ☁️ `{c.metadata['storage_uri']}`" if c.metadata.get("storage_uri") else ""
                                st.markdown(f"**• [{c.source} - {loc}]** `Format: {c.file_format.upper()}`{adls_tag}")
                                st.caption(f"Scores: Dense={res.dense_score} | Sparse={res.sparse_score} | RRF={res.rrf_score} | Re-rank={res.final_score}")
                                st.code(c.text[:300] + ("..." if len(c.text) > 300 else ""))
                else:
                    response_box.warning("No answer generated.")
            except Exception as e:
                err = str(e)
                if "api_key" in err.lower() or "authentication" in err.lower():
                    response_box.error(
                        f"🔑 **Authentication Required**\n\n"
                        f"Please enter your **Azure OpenAI Endpoint & API Key** (or standard OpenAI API Key) in the sidebar.\n\n"
                        f"Error details: `{err}`"
                    )
                else:
                    response_box.error(f"⚠️ Error: {err}")

def main() -> None:
    """
    Explanation:
        Main application orchestration entry point coordinating page layout initialization,
        sidebar configuration, pipeline lifecycle management, ingestion, and UI dashboard rendering.

    :param None: Reads execution configuration and coordinates Streamlit application lifecycle.
    :return None: Executes Streamlit application cycle.
    """
    init_page_config()
    init_session_state()
    cfg = render_sidebar()
    pipeline = get_or_create_pipeline(
        platform_mode=cfg["platform_mode"],
        chat_deployment=cfg["chat_deployment"],
        adls_container=cfg["adls_container"],
        azure_endpoint=cfg["azure_endpoint"],
        azure_key=cfg["azure_key"],
        openai_key=cfg["openai_key"],
    )
    handle_document_ingestion(
        pipeline=pipeline,
        uploaded_files=cfg["uploaded_files"],
        adls_container=cfg["adls_container"],
    )
    render_dashboard_and_chat(
        pipeline=pipeline,
        selected_role_key=cfg["selected_role_key"],
    )

if __name__ == "__main__":
    main()
