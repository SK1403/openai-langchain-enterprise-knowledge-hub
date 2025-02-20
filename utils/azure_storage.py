#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##
# Description:-
#
#       Azure Storage (ADLS Gen2 & Blob Storage) integration module.
#       Provides container creation, binary file upload with MIME type resolution,
#       and blob enumeration via Connection Strings or Managed Identity.
#
##
# Development date    Developed by       Comments
# ----------------    ------------       ---------
# 16/01/2025          Saddam Khan        Initial implementation
# 20/02/2025          Saddam Khan        Added ADLS Gen2 blob upload and container listing
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import mimetypes
from typing import Tuple, List, Optional
from config import settings

def get_blob_service_client():
    """
    Explanation: Initializes Azure BlobServiceClient using Connection String or DefaultAzureCredential
    :return client Optional[BlobServiceClient]: Authenticated Azure storage client or None
    """
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", settings.azure_storage_connection_string)
    if conn_str and conn_str.strip():
        from azure.storage.blob import BlobServiceClient
        return BlobServiceClient.from_connection_string(conn_str.strip())

    account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", settings.azure_storage_account_name)
    if account_name and account_name.strip():
        from azure.storage.blob import BlobServiceClient
        from azure.identity import DefaultAzureCredential
        account_url = f"https://{account_name.strip()}.blob.core.windows.net"
        return BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())

    return None

def upload_to_adls(
    file_bytes: bytes,
    filename: str,
    container_name: str = "",
    folder: str = "raw",
) -> Tuple[bool, str]:
    """
    Explanation: Uploads raw binary file to Azure Data Lake Storage Gen2 / Blob container
    :param  file_bytes bytes: Document byte array
    :param  filename str: Destination file name
    :param  container_name str: Target ADLS Gen2 container name
    :param  folder str: Target directory folder within container
    :return success bool: True if upload succeeded, False otherwise
    :return blob_url_or_error str: Azure storage blob URL or error message
    """
    target_container = container_name or settings.azure_storage_container
    if not target_container:
        return False, "No Azure Storage container specified."

    try:
        client = get_blob_service_client()
        if not client:
            return False, "Azure Storage credentials not configured (Connection String or Account Name missing)."

        container_client = client.get_container_client(target_container)
        if not container_client.exists():
            container_client.create_container()

        blob_path = f"{folder.strip('/')}/{filename}" if folder else filename
        blob_client = container_client.get_blob_client(blob_path)

        content_type, _ = mimetypes.guess_type(filename)
        from azure.storage.blob import ContentSettings
        blob_client.upload_blob(
            file_bytes,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type or "application/octet-stream"),
        )
        return True, f"https://{client.account_name}.blob.core.windows.net/{target_container}/{blob_path}"
    except Exception as e:
        return False, f"ADLS Upload failed: {str(e)}"

def list_adls_blobs(container_name: str = "", folder: str = "raw") -> List[str]:
    """
    Explanation: Enumerates blob names present under specified directory prefix in ADLS Gen2 container
    :param  container_name str: Target storage container name
    :param  folder str: Directory prefix filter
    :return blob_names List[str]: List of discovered blob names
    """
    target_container = container_name or settings.azure_storage_container
    if not target_container:
        return []

    try:
        client = get_blob_service_client()
        if not client:
            return []

        container_client = client.get_container_client(target_container)
        if not container_client.exists():
            return []

        prefix = f"{folder.strip('/')}/" if folder else ""
        blobs = container_client.list_blobs(name_starts_with=prefix)
        return [b.name for b in blobs]
    except Exception:
        return []
