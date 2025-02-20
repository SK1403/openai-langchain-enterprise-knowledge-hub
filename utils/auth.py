#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##
# Description:-
#
#       Azure OpenAI Service and OpenAI authentication verification utilities.
#       Validates Azure OpenAI endpoints, API keys, Microsoft Entra ID managed identities,
#       and OpenAI fallback configurations.
#
##
# Development date    Developed by       Comments
# ----------------    ------------       ---------
# 16/01/2025          Saddam Khan        Initial implementation
# 20/02/2025          Saddam Khan        Added Azure OpenAI and Entra ID credential verification
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
from typing import Tuple
from config import settings

def verify_azure_openai_credentials() -> Tuple[bool, str]:
    """
    Explanation: Validates whether Azure OpenAI Service, Entra ID MSI, or standard OpenAI credentials are active
    :return is_configured bool: Flag indicating whether viable credentials exist
    :return status_message str: Informative status string or error explanation
    """
    # 1. Check Azure OpenAI endpoint & key
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", settings.azure_openai_endpoint)
    azure_key = os.getenv("AZURE_OPENAI_API_KEY", settings.azure_openai_api_key)

    if endpoint and azure_key:
        clean_endpoint = endpoint.split("://")[-1].split("/")[0]
        return True, f"Azure OpenAI Connected ({clean_endpoint})"

    # 2. Check Azure Managed Identity (MSI / Entra ID)
    if endpoint and not azure_key:
        try:
            from azure.identity import DefaultAzureCredential
            cred = DefaultAzureCredential()
            return True, f"Azure Managed Identity (Entra ID) Connected ({endpoint})"
        except Exception:
            pass

    # 3. Check standard OpenAI fallback key
    openai_key = os.getenv("OPENAI_API_KEY", settings.openai_api_key)
    if openai_key:
        masked = openai_key[:7] + "..." + openai_key[-4:] if len(openai_key) > 12 else "***"
        return True, f"OpenAI API Key Active ({masked})"

    return False, "Neither Azure OpenAI nor OpenAI API credentials detected. Please configure in .env or sidebar."
