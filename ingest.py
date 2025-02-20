#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##
# Description:-
#
#       Batch ingestion CLI utility for Azure & Databricks Enterprise Knowledge Hub.
#       Scans target folder for multi-format files, executes parsing, and populates
#       the AzureEnterpriseHybridIndex with Entra ID access controls.
#
##
# Development date    Developed by       Comments
# ----------------    ------------       ---------
# 16/01/2025          Saddam Khan        Initial implementation
# 20/02/2025          Saddam Khan        Added multi-format document chunking and ADLS Gen2 ingestion
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys
import glob
from parsers.factory import ParserFactory
from indexing.hybrid_index import AzureEnterpriseHybridIndex
from utils.auth import verify_azure_openai_credentials

def batch_ingest_directory(dir_path: str, index: AzureEnterpriseHybridIndex):
    """
    Explanation: Scans directory for enterprise files, extracts structured chunks, and indexes them
    :param  dir_path str: Local path to folder containing source documents
    :param  index AzureEnterpriseHybridIndex: Target hybrid index instance
    :return None: Modifies index in place
    """
    print(f"[*] Scanning enterprise directory: {dir_path}")
    files = glob.glob(os.path.join(dir_path, "*.*"))
    if not files:
        print("[!] No files found.")
        return

    total_chunks = 0
    for fpath in sorted(files):
        fname = os.path.basename(fpath)
        _, ext = os.path.splitext(fname.lower())
        if ext not in ParserFactory.supported_extensions():
            continue

        print(f" -> Ingesting '{fname}' ({ext})... ", end="", flush=True)
        try:
            with open(fpath, "rb") as f:
                content = f.read()
            parser = ParserFactory.get_parser(fname)
            chunks = parser.parse_bytes(content, fname)
            index.add_chunks(chunks)
            total_chunks += len(chunks)
            print(f"[✓] {len(chunks)} chunks indexed.")
        except Exception as e:
            print(f"[x] Error: {e}")

    print(f"\n[✓] Ingestion complete. Indexed {total_chunks} total chunks into {index.backend_name} index.")

def main():
    """
    Explanation: CLI entrypoint for running batch file ingestion and displaying index statistics
    :return None: Executes batch ingestion routine
    """
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "sample_data"
    is_valid, status = verify_azure_openai_credentials()
    print("=" * 75)
    print("     Azure & Databricks Enterprise RAG Batch Ingestion Utility")
    print("=" * 75)
    print(f"[*] Credential Status: {status}")
    
    index = AzureEnterpriseHybridIndex()
    batch_ingest_directory(target_dir, index)
    stats = index.get_stats()
    print("\n--- Knowledge Base Index Statistics ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
