#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##
# Description:-
#
#       Enterprise JSON & JSONL Parser with Schema Flattening and Entity Chunking.
#       Parses nested JSON and JSON Lines payloads into semantic entities for Azure OpenAI RAG.
#
##
# Development date    Developed by       Comments
# ----------------    ------------       ---------
# 16/01/2025          Saddam Khan        Initial implementation
# 20/02/2025          Saddam Khan        Added recursive JSON hierarchy flattening and key indexing
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""
Explanation:
    This module parses standard JSON and JSON Lines (JSONL) files, performing
    recursive schema flattening and entity chunking for semantic knowledge graphs.

:param None: Module definitions.
:return None: Parser class and dictionary flattening functions.
"""

import json
from typing import List, Dict, Any
from parsers.base import BaseParser, DocumentChunk


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Explanation:
        Flattens a nested dictionary into dot-separated paths.

    :param d <Dict[str, Any]>: Nested dictionary to flatten.
    :param parent_key <str>: Current recursion parent key path.
    :param sep <str>: Key delimiter separator.
    :return Dict[str, Any]: Flattened key-value dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            if all(not isinstance(i, (dict, list)) for i in v):
                items.append((new_key, ", ".join(map(str, v))))
            else:
                for idx, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(flatten_dict(item, f"{new_key}[{idx}]", sep=sep).items())
                    else:
                        items.append((f"{new_key}[{idx}]", item))
        else:
            items.append((new_key, v))
    return dict(items)

def record_to_semantic_text(record: Dict[str, Any], title: str = "Record") -> str:
    flat = flatten_dict(record) if any(isinstance(v, (dict, list)) for v in record.values()) else record
    lines = [f"=== {title} ==="]
    for k, v in flat.items():
        clean_key = k.replace("_", " ").replace(".", " > ").title()
        lines.append(f"- {clean_key}: {v}")
    return "\n".join(lines)

class JSONParser(BaseParser):
    """Enterprise JSON & JSONL parser with schema unwinding and entity chunking."""

    def parse_text(self, content: str, filename: str, **kwargs) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        allowed_roles = kwargs.get("allowed_roles", ["all"])
        
        lower_name = filename.lower()
        if "finance" in lower_name:
            allowed_roles = ["finance", "executive", "all"]
        elif "infra" in lower_name or "azure" in lower_name or "adf" in lower_name or "databrick" in lower_name:
            allowed_roles = ["engineering", "all"]

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = []
            for line in content.splitlines():
                if line.strip():
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue

        if isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    title = item.get("name") or item.get("id") or item.get("title") or f"Record #{idx+1}"
                    text = record_to_semantic_text(item, title=str(title))
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{filename}#rec_{idx+1}",
                            text=text,
                            source=filename,
                            file_format="json",
                            section=str(title),
                            allowed_roles=allowed_roles,
                            metadata={"record_index": idx, "keys": list(item.keys())[:10]},
                        )
                    )
        elif isinstance(data, dict):
            has_sub_collections = any(isinstance(v, (list, dict)) for v in data.values())
            if has_sub_collections:
                for key, val in data.items():
                    section_title = key.replace("_", " ").title()
                    if isinstance(val, dict):
                        text = record_to_semantic_text(val, title=f"Section: {section_title}")
                    elif isinstance(val, list):
                        lines = [f"=== Section: {section_title} ({len(val)} items) ==="]
                        for i, elem in enumerate(val):
                            if isinstance(elem, dict):
                                lines.append(record_to_semantic_text(elem, title=f"Item {i+1}"))
                            else:
                                lines.append(f"- {elem}")
                        text = "\n\n".join(lines)
                    else:
                        text = f"{section_title}: {val}"

                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{filename}#{key}",
                            text=text,
                            source=filename,
                            file_format="json",
                            section=section_title,
                            allowed_roles=allowed_roles,
                            metadata={"entity_key": key},
                        )
                    )
            else:
                text = record_to_semantic_text(data, title="Configuration Object")
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{filename}#root",
                        text=text,
                        source=filename,
                        file_format="json",
                        section="Configuration",
                        allowed_roles=allowed_roles,
                        metadata={"keys": list(data.keys())},
                    )
                )

        return chunks

    def parse_bytes(self, content: bytes, filename: str, **kwargs) -> List[DocumentChunk]:
        return self.parse_text(content.decode("utf-8", errors="ignore"), filename, **kwargs)
