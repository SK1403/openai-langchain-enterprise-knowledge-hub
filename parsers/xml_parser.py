#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##
# Description:-
#
#       Enterprise XML Parser with Tree Flattening and Entity Extraction.
#       Recursively converts XML schemas, configuration manifests, and data trees
#       into clean semantic text chunks for Azure OpenAI RAG.
#
##
# Development date    Developed by       Comments
# ----------------    ------------       ---------
# 16/01/2025          Saddam Khan        Initial implementation
# 20/02/2025          Saddam Khan        Added XML node traversal and structured attribute extraction
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""
Explanation:
    This module parses complex XML documents, converting hierarchical XML
    elements and attributes into semantic text chunks suitable for enterprise RAG.

:param None: Module definitions.
:return None: Parser class and element recursive parsing helper functions.
"""

import xml.etree.ElementTree as ET
from typing import List
from parsers.base import BaseParser, DocumentChunk


def element_to_semantic_text(elem: ET.Element, indent: int = 0) -> List[str]:
    """
    Explanation:
        Recursively converts an XML element and attributes into clean semantic text.

    :param elem <ET.Element>: XML Element node to serialize.
    :param indent <int>: Current indentation depth.
    :return List[str]: Array of formatted semantic string lines.
    """
    lines = []
    prefix = "  " * indent
    tag_clean = elem.tag.split("}")[-1].replace("_", " ").title()
    
    attr_str = ""
    if elem.attrib:
        attrs = [f"{k}={v}" for k, v in elem.attrib.items()]
        attr_str = f" [{', '.join(attrs)}]"

    text_content = elem.text.strip() if elem.text and elem.text.strip() else ""

    if len(elem) == 0:
        if text_content:
            lines.append(f"{prefix}- {tag_clean}{attr_str}: {text_content}")
        elif attr_str:
            lines.append(f"{prefix}- {tag_clean}{attr_str}")
    else:
        lines.append(f"{prefix}### {tag_clean}{attr_str}")
        if text_content:
            lines.append(f"{prefix}  {text_content}")
        for child in elem:
            lines.extend(element_to_semantic_text(child, indent + 1))

    return lines

class XMLParser(BaseParser):
    """Enterprise XML parser with tree traversal and attribute preservation."""

    def parse_text(self, content: str, filename: str, **kwargs) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        allowed_roles = kwargs.get("allowed_roles", ["all"])

        lower_name = filename.lower()
        if "finance" in lower_name or "bank" in lower_name:
            allowed_roles = ["finance", "executive", "all"]
        elif "legal" in lower_name:
            allowed_roles = ["legal", "executive", "all"]

        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            return [
                DocumentChunk(
                    chunk_id=f"{filename}#raw_fragment",
                    text="\n".join(lines[:100]),
                    source=filename,
                    file_format="xml",
                    section="XML Fragment",
                    allowed_roles=allowed_roles,
                    metadata={"error": str(e)},
                )
            ]

        children = list(root)
        if len(children) > 1 and len(set(c.tag for c in children)) <= 3:
            for idx, child in enumerate(children, start=1):
                record_id = child.attrib.get("id") or child.attrib.get("name") or f"Record_{idx}"
                text_lines = element_to_semantic_text(child)
                text = "\n".join(text_lines)
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{filename}#{record_id}",
                        text=f"=== XML Record: {record_id} ===\n{text}",
                        source=filename,
                        file_format="xml",
                        section=str(record_id),
                        allowed_roles=allowed_roles,
                        metadata={"tag": child.tag, "attributes": child.attrib},
                    )
                )
        else:
            if len(children) > 0:
                for idx, child in enumerate(children, start=1):
                    section_name = child.attrib.get("name") or child.tag.split("}")[-1].replace("_", " ").title()
                    text_lines = element_to_semantic_text(child)
                    text = "\n".join(text_lines)
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{filename}#sec_{idx}",
                            text=f"=== XML Section: {section_name} ===\n{text}",
                            source=filename,
                            file_format="xml",
                            section=section_name,
                            allowed_roles=allowed_roles,
                            metadata={"tag": child.tag},
                        )
                    )
            else:
                text = "\n".join(element_to_semantic_text(root))
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{filename}#root",
                        text=text,
                        source=filename,
                        file_format="xml",
                        section="Document Root",
                        allowed_roles=allowed_roles,
                    )
                )

        return chunks

    def parse_bytes(self, content: bytes, filename: str, **kwargs) -> List[DocumentChunk]:
        return self.parse_text(content.decode("utf-8", errors="ignore"), filename, **kwargs)
