#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##
# Description:-
#
#       Enterprise Text & Markdown Parser with Heading-Aware Chunking.
#       Parses plain text, logs, and Markdown documentation, segmenting by
#       headings and contextual blocks for Azure OpenAI RAG.
#
##
# Development date    Developed by       Comments
# ----------------    ------------       ---------
# 16/01/2025          Saddam Khan        Initial implementation
# 20/02/2025          Saddam Khan        Added markdown and plain-text sliding window chunking
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""
Explanation:
    This module implements heading-aware Markdown and plain-text document
    chunking, preserving semantic sections and headers for contextual indexing.

:param None: Module definitions.
:return None: Parser class export.
"""

import re
from typing import List
from parsers.base import BaseParser, DocumentChunk
from parsers.pdf_parser import chunk_text
from config import settings


class TextParser(BaseParser):
    """
    Explanation:
        Enterprise text & markdown parser with heading-aware chunking.

    :return TextParser: Initialized text and markdown parser.
    """

    def parse_text(self, content: str, filename: str, **kwargs) -> List[DocumentChunk]:
        """
        Explanation:
            Parses text and markdown strings, segmenting on markdown headers or paragraphs.

        :param content <str>: Raw text or markdown string content.
        :param filename <str>: Source file name or blob URI.
        :param kwargs <dict>: Optional keyword arguments including 'allowed_roles'.
        :return List[DocumentChunk]: List of segmented document chunks with section metadata.
        """
        chunks: List[DocumentChunk] = []
        allowed_roles = kwargs.get("allowed_roles", ["all"])

        lower_name = filename.lower()
        if "legal" in lower_name:
            allowed_roles = ["legal", "executive", "all"]
        elif "finance" in lower_name:
            allowed_roles = ["finance", "executive", "all"]
        elif "runbook" in lower_name or "azure" in lower_name or "databrick" in lower_name:
            allowed_roles = ["engineering", "all"]

        if filename.endswith((".md", ".markdown")):
            sections = re.split(r"(?m)(?=^#{1,3}\s+)", content)
            for idx, sec in enumerate(sections):
                sec_clean = sec.strip()
                if not sec_clean:
                    continue

                lines = sec_clean.splitlines()
                header = lines[0].strip("# ").strip() if lines else f"Section {idx+1}"
                sub_chunks = chunk_text(sec_clean, settings.chunk_size, settings.chunk_overlap)
                for c_idx, sub in enumerate(sub_chunks):
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{filename}#sec_{idx+1}_c{c_idx+1}",
                            text=sub,
                            source=filename,
                            file_format="markdown",
                            section=header,
                            allowed_roles=allowed_roles,
                            metadata={"header": header},
                        )
                    )
        else:
            raw_chunks = chunk_text(content, settings.chunk_size, settings.chunk_overlap)
            for idx, c in enumerate(raw_chunks):
                first_line = c.split("\n")[0][:80].strip()
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{filename}#c{idx+1}",
                        text=c,
                        source=filename,
                        file_format="text",
                        section=first_line or f"Part {idx+1}",
                        allowed_roles=allowed_roles,
                    )
                )

        return chunks

    def parse_bytes(self, content: bytes, filename: str, **kwargs) -> List[DocumentChunk]:
        return self.parse_text(content.decode("utf-8", errors="ignore"), filename, **kwargs)
