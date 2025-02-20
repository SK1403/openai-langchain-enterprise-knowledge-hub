#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##
# Description:-
#
#       Enterprise PDF Parser with Page Extraction and Boundary-Aware Chunking.
#       Processes enterprise PDFs while retaining page metadata, security roles,
#       and paragraph structure for Azure OpenAI RAG.
#
##
# Development date    Developed by       Comments
# ----------------    ------------       ---------
# 16/01/2025          Saddam Khan        Initial implementation
# 20/02/2025          Saddam Khan        Added page-level text extraction and metadata preservation
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""
Explanation:
    This module provides boundary-aware document splitting and page-level
    parsing for PDF documents, supporting role-based access control metadata.

:param None: Module definitions.
:return None: Parser class and chunking helper functions.
"""

import io
import re
from typing import List
from pypdf import PdfReader
from parsers.base import BaseParser, DocumentChunk
from config import settings


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Explanation:
        Hierarchical text splitter preserving paragraph and sentence boundaries.

    :param text <str>: Raw text content to split into chunks.
    :param chunk_size <int>: Maximum character length per chunk.
    :param chunk_overlap <int>: Overlap character count between consecutive chunks.
    :return List[str]: Array of extracted text chunks.
    """
    if not text.strip():
        return []
    
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = (current_chunk + "\n\n" + para) if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(para) > chunk_size:
                sentences = re.split(r"(?<=[.?!])\s+", para)
                sub_chunk = ""
                for s in sentences:
                    if len(sub_chunk) + len(s) + 1 <= chunk_size:
                        sub_chunk = (sub_chunk + " " + s) if sub_chunk else s
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk)
                        sub_chunk = s
                if sub_chunk:
                    current_chunk = sub_chunk
                else:
                    current_chunk = ""
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

class PDFParser(BaseParser):
    """Enterprise PDF parser with page extraction and Entra ID role inference."""

    def parse_bytes(self, content: bytes, filename: str, **kwargs) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        reader = PdfReader(io.BytesIO(content))
        allowed_roles = kwargs.get("allowed_roles", ["all"])
        
        lower_name = filename.lower()
        if "legal" in lower_name:
            allowed_roles = ["legal", "executive", "all"]
        elif "finance" in lower_name:
            allowed_roles = ["finance", "executive", "all"]
        elif "engineer" in lower_name or "azure" in lower_name or "databrick" in lower_name:
            allowed_roles = ["engineering", "all"]

        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue

            sub_chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
            for idx, c in enumerate(sub_chunks):
                first_line = c.split("\n")[0][:80].strip()
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{filename}#p{page_idx}_c{idx}",
                        text=c,
                        source=filename,
                        file_format="pdf",
                        section=first_line or f"Page {page_idx}",
                        page=page_idx,
                        allowed_roles=allowed_roles,
                        metadata={"total_pages": len(reader.pages), "chunk_len": len(c)},
                    )
                )

        return chunks

    def parse_text(self, content: str, filename: str, **kwargs) -> List[DocumentChunk]:
        return self.parse_bytes(content.encode("utf-8"), filename, **kwargs)
