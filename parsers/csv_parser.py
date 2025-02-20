#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##
# Description:-
#
#       Enterprise CSV/TSV Parser with Markdown Table Chunking and Header Preservation.
#       Transforms tabular records into structured markdown tables for Azure OpenAI RAG.
#
##
# Development date    Developed by       Comments
# ----------------    ------------       ---------
# 16/01/2025          Saddam Khan        Initial implementation
# 20/02/2025          Saddam Khan        Added tabular row-group serialization and header retention
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""
Explanation:
    This module parses CSV and TSV files into chunked Markdown table representations
    while preserving column headers across batches for high-precision semantic search.

:param None: Module definitions.
:return None: Parser class export.
"""

import csv
import io
from typing import List
from parsers.base import BaseParser, DocumentChunk


class CSVParser(BaseParser):
    """
    Explanation:
        Enterprise CSV/TSV parser with markdown table chunking and header preservation.

    :param rows_per_chunk <int>: Number of tabular rows per document chunk.
    :return CSVParser: Initialized parser instance.
    """

    def __init__(self, rows_per_chunk: int = 15):
        self.rows_per_chunk = rows_per_chunk

    def parse_text(self, content: str, filename: str, **kwargs) -> List[DocumentChunk]:
        """
        Explanation:
            Parses raw CSV/TSV text content into structured markdown table chunks.

        :param content <str>: Raw CSV or TSV string content.
        :param filename <str>: Name or URI of the source file.
        :param kwargs <dict>: Additional options such as 'allowed_roles'.
        :return List[DocumentChunk]: List of extracted document chunks with row metadata.
        """
        chunks: List[DocumentChunk] = []
        allowed_roles = kwargs.get("allowed_roles", ["all"])
        delimiter = "\t" if filename.endswith(".tsv") else ","

        reader = csv.reader(io.StringIO(content), delimiter=delimiter)
        try:
            headers = next(reader)
        except StopIteration:
            return []

        clean_headers = [h.strip() for h in headers]
        markdown_header = "| " + " | ".join(clean_headers) + " |\n| " + " | ".join(["---"] * len(clean_headers)) + " |"

        rows: List[List[str]] = []
        for row in reader:
            if any(cell.strip() for cell in row):
                rows.append(row)

        for i in range(0, len(rows), self.rows_per_chunk):
            batch = rows[i : i + self.rows_per_chunk]
            table_lines = [markdown_header]
            for r in batch:
                padded = r + [""] * (len(clean_headers) - len(r))
                clean_row = [c.replace("|", "\\|").strip() for c in padded[:len(clean_headers)]]
                table_lines.append("| " + " | ".join(clean_row) + " |")

            table_text = f"=== Tabular Dataset: {filename} (Rows {i+1} to {i+len(batch)}) ===\n" + "\n".join(table_lines)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{filename}#rows_{i+1}_{i+len(batch)}",
                    text=table_text,
                    source=filename,
                    file_format="csv",
                    section=f"Rows {i+1}-{i+len(batch)}",
                    allowed_roles=allowed_roles,
                    metadata={"columns": clean_headers, "start_row": i + 1, "end_row": i + len(batch)},
                )
            )

        return chunks

    def parse_bytes(self, content: bytes, filename: str, **kwargs) -> List[DocumentChunk]:
        return self.parse_text(content.decode("utf-8", errors="ignore"), filename, **kwargs)
