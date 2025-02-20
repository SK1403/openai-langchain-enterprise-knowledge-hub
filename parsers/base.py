#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##
# Description:-
#
#       Base abstraction module for Azure multi-format document parsers.
#       Defines the DocumentChunk dataclass containing content, provenance metadata,
#       and Microsoft Entra ID (Azure AD) RBAC permission annotations.
#
##
# Development date    Developed by       Comments
# ----------------    ------------       ---------
# 16/01/2025          Saddam Khan        Initial implementation
# 20/02/2025          Saddam Khan        Added structured chunking metadata and token boundary logic
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import datetime

@dataclass
class DocumentChunk:
    """
    Explanation: Canonical enterprise document chunk holding extracted content and Entra ID metadata
    :param  chunk_id str: Unique identifier for the chunk
    :param  text str: Extracted text content
    :param  source str: Originating file name or ADLS Gen2 path
    :param  file_format str: Document format extension
    :param  section str: Structural section, header, or worksheet identifier
    :param  page Optional[int]: 1-based page number if applicable
    :param  allowed_roles List[str]: List of Entra ID security roles permitted access
    :param  metadata Dict[str, Any]: Additional document metadata
    :param  timestamp str: Ingestion ISO timestamp
    """
    chunk_id: str
    text: str
    source: str
    file_format: str
    section: str = "General"
    page: Optional[int] = None
    allowed_roles: List[str] = field(default_factory=lambda: ["all"])
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())

class BaseParser(ABC):
    """
    Explanation: Abstract base class defining document parsing interface across enterprise formats
    """

    @abstractmethod
    def parse_bytes(self, content: bytes, filename: str, **kwargs) -> List[DocumentChunk]:
        """
        Explanation: Parses binary file byte stream into structured document chunks
        :param  content bytes: Binary payload
        :param  filename str: Source filename
        :param  kwargs: Optional parser arguments including allowed_roles
        :return chunks List[DocumentChunk]: Extracted and structured document chunks
        """
        pass

    @abstractmethod
    def parse_text(self, content: str, filename: str, **kwargs) -> List[DocumentChunk]:
        """
        Explanation: Parses text string into structured document chunks
        :param  content str: Text content to chunk
        :param  filename str: Source filename
        :param  kwargs: Optional parser arguments
        :return chunks List[DocumentChunk]: Extracted document chunks
        """
        pass
