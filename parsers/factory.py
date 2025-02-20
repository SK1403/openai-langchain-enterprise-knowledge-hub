#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##
# Description:-
#
#       Parser Factory registry pattern module for Azure Enterprise Knowledge Hub.
#       Maps file extensions (.pdf, .json, .xml, .csv, .md, .txt) to dedicated
#       specialized parser engines for transparent ingestion.
#
##
# Development date    Developed by       Comments
# ----------------    ------------       ---------
# 16/01/2025          Saddam Khan        Initial implementation
# 20/02/2025          Saddam Khan        Added MIME-type dispatch and multi-format parser registry
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
from typing import Dict
from parsers.base import BaseParser
from parsers.pdf_parser import PDFParser
from parsers.json_parser import JSONParser
from parsers.xml_parser import XMLParser
from parsers.csv_parser import CSVParser
from parsers.text_parser import TextParser

class ParserFactory:
    """
    Explanation: Factory class maintaining registry of file extension to parser associations
    """

    _registry: Dict[str, BaseParser] = {
        ".pdf": PDFParser(),
        ".json": JSONParser(),
        ".jsonl": JSONParser(),
        ".xml": XMLParser(),
        ".xsd": XMLParser(),
        ".csv": CSVParser(),
        ".tsv": CSVParser(),
        ".md": TextParser(),
        ".markdown": TextParser(),
        ".txt": TextParser(),
        ".log": TextParser(),
    }

    @classmethod
    def get_parser(cls, filename: str) -> BaseParser:
        """
        Explanation: Resolves appropriate parser instance based on file extension
        :param  filename str: Source document path or filename
        :return parser BaseParser: Initialized parser instance matching extension
        """
        _, ext = os.path.splitext(filename.lower())
        return cls._registry.get(ext, cls._registry[".txt"])

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """
        Explanation: Enumerates all supported file extensions registered in factory
        :return extensions list[str]: List of valid file extension strings
        """
        return list(cls._registry.keys())
