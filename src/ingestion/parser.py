import os
import re
from typing import List, Dict, Any, Optional
from src.common.models import Document

class DocumentParser:
    """
    Structure-aware document parser supporting Markdown, PDF, Plaintext, and directories.
    Extracts sections, headers, and tabular elements.
    """

    @staticmethod
    def parse_markdown(raw_text: str, title: str = "Untitled Document", metadata: Dict[str, Any] = None, allowed_roles: List[str] = None) -> Document:
        metadata = metadata or {}
        allowed_roles = allowed_roles or ["public"]
        
        cleaned_text = raw_text.strip()
        
        # Extract headings and table structures
        headings = re.findall(r'^(#{1,6})\s+(.+)$', cleaned_text, flags=re.MULTILINE)
        has_tables = bool(re.search(r'\|.+\|[\r\n]+\|[-:| ]+\|', cleaned_text))
        
        extracted_meta = {
            **metadata,
            "headings_count": len(headings),
            "headings": [h[1] for h in headings],
            "contains_tables": has_tables,
            "char_count": len(cleaned_text),
            "word_count": len(cleaned_text.split()),
        }
        
        return Document(
            title=title,
            content=cleaned_text,
            metadata=extracted_meta,
            allowed_roles=allowed_roles
        )

    @staticmethod
    def parse_pdf(file_path: str, title: Optional[str] = None, allowed_roles: Optional[List[str]] = None) -> Document:
        """
        Parses PDF documents page-by-page into structured markdown sections.
        """
        import pypdf
        filename = os.path.basename(file_path)
        doc_title = title or os.path.splitext(filename)[0]
        
        reader = pypdf.PdfReader(file_path)
        pages_content = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_content.append(f"## Page {idx + 1}\n{text.strip()}\n")
                
        full_text = "\n".join(pages_content)
        return DocumentParser.parse_markdown(
            full_text,
            title=doc_title,
            metadata={"source_path": file_path, "page_count": len(reader.pages), "file_type": "pdf"},
            allowed_roles=allowed_roles
        )

    @staticmethod
    def parse_docx(file_path: str, title: Optional[str] = None, allowed_roles: Optional[List[str]] = None) -> Document:
        """
        Parses DOCX documents by extracting XML paragraphs directly from the zip container.
        """
        import zipfile
        import xml.etree.ElementTree as ET

        filename = os.path.basename(file_path)
        doc_title = title or os.path.splitext(filename)[0]

        paragraphs = []
        try:
            with zipfile.ZipFile(file_path) as z:
                xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            for p in tree.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
                texts = [
                    node.text for node in p.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
                    if node.text
                ]
                if texts:
                    paragraphs.append("".join(texts))
        except Exception as e:
            paragraphs.append(f"Error parsing DOCX file: {e}")

        full_text = "\n\n".join(paragraphs)
        return DocumentParser.parse_markdown(
            full_text,
            title=doc_title,
            metadata={"source_path": file_path, "paragraph_count": len(paragraphs), "file_type": "docx"},
            allowed_roles=allowed_roles
        )

    @classmethod
    def parse_file(cls, file_path: str, title: str = None, allowed_roles: List[str] = None) -> Document:
        filename = os.path.basename(file_path)
        doc_title = title or os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            return cls.parse_pdf(file_path, title=doc_title, allowed_roles=allowed_roles)
        elif ext == ".docx":
            return cls.parse_docx(file_path, title=doc_title, allowed_roles=allowed_roles)

        # Markdown / Plain text / Code files
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
            
        return cls.parse_markdown(
            content, 
            title=doc_title, 
            metadata={"source_path": file_path, "file_type": ext.lstrip(".")}, 
            allowed_roles=allowed_roles
        )

    @classmethod
    def parse_directory(cls, dir_path: str, allowed_roles: List[str] = None) -> List[Document]:
        """
        Recursively scans and parses all supported documents in a directory.
        """
        supported_exts = {".pdf", ".docx", ".md", ".markdown", ".txt", ".rst"}
        documents = []

        for root, _, files in os.walk(dir_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in supported_exts:
                    full_path = os.path.join(root, file)
                    try:
                        doc = cls.parse_file(full_path, allowed_roles=allowed_roles)
                        documents.append(doc)
                    except Exception as e:
                        print(f"Skipping file {full_path}: {e}")

        return documents

