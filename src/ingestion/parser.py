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
        Word headings become Markdown headings (so the chunker can split by section) and
        tables are flattened into pipe-delimited rows.
        """
        import zipfile
        import xml.etree.ElementTree as ET

        W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        filename = os.path.basename(file_path)
        doc_title = title or os.path.splitext(filename)[0]

        def paragraph_text(p) -> str:
            parts = []
            for node in p.iter():
                if node.tag == W + "t" and node.text:
                    parts.append(node.text)
                elif node.tag == W + "tab":
                    parts.append("\t")
                elif node.tag in (W + "br", W + "cr"):
                    parts.append("\n")
            return "".join(parts).strip()

        def heading_level(p) -> int:
            ppr = p.find(W + "pPr")
            if ppr is None:
                return 0
            style = ppr.find(W + "pStyle")
            if style is not None:
                val = (style.get(W + "val") or "").lower()
                if val == "title":
                    return 1
                m = re.match(r'^heading\s*(\d)$', val)
                if m:
                    return int(m.group(1))
            outline = ppr.find(W + "outlineLvl")
            if outline is not None:
                try:
                    return int(outline.get(W + "val", "9")) + 1
                except ValueError:
                    pass
            return 0

        def walk(container, out: List[str]):
            for child in container:
                if child.tag == W + "p":
                    text = paragraph_text(child)
                    if not text:
                        continue
                    level = heading_level(child)
                    if 1 <= level <= 6:
                        out.append(f"{'#' * level} {' '.join(text.split())}")
                    else:
                        out.append(text)
                elif child.tag == W + "tbl":
                    rows = []
                    for tr in child.iter(W + "tr"):
                        cells = []
                        for tc in tr.findall(W + "tc"):
                            cell = " ".join(paragraph_text(p) for p in tc.iter(W + "p")).strip()
                            cells.append(" ".join(cell.split()))
                        if any(cells):
                            rows.append("| " + " | ".join(cells) + " |")
                    if rows:
                        out.append("\n".join(rows))
                elif child.tag in (W + "sdt", W + "sdtContent", W + "body"):
                    walk(child, out)

        paragraphs: List[str] = []
        try:
            with zipfile.ZipFile(file_path) as z:
                xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            body = tree.find(W + "body")
            walk(body if body is not None else tree, paragraphs)
        except Exception as e:
            raise ValueError(f"Could not read DOCX file '{filename}': {e}") from e

        if not paragraphs:
            raise ValueError(f"No readable text found in DOCX file '{filename}'.")

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
        elif ext == ".doc":
            raise ValueError("Legacy .doc files are not supported. Save the file as .docx in Word and upload again.")

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

