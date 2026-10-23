import pytest
from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import StructureAwareChunker
from src.ingestion.vector_indexer import VectorIndexer
from src.ingestion.keyword_indexer import KeywordIndexer
from src.ingestion.graph_indexer import KnowledgeGraphIndexer

SAMPLE_DOC = """# Cloud Infrastructure SLA

## Backup Policy
BackupManager handles automated daily snapshots at 02:00 UTC. 
Data retention is strictly enforced for 30 days.

## Database Dependencies
Database cluster depends on PostgreSQL. 
AuthService requires Redis for session tokens.
"""

def test_document_parser():
    doc = DocumentParser.parse_markdown(SAMPLE_DOC, title="Cloud SLA")
    assert doc.title == "Cloud SLA"
    assert doc.metadata["headings_count"] == 3
    assert "Backup Policy" in doc.metadata["headings"]

def test_docx_parser(tmp_path):
    import zipfile
    docx_file = tmp_path / "test.docx"
    xml_content = b'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:t>Docx paragraph 1 content.</w:t></w:p><w:p><w:t>Docx paragraph 2 with SLA policy.</w:t></w:p></w:body></w:document>'
    with zipfile.ZipFile(docx_file, "w") as z:
        z.writestr("word/document.xml", xml_content)
    
    doc = DocumentParser.parse_file(str(docx_file), title="Test Docx")
    assert doc.title == "Test Docx"
    assert "Docx paragraph 1 content." in doc.content
    assert "Docx paragraph 2 with SLA policy." in doc.content


def test_hierarchical_chunker():
    doc = DocumentParser.parse_markdown(SAMPLE_DOC, title="Cloud SLA")
    chunker = StructureAwareChunker(parent_max_words=100, child_max_words=25, overlap_words=5)
    parents, children = chunker.chunk_document(doc)

    assert len(parents) >= 2
    assert len(children) >= 2
    for child in children:
        assert child.parent_id is not None
        assert child.doc_id == doc.doc_id

def test_tri_indexing():
    doc = DocumentParser.parse_markdown(SAMPLE_DOC, title="Cloud SLA")
    chunker = StructureAwareChunker()
    parents, children = chunker.chunk_document(doc)

    # 1. Vector Index
    v_idx = VectorIndexer()
    v_idx.index_chunks(children)
    v_results = v_idx.search("backup schedule", top_k=2)
    assert len(v_results) > 0

    # 2. Keyword Index
    k_idx = KeywordIndexer()
    k_idx.index_chunks(children)
    k_results = k_idx.search("PostgreSQL", top_k=2)
    assert len(k_results) > 0
    assert any("PostgreSQL" in c.text for c, s in k_results)

    # 3. Knowledge Graph Index
    g_idx = KnowledgeGraphIndexer()
    g_idx.index_chunks(children)
    subgraph = g_idx.query_subgraph(["Database", "AuthService"])
    assert len(subgraph["entities"]) > 0
