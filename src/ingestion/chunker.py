import re
import uuid
from typing import List, Tuple
from src.common.models import Document, Chunk

class StructureAwareChunker:
    """
    Splits documents into hierarchical Parent-Child chunks.
    - Parent chunks: ~512-1024 words (logical section context).
    - Child chunks: ~100-150 words (high-precision retrieval units).
    - Preserves Markdown heading lineages and metadata.
    """

    def __init__(self, parent_max_words: int = 500, child_max_words: int = 120, overlap_words: int = 20):
        self.parent_max_words = parent_max_words
        self.child_max_words = child_max_words
        self.overlap_words = overlap_words

    def chunk_document(self, doc: Document) -> Tuple[List[Chunk], List[Chunk]]:
        """
        Returns (parent_chunks, child_chunks)
        """
        parent_chunks: List[Chunk] = []
        child_chunks: List[Chunk] = []

        # Split by markdown headers (#, ##, ###)
        sections = re.split(r'\n(?=#{1,4}\s)', doc.content)
        if not sections or (len(sections) == 1 and not sections[0].startswith('#')):
            # Fallback to paragraph-based split
            sections = [s for s in doc.content.split('\n\n') if s.strip()]

        for sec_idx, section in enumerate(sections):
            sec_text = section.strip()
            if not sec_text:
                continue

            # Create Parent Chunk
            parent_id = f"p_{doc.doc_id}_{sec_idx}_{uuid.uuid4().hex[:4]}"
            parent_chunk = Chunk(
                chunk_id=parent_id,
                doc_id=doc.doc_id,
                parent_id=None,
                text=sec_text,
                metadata={
                    **doc.metadata,
                    "is_parent": True,
                    "section_index": sec_idx,
                    "title": doc.title
                },
                allowed_roles=doc.allowed_roles
            )
            parent_chunks.append(parent_chunk)

            # Subdivide parent into overlapping child chunks
            words = sec_text.split()
            if len(words) <= self.child_max_words:
                # Section is small enough to be its own child chunk
                child_chunk = Chunk(
                    chunk_id=f"c_{doc.doc_id}_{sec_idx}_0",
                    doc_id=doc.doc_id,
                    parent_id=parent_id,
                    text=sec_text,
                    metadata={
                        **doc.metadata,
                        "is_parent": False,
                        "parent_id": parent_id,
                        "title": doc.title
                    },
                    allowed_roles=doc.allowed_roles
                )
                child_chunks.append(child_chunk)
            else:
                step = max(1, self.child_max_words - self.overlap_words)
                child_idx = 0
                for start in range(0, len(words), step):
                    chunk_words = words[start:start + self.child_max_words]
                    if not chunk_words:
                        break
                    child_chunk = Chunk(
                        chunk_id=f"c_{doc.doc_id}_{sec_idx}_{child_idx}",
                        doc_id=doc.doc_id,
                        parent_id=parent_id,
                        text=" ".join(chunk_words),
                        metadata={
                            **doc.metadata,
                            "is_parent": False,
                            "parent_id": parent_id,
                            "child_index": child_idx,
                            "title": doc.title
                        },
                        allowed_roles=doc.allowed_roles
                    )
                    child_chunks.append(child_chunk)
                    child_idx += 1
                    if start + self.child_max_words >= len(words):
                        break

        return parent_chunks, child_chunks
