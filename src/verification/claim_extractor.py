import re
from typing import List
from src.common.models import AtomicClaim

class ClaimExtractor:
    """
    Parses a generated response into discrete, verifiable atomic claims with citation tags.
    Robustly handles ASCII brackets [...], fullwidth brackets 【...】, internal whitespace,
    and Unicode non-breaking spaces.
    """

    CITATION_PATTERN = re.compile(r'[\[\u3010]\s*([a-zA-Z0-9_\-]+#[a-zA-Z0-9_\-]+)\s*[\]\u3011]')

    @classmethod
    def extract_claims(cls, answer_text: str) -> List[AtomicClaim]:
        if not answer_text:
            return []

        # Normalize unicode spaces (narrow no-break space, non-breaking space)
        normalized = answer_text.replace('\u202f', ' ').replace('\xa0', ' ')

        # Split into sentences without breaking bracketed citation tags
        raw_sentences = re.split(r'(?<=[.!?])\s+(?![^\[\u3010]*[\]\u3011])', normalized)
        claims: List[AtomicClaim] = []

        for sent in raw_sentences:
            sent_clean = sent.strip()
            if not sent_clean:
                continue

            # Check if this snippet is strictly an orphaned citation tag [doc#chunk] or 【doc#chunk】
            if re.fullmatch(r'[\[\u3010]\s*[a-zA-Z0-9_\-]+#[a-zA-Z0-9_\-]+\s*[\]\u3011]\.?', sent_clean):
                if claims:
                    orphan_citations = cls.CITATION_PATTERN.findall(sent_clean)
                    for c in orphan_citations:
                        if c not in claims[-1].cited_chunk_ids:
                            claims[-1].cited_chunk_ids.append(c)
                continue

            # Check if snippet starts with a citation intended for previous sentence (e.g., "[doc#chunk] Next sentence")
            leading_match = re.match(r'^([\[\u3010]\s*[a-zA-Z0-9_\-]+#[a-zA-Z0-9_\-]+\s*[\]\u3011]\.?\s*)(.*)', sent_clean)
            if leading_match and claims:
                leading_tag = cls.CITATION_PATTERN.findall(leading_match.group(1))
                for c in leading_tag:
                    if c not in claims[-1].cited_chunk_ids:
                        claims[-1].cited_chunk_ids.append(c)
                sent_clean = leading_match.group(2).strip()
                if not sent_clean:
                    continue

            if len(sent_clean) < 10:
                continue

            # Find citation tags like [doc_id#chunk_id] or 【doc_id#chunk_id】
            citations = cls.CITATION_PATTERN.findall(sent_clean)
            # Deduplicate while preserving order
            unique_citations = list(dict.fromkeys(citations))

            # Clean text by removing brackets for cleaner entailment inspection
            claim_body = cls.CITATION_PATTERN.sub('', sent_clean).strip()
            # Clean markdown bold/italics markers
            claim_body = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', claim_body)

            claims.append(
                AtomicClaim(
                    claim_text=claim_body if claim_body else sent_clean,
                    cited_chunk_ids=unique_citations,
                    entailment_status="neutral",
                    entailment_score=0.0
                )
            )

        return claims


