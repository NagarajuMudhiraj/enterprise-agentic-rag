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

    UNICODE_PUNCTUATION = str.maketrans({
        '\u202f': ' ', '\xa0': ' ',
        '\u2010': '-', '\u2011': '-', '\u2012': '-', '\u2013': '-', '\u2014': '-', '\u2015': '-', '\u2212': '-',
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
    })

    @classmethod
    def normalize_text(cls, text: str) -> str:
        """
        Maps typographic Unicode spaces, hyphens and quotes (common in LLM output) to ASCII.
        """
        return text.translate(cls.UNICODE_PUNCTUATION) if text else text

    @classmethod
    def extract_claims(cls, answer_text: str) -> List[AtomicClaim]:
        if not answer_text:
            return []

        normalized = cls.normalize_text(answer_text)

        claims: List[AtomicClaim] = []
        # Paragraph by paragraph: LLMs often cite once at the end of a paragraph
        for paragraph in re.split(r'\n\s*\n', normalized):
            para_claims = cls._extract_paragraph_claims(paragraph, claims)
            # Uncited sentences inherit the nearest following (else preceding) citation in the paragraph
            for idx, claim in enumerate(para_claims):
                if claim.cited_chunk_ids:
                    continue
                donor = next((c for c in para_claims[idx + 1:] if c.cited_chunk_ids), None) or \
                    next((c for c in reversed(para_claims[:idx]) if c.cited_chunk_ids), None)
                if donor:
                    claim.cited_chunk_ids = list(donor.cited_chunk_ids)
            claims.extend(para_claims)

        return claims

    @classmethod
    def _extract_paragraph_claims(cls, paragraph: str, previous_claims: List[AtomicClaim]) -> List[AtomicClaim]:
        # Split into sentences without breaking bracketed citation tags
        raw_sentences = re.split(r'(?<=[.!?])\s+(?![^\[\u3010]*[\]\u3011])', paragraph)
        claims: List[AtomicClaim] = []

        for sent in raw_sentences:
            sent_clean = sent.strip()
            if not sent_clean:
                continue
            last_claim = claims[-1] if claims else (previous_claims[-1] if previous_claims else None)

            # Check if this snippet is strictly an orphaned citation tag [doc#chunk] or 【doc#chunk】
            if re.fullmatch(r'[\[\u3010]\s*[a-zA-Z0-9_\-]+#[a-zA-Z0-9_\-]+\s*[\]\u3011]\.?', sent_clean):
                if last_claim:
                    orphan_citations = cls.CITATION_PATTERN.findall(sent_clean)
                    for c in orphan_citations:
                        if c not in last_claim.cited_chunk_ids:
                            last_claim.cited_chunk_ids.append(c)
                continue

            # Check if snippet starts with a citation intended for previous sentence (e.g., "[doc#chunk] Next sentence")
            leading_match = re.match(r'^([\[\u3010]\s*[a-zA-Z0-9_\-]+#[a-zA-Z0-9_\-]+\s*[\]\u3011]\.?\s*)(.*)', sent_clean)
            if leading_match and last_claim:
                leading_tag = cls.CITATION_PATTERN.findall(leading_match.group(1))
                for c in leading_tag:
                    if c not in last_claim.cited_chunk_ids:
                        last_claim.cited_chunk_ids.append(c)
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


