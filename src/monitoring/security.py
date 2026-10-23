import re
from typing import List, Optional
from src.common.models import Chunk

class SecurityManager:
    """
    Enterprise security guardrails:
    1. Document & Chunk-level Access Control List (ACL) validation.
    2. PII Sanitization & Masking.
    """

    EMAIL_REGEX = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    API_KEY_REGEX = r'(?:api[_-]?key|secret|token)[\s:=]+([a-zA-Z0-9_\-]{16,})'
    PHONE_REGEX = r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'

    @staticmethod
    def check_access(chunk: Chunk, user_roles: Optional[List[str]] = None) -> bool:
        roles = set(user_roles or ["public"])
        allowed = set(chunk.allowed_roles)
        return bool(roles.intersection(allowed))

    @classmethod
    def mask_pii(cls, text: str) -> str:
        # Mask emails
        sanitized = re.sub(cls.EMAIL_REGEX, "[EMAIL_MASKED]", text)
        # Mask phone numbers
        sanitized = re.sub(cls.PHONE_REGEX, "[PHONE_MASKED]", sanitized)
        # Mask API keys
        sanitized = re.sub(cls.API_KEY_REGEX, "api_key=[SECRET_MASKED]", sanitized, flags=re.IGNORECASE)
        return sanitized
