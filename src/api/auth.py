"""API-key authentication and server-side role resolution.

Authorization is always derived from the credential, never from the request
body. A caller cannot widen its own access by sending roles in the payload — a
mistake the previous `user_roles` request field invited.
"""

import hashlib
import hmac
from dataclasses import dataclass
from typing import List, Optional, Tuple

from fastapi import Depends, Header, HTTPException, status

from src.common.config import settings
from src.common.logging_config import get_logger

logger = get_logger(__name__)


def _fingerprint(key: str) -> str:
    """Stable, non-reversible identifier used for logs and rate limiting."""
    return "key_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Principal:
    key_id: str
    roles: Tuple[str, ...]
    authenticated: bool

    @property
    def can_write(self) -> bool:
        return bool(set(self.roles) & set(settings.write_roles))

    @property
    def is_admin(self) -> bool:
        return "admin" in self.roles

    @property
    def label(self) -> str:
        return self.key_id


ANONYMOUS = "anonymous"


def _unauthorized(reason: str) -> None:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=reason,
        headers={"WWW-Authenticate": 'ApiKey realm="enterprise-rag"'},
    )


def _match_key(supplied: str) -> Optional[Tuple[str, List[str]]]:
    for key, roles in settings.api_key_roles().items():
        if hmac.compare_digest(key, supplied):
            return key, roles
    return None


async def get_principal(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Principal:
    """Resolve the caller, or raise 401.

    When AUTH_ENABLED is false the caller is granted the configured development
    roles. That escape hatch is refused outright in production.
    """
    if not settings.AUTH_ENABLED:
        if settings.is_production:
            logger.error("AUTH_ENABLED=false is not permitted in production")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication is disabled in a production environment.",
            )
        return Principal(
            key_id=ANONYMOUS,
            roles=tuple(settings.dev_anonymous_roles),
            authenticated=False,
        )

    configured = settings.api_key_roles()
    if not configured:
        if settings.is_production:
            logger.error("No API keys provisioned for a production deployment")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server is not provisioned with any API keys.",
            )
        logger.warning("No API keys configured; falling back to development access")
        return Principal(
            key_id=ANONYMOUS,
            roles=tuple(settings.dev_anonymous_roles),
            authenticated=False,
        )

    if not x_api_key:
        _unauthorized("Missing X-API-Key header.")

    match = _match_key(x_api_key)
    if not match:
        logger.warning("Rejected unknown API key", extra={"principal": _fingerprint(x_api_key)})
        _unauthorized("Invalid API key.")

    key, roles = match
    return Principal(key_id=_fingerprint(key), roles=tuple(roles), authenticated=True)


async def require_write(principal: Principal = Depends(get_principal)) -> Principal:
    """Dependency guarding mutating endpoints (ingest, delete)."""
    if not principal.can_write:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This operation requires a role with write access.",
        )
    return principal


async def require_admin(principal: Principal = Depends(get_principal)) -> Principal:
    """Dependency guarding operator-only endpoints (metrics, audit log)."""
    if not principal.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This operation requires the admin role.",
        )
    return principal
