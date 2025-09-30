"""
Middleware package for UpskillAdvisor.

This package contains middleware components for the FastAPI application,
including PII redaction, logging, and other cross-cutting concerns.
"""

from .pii_middleware import PIIRedactionMiddleware, create_pii_middleware_config

__all__ = [
    'PIIRedactionMiddleware',
    'create_pii_middleware_config'
]
