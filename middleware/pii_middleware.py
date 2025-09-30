"""
PII Redaction Middleware for FastAPI

This middleware automatically redacts PII from request/response data
to ensure privacy compliance across all API endpoints.
"""

import logging
import json
from typing import Dict, Any, Optional
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from services.pii_redaction import PIIRedactor, create_safe_logging_config

logger = logging.getLogger(__name__)


class PIIRedactionMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically redact PII from API requests and responses."""
    
    def __init__(self, app, config: Optional[Dict[str, Any]] = None):
        super().__init__(app)
        self.pii_redactor = PIIRedactor(create_safe_logging_config())
        self.config = config or {}
        
        # Endpoints to exclude from PII redaction (if any)
        self.exclude_paths = self.config.get('exclude_paths', [])
        
        # Whether to redact request bodies
        self.redact_requests = self.config.get('redact_requests', True)
        
        # Whether to redact response bodies
        self.redact_responses = self.config.get('redact_responses', False)  # Usually not needed for responses
    
    async def dispatch(self, request: Request, call_next):
        """Process request and response with PII redaction."""
        
        # Skip PII redaction for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Log request with PII redaction
        if self.redact_requests:
            await self._log_request_safely(request)
        
        # Process the request
        response = await call_next(request)
        
        # Log response with PII redaction (if enabled)
        if self.redact_responses:
            await self._log_response_safely(response)
        
        return response
    
    async def _log_request_safely(self, request: Request):
        """Log request data with PII redaction."""
        try:
            # Get request body if it exists
            body = None
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.body()
                    if body:
                        # Try to parse as JSON
                        try:
                            body_data = json.loads(body.decode('utf-8'))
                            redacted_body = self.pii_redactor.redact_request_data(body_data)
                            logger.debug(f"Request to {request.url.path}: {redacted_body}")
                        except json.JSONDecodeError:
                            # If not JSON, redact as text
                            redacted_text = self.pii_redactor.redact_text(body.decode('utf-8'))
                            logger.debug(f"Request to {request.url.path}: {redacted_text}")
                except Exception as e:
                    logger.warning(f"Failed to process request body for PII redaction: {e}")
            
            # Log basic request info
            logger.info(f"Request: {request.method} {request.url.path}")
            
        except Exception as e:
            logger.warning(f"Failed to log request safely: {e}")
    
    async def _log_response_safely(self, response: Response):
        """Log response data with PII redaction."""
        try:
            if isinstance(response, JSONResponse):
                # Get response body
                body = response.body
                if body:
                    try:
                        body_data = json.loads(body.decode('utf-8'))
                        redacted_body = self.pii_redactor.redact_dict(body_data)
                        logger.debug(f"Response from {response.status_code}: {redacted_body}")
                    except json.JSONDecodeError:
                        # If not JSON, redact as text
                        redacted_text = self.pii_redactor.redact_text(body.decode('utf-8'))
                        logger.debug(f"Response from {response.status_code}: {redacted_text}")
            
            logger.info(f"Response: {response.status_code}")
            
        except Exception as e:
            logger.warning(f"Failed to log response safely: {e}")


def create_pii_middleware_config() -> Dict[str, Any]:
    """Create default configuration for PII redaction middleware."""
    return {
        'exclude_paths': [
            '/health',  # Health check endpoints
            '/metrics',  # Metrics endpoints
            '/docs',  # API documentation
            '/openapi.json',  # OpenAPI schema
        ],
        'redact_requests': True,
        'redact_responses': False,  # Usually not needed for responses
    }
