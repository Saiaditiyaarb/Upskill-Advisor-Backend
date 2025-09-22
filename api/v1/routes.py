"""
Versioned API v1 routes.

Design choices:
- The router does not hardcode a version prefix; main.py mounts it using settings.api_v1_prefix. This allows changing the prefix centrally.
- Responses are wrapped in the generic ApiResponse to keep a stable envelope while inner data evolves.
"""
from __future__ import annotations

from uuid import uuid4
import logging

from fastapi import APIRouter, Depends

from schemas.api import AdviseRequest, AdviseResult, ApiResponse
from services.advisor_service import advise
from services.retriever import Retriever, get_retriever
from core.logging_config import set_request_id

# Optional caching decorator; no-op if fastapi-cache2 is not installed
try:
    from fastapi_cache.decorator import cache  # type: ignore
except Exception:  # pragma: no cover
    def cache(expire: int = 60):  # type: ignore
        def _wrap(func):
            return func
        return _wrap

router = APIRouter(tags=["advisor"])  # mounted under /api/v1 by main.py
logger = logging.getLogger("api")


@router.post("/advise", response_model=ApiResponse[AdviseResult])
@cache(expire=60)
async def post_advise(request: AdviseRequest, retriever: Retriever = Depends(get_retriever)) -> ApiResponse[AdviseResult]:
    """Return course recommendations and a learning plan based on the user's profile.

    The response is wrapped in ApiResponse to future-proof client integrations by keeping
    request_id and status fields stable across versions.
    """
    req_id = str(uuid4())
    set_request_id(req_id)
    logger.info("advise_request_received", extra={"request_id": req_id})
    result = await advise(request, retriever)
    logger.info("advise_request_completed", extra={"request_id": req_id})
    return ApiResponse[AdviseResult](request_id=req_id, status="ok", data=result)
