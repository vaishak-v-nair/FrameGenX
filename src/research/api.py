"""Read-only research results; existing local API trust scope still applies."""
from fastapi import APIRouter, HTTPException, Query
from src.research.reports import list_experiments

router = APIRouter(prefix="/api/research", tags=["experimental research"])


@router.get("/experiments")
def research_experiments(limit: int = Query(default=10, ge=1, le=25)):
    try:
        return list_experiments(limit=limit)
    except (OSError, ValueError):
        raise HTTPException(status_code=503, detail={"code": "FGX_RESEARCH_STORAGE_UNAVAILABLE",
            "message": "Research evidence storage is unavailable; retry this read."}) from None
