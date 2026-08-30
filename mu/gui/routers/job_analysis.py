"""Retrospective engineering-job analysis endpoints.

This router is mounted beneath the existing /api/jobs prefix by
mu.gui.routers.__init__, keeping analysis calculations control-plane neutral.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from mu.jobs.analysis import compare_job_analyses
from mu.jobs.performance import build_job_performance


router = APIRouter()


def _service(request: Request):
    service = getattr(request.app.state, "job_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="job service is unavailable")
    return service


@router.get("/analysis/compare")
async def compare_job_endpoint(
    request: Request,
    job_id: str = Query(...),
    compare_id: str = Query(...),
    timeline_limit: int = Query(default=5000, ge=100, le=20000),
):
    """Compare two jobs: delta = primary - reference for every metric."""
    service = _service(request)
    try:
        primary = build_job_performance(service, job_id, timeline_limit=timeline_limit)
        reference = build_job_performance(
            service, compare_id, timeline_limit=timeline_limit
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Job not found: {exc}") from exc
    return {
        "reference": {"job": reference.get("job") or {}},
        "comparison": compare_job_analyses(primary, reference),
    }


@router.get("/{job_id}/analysis")
async def get_job_analysis(
    job_id: str,
    request: Request,
    timeline_limit: int = Query(default=5000, ge=100, le=20000),
):
    try:
        return {
            "analysis": build_job_performance(
                _service(request),
                job_id,
                timeline_limit=timeline_limit,
            )
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found") from exc
