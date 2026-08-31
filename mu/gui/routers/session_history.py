"""Authoritative session-history selection for the web GUI.

Opening a saved session must not depend on the freshly reconstructed in-memory
Session already containing its transcript.  Named history requests compare the
live and durable copies and render whichever contains the more complete
conversation.  This recovers empty/partially hydrated live sessions without
throwing away newer in-memory turns that have not yet reached disk.
"""

from __future__ import annotations

import os
from functools import partial
from types import SimpleNamespace
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query, Request

from mu.gui.async_utils import run_sync_responsive

from . import sessions as sessions_router
from ._session_summary import read_session_summary


router = APIRouter()


def _saved_history_session(name: str):
    """Return a minimal session facade backed by the durable session JSON."""
    data = sessions_router._read_session_data(name)
    if data is None:
        return None

    if isinstance(data, list):
        history = data
    elif isinstance(data, dict):
        history = data.get("history", [])
    else:
        history = []

    if not isinstance(history, list):
        history = []

    manager = SimpleNamespace(
        current_session_name=name,
        history=history,
    )
    return SimpleNamespace(session_manager=manager)


def _history_length(session) -> int:
    manager = getattr(session, "session_manager", None)
    history = getattr(manager, "history", None)
    return len(history) if isinstance(history, list) else 0


def _saved_history_candidate(name: str, live_session):
    """Read durable history only when it can be newer than the live copy."""
    if live_session is None or _history_length(live_session) == 0:
        return _saved_history_session(name)

    manager = getattr(live_session, "session_manager", None)
    live_revision = getattr(manager, "revision", None)
    if live_revision is None:
        # Lightweight facades and legacy managers have no revision marker, so
        # retain the conservative length comparison used before revisions.
        return _saved_history_session(name)

    path = os.path.join(sessions_router._safe_session_dir(name), "session.json")
    if not os.path.isfile(path):
        return None
    summary = read_session_summary(path)
    if "revision" not in summary:
        return _saved_history_session(name)
    try:
        durable_revision = int(summary.get("revision", 0) or 0)
        current_revision = int(live_revision or 0)
    except (TypeError, ValueError):
        return _saved_history_session(name)
    if durable_revision > current_revision:
        return _saved_history_session(name)
    return None


def _request_for_session(session):
    """Build the tiny request facade consumed by sessions.get_history()."""
    state = SimpleNamespace(session_by_name=lambda _name=None: session)
    return SimpleNamespace(app=SimpleNamespace(state=state))


@router.get("/current/history")
async def get_authoritative_history(
    request: Request,
    session_name: Optional[str] = None,
    limit_turns: Optional[int] = Query(default=None, ge=1, le=500),
    artifact_limit: Optional[int] = Query(default=None, ge=0, le=100),
    before_index: Optional[int] = Query(default=None, ge=0),
    after_index: Optional[int] = None,
    checkpoint_count: Optional[int] = Query(default=None, ge=1, le=20),
    full: bool = False,
) -> Dict[str, Any]:
    """Return the most complete timeline available for a named session."""
    selected_request = request
    history_source = "live_session"
    recovered = False

    if session_name:
        live_session = request.app.state.session_by_name(session_name)
        # A saved session can be very large. Keep both the bounded summary scan
        # and the occasional full recovery decode off the ASGI event loop so a
        # history page never stalls SSE, interrupts, or the rest of the GUI.
        saved_session = await run_sync_responsive(
            partial(
                _saved_history_candidate,
                session_name,
                live_session,
            )
        )
        live_count = _history_length(live_session)
        saved_count = _history_length(saved_session)

        # A newly reconstructed Session can briefly exist with no/partial
        # history even though session.json still contains the full transcript.
        # Prefer durable state only when it is strictly more complete.  If the
        # live copy has newer turns, keep it so a just-finished response is not
        # replaced with an older disk snapshot.
        if saved_session is not None and saved_count > live_count:
            selected_request = _request_for_session(saved_session)
            history_source = "durable_session"
            recovered = live_session is not None

    payload = await sessions_router.get_history(
        selected_request,
        session_name=session_name,
        limit_turns=limit_turns,
        artifact_limit=artifact_limit,
        before_index=before_index,
        after_index=after_index,
        checkpoint_count=(
            checkpoint_count if isinstance(checkpoint_count, int) else None
        ),
        full=full,
    )
    payload["history_source"] = history_source
    payload["history_recovered"] = recovered
    return payload
