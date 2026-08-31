"""Shared pytest fixtures for test isolation.

- Redirects every MuCLI runtime write to a disposable suite-owned home.
- Resets SemanticCodeIndex singleton between tests to prevent cross-test state leaks.
- Cleans up FolderContext snapshots to prevent memory accumulation.
- Scrubs any `courses/*` directories created during the session —
  teacher engines resolve their workspace root from `os.getcwd()` when
  `folder_context=None`, which in CI is the repo root, so tests that
  don't explicitly chdir end up polluting the working tree.
"""
import atexit
import glob
import os
import shutil
import tempfile
from pathlib import Path


# This must happen before importing any project module. ``utils.config``
# resolves and creates HISTORY_DIR at import time; setting MUCLI_HOME from a
# fixture is therefore too late and lets tests write sessions/traces into the
# user's real ~/.mucli. Subprocesses inherit this environment automatically.
_TEST_MUCLI_HOME = tempfile.mkdtemp(prefix="mucli-pytest-")
os.environ["MUCLI_HOME"] = _TEST_MUCLI_HOME


def _remove_test_mucli_home() -> None:
    shutil.rmtree(_TEST_MUCLI_HOME, ignore_errors=True)


# The session fixture handles normal shutdown. The atexit fallback also covers
# collection failures and interrupted runs where fixture teardown is skipped.
atexit.register(_remove_test_mucli_home)

import pytest
from mu.retrieval.index import SemanticCodeIndex
from mu.retrieval.index import RETRIEVAL_INDEX as _RETRIEVAL_INDEX
from mu.workspace.folder_context import FolderContext


# Directories the feature / teacher engines may create relative to cwd
# when no folder_context is attached. The session-end cleanup walks the
# repo and removes anything in these patterns that wasn't present at
# session start.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TEST_ARTIFACT_PATTERNS = (
    "courses",
    ".mucli",
)


def _snapshot_artifacts() -> set[str]:
    found: set[str] = set()
    for pattern in _TEST_ARTIFACT_PATTERNS:
        for path in glob.glob(str(_REPO_ROOT / pattern)):
            found.add(path)
    return found


@pytest.fixture(scope="session", autouse=True)
def _cleanup_repo_test_artifacts():
    """Session-end safety net: remove any feature/teacher artifacts that
    weren't on disk before the suite ran.

    Existing artifacts (manual development state) are preserved — we
    only delete what tests created. Belt-and-suspenders alongside the
    per-module autouse chdir fixtures that prevent the writes in the
    first place."""
    pre_existing = _snapshot_artifacts()
    try:
        yield
    finally:
        post = _snapshot_artifacts()
        for path in sorted(post - pre_existing):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                elif os.path.isfile(path):
                    os.remove(path)
            except OSError:
                pass
        _remove_test_mucli_home()


@pytest.fixture(autouse=True)
def _reset_global_singletons():
    """Reset module-level singletons that accumulate state across tests.

    The SemanticCodeIndex singleton (_RETRIEVAL_INDEX in mu.tools) indexes
    workspace files and caches them in .documents. Without resetting, state
    from one test leaks into the next, causing memory growth and flaky tests.
    """
    _RETRIEVAL_INDEX.reset()
    yield
    _RETRIEVAL_INDEX.reset()


@pytest.fixture(autouse=True)
def _cleanup_folder_context():
    """Clear FolderContext state between tests to prevent memory accumulation.

    FolderContext tracks all live instances in _instances. Without cleanup,
    snapshots and folder references accumulate across tests, causing memory
    growth and potential hangs.
    """
    yield
    FolderContext.reset_all()
