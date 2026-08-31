#!/usr/bin/env bash
# Self-reset: stop GUI, wait, restart on 0.0.0.0.
# Runs detached so mucli shutdown won't kill it.
set -euo pipefail

MUCLI="$(command -v mucli)"

# detach from parent process group entirely
setsid --wait "$MUCLI" --gui-stop >/dev/null 2>&1 || true

# give the daemon time to release the port
sleep 3

# restart — this stays in our background session
exec env MUCLI_GUI_ALLOW_REMOTE=1 "$MUCLI" --gui --host 0.0.0.0
