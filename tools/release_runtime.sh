#!/usr/bin/env bash
# Sourced by both release cycles before they create or update any release files.
if ! OPEN_CAKE_PYTHON=$(command -v "${OPEN_CAKE_PYTHON:-python3}"); then
  echo "release cycles require Python >= 3.10; set OPEN_CAKE_PYTHON to a supported executable" >&2
  exit 2
fi
# Keep a relative executable valid when the cycle changes to the project root.
case "$OPEN_CAKE_PYTHON" in
  /*) ;;
  *) OPEN_CAKE_PYTHON="$PWD/$OPEN_CAKE_PYTHON" ;;
esac
export OPEN_CAKE_PYTHON
if ! "$OPEN_CAKE_PYTHON" -c '
import sys
if sys.version_info < (3, 10):
    sys.stderr.write(
        "release cycles require Python >= 3.10; selected "
        + sys.executable + " (" + sys.version.split()[0] + ")\n"
    )
    sys.exit(2)
'; then
  exit 2
fi
