#!/usr/bin/env bash
# Kill processes whose command line matches the pattern, never this script or its parent shell.
for p in $(pgrep -f "$1"); do [ "$p" != "$$" ] && [ "$p" != "$PPID" ] && kill "$p" 2>/dev/null && echo "killed $p"; done
