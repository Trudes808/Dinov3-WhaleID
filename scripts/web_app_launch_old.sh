#!/usr/bin/env bash
# Start the PREVIOUS version of the Whale Review web app (the one before the 2026-11-24 beta, code of 2026-11-23 08:42)
# next to the current one, without rolling back. Operations page: docs/web_app_versions.md.
#
#   scripts/web_app_launch_old.sh            build the copy on first use, start it on 127.0.0.1:8766
#   scripts/web_app_launch_old.sh --stop     stop it (by the pid that listens on the port)
#   scripts/web_app_launch_old.sh --status   show whether it runs
#   OLD_PORT=8767 OLD_APP_ROOT=/some/folder scripts/web_app_launch_old.sh     other port / other folder
#
# WARNING: the old app has NO sign-in. It is started on 127.0.0.1 only (never change that). It reads and writes the SAME
# researcher state files as the beta (data/<day>/labeling.json, ...): do not sort in both at the same time.
set -euo pipefail
NW=/home/sat3737/NOAA_whales
BACKUP="${OLD_BACKUP:-$NW/moonshot_cache/switch_root/backup_beta_golive_2026-11-24}"
ROOT="${OLD_APP_ROOT:-$NW/app_old_2026-11-23}"     # outside both repositories; the app is $ROOT/Whale-Review-Web
PORT="${OLD_PORT:-8766}"
APP="$ROOT/Whale-Review-Web"
LOG="$ROOT/serve_old.log"
listener_pid() { ss -ltnpH "sport = :$PORT" | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1; }

case "${1:-}" in
  --stop)
    pid=$(listener_pid)
    if [ -z "$pid" ]; then echo "nothing listens on port $PORT"; exit 0; fi
    if ! tr '\0' ' ' < /proc/$pid/cmdline | grep -q 'backend.app'; then echo "pid $pid on port $PORT is not a Whale Review server; not touching it"; exit 1; fi
    if [ "$(readlink -f /proc/$pid/cwd)" != "$(readlink -f "$APP")" ]; then echo "pid $pid on port $PORT runs from another folder; not touching it"; exit 1; fi
    kill -TERM "$pid"; echo "stopped pid $pid (port $PORT)"; exit 0 ;;
  --status)
    pid=$(listener_pid); if [ -n "$pid" ]; then echo "listening on port $PORT, pid $pid"; ss -ltnp "sport = :$PORT"; else echo "not running on port $PORT"; fi; exit 0 ;;
  "" ) ;;
  *) echo "usage: $0 [--stop|--status]"; exit 2 ;;
esac

[ "$PORT" != 8765 ] || { echo "port 8765 belongs to the current app; choose another OLD_PORT"; exit 1; }
[ -d "$BACKUP/code_before" ] || { echo "missing $BACKUP/code_before"; exit 1; }
[ -z "$(listener_pid)" ] || { echo "port $PORT is already in use (pid $(listener_pid)); run with --stop first"; exit 1; }

# 1. a runnable copy of the old code (first use only; the original in the backup folder is never touched)
if [ ! -f "$APP/backend/app.py" ]; then
  mkdir -p "$APP"
  cp -a "$BACKUP/code_before/." "$APP/"
  (cd "$APP" && sha256sum -c --quiet "$BACKUP/sha256_code_before.txt") && echo "copy of the old code verified against sha256_code_before.txt"
fi
# 2. links: the old code finds its data in ./data and its sibling trees in ../Dinov3-WhaleID, ../Whale-Image-Data-Factory
#    (the mesh of the Interpret page) and ../moonshot_cache; they all point to the live ones.
link() { [ -e "$2" ] || [ -L "$2" ] || ln -s "$1" "$2"; }
link "$NW/Whale-Review-Web/data"      "$APP/data"
link "$NW/Dinov3-WhaleID"             "$ROOT/Dinov3-WhaleID"
link "$NW/Whale-Image-Data-Factory"   "$ROOT/Whale-Image-Data-Factory"
link "$NW/moonshot_cache"             "$ROOT/moonshot_cache"

# 3. start it: the old serve.sh sets the same environment as it did on 2026-11-23; --host 127.0.0.1 replaces its 0.0.0.0
cd "$APP"
setsid nohup ./serve.sh --port "$PORT" --host 127.0.0.1 >> "$LOG" 2>&1 &
for _ in $(seq 1 30); do sleep 1; [ -n "$(listener_pid)" ] && break; done
pid=$(listener_pid)
if [ -z "$pid" ]; then echo "did not start; see $LOG"; tail -5 "$LOG"; exit 1; fi
echo "old app (no sign-in, 127.0.0.1 only): http://localhost:$PORT/   pid $pid   log $LOG"
echo "stop it: $0 --stop     (it shares the sorting state with the beta: do not sort in both at once)"
