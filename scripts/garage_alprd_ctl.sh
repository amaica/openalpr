#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
DAEMON="${DAEMON:-$ROOT/build/src/garage_alpr_daemon}"
CFG="${CFG:-$ROOT/config/openalpr.conf.defaults}"
ENH="${ENH:-$ROOT/mini_enhancer/build/mini_enhancer}"

FIFO="${FIFO:-/tmp/garage_alprd.in}"
PIDFILE="${PIDFILE:-/tmp/garage_alprd.pid}"
OUTLOG="${OUTLOG:-/tmp/garage_alprd.out.jsonl}"
ERRLOG="${ERRLOG:-/tmp/garage_alprd.err.jsonl}"

usage() {
  cat <<EOF
Usage:
  $0 start
  $0 stop
  $0 status
  $0 ping
  $0 stats
  $0 send <image_path>

Env (opcional):
  DAEMON=$DAEMON
  CFG=$CFG
  ENH=$ENH
  FIFO=$FIFO
  PIDFILE=$PIDFILE
  OUTLOG=$OUTLOG
  ERRLOG=$ERRLOG
EOF
}

is_running() {
  [[ -f "$PIDFILE" ]] || return 1
  local pid
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  if kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  # stale pidfile
  rm -f "$PIDFILE"
  return 1
}

start() {
  if is_running; then
    echo "already running pid=$(cat "$PIDFILE")"
    return 0
  fi
  rm -f "$FIFO"
  mkfifo "$FIFO"
  : > "$OUTLOG"
  : > "$ERRLOG"
  # Keep the FIFO write-end open so the daemon doesn't see EOF
  # when a client writes one line and closes.
  # Open FIFO read+write to avoid blocking on open().
  nohup bash -c "exec 3<>\"$FIFO\"; exec \"$DAEMON\" --country br --config \"$CFG\" --enh-bin \"$ENH\" --external-enhancer <\"$FIFO\" >\"$OUTLOG\" 2>\"$ERRLOG\"" &
  echo $! > "$PIDFILE"
  echo "started pid=$! fifo=$FIFO"
  # Wait for daemon to become ready (loads models on first start).
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    send_cmd "__ping" >/dev/null 2>&1 || true
    sleep 0.2
    if [[ -s "$OUTLOG" ]]; then
      return 0
    fi
  done
  echo "warning: daemon may still be starting; check logs: $ERRLOG"
}

send_cmd() {
  local s="$1"
  [[ -p "$FIFO" ]] || { echo "FIFO missing: $FIFO (start first)"; return 1; }
  # Writing to a FIFO blocks if daemon hasn't opened it yet.
  # Use a short timeout so ctl doesn't hang.
  if command -v timeout >/dev/null 2>&1; then
    timeout 10s bash -c "printf '%s\n' \"\$1\" > \"\$2\"" _ "$s" "$FIFO" || {
      echo "write timeout (daemon not ready?)"
      return 1
    }
  else
    printf '%s\n' "$s" > "$FIFO"
  fi
}

stop() {
  if ! is_running; then
    echo "not running"
    return 0
  fi
  send_cmd "__quit" || true
  local pid
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PIDFILE"
      echo "stopped"
      return 0
    fi
    sleep 0.1
  done
  echo "forcing kill pid=$pid"
  kill "$pid" 2>/dev/null || true
  rm -f "$PIDFILE"
}

status() {
  if is_running; then
    echo "running pid=$(cat "$PIDFILE")"
    echo "fifo=$FIFO"
    echo "out=$OUTLOG"
    echo "err=$ERRLOG"
  else
    echo "not running"
  fi
}

cmd="${1:-}"
case "$cmd" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  ping) send_cmd "__ping" ;;
  stats) send_cmd "__stats" ;;
  send)
    [[ -n "${2:-}" ]] || { usage; exit 1; }
    send_cmd "$2"
    ;;
  -h|--help|"") usage; exit 0 ;;
  *) usage; exit 1 ;;
esac

