#!/usr/bin/env bash
set -Eeuo pipefail

readonly READRAFT_APP_DIR="${READRAFT_APP_DIR:-/opt/readraft}"
readonly READRAFT_SERVICE="${READRAFT_SERVICE:-readraft}"
readonly READRAFT_UPDATE_CONFIG="${READRAFT_UPDATE_CONFIG:-/etc/readraft/update.env}"

fail() {
  printf 'Readraft auto-update installation failed: %s\n' "$*" >&2
  exit 1
}

if (( EUID != 0 )); then
  fail "run this script with sudo"
fi

command -v systemctl >/dev/null || fail "systemctl is not installed"
[[ -x "$READRAFT_APP_DIR/deploy/update.sh" ]] \
  || fail "$READRAFT_APP_DIR/deploy/update.sh is missing"
[[ -f "$READRAFT_APP_DIR/deploy/readraft-update.service" ]] \
  || fail "readraft-update.service is missing"
[[ -f "$READRAFT_APP_DIR/deploy/readraft-update.timer" ]] \
  || fail "readraft-update.timer is missing"
systemctl is-active --quiet "$READRAFT_SERVICE" \
  || fail "$READRAFT_SERVICE is not active"

install -d -m 0755 "${READRAFT_UPDATE_CONFIG%/*}"
if [[ ! -e "$READRAFT_UPDATE_CONFIG" ]]; then
  temporary_config="$(mktemp -t readraft-update-env.XXXXXX)"
  trap 'rm -f -- "$temporary_config"' EXIT
  cat >"$temporary_config" <<'EOF'
# Stable annotated vMAJOR.MINOR.PATCH tags only. Use "main" only on a
# development server that can tolerate every pushed commit.
READRAFT_UPDATE_CHANNEL=release
READRAFT_BACKUP_RETENTION_DAYS=30
READRAFT_VENV_RETENTION=3
EOF
  install -o root -g root -m 0600 \
    "$temporary_config" "$READRAFT_UPDATE_CONFIG"
fi

install -m 0644 "$READRAFT_APP_DIR/deploy/readraft-update.service" \
  /etc/systemd/system/readraft-update.service
install -m 0644 "$READRAFT_APP_DIR/deploy/readraft-update.timer" \
  /etc/systemd/system/readraft-update.timer
systemctl daemon-reload

"$READRAFT_APP_DIR/deploy/update.sh" --check
systemctl enable --now readraft-update.timer

printf 'Readraft automatic updates are enabled. Next run:\n'
systemctl list-timers readraft-update.timer --all --no-pager
