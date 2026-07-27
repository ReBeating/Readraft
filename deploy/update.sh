#!/usr/bin/env bash
set -Eeuo pipefail

readonly READRAFT_APP_DIR="/opt/readraft"
readonly READRAFT_APP_USER="readraft"
readonly READRAFT_SERVICE="readraft"
readonly READRAFT_BACKUP_DIR="/var/backups/readraft"
readonly READRAFT_HEALTH_URL="http://127.0.0.1:8010/healthz"

fail() {
  printf 'Readraft update failed: %s\n' "$*" >&2
  exit 1
}

run_as_readraft() {
  runuser -u "$READRAFT_APP_USER" -- "$@"
}

if (( EUID != 0 )); then
  fail "run this script with sudo"
fi

command -v git >/dev/null || fail "git is not installed"
command -v curl >/dev/null || fail "curl is not installed"
command -v runuser >/dev/null || fail "runuser is not installed"
command -v systemctl >/dev/null || fail "systemctl is not installed"

[[ -d "$READRAFT_APP_DIR/.git" ]] \
  || fail "$READRAFT_APP_DIR is not a Git checkout"
[[ -f "$READRAFT_APP_DIR/.env" ]] \
  || fail "$READRAFT_APP_DIR/.env is missing"
[[ -x "$READRAFT_APP_DIR/.venv/bin/python" ]] \
  || fail "$READRAFT_APP_DIR/.venv is missing"

if [[ -n "$(run_as_readraft git -C "$READRAFT_APP_DIR" status --porcelain)" ]]; then
  fail "the deployment checkout has uncommitted changes"
fi

upstream="$(
  run_as_readraft git -C "$READRAFT_APP_DIR" \
    rev-parse --abbrev-ref --symbolic-full-name '@{upstream}'
)" || fail "the current branch has no upstream"

run_as_readraft git -C "$READRAFT_APP_DIR" fetch --prune

current_commit="$(
  run_as_readraft git -C "$READRAFT_APP_DIR" rev-parse HEAD
)"
target_commit="$(
  run_as_readraft git -C "$READRAFT_APP_DIR" rev-parse "$upstream"
)"

if [[ "$current_commit" == "$target_commit" ]]; then
  printf 'Readraft is already current at %s.\n' "${current_commit:0:12}"
  exit 0
fi

run_as_readraft git -C "$READRAFT_APP_DIR" \
  merge-base --is-ancestor "$current_commit" "$target_commit" \
  || fail "the upstream update is not a fast-forward"

target_lock="$(mktemp -t readraft-requirements.XXXXXX)"
cleanup() {
  rm -f "$target_lock"
}
trap cleanup EXIT

run_as_readraft git -C "$READRAFT_APP_DIR" \
  show "$target_commit:requirements.lock" >"$target_lock"
chmod 0644 "$target_lock"

printf 'Installing dependencies for %s before stopping the service...\n' \
  "${target_commit:0:12}"
run_as_readraft "$READRAFT_APP_DIR/.venv/bin/python" -m pip install \
  --require-hashes -r "$target_lock"

install -d -o "$READRAFT_APP_USER" -g "$READRAFT_APP_USER" -m 0700 \
  "$READRAFT_BACKUP_DIR"
backup_path="$READRAFT_BACKUP_DIR/readraft-$(
  date -u +%Y%m%dT%H%M%SZ
)-${current_commit:0:12}.zip"

was_active=false
if systemctl is-active --quiet "$READRAFT_SERVICE"; then
  was_active=true
fi
systemctl stop "$READRAFT_SERVICE"

if ! run_as_readraft env PYTHONPATH="$READRAFT_APP_DIR" \
  "$READRAFT_APP_DIR/.venv/bin/python" -m app.backup \
  create "$backup_path"; then
  if [[ "$was_active" == true ]]; then
    systemctl start "$READRAFT_SERVICE" || true
  fi
  fail "backup failed; the Git checkout was not changed"
fi

printf 'Backup created at %s.\n' "$backup_path"
if ! run_as_readraft git -C "$READRAFT_APP_DIR" pull --ff-only; then
  fail "git pull failed; the service remains stopped and backup is $backup_path"
fi

run_as_readraft "$READRAFT_APP_DIR/.venv/bin/python" -m pip install \
  --require-hashes -r "$READRAFT_APP_DIR/requirements.lock"

install -m 0644 "$READRAFT_APP_DIR/deploy/readraft.service" \
  "/etc/systemd/system/$READRAFT_SERVICE.service"
systemctl daemon-reload
systemctl start "$READRAFT_SERVICE"

for _attempt in {1..30}; do
  if curl --fail --silent "$READRAFT_HEALTH_URL" >/dev/null; then
    printf 'Readraft updated to %s and passed its health check.\n' \
      "${target_commit:0:12}"
    exit 0
  fi
  sleep 1
done

printf 'Readraft did not become healthy. Backup: %s\n' "$backup_path" >&2
printf 'Inspect with: journalctl -u %s -n 200 --no-pager\n' \
  "$READRAFT_SERVICE" >&2
exit 1
