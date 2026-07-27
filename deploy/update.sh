#!/usr/bin/env bash
set -Eeuo pipefail

readonly READRAFT_APP_DIR="${READRAFT_APP_DIR:-/opt/readraft}"
readonly READRAFT_APP_USER="${READRAFT_APP_USER:-readraft}"
readonly READRAFT_SERVICE="${READRAFT_SERVICE:-readraft}"
readonly READRAFT_UPDATE_SERVICE="${READRAFT_UPDATE_SERVICE:-${READRAFT_SERVICE}-update}"
readonly READRAFT_UPDATE_TIMER="${READRAFT_UPDATE_TIMER:-${READRAFT_SERVICE}-update}"
readonly READRAFT_BACKUP_DIR="${READRAFT_BACKUP_DIR:-/var/backups/readraft}"
readonly READRAFT_HEALTH_URL="${READRAFT_HEALTH_URL:-http://127.0.0.1:8010/healthz}"
readonly READRAFT_UPDATE_REMOTE="${READRAFT_UPDATE_REMOTE:-origin}"
readonly READRAFT_UPDATE_CHANNEL="${READRAFT_UPDATE_CHANNEL:-release}"
readonly READRAFT_UPDATE_BRANCH="${READRAFT_UPDATE_BRANCH:-main}"
readonly READRAFT_UPDATE_LOCK="${READRAFT_UPDATE_LOCK:-/run/lock/readraft-update.lock}"
readonly READRAFT_BACKUP_RETENTION_DAYS="${READRAFT_BACKUP_RETENTION_DAYS:-30}"
readonly READRAFT_VENV_RETENTION="${READRAFT_VENV_RETENTION:-3}"
readonly READRAFT_HEALTH_ATTEMPTS="${READRAFT_HEALTH_ATTEMPTS:-60}"
readonly READRAFT_VENV_DIR="$READRAFT_APP_DIR/.venvs"

mode="apply"
explicit_target=""
target_lock=""
unit_snapshot=""
build_env=""
backup_path=""
current_commit=""
target_commit=""
target_label=""
previous_runtime_dir=""
service_stopped=false
rollback_needed=false

usage() {
  cat <<'EOF'
Usage: update.sh [--check] [--target REF]

Without arguments, update to the newest stable annotated vMAJOR.MINOR.PATCH tag.
Set READRAFT_UPDATE_CHANNEL=main to follow the configured development branch.

  --check       Fetch and report whether an update is available.
  --target REF  Update to one explicit descendant commit or tag.
  -h, --help    Show this help.
EOF
}

fail() {
  printf 'Readraft update failed: %s\n' "$*" >&2
  exit 1
}

run_as_readraft() {
  runuser -u "$READRAFT_APP_USER" -- "$@"
}

is_integer_between() {
  local value="$1"
  local minimum="$2"
  local maximum="$3"
  [[ "$value" =~ ^[0-9]+$ ]] \
    && (( value >= minimum && value <= maximum ))
}

wait_for_health() {
  local attempt
  for ((attempt = 1; attempt <= READRAFT_HEALTH_ATTEMPTS; attempt += 1)); do
    if curl --fail --silent "$READRAFT_HEALTH_URL" >/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

replace_runtime_link() {
  local runtime_dir="$1"
  local temporary_link="$READRAFT_APP_DIR/.venv.next.$$"
  rm -f -- "$temporary_link"
  ln -s "$runtime_dir" "$temporary_link"
  mv -Tf -- "$temporary_link" "$READRAFT_APP_DIR/.venv"
  chown -h "$READRAFT_APP_USER:$READRAFT_APP_USER" "$READRAFT_APP_DIR/.venv"
}

restore_previous_release() {
  local rollback_failed=false
  local rollback_service_stopped=true

  printf 'Update failed after the release switch; restoring %s.\n' \
    "${current_commit:0:12}" >&2
  if ! systemctl stop "$READRAFT_SERVICE"; then
    rollback_failed=true
    rollback_service_stopped=false
  fi

  if [[ "$rollback_service_stopped" == true ]]; then
    if ! run_as_readraft git -C "$READRAFT_APP_DIR" \
      checkout --detach "$current_commit"; then
      rollback_failed=true
    fi
    if [[ -n "$previous_runtime_dir" && -d "$previous_runtime_dir" ]]; then
      replace_runtime_link "$previous_runtime_dir" || rollback_failed=true
    else
      printf 'Previous Python environment is unavailable: %s\n' \
        "$previous_runtime_dir" >&2
      rollback_failed=true
    fi
    if [[ -f "$unit_snapshot" ]]; then
      install -m 0644 "$unit_snapshot" \
        "/etc/systemd/system/$READRAFT_SERVICE.service" \
        || rollback_failed=true
    else
      rollback_failed=true
    fi
    systemctl daemon-reload || rollback_failed=true
  fi

  if [[ "$rollback_service_stopped" == true \
    && -x "$previous_runtime_dir/bin/python" \
    && -f "$backup_path" ]]; then
    if ! run_as_readraft env PYTHONPATH="$READRAFT_APP_DIR" \
      "$previous_runtime_dir/bin/python" -m app.backup \
      restore "$backup_path" --replace; then
      rollback_failed=true
    fi
  else
    rollback_failed=true
  fi

  if [[ "$rollback_service_stopped" == true ]]; then
    systemctl start "$READRAFT_SERVICE" || rollback_failed=true
    if ! wait_for_health; then
      rollback_failed=true
    fi
  fi

  if [[ "$rollback_failed" == true ]]; then
    printf 'Automatic rollback was incomplete. Keep backup %s and inspect: %s\n' \
      "$backup_path" \
      "journalctl -u $READRAFT_SERVICE -n 200 --no-pager" >&2
    return 1
  fi
  printf 'Previous release restored and healthy. Failed backup retained at %s.\n' \
    "$backup_path" >&2
  return 0
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM

  if (( status != 0 )); then
    if [[ "$rollback_needed" == true ]]; then
      restore_previous_release || true
    elif [[ "$service_stopped" == true ]]; then
      systemctl start "$READRAFT_SERVICE" || true
    fi
  fi

  rm -f -- "$target_lock" "$unit_snapshot"
  if [[ -n "$build_env" && "$build_env" == "$READRAFT_VENV_DIR"/.build-* ]]; then
    rm -rf -- "$build_env"
  fi
  exit "$status"
}

prune_successful_backups() {
  find "$READRAFT_BACKUP_DIR" -maxdepth 1 -type f \
    -name 'readraft-update-*.zip' \
    -mtime "+$READRAFT_BACKUP_RETENTION_DAYS" -print -delete
}

prune_old_environments() {
  local kept=0
  local entry
  local directory
  local basename

  while IFS= read -r entry; do
    directory="${entry#* }"
    basename="${directory##*/}"
    [[ "$basename" =~ ^[0-9a-f]{40}$ ]] || continue
    if [[ "$directory" == "$READRAFT_VENV_DIR/$target_commit" \
      || "$directory" == "$previous_runtime_dir" ]]; then
      kept=$((kept + 1))
      continue
    fi
    kept=$((kept + 1))
    if (( kept > READRAFT_VENV_RETENTION )); then
      rm -rf -- "$directory"
    fi
  done < <(
    find "$READRAFT_VENV_DIR" -mindepth 1 -maxdepth 1 -type d \
      -name '[0-9a-f]*' -printf '%T@ %p\n' | sort -nr
  )
}

while (( $# > 0 )); do
  case "$1" in
    --check)
      mode="check"
      shift
      ;;
    --target)
      (( $# >= 2 )) || fail "--target requires a ref"
      explicit_target="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
done

if (( EUID != 0 )); then
  fail "run this script with sudo"
fi

for command_name in curl find flock git install runuser systemctl; do
  command -v "$command_name" >/dev/null \
    || fail "$command_name is not installed"
done

is_integer_between "$READRAFT_BACKUP_RETENTION_DAYS" 1 3650 \
  || fail "READRAFT_BACKUP_RETENTION_DAYS must be between 1 and 3650"
is_integer_between "$READRAFT_VENV_RETENTION" 2 20 \
  || fail "READRAFT_VENV_RETENTION must be between 2 and 20"
is_integer_between "$READRAFT_HEALTH_ATTEMPTS" 5 600 \
  || fail "READRAFT_HEALTH_ATTEMPTS must be between 5 and 600"
[[ "$READRAFT_UPDATE_CHANNEL" == "release" \
  || "$READRAFT_UPDATE_CHANNEL" == "main" ]] \
  || fail "READRAFT_UPDATE_CHANNEL must be release or main"

install -d -m 0755 "${READRAFT_UPDATE_LOCK%/*}"
exec 9>"$READRAFT_UPDATE_LOCK"
flock -n 9 || fail "another Readraft update is already running"
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ -d "$READRAFT_APP_DIR/.git" ]] \
  || fail "$READRAFT_APP_DIR is not a Git checkout"
[[ -f "$READRAFT_APP_DIR/.env" ]] \
  || fail "$READRAFT_APP_DIR/.env is missing"
[[ -x "$READRAFT_APP_DIR/.venv/bin/python" ]] \
  || fail "$READRAFT_APP_DIR/.venv is missing"
id "$READRAFT_APP_USER" >/dev/null 2>&1 \
  || fail "system user $READRAFT_APP_USER does not exist"

if [[ -n "$(run_as_readraft git -C "$READRAFT_APP_DIR" status --porcelain)" ]]; then
  fail "the deployment checkout has uncommitted changes"
fi
run_as_readraft git -C "$READRAFT_APP_DIR" \
  remote get-url "$READRAFT_UPDATE_REMOTE" >/dev/null \
  || fail "Git remote $READRAFT_UPDATE_REMOTE is unavailable"

run_as_readraft git -C "$READRAFT_APP_DIR" \
  fetch --prune --tags "$READRAFT_UPDATE_REMOTE"
current_commit="$(
  run_as_readraft git -C "$READRAFT_APP_DIR" rev-parse HEAD
)"

if [[ -n "$explicit_target" ]]; then
  [[ "$explicit_target" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$ ]] \
    || fail "invalid target ref"
  target_label="$explicit_target"
elif [[ "$READRAFT_UPDATE_CHANNEL" == "main" ]]; then
  target_label="$READRAFT_UPDATE_REMOTE/$READRAFT_UPDATE_BRANCH"
else
  target_label="$(
    run_as_readraft git -C "$READRAFT_APP_DIR" \
      ls-remote --tags --refs "$READRAFT_UPDATE_REMOTE" 'v*' \
      | awk '{sub("refs/tags/", "", $2); print $2}' \
      | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' \
      | sort -V \
      | tail -n 1
  )"
  [[ -n "$target_label" ]] || fail "no stable release tag is available"
  run_as_readraft git -C "$READRAFT_APP_DIR" fetch \
    "$READRAFT_UPDATE_REMOTE" \
    "refs/tags/$target_label:refs/tags/$target_label"
  [[ "$(
    run_as_readraft git -C "$READRAFT_APP_DIR" \
      cat-file -t "refs/tags/$target_label"
  )" == "tag" ]] || fail "release tag $target_label is not annotated"
fi

target_commit="$(
  run_as_readraft git -C "$READRAFT_APP_DIR" \
    rev-parse --verify "$target_label^{commit}"
)" || fail "cannot resolve update target $target_label"

if [[ "$current_commit" == "$target_commit" ]]; then
  printf 'Readraft is already current at %s (%s).\n' \
    "${current_commit:0:12}" "$target_label"
  exit 0
fi

run_as_readraft git -C "$READRAFT_APP_DIR" \
  merge-base --is-ancestor "$current_commit" "$target_commit" \
  || fail "target $target_label is not a fast-forward from the deployed commit"

if [[ "$mode" == "check" ]]; then
  printf 'Readraft update available: %s -> %s (%s).\n' \
    "${current_commit:0:12}" "${target_commit:0:12}" "$target_label"
  exit 0
fi

systemctl is-active --quiet "$READRAFT_SERVICE" \
  || fail "$READRAFT_SERVICE is not active"
curl --fail --silent "$READRAFT_HEALTH_URL" >/dev/null \
  || fail "the current Readraft service is not healthy"

target_lock="$(mktemp -t readraft-requirements.XXXXXX)"
run_as_readraft git -C "$READRAFT_APP_DIR" \
  show "$target_commit:requirements.lock" >"$target_lock"
chmod 0644 "$target_lock"

python_base="$(
  run_as_readraft "$READRAFT_APP_DIR/.venv/bin/python" \
    -c 'import os, sys; print(os.path.realpath(sys._base_executable))'
)"
[[ -x "$python_base" ]] || fail "base Python is unavailable: $python_base"

install -d -o "$READRAFT_APP_USER" -g "$READRAFT_APP_USER" -m 0755 \
  "$READRAFT_VENV_DIR"
target_env="$READRAFT_VENV_DIR/$target_commit"
if [[ -f "$target_env/.readraft-ready" \
  && "$(tr -d '\n' <"$target_env/.readraft-ready")" == "$target_commit" \
  && -x "$target_env/bin/python" ]]; then
  printf 'Reusing prepared Python environment for %s.\n' \
    "${target_commit:0:12}"
elif [[ -e "$target_env" ]]; then
  fail "incomplete target environment exists: $target_env"
else
  build_env="$(
    run_as_readraft mktemp -d \
      "$READRAFT_VENV_DIR/.build-${target_commit:0:12}.XXXXXX"
  )"
  printf 'Preparing isolated dependencies for %s before downtime.\n' \
    "${target_commit:0:12}"
  run_as_readraft "$python_base" -m venv "$build_env"
  run_as_readraft "$build_env/bin/python" -m pip install \
    --require-hashes -r "$target_lock"
  printf '%s\n' "$target_commit" >"$build_env/.readraft-ready"
  chown "$READRAFT_APP_USER:$READRAFT_APP_USER" \
    "$build_env/.readraft-ready"
  run_as_readraft mv "$build_env" "$target_env"
  build_env=""
fi

install -d -o "$READRAFT_APP_USER" -g "$READRAFT_APP_USER" -m 0700 \
  "$READRAFT_BACKUP_DIR"
backup_path="$READRAFT_BACKUP_DIR/readraft-update-$(
  date -u +%Y%m%dT%H%M%SZ
)-${current_commit:0:12}.zip"
unit_snapshot="$(mktemp -t readraft-service.XXXXXX)"
cp "/etc/systemd/system/$READRAFT_SERVICE.service" "$unit_snapshot"

systemctl stop "$READRAFT_SERVICE"
service_stopped=true

run_as_readraft env PYTHONPATH="$READRAFT_APP_DIR" \
  "$READRAFT_APP_DIR/.venv/bin/python" -m app.backup \
  create "$backup_path"
run_as_readraft env PYTHONPATH="$READRAFT_APP_DIR" \
  "$READRAFT_APP_DIR/.venv/bin/python" -m app.backup \
  verify "$backup_path"
printf 'Verified backup created at %s.\n' "$backup_path"

if [[ -L "$READRAFT_APP_DIR/.venv" ]]; then
  previous_runtime_dir="$(readlink -f "$READRAFT_APP_DIR/.venv")"
  rollback_needed=true
else
  previous_runtime_dir="$READRAFT_VENV_DIR/$current_commit"
  [[ ! -e "$previous_runtime_dir" ]] \
    || fail "cannot preserve current environment: $previous_runtime_dir exists"
  mv "$READRAFT_APP_DIR/.venv" "$previous_runtime_dir"
  rollback_needed=true
  printf '%s\n' "$current_commit" >"$previous_runtime_dir/.readraft-ready"
  chown "$READRAFT_APP_USER:$READRAFT_APP_USER" \
    "$previous_runtime_dir/.readraft-ready"
fi
[[ -x "$previous_runtime_dir/bin/python" ]] \
  || fail "previous Python environment is invalid"

run_as_readraft git -C "$READRAFT_APP_DIR" \
  checkout --detach "$target_commit"
replace_runtime_link "$target_env"

install -m 0644 "$READRAFT_APP_DIR/deploy/readraft.service" \
  "/etc/systemd/system/$READRAFT_SERVICE.service"
systemctl daemon-reload
systemctl start "$READRAFT_SERVICE"

if ! wait_for_health; then
  fail "release $target_label did not pass its health check"
fi

install -m 0644 "$READRAFT_APP_DIR/deploy/readraft-update.service" \
  "/etc/systemd/system/$READRAFT_UPDATE_SERVICE.service"
install -m 0644 "$READRAFT_APP_DIR/deploy/readraft-update.timer" \
  "/etc/systemd/system/$READRAFT_UPDATE_TIMER.timer"
systemctl daemon-reload

rollback_needed=false
service_stopped=false
prune_successful_backups
prune_old_environments
printf 'Readraft updated to %s (%s) and passed its health check.\n' \
  "${target_commit:0:12}" "$target_label"
