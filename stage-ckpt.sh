#!/usr/bin/env bash
# ============================================================================
# Atomically stage the latest verified checkpoint generation between a primary
# training host, a rented GPU host, a NAS burst directory, or two local paths.
#
# Usage:
#   ./stage-ckpt.sh to-nas [--allow-rollback]                   # PRIMARY_HOST -> NAS_HOST
#   ./stage-ckpt.sh from-nas [--allow-rollback]                 # NAS_HOST     -> PRIMARY_HOST
#   ./stage-ckpt.sh push-nas [local_ckpt_dir] [--allow-rollback] # local rental -> NAS_HOST
#   ./stage-ckpt.sh pull-nas [local_ckpt_dir] [--allow-rollback] # NAS_HOST     -> local dir
#   ./stage-ckpt.sh local <src_dir> <dst_dir> [--allow-rollback] # local atomic stage + verify
#
# Required env vars for remote modes:
#   to-nas / from-nas:
#     PRIMARY_HOST       SSH target for primary training host (e.g. user@primary.example, or 'local')
#     PRIMARY_CKPT_DIR   Checkpoint directory on primary host
#     NAS_HOST           SSH target for NAS host (e.g. user@nas.example or host when NAS_USER is set)
#     NAS_CKPT_DIR       Checkpoint directory on NAS host (falls back to NAS_CKPT)
#   push-nas / pull-nas:
#     NAS_HOST           SSH target for NAS host (or hostname + NAS_USER)
#     NAS_CKPT_DIR       Checkpoint directory on NAS host (falls back to NAS_CKPT)
#     CKPT_DIR           Local checkpoint directory (if positional arg omitted)
#
# Optional:
#   EXPECTED_STEP        Fail closed if source LATEST.json completed_step != EXPECTED_STEP
#   EXPECTED_GENERATION  Fail closed if source LATEST.json generation != EXPECTED_GENERATION
#   EXPECTED_SHA256      Fail closed if source LATEST.json sha256 != EXPECTED_SHA256
#   ALLOW_ROLLBACK       Set to 1 (or pass --allow-rollback) to permit rolling LATEST.json backward
#   NAS_USER             Optional username prepended when NAS_HOST omits user@
#   NAS_KEY              Optional SSH private key path (-i NAS_KEY -o IdentitiesOnly=yes)
#   SSH_BIN              Override ssh binary (default: ssh; e.g. $WORK/_nassh on rental hosts)
#   RSYNC_BIN            Override rsync binary (default: rsync)
#   PYTHON_BIN           Override local python3 binary (default: python3)
#   REMOTE_PYTHON_BIN    Override remote python3 binary on destination (default: python3)
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HERE/secrets.env" ] && { set -a; . "$HERE/secrets.env"; set +a; }

SSH_BIN="${SSH_BIN:-ssh}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REMOTE_PYTHON_BIN="${REMOTE_PYTHON_BIN:-python3}"
ALLOW_ROLLBACK="${ALLOW_ROLLBACK:-0}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

resolve_nas_target() {
  local host="${NAS_HOST:?Set NAS_HOST}"
  case "$host" in
    *@*) printf '%s\n' "$host" ;;
    *)
      if [ -n "${NAS_USER:-}" ]; then
        printf '%s@%s\n' "$NAS_USER" "$host"
      else
        printf '%s\n' "$host"
      fi
      ;;
  esac
}

resolve_nas_dir() {
  local dir="${NAS_CKPT_DIR:-${NAS_CKPT:-}}"
  [ -n "$dir" ] || die "Set NAS_CKPT_DIR (or NAS_CKPT)"
  printf '%s\n' "$dir"
}

prepare_ssh_cmd() {
  local meta_dir="$1"
  if [ -n "${NAS_KEY:-}" ]; then
    local wrapper="$meta_dir/_ssh_cmd"
    {
      printf '#!/usr/bin/env bash\n'
      printf 'set -euo pipefail\n'
      printf 'exec %q -i %q -o IdentitiesOnly=yes -o BatchMode=yes "$@"\n' "$SSH_BIN" "$NAS_KEY"
    } > "$wrapper"
    chmod +x "$wrapper"
    printf '%s\n' "$wrapper"
  else
    printf '%s\n' "$SSH_BIN"
  fi
}

check_source_pins() {
  local src_label="$1"
  local gen_name="$2"
  local src_step="$3"
  local src_sha="$4"

  if [ -n "${EXPECTED_STEP:-}" ] && [ "$src_step" != "$EXPECTED_STEP" ]; then
    die "Source step mismatch on ${src_label}: expected ${EXPECTED_STEP}, found ${src_step}"
  fi
  if [ -n "${EXPECTED_GENERATION:-}" ] && [ "$gen_name" != "$EXPECTED_GENERATION" ]; then
    die "Source generation mismatch on ${src_label}: expected ${EXPECTED_GENERATION}, found ${gen_name}"
  fi
  if [ -n "${EXPECTED_SHA256:-}" ] && [ "$src_sha" != "$EXPECTED_SHA256" ]; then
    die "Source sha256 mismatch on ${src_label}: expected ${EXPECTED_SHA256}, found ${src_sha}"
  fi
}

remote_commit_staged() {
  local ssh_cmd="$1"
  local dst_host="$2"
  local dst_dir="$3"
  local stage_dir="$4"
  local tmp_meta="$5"
  local gen_name="$6"
  local src_step="$7"
  local src_sha="$8"

  # Transfer LATEST.json and stdlib-only verifier into the fresh unique destination staging dir.
  if ! "$RSYNC_BIN" -rt --no-o --no-g -e "$ssh_cmd" "$tmp_meta/LATEST.json" "${dst_host}:${stage_dir}/LATEST.json"; then
    "$ssh_cmd" "$dst_host" "rm -rf '$stage_dir'" >/dev/null 2>&1 || true
    die "Failed to transfer LATEST.json to ${dst_host}:${dst_dir}"
  fi

  cp "$HERE/checkpoint_io.py" "$tmp_meta/_checkpoint_io.py"
  if ! "$RSYNC_BIN" -rt --no-o --no-g -e "$ssh_cmd" "$tmp_meta/_checkpoint_io.py" "${dst_host}:${stage_dir}/_checkpoint_io.py"; then
    "$ssh_cmd" "$dst_host" "rm -rf '$stage_dir'" >/dev/null 2>&1 || true
    die "Failed to transfer checkpoint verifier to ${dst_host}:${dst_dir}"
  fi

  local rollback_flag=""
  if [ "$ALLOW_ROLLBACK" = "1" ]; then
    rollback_flag="--allow-rollback"
  fi

  # Run full manifest + SHA-256 + collision + monotonic-pointer verification & atomic commit on destination.
  "$ssh_cmd" "$dst_host" "
    set -euo pipefail
    trap 'rm -rf '\''${stage_dir}'\''' EXIT
    '${REMOTE_PYTHON_BIN}' '${stage_dir}/_checkpoint_io.py' commit-staged \
      --staging-dir '${stage_dir}' \
      --ckpt-dir '${dst_dir}' \
      --expected-step '${src_step}' \
      --expected-generation '${gen_name}' \
      --expected-sha256 '${src_sha}' \
      ${rollback_flag}
  " || die "Destination verification/atomic commit failed on ${dst_host}:${dst_dir}"
}

stage_local_to_remote() {
  local src_dir="$1"
  local dst_host="$2"
  local dst_dir="$3"

  local tmp_meta
  tmp_meta="$(mktemp -d "${TMPDIR:-/tmp}/stage_ckpt_meta_XXXXXX")"
  trap 'rm -rf "$tmp_meta"' RETURN

  local -a verify_args=("$HERE/checkpoint_io.py" verify --ckpt-dir "$src_dir")
  [ -n "${EXPECTED_STEP:-}" ] && verify_args+=(--expected-step "$EXPECTED_STEP")
  [ -n "${EXPECTED_GENERATION:-}" ] && verify_args+=(--expected-generation "$EXPECTED_GENERATION")
  [ -n "${EXPECTED_SHA256:-}" ] && verify_args+=(--expected-sha256 "$EXPECTED_SHA256")
  "$PYTHON_BIN" "${verify_args[@]}" >/dev/null \
    || die "Source checkpoint verification failed in ${src_dir}"

  cp "${src_dir}/LATEST.json" "$tmp_meta/LATEST.json"

  local gen_name src_step src_sha
  gen_name="$("$PYTHON_BIN" "$HERE/checkpoint_io.py" inspect --pointer "$tmp_meta/LATEST.json" --field generation)" \
    || die "Invalid LATEST.json in ${src_dir}"
  src_step="$("$PYTHON_BIN" "$HERE/checkpoint_io.py" inspect --pointer "$tmp_meta/LATEST.json" --field completed_step)"
  src_sha="$("$PYTHON_BIN" "$HERE/checkpoint_io.py" inspect --pointer "$tmp_meta/LATEST.json" --field sha256)"

  check_source_pins "$src_dir" "$gen_name" "$src_step" "$src_sha"

  local ssh_cmd
  ssh_cmd="$(prepare_ssh_cmd "$tmp_meta")"

  echo "staging generation=${gen_name} (completed_step=${src_step}, sha256=${src_sha:0:12}) ${src_dir} -> ${dst_host}:${dst_dir}"

  local stage_dir
  stage_dir="$("$ssh_cmd" "$dst_host" "mkdir -p '$dst_dir' && mktemp -d '$dst_dir/.stage_ckpt_XXXXXX'")" \
    || die "Failed to prepare unique staging directory on ${dst_host}:${dst_dir}"
  case "$stage_dir" in
    "$dst_dir"/.stage_ckpt_*) ;;
    *) die "Unexpected staging directory path returned by ${dst_host}: ${stage_dir}" ;;
  esac

  if ! "$RSYNC_BIN" -rt --no-o --no-g -e "$ssh_cmd" "${src_dir}/${gen_name}/" "${dst_host}:${stage_dir}/${gen_name}/"; then
    "$ssh_cmd" "$dst_host" "rm -rf '$stage_dir'" >/dev/null 2>&1 || true
    die "Failed to transfer generation ${gen_name} from ${src_dir} to ${dst_host}:${dst_dir}"
  fi

  remote_commit_staged "$ssh_cmd" "$dst_host" "$dst_dir" "$stage_dir" "$tmp_meta" "$gen_name" "$src_step" "$src_sha"
  echo "done: ${dst_host}:${dst_dir} now points atomically to ${gen_name} (completed_step=${src_step})"
}

stage_remote_to_local() {
  local src_host="$1"
  local src_dir="$2"
  local dst_dir="$3"

  local tmp_meta
  tmp_meta="$(mktemp -d "${TMPDIR:-/tmp}/stage_ckpt_meta_XXXXXX")"
  trap 'rm -rf "$tmp_meta"' RETURN

  local ssh_cmd
  ssh_cmd="$(prepare_ssh_cmd "$tmp_meta")"

  mkdir -p "$dst_dir"
  local stage_dir
  stage_dir="$(mktemp -d "${dst_dir}/.stage_ckpt_XXXXXX")"

  if ! "$RSYNC_BIN" -rt --no-o --no-g -e "$ssh_cmd" "${src_host}:${src_dir}/LATEST.json" "$stage_dir/LATEST.json"; then
    rm -rf "$stage_dir"
    die "Failed to fetch LATEST.json from ${src_host}:${src_dir}"
  fi

  local gen_name src_step src_sha
  gen_name="$("$PYTHON_BIN" "$HERE/checkpoint_io.py" inspect --pointer "$stage_dir/LATEST.json" --field generation)" \
    || { rm -rf "$stage_dir"; die "Invalid LATEST.json on ${src_host}:${src_dir}"; }
  src_step="$("$PYTHON_BIN" "$HERE/checkpoint_io.py" inspect --pointer "$stage_dir/LATEST.json" --field completed_step)"
  src_sha="$("$PYTHON_BIN" "$HERE/checkpoint_io.py" inspect --pointer "$stage_dir/LATEST.json" --field sha256)"

  if ! check_source_pins "${src_host}:${src_dir}" "$gen_name" "$src_step" "$src_sha"; then
    rm -rf "$stage_dir"
    exit 1
  fi

  echo "staging generation=${gen_name} (completed_step=${src_step}, sha256=${src_sha:0:12}) ${src_host}:${src_dir} -> ${dst_dir}"

  if ! "$RSYNC_BIN" -rt --no-o --no-g -e "$ssh_cmd" "${src_host}:${src_dir}/${gen_name}/" "${stage_dir}/${gen_name}/"; then
    rm -rf "$stage_dir"
    die "Failed to transfer generation ${gen_name} from ${src_host}:${src_dir} to ${dst_dir}"
  fi

  local -a commit_args=(
    "$HERE/checkpoint_io.py" commit-staged
    --staging-dir "$stage_dir"
    --ckpt-dir "$dst_dir"
    --expected-step "$src_step"
    --expected-generation "$gen_name"
    --expected-sha256 "$src_sha"
  )
  [ "$ALLOW_ROLLBACK" = "1" ] && commit_args+=(--allow-rollback)

  if ! "$PYTHON_BIN" "${commit_args[@]}"; then
    rm -rf "$stage_dir"
    die "Destination verification/atomic commit failed in ${dst_dir}"
  fi
  rm -rf "$stage_dir"
  echo "done: ${dst_dir} now points atomically to ${gen_name} (completed_step=${src_step})"
}

stage_remote_pair() {
  local src_host="$1"
  local src_dir="$2"
  local dst_host="$3"
  local dst_dir="$4"

  if [ "$src_host" = "local" ]; then
    stage_local_to_remote "$src_dir" "$dst_host" "$dst_dir"
    return
  fi
  if [ "$dst_host" = "local" ]; then
    stage_remote_to_local "$src_host" "$src_dir" "$dst_dir"
    return
  fi

  local tmp_meta
  tmp_meta="$(mktemp -d "${TMPDIR:-/tmp}/stage_ckpt_meta_XXXXXX")"
  trap 'rm -rf "$tmp_meta"' RETURN

  local ssh_cmd
  ssh_cmd="$(prepare_ssh_cmd "$tmp_meta")"

  "$RSYNC_BIN" -rt --no-o --no-g -e "$ssh_cmd" "${src_host}:${src_dir}/LATEST.json" "$tmp_meta/LATEST.json" \
    || die "Failed to fetch LATEST.json from ${src_host}:${src_dir}"

  local gen_name src_step src_sha
  gen_name="$("$PYTHON_BIN" "$HERE/checkpoint_io.py" inspect --pointer "$tmp_meta/LATEST.json" --field generation)" \
    || die "Invalid LATEST.json on ${src_host}:${src_dir}"
  src_step="$("$PYTHON_BIN" "$HERE/checkpoint_io.py" inspect --pointer "$tmp_meta/LATEST.json" --field completed_step)"
  src_sha="$("$PYTHON_BIN" "$HERE/checkpoint_io.py" inspect --pointer "$tmp_meta/LATEST.json" --field sha256)"

  check_source_pins "${src_host}:${src_dir}" "$gen_name" "$src_step" "$src_sha"

  echo "staging generation=${gen_name} (completed_step=${src_step}, sha256=${src_sha:0:12}) ${src_host}:${src_dir} -> ${dst_host}:${dst_dir}"

  local stage_dir
  stage_dir="$("$ssh_cmd" "$dst_host" "mkdir -p '$dst_dir' && mktemp -d '$dst_dir/.stage_ckpt_XXXXXX'")" \
    || die "Failed to prepare unique staging directory on ${dst_host}:${dst_dir}"
  case "$stage_dir" in
    "$dst_dir"/.stage_ckpt_*) ;;
    *) die "Unexpected staging directory path returned by ${dst_host}: ${stage_dir}" ;;
  esac

  # Copy immutable generation dir into fresh unique hidden staging dir on destination.
  if ! "$ssh_cmd" "$src_host" "rsync -rt --no-o --no-g '${src_dir}/${gen_name}/' '${dst_host}:${stage_dir}/${gen_name}/'"; then
    "$ssh_cmd" "$dst_host" "rm -rf '$stage_dir'" >/dev/null 2>&1 || true
    die "Failed to transfer generation ${gen_name} from ${src_host} to ${dst_host}"
  fi

  remote_commit_staged "$ssh_cmd" "$dst_host" "$dst_dir" "$stage_dir" "$tmp_meta" "$gen_name" "$src_step" "$src_sha"
  echo "done: ${dst_host}:${dst_dir} now points atomically to ${gen_name} (completed_step=${src_step})"
}

POSITIONAL=()
for arg in "$@"; do
  case "$arg" in
    --allow-rollback)
      ALLOW_ROLLBACK=1
      ;;
    *)
      POSITIONAL+=("$arg")
      ;;
  esac
done
if [ "${#POSITIONAL[@]}" -gt 0 ]; then
  set -- "${POSITIONAL[@]}"
else
  set --
fi

case "${1:-}" in
  to-nas)
    : "${PRIMARY_HOST:?Set PRIMARY_HOST}"
    : "${PRIMARY_CKPT_DIR:?Set PRIMARY_CKPT_DIR}"
    NAS_TARGET="$(resolve_nas_target)"
    NAS_DIR="$(resolve_nas_dir)"
    stage_remote_pair "$PRIMARY_HOST" "$PRIMARY_CKPT_DIR" "$NAS_TARGET" "$NAS_DIR"
    ;;
  from-nas)
    : "${PRIMARY_HOST:?Set PRIMARY_HOST}"
    : "${PRIMARY_CKPT_DIR:?Set PRIMARY_CKPT_DIR}"
    NAS_TARGET="$(resolve_nas_target)"
    NAS_DIR="$(resolve_nas_dir)"
    stage_remote_pair "$NAS_TARGET" "$NAS_DIR" "$PRIMARY_HOST" "$PRIMARY_CKPT_DIR"
    ;;
  push-nas)
    SRC_DIR="${2:-${CKPT_DIR:-}}"
    [ -n "$SRC_DIR" ] || die "usage: $0 push-nas [local_ckpt_dir] (or set CKPT_DIR)"
    NAS_TARGET="$(resolve_nas_target)"
    NAS_DIR="$(resolve_nas_dir)"
    stage_local_to_remote "$SRC_DIR" "$NAS_TARGET" "$NAS_DIR"
    ;;
  pull-nas)
    DST_DIR="${2:-${CKPT_DIR:-}}"
    [ -n "$DST_DIR" ] || die "usage: $0 pull-nas [local_ckpt_dir] (or set CKPT_DIR)"
    NAS_TARGET="$(resolve_nas_target)"
    NAS_DIR="$(resolve_nas_dir)"
    stage_remote_to_local "$NAS_TARGET" "$NAS_DIR" "$DST_DIR"
    ;;
  local)
    SRC_DIR="${2:-}"
    DST_DIR="${3:-}"
    [ -n "$SRC_DIR" ] && [ -n "$DST_DIR" ] || die "usage: $0 local <src_ckpt_dir> <dst_ckpt_dir> [--allow-rollback]"
    commit_args=(
      "$HERE/checkpoint_io.py" commit-staged
      --staging-dir "$SRC_DIR"
      --ckpt-dir "$DST_DIR"
    )
    [ -n "${EXPECTED_STEP:-}" ] && commit_args+=(--expected-step "$EXPECTED_STEP")
    [ -n "${EXPECTED_GENERATION:-}" ] && commit_args+=(--expected-generation "$EXPECTED_GENERATION")
    [ -n "${EXPECTED_SHA256:-}" ] && commit_args+=(--expected-sha256 "$EXPECTED_SHA256")
    [ "$ALLOW_ROLLBACK" = "1" ] && commit_args+=(--allow-rollback)
    "$PYTHON_BIN" "${commit_args[@]}"
    ;;
  *)
    echo "usage: $0 {to-nas|from-nas|push-nas [local_dir]|pull-nas [local_dir]|local <src_dir> <dst_dir>} [--allow-rollback]" >&2
    exit 1
    ;;
esac
