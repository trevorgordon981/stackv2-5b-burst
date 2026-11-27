#!/usr/bin/env bash
# ============================================================================
# Atomically stage the latest verified checkpoint generation between a primary
# training host and a NAS burst directory, or between two local directories.
#
# Usage:
#   ./stage-ckpt.sh to-nas                 # PRIMARY_HOST -> NAS_HOST
#   ./stage-ckpt.sh from-nas               # NAS_HOST     -> PRIMARY_HOST
#   ./stage-ckpt.sh local <src_dir> <dst>  # local atomic stage + verify
#
# Required env vars for remote modes:
#   PRIMARY_HOST       SSH target for primary training host (e.g. user@primary.example)
#   PRIMARY_CKPT_DIR   Checkpoint directory on primary host
#   NAS_HOST           SSH target for NAS host (e.g. user@nas.example)
#   NAS_CKPT_DIR       Checkpoint directory on NAS host
#
# Optional:
#   EXPECTED_STEP      Fail closed if source LATEST.json completed_step != EXPECTED_STEP
#   SSH_BIN            Override ssh binary (default: ssh)
#   RSYNC_BIN          Override rsync binary (default: rsync)
#   PYTHON_BIN         Override python3 binary (default: python3)
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HERE/secrets.env" ] && { set -a; . "$HERE/secrets.env"; set +a; }

SSH_BIN="${SSH_BIN:-ssh}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

stage_remote_pair() {
  local src_host="$1"
  local src_dir="$2"
  local dst_host="$3"
  local dst_dir="$4"

  local tmp_meta
  tmp_meta="$(mktemp -d "${TMPDIR:-/tmp}/stage_ckpt_meta_XXXXXX")"
  trap 'rm -rf "$tmp_meta"' RETURN

  "$RSYNC_BIN" -rt --no-o --no-g -e "$SSH_BIN" "${src_host}:${src_dir}/LATEST.json" "$tmp_meta/LATEST.json" \
    || die "Failed to fetch LATEST.json from ${src_host}:${src_dir}"

  local gen_name src_step src_sha
  gen_name="$("$PYTHON_BIN" "$HERE/checkpoint_io.py" inspect --pointer "$tmp_meta/LATEST.json" --field generation)" \
    || die "Invalid LATEST.json on ${src_host}:${src_dir}"
  src_step="$("$PYTHON_BIN" "$HERE/checkpoint_io.py" inspect --pointer "$tmp_meta/LATEST.json" --field completed_step)"
  src_sha="$("$PYTHON_BIN" "$HERE/checkpoint_io.py" inspect --pointer "$tmp_meta/LATEST.json" --field sha256)"

  if [ -n "${EXPECTED_STEP:-}" ] && [ "$src_step" != "$EXPECTED_STEP" ]; then
    die "Source step mismatch on ${src_host}:${src_dir}: expected ${EXPECTED_STEP}, found ${src_step}"
  fi

  echo "staging generation=${gen_name} (completed_step=${src_step}, sha256=${src_sha:0:12}) ${src_host}:${src_dir} -> ${dst_host}:${dst_dir}"

  local stage_token="stage_${gen_name}_$$"
  "$SSH_BIN" "$dst_host" "mkdir -p '$dst_dir' && rm -rf '$dst_dir/.${stage_token}' && mkdir -p '$dst_dir/.${stage_token}'" \
    || die "Failed to prepare staging directory on ${dst_host}:${dst_dir}"

  # Copy immutable generation dir into hidden staging dir on destination.
  "$SSH_BIN" "$src_host" "rsync -rt --no-o --no-g '${src_dir}/${gen_name}/' '${dst_host}:${dst_dir}/.${stage_token}/${gen_name}/'" \
    || die "Failed to transfer generation ${gen_name} from ${src_host} to ${dst_host}"

  # Copy pointer into hidden staging dir on destination.
  "$RSYNC_BIN" -rt --no-o --no-g -e "$SSH_BIN" "$tmp_meta/LATEST.json" "${dst_host}:${dst_dir}/.${stage_token}/LATEST.json" \
    || die "Failed to transfer LATEST.json to ${dst_host}:${dst_dir}"

  # Verify SHA-256 on destination BEFORE promoting immutable dir & atomically swapping LATEST.json.
  "$SSH_BIN" "$dst_host" "
    set -euo pipefail
    cd '$dst_dir'
    actual=\$(sha256sum '.${stage_token}/${gen_name}/checkpoint.pt' | awk '{print \$1}')
    if [ \"\$actual\" != '${src_sha}' ]; then
      rm -rf '.${stage_token}'
      echo 'ERROR: destination SHA-256 mismatch' >&2
      exit 1
    fi
    if [ ! -d '${gen_name}' ]; then
      mv '.${stage_token}/${gen_name}' '${gen_name}'
    fi
    cp '.${stage_token}/LATEST.json' '.LATEST.json.${stage_token}.tmp'
    mv -f '.LATEST.json.${stage_token}.tmp' 'LATEST.json'
    rm -rf '.${stage_token}'
  " || die "Destination verification/atomic commit failed on ${dst_host}:${dst_dir}"

  echo "done: ${dst_host}:${dst_dir} now points atomically to ${gen_name} (completed_step=${src_step})"
}

case "${1:-}" in
  to-nas)
    : "${PRIMARY_HOST:?Set PRIMARY_HOST}"
    : "${PRIMARY_CKPT_DIR:?Set PRIMARY_CKPT_DIR}"
    : "${NAS_HOST:?Set NAS_HOST}"
    : "${NAS_CKPT_DIR:?Set NAS_CKPT_DIR}"
    stage_remote_pair "$PRIMARY_HOST" "$PRIMARY_CKPT_DIR" "$NAS_HOST" "$NAS_CKPT_DIR"
    ;;
  from-nas)
    : "${PRIMARY_HOST:?Set PRIMARY_HOST}"
    : "${PRIMARY_CKPT_DIR:?Set PRIMARY_CKPT_DIR}"
    : "${NAS_HOST:?Set NAS_HOST}"
    : "${NAS_CKPT_DIR:?Set NAS_CKPT_DIR}"
    stage_remote_pair "$NAS_HOST" "$NAS_CKPT_DIR" "$PRIMARY_HOST" "$PRIMARY_CKPT_DIR"
    ;;
  local)
    SRC_DIR="${2:-}"
    DST_DIR="${3:-}"
    [ -n "$SRC_DIR" ] && [ -n "$DST_DIR" ] || die "usage: $0 local <src_ckpt_dir> <dst_ckpt_dir>"
    commit_args=(
      "$HERE/checkpoint_io.py" commit-staged
      --staging-dir "$SRC_DIR"
      --ckpt-dir "$DST_DIR"
    )
    [ -n "${EXPECTED_STEP:-}" ] && commit_args+=(--expected-step "$EXPECTED_STEP")
    [ -n "${EXPECTED_GENERATION:-}" ] && commit_args+=(--expected-generation "$EXPECTED_GENERATION")
    [ -n "${EXPECTED_SHA256:-}" ] && commit_args+=(--expected-sha256 "$EXPECTED_SHA256")
    "$PYTHON_BIN" "${commit_args[@]}"
    ;;
  *)
    echo "usage: $0 {to-nas|from-nas|local <src_dir> <dst_dir>}" >&2
    exit 1
    ;;
esac
