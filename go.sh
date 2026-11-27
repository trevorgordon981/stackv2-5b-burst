#!/usr/bin/env bash
# ============================================================================
# Single supported burst launcher for stackv2-5b-burst.
#
# Modes:
#   BURST_MODE=nas   (default) Join Tailscale, pull data + latest checkpoint
#                    generation from NAS over SSH+rsync, verify integrity and
#                    expected handoff, and launch train.py.
#   BURST_MODE=local Offline/local staged mode (uses local DATA_ROOT + CKPT_DIR,
#                    optional LOCAL_CKPT_SRC atomic commit, no Tailscale/SSH).
#
# Safety contract (fail-closed):
#   - Failed data or checkpoint transfer ALWAYS aborts non-zero; never silently
#     restarts from step 0 and never launches on a stale preexisting checkpoint.
#   - Remote checkpoints are pulled into a staging directory, verified by
#     checkpoint_io.py (LATEST.json + manifest.json + SHA-256 + expected pins),
#     and atomically committed to CKPT_DIR before training starts.
#   - Monotonic LATEST.json protection refuses rolling CKPT_DIR backward to an
#     older completed_step unless ALLOW_ROLLBACK=1 is explicitly set.
#   - Cold start from step 0 requires explicit ALLOW_COLD_START=1 (and never
#     triggers after a failed pull).
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HERE/secrets.env" ] && { set -a; . "$HERE/secrets.env"; set +a; }

BURST_MODE="${BURST_MODE:-nas}"
WORK="${WORK:-$HOME/stackv2-burst-run}"
export DATA_ROOT="${DATA_ROOT:-$WORK/data}"
export CKPT_DIR="${CKPT_DIR:-$WORK/ckpt}"
export TOKENIZER="${TOKENIZER:-$HERE/tokenizer}"
export ALLOW_COLD_START="${ALLOW_COLD_START:-0}"

# 96GB-card defaults; BS*ACC=48 keeps global batch aligned across hardware.
export SEQ_LEN="${SEQ_LEN:-8192}"
export ROPE_THETA="${ROPE_THETA:-500000}"
export BS="${BS:-16}"
export ACC="${ACC:-3}"
export GRAD_CKPT="${GRAD_CKPT:-1}"
export SAVE="${SAVE:-200}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

TORCH_SPEC="${TORCH_SPEC:-torch==2.11.*}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"
SSH_BIN="${SSH_BIN:-ssh}"
TAILSCALE_BIN="${TAILSCALE_BIN:-tailscale}"
SKIP_DEPS="${SKIP_DEPS:-0}"
SKIP_TAILSCALE="${SKIP_TAILSCALE:-0}"
SKIP_CKPT_PULL="${SKIP_CKPT_PULL:-0}"

mkdir -p "$WORK" "$DATA_ROOT" "$CKPT_DIR"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

if [ "$BURST_MODE" != "nas" ] && [ "$BURST_MODE" != "local" ]; then
  die "Unsupported BURST_MODE='$BURST_MODE' (expected 'nas' or 'local')"
fi

if [ "$SKIP_DEPS" != "1" ]; then
  echo "== [1/4] deps ($TORCH_SPEC from $TORCH_INDEX_URL) =="
  "$PYTHON_BIN" -m pip install -q --upgrade pip
  "$PYTHON_BIN" -c "import torch,sys; v=tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); sys.exit(0 if (torch.cuda.is_available() and v>=(2,7)) else 1)" 2>/dev/null \
    || "$PYTHON_BIN" -m pip install -q "$TORCH_SPEC" --index-url "$TORCH_INDEX_URL"
  "$PYTHON_BIN" -m pip install -q -r "$HERE/requirements.txt"
  "$PYTHON_BIN" -c "import torch; print('  torch', torch.__version__, '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
else
  echo "== [1/4] deps (skipped: SKIP_DEPS=1) =="
fi

stage_and_commit_checkpoint() {
  local staging_dir="$1"
  local -a commit_args=(
    "$HERE/checkpoint_io.py" commit-staged
    --staging-dir "$staging_dir"
    --ckpt-dir "$CKPT_DIR"
  )
  [ -n "${EXPECTED_STEP:-}" ] && commit_args+=(--expected-step "$EXPECTED_STEP")
  [ -n "${EXPECTED_GENERATION:-}" ] && commit_args+=(--expected-generation "$EXPECTED_GENERATION")
  [ -n "${EXPECTED_SHA256:-}" ] && commit_args+=(--expected-sha256 "$EXPECTED_SHA256")
  [ "${ALLOW_ROLLBACK:-0}" = "1" ] && commit_args+=(--allow-rollback)

  local committed_json
  if ! committed_json="$("$PYTHON_BIN" "${commit_args[@]}")"; then
    die "Staged checkpoint verification/commit failed; refusing to launch training"
  fi

  local committed_gen
  committed_gen="$("$PYTHON_BIN" -c 'import json,sys; print(json.loads(sys.argv[1])["generation"])' "$committed_json")"
  export EXPECTED_GENERATION="$committed_gen"
  echo "  committed verified generation: $committed_gen"
}

if [ "$BURST_MODE" = "nas" ]; then
  : "${NAS_HOST:?Set NAS_HOST (e.g. nas.example.ts.net) via env or local secrets.env}"
  : "${NAS_USER:?Set NAS_USER via env or local secrets.env}"
  : "${NAS_DATA:?Set NAS_DATA (remote dataset directory) via env or local secrets.env}"
  : "${NAS_CKPT:?Set NAS_CKPT (remote burst checkpoint directory) via env or local secrets.env}"

  if [ "$SKIP_TAILSCALE" != "1" ]; then
    : "${TS_AUTHKEY:?Set TS_AUTHKEY env var (or use BURST_MODE=local / SKIP_TAILSCALE=1)}"
    echo "== [2/4] join tailnet =="
    command -v "$TAILSCALE_BIN" >/dev/null || curl -fsSL https://tailscale.com/install.sh | sh
    if ! sudo "$TAILSCALE_BIN" status >/dev/null 2>&1; then
      if [ -e /dev/net/tun ]; then
        echo "  starting tailscaled (TUN)"
        sudo tailscaled --state=/var/lib/tailscale/tailscaled.state >/tmp/tailscaled.log 2>&1 &
      else
        echo "  no /dev/net/tun -> userspace networking"
        sudo tailscaled --tun=userspace-networking --socks5-server=localhost:1055 --state=/var/lib/tailscale/tailscaled.state >/tmp/tailscaled.log 2>&1 &
      fi
      sleep 5
    fi
    sudo "$TAILSCALE_BIN" up --authkey "$TS_AUTHKEY" --hostname "${TS_HOSTNAME:-gpu-burst}"
    sudo "$TAILSCALE_BIN" status 2>/dev/null | head -5 || true
    NAS_IP="$(sudo "$TAILSCALE_BIN" status 2>/dev/null | awk -v n="$NAS_HOST" '$2==n{print $1; exit}')"
    NAS_IP="${NAS_IP:-$NAS_HOST}"
  else
    echo "== [2/4] tailnet (skipped: SKIP_TAILSCALE=1) =="
    NAS_IP="$NAS_HOST"
  fi

  echo "== [3/4] pull data + verified checkpoint generation from NAS =="
  NASU="${NAS_USER}@${NAS_IP}"

  SSHC="$WORK/_nassh"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'args=(-o StrictHostKeyChecking=accept-new -o BatchMode=yes)\n'
    if [ "$SKIP_TAILSCALE" != "1" ]; then
      printf 'args+=(-o "ProxyCommand=%s nc %%h %%p")\n' "$TAILSCALE_BIN"
    fi
    if [ -n "${NAS_KEY:-}" ]; then
      printf 'args+=(-i %q -o IdentitiesOnly=yes)\n' "$NAS_KEY"
    fi
    printf 'exec %q "${args[@]}" "$@"\n' "$SSH_BIN"
  } > "$SSHC"
  chmod +x "$SSHC"

  echo "  NAS=$NASU key=${NAS_KEY:-<tailscale-ssh/agent>}"

  LANGS="${LANGS:-Python Java Go C-Sharp PHP}"
  SHARDS_PER_LANG="${SHARDS_PER_LANG:-1}"
  if [ -z "$(find "$DATA_ROOT" -type f -name '*.zst' 2>/dev/null | head -n 1)" ]; then
    for L in $LANGS; do
      mkdir -p "$DATA_ROOT/$L"
      shard_list="$("$SSHC" "$NASU" "ls -1 '$NAS_DATA/$L'/*.zst 2>/dev/null | head -n '$SHARDS_PER_LANG'")" \
        || die "Failed to list data shards on $NASU:$NAS_DATA/$L"
      [ -n "$shard_list" ] || die "No .zst shards found on $NASU:$NAS_DATA/$L"
      while IFS= read -r remote_shard; do
        [ -n "$remote_shard" ] || continue
        echo "  data <- $L/$(basename "$remote_shard")"
        "$RSYNC_BIN" -rt --no-o --no-g -e "$SSHC" "${NASU}:${remote_shard}" "$DATA_ROOT/$L/" \
          || die "Failed to pull data shard $remote_shard from $NASU"
      done <<< "$shard_list"
    done
  fi

  if [ "$SKIP_CKPT_PULL" = "1" ]; then
    if [ "$ALLOW_COLD_START" != "1" ]; then
      die "SKIP_CKPT_PULL=1 requires ALLOW_COLD_START=1 (or an already-verified local checkpoint in BURST_MODE=local)"
    fi
    echo "  skipping remote checkpoint pull (SKIP_CKPT_PULL=1, ALLOW_COLD_START=1)"
  else
    PULL_STAGE="$(mktemp -d "$WORK/.ckpt_pull_staging_XXXXXX")"
    trap 'rm -rf "$PULL_STAGE"' EXIT

    if ! "$RSYNC_BIN" -rt --no-o --no-g -e "$SSHC" "${NASU}:${NAS_CKPT}/LATEST.json" "$PULL_STAGE/LATEST.json"; then
      die "Failed to pull ${NAS_CKPT}/LATEST.json from $NASU — refusing cold start or stale resume"
    fi

    REMOTE_GEN="$("$PYTHON_BIN" "$HERE/checkpoint_io.py" inspect --pointer "$PULL_STAGE/LATEST.json" --field generation)" \
      || die "Invalid LATEST.json pulled from $NASU — refusing launch"

    if ! "$RSYNC_BIN" -rt --no-o --no-g -e "$SSHC" "${NASU}:${NAS_CKPT}/${REMOTE_GEN}/" "$PULL_STAGE/${REMOTE_GEN}/"; then
      die "Failed to pull generation ${REMOTE_GEN} from $NASU — refusing cold start or stale resume"
    fi

    stage_and_commit_checkpoint "$PULL_STAGE"
    rm -rf "$PULL_STAGE"
    trap - EXIT
  fi
else
  echo "== [2/4] tailnet (skipped: BURST_MODE=local) =="
  echo "== [3/4] local staging & preflight =="
  if [ -n "${LOCAL_CKPT_SRC:-}" ]; then
    stage_and_commit_checkpoint "$LOCAL_CKPT_SRC"
  fi
fi

SHARD_EXT="${SHARD_EXT:-.zst}"
SHARD_COUNT="$(find "$DATA_ROOT" -type f -name "*${SHARD_EXT}" 2>/dev/null | wc -l | tr -d ' ')"
[ "$SHARD_COUNT" -gt 0 ] || die "No data shards (*${SHARD_EXT}) found in DATA_ROOT=$DATA_ROOT"

verify_args=("$HERE/checkpoint_io.py" verify --ckpt-dir "$CKPT_DIR")
if [ "$ALLOW_COLD_START" = "1" ] && { [ "$BURST_MODE" = "local" ] || [ "$SKIP_CKPT_PULL" = "1" ]; }; then
  verify_args+=(--allow-cold-start)
fi
[ -n "${EXPECTED_STEP:-}" ] && verify_args+=(--expected-step "$EXPECTED_STEP")
[ -n "${EXPECTED_GENERATION:-}" ] && verify_args+=(--expected-generation "$EXPECTED_GENERATION")
[ -n "${EXPECTED_SHA256:-}" ] && verify_args+=(--expected-sha256 "$EXPECTED_SHA256")

"$PYTHON_BIN" "${verify_args[@]}" || die "Checkpoint preflight failed in $CKPT_DIR — refusing to launch"

echo "  shards: $SHARD_COUNT | tokenizer: $TOKENIZER | ckpt_dir: $CKPT_DIR"
echo "== [4/4] launch training =="
cd "$HERE"
if [ -n "${TRAIN_SCRIPT:-}" ]; then
  exec "$PYTHON_BIN" "$TRAIN_SCRIPT"
else
  exec "$PYTHON_BIN" "$HERE/train.py"
fi
