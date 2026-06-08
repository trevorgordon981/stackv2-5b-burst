#!/usr/bin/env bash
# ============================================================================
# Resume the stackv2-5b pretrain on a rented GPU (e.g. Vast.ai RTX PRO 6000 Blackwell).
# Goal: `git clone <repo> && bash setup-on-rental.sh` -> training resumes in ~10 min.
#
# YOU PROVIDE (env vars before running):
#   HF_TOKEN    HuggingFace token (for the bigcode/starcoder2-3b tokenizer)   [required]
#   TS_AUTHKEY  Tailscale auth key -> joins tailnet to pull data+ckpt from NAS [recommended]
#   NAS         NAS/relay tailnet host for rsync (default: batcloud)
#
# DATA + CHECKPOINT options (pick one — see README):
#   A) NAS over tailnet (default): set TS_AUTHKEY; script rsyncs shards + latest ckpt.
#   B) Manual: pre-place .zst shards in $DATA_ROOT/<lang>/ and latest ckpt in $CKPT_DIR.
#
# TUNABLES (sensible 96GB-card burst defaults; keep BS*ACC=48 for LR-schedule continuity):
#   BS=16 ACC=3  -> 48 seqs/step = SAME global batch as the Spark (BS6*ACC8), so the
#                   cosine LR schedule stays aligned step-for-step. Faster card = more tok/s,
#                   NOT a bigger batch. Bump BS only if you also drop ACC to keep 48.
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

WORK="${WORK:-$HOME/stackv2-burst-run}"
export DATA_ROOT="${DATA_ROOT:-$WORK/data}"
export CKPT_DIR="${CKPT_DIR:-$WORK/ckpt}"
NAS="${NAS:-batcloud}"
NAS_DATA="${NAS_DATA:-/volume1/Datasets/pretrain-data/the-stack-v2-content}"
NAS_CKPT="${NAS_CKPT:-/volume1/Datasets/pretrain-checkpoints/burst}"
SHARD_DIRS="${SHARD_DIRS:-8}"   # how many language subdirs of shards to stage (each is sizable!)

# --- training config (override as needed) ---
export SEQ_LEN="${SEQ_LEN:-8192}" ROPE_THETA="${ROPE_THETA:-500000}"
export BS="${BS:-16}" ACC="${ACC:-3}" NW="${NW:-12}" GRAD_CKPT="${GRAD_CKPT:-1}" SAVE="${SAVE:-200}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZER="${TOKENIZER:-bigcode/starcoder2-3b}"

mkdir -p "$WORK" "$DATA_ROOT" "$CKPT_DIR"
echo "WORK=$WORK  DATA_ROOT=$DATA_ROOT  CKPT_DIR=$CKPT_DIR  BS=$BS ACC=$ACC SEQ_LEN=$SEQ_LEN"

echo "== [1/5] python deps =="
pip install -q --upgrade pip
# torch matched to the box CUDA (Blackwell sm_120 -> cu128/cu130). Edit index-url if your box differs.
python -c "import torch,sys; v=tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); sys.exit(0 if (torch.cuda.is_available() and v>=(2,7)) else 1)" 2>/dev/null \
  || pip install -q "torch==2.11.*" --index-url https://download.pytorch.org/whl/cu128
pip install -q -r "$HERE/requirements.txt"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

echo "== [2/5] tokenizer auth =="
if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; huggingface-cli login --token "$HF_TOKEN" >/dev/null 2>&1 || true; fi

echo "== [3/5] tailnet (for NAS pull) =="
if [ -n "${TS_AUTHKEY:-}" ] && ! command -v tailscale >/dev/null; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi
if [ -n "${TS_AUTHKEY:-}" ]; then sudo tailscale up --authkey "$TS_AUTHKEY" --hostname vast-burst --ssh || true; fi

echo "== [4/5] stage data + latest checkpoint =="
if [ -z "$(ls -A "$DATA_ROOT" 2>/dev/null)" ]; then
  if command -v tailscale >/dev/null; then
    echo "  rsyncing $SHARD_DIRS shard-dirs from $NAS:$NAS_DATA ..."
    ssh -o StrictHostKeyChecking=accept-new "$NAS" "ls $NAS_DATA" | head -n "$SHARD_DIRS" | while read -r L; do
      echo "   -> $L"; rsync -a "$NAS:$NAS_DATA/$L/" "$DATA_ROOT/$L/" || true
    done
  else
    echo "  !! DATA_ROOT empty and no tailnet. Place .zst shards in $DATA_ROOT/<lang>/ then re-run." ; exit 1
  fi
fi
if [ -z "$(ls -A "$CKPT_DIR" 2>/dev/null)" ] && command -v tailscale >/dev/null; then
  rsync -a "$NAS:$NAS_CKPT/" "$CKPT_DIR/" 2>/dev/null || echo "  (no ckpt on NAS -> cold start from step 0)"
fi
echo "  shards: $(find "$DATA_ROOT" -name '*.zst' 2>/dev/null | wc -l) | ckpts: $(ls "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | wc -l)"

echo "== [5/5] launch (auto-resumes from latest ckpt in $CKPT_DIR) =="
cd "$HERE"
exec python train.py
