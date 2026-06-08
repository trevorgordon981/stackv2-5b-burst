#!/usr/bin/env bash
# ============================================================================
# Burst launcher.  On a fresh rented GPU box (Vast.ai etc.):
#
#     git clone https://github.com/trevorgordon981/stackv2-5b-burst
#     cd stackv2-5b-burst
#     TS_AUTHKEY=tskey-auth-...  bash go.sh
#
# NO secrets live in this repo. The only thing you pass is a Tailscale auth key
# (env var) so the box joins your tailnet and pulls data+ckpt from the NAS over
# Tailscale SSH. Mint a reusable+ephemeral+tagged key at:
#   https://login.tailscale.com/admin/settings/keys
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HERE/secrets.env" ] && { set -a; . "$HERE/secrets.env"; set +a; }   # optional LOCAL override (gitignored)
: "${TS_AUTHKEY:?Set TS_AUTHKEY env, e.g.  TS_AUTHKEY=tskey-auth-... bash go.sh  (mint at tailscale admin -> Keys)}"

WORK="${WORK:-$HOME/stackv2-burst-run}"
export DATA_ROOT="$WORK/data" CKPT_DIR="$WORK/ckpt"
export TOKENIZER="$HERE/tokenizer"                       # bundled -> no HF token
NAS="${NAS:-batcloud}"; NAS_USER="${NAS_USER:-trevorbg}"
NAS_DATA="/volume1/Datasets/pretrain-data/the-stack-v2-content"
NAS_CKPT="/volume1/Datasets/pretrain-checkpoints/burst"
SHARD_DIRS="${SHARD_DIRS:-8}"
# 96GB-card defaults; BS*ACC=48 = SAME global batch as the Spark (LR schedule stays aligned)
export SEQ_LEN=8192 ROPE_THETA=500000 BS=16 ACC=3 NW=12 GRAD_CKPT=1 SAVE=200
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$WORK" "$DATA_ROOT" "$CKPT_DIR"

echo "== [1/4] deps =="
pip install -q --upgrade pip
python -c "import torch,sys; v=tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); sys.exit(0 if (torch.cuda.is_available() and v>=(2,7)) else 1)" 2>/dev/null \
  || pip install -q "torch==2.11.*" --index-url https://download.pytorch.org/whl/cu128
pip install -q -r "$HERE/requirements.txt"
python -c "import torch;print(' torch',torch.__version__,'|',torch.cuda.get_device_name(0))"

echo "== [2/4] join tailnet =="
command -v tailscale >/dev/null || curl -fsSL https://tailscale.com/install.sh | sh
# containers have no systemd, so start tailscaled ourselves if it isn't up
if ! sudo tailscale status >/dev/null 2>&1; then
  if [ -e /dev/net/tun ]; then
    echo "  starting tailscaled (TUN)"
    sudo tailscaled --state=/var/lib/tailscale/tailscaled.state >/tmp/tailscaled.log 2>&1 &
  else
    echo "  no /dev/net/tun -> userspace networking (NAS pull will need the SOCKS proxy; tell the operator if this box has no TUN)"
    sudo tailscaled --tun=userspace-networking --socks5-server=localhost:1055 --state=/var/lib/tailscale/tailscaled.state >/tmp/tailscaled.log 2>&1 &
  fi
  sleep 5
fi
sudo tailscale up --authkey "$TS_AUTHKEY" --hostname vast-burst
sudo tailscale status 2>/dev/null | head -5 || true

echo "== [3/4] pull data + latest checkpoint from NAS =="
NAS_IP=$(sudo tailscale status 2>/dev/null | awk -v n="$NAS" '$2==n{print $1; exit}'); NAS_IP="${NAS_IP:-$NAS}"
NASU="$NAS_USER@$NAS_IP"
# universal NAS-ssh wrapper: tunnels via `tailscale nc` (needed in userspace mode, harmless under TUN);
# uses a deploy key if NAS_KEY is set (the NAS is key-only sshd).
cat > "$WORK/_nassh" <<SSHW
#!/usr/bin/env bash
exec ssh -o StrictHostKeyChecking=accept-new -o ProxyCommand="tailscale nc %h %p" ${NAS_KEY:+-i $NAS_KEY -o IdentitiesOnly=yes} "\$@"
SSHW
chmod +x "$WORK/_nassh"; SSHC="$WORK/_nassh"
echo "  NAS=$NASU  key=${NAS_KEY:-<none/agent>}"
if [ -z "$(ls -A "$DATA_ROOT" 2>/dev/null)" ]; then
  "$SSHC" "$NASU" "ls $NAS_DATA" | head -n "$SHARD_DIRS" | while read -r L; do
    echo "  data <- $L"; rsync -rt --no-o --no-g -e "$SSHC" "$NASU:$NAS_DATA/$L/" "$DATA_ROOT/$L/" || true
  done
fi
rsync -rt --no-o --no-g -e "$SSHC" "$NASU:$NAS_CKPT/" "$CKPT_DIR/" 2>/dev/null \
  || echo "  (NAS pull failed -> authorize this box: add its key to the NAS and re-run with NAS_KEY=~/.ssh/nas)"
echo "  shards: $(find "$DATA_ROOT" -name '*.zst' 2>/dev/null | wc -l) | ckpts: $(ls "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | wc -l)"

echo "== [4/4] launch (auto-resumes from latest ckpt) =="
cd "$HERE" && exec python train.py
