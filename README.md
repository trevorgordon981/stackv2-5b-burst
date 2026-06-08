# stackv2-5b burst kit

Resume the **stackv2-5b** pretrain (the one running on the DGX Spark `node1`) on a
**rented GPU** for a burst — e.g. a Vast.ai RTX PRO 6000 Blackwell (96GB) — then bring
the checkpoint back to the Spark. Free hardware for the slow trickle; paid bursts for
the real tokens.

## Why this works
`train.py` is hardware-agnostic: it reads all config from env vars and **auto-resumes
from the highest-numbered `ckpt_*.pt` in `$CKPT_DIR`**. A checkpoint is a plain
`torch.save` dict (`model` + 8-bit AdamW state + `step` + `cfg`) that loads on any CUDA
box with matching deps. So the loop is just: move the checkpoint, run, move it back.

## Clone and go
```bash
git clone https://github.com/trevorgordon981/stackv2-5b-burst
cd stackv2-5b-burst
TS_AUTHKEY=tskey-auth-...  bash go.sh
```
`go.sh` handles everything — deps, **bundled tokenizer** (no HF token), training config,
NAS paths, and the data+checkpoint pull. **No secrets are stored in this repo.** The only
thing you pass is a **Tailscale auth key** (env var) so the box joins your tailnet and
pulls data+ckpt from the NAS over **Tailscale SSH**. Mint a reusable+ephemeral+tagged key
at <https://login.tailscale.com/admin/settings/keys>.

> Prereq: the NAS must accept the box — i.e. **Tailscale SSH enabled on the Synology**
> (`tailscale up --ssh`) and your **ACL** allowing the key's tag (e.g. `tag:ci`) to SSH
> `batcloud`. Nothing long-lived is left on the rented box (ephemeral key auto-removes).

Defaults: `BS=16 ACC=3 SEQ_LEN=8192 ROPE_THETA=500000` → **48 seqs/step, the same global
batch as the Spark** (BS6×ACC8), so the cosine LR schedule stays aligned step-for-step.
The faster card shows up as higher tok/s, not a bigger batch.

(`setup-on-rental.sh` is the manual/parameterized variant if you ever want to override via env.)

## The burst loop
```
# 1. before the burst — push the Spark's latest checkpoint to the NAS:
./stage-ckpt.sh to-nas

# 2. on the rental — run setup-on-rental.sh (pulls that ckpt, trains fast, saves every SAVE steps)

# 3. when done — let it write a final ckpt, then push the rental's ckpt back to the NAS
#    (rsync $CKPT_DIR/ckpt_*.pt -> NAS:/volume1/Datasets/pretrain-checkpoints/burst/)

# 4. back home — pull it to node1 and resume the Spark:
./stage-ckpt.sh from-nas
#    stop stackv2-train.service, relaunch -> it resumes from the rental's higher step.
```

## Data
The rental needs Stack v2 shards in `$DATA_ROOT/<lang>/*.zst`. `setup-on-rental.sh`
rsyncs a subset (`SHARD_DIRS=8` language dirs) from the NAS over the tailnet — reusing
what you already fetched, no re-downloading from HF. Bump `SHARD_DIRS` for a longer burst
so the stream doesn't repeat. (Each lang dir is sizable; stage enough for the burst length.)

## Gotchas
- **CUDA**: Blackwell needs CUDA 12.8+. The script installs torch from the cu130 wheel; if
  your Vast image has a different CUDA, edit the `--index-url` in step [1/5].
- **Deps must match** the Spark (`requirements.txt` is pinned) so optimizer state reloads.
- **96GB fits easily** — no `MemoryMax` gymnastics; you can raise BS (keep BS×ACC=48).
- Keep `SEQ_LEN`/`ROPE_THETA` identical to the Spark run or the schedule/positions drift.
