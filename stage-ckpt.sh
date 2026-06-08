#!/usr/bin/env bash
# Move checkpoints between node1 (Spark) and the NAS burst dir, so a rented box can
# pick up where the Spark left off — and the Spark can pick up the rental's progress.
# Run from your Mac (needs SSH to node1 + the NAS). train.py always resumes from the
# HIGHEST-numbered ckpt_*.pt, so round-trips "just work" as long as steps only go up.
#
#   ./stage-ckpt.sh to-nas     # before a burst: node1 local ckpt  -> NAS burst dir (rental pulls this)
#   ./stage-ckpt.sh from-nas   # after a burst:  NAS burst dir     -> node1 local (Spark resumes from it)
set -euo pipefail
NODE1="${NODE1:-node1}"
NODE1_CKPT="${NODE1_CKPT:-/home/node1/runs/stackv2-5b}"
NAS="${NAS:-batcloud}"
NAS_CKPT="${NAS_CKPT:-/volume1/Datasets/pretrain-checkpoints/burst}"

case "${1:-}" in
  to-nas)
    latest=$(ssh "$NODE1" "ls -1 $NODE1_CKPT/ckpt_*.pt 2>/dev/null | tail -1")
    [ -n "$latest" ] || { echo "no ckpt on node1"; exit 1; }
    echo "staging $latest -> $NAS:$NAS_CKPT/"
    ssh "$NAS" "mkdir -p $NAS_CKPT"
    ssh "$NODE1" "rsync -a '$latest' $NAS:$NAS_CKPT/"   # node1->NAS over LAN
    echo "done. Point the rental's CKPT_DIR rsync at $NAS:$NAS_CKPT/"
    ;;
  from-nas)
    latest=$(ssh "$NAS" "ls -1 $NAS_CKPT/ckpt_*.pt 2>/dev/null | tail -1")
    [ -n "$latest" ] || { echo "no ckpt on NAS"; exit 1; }
    echo "pulling $latest -> $NODE1:$NODE1_CKPT/  (stop stackv2-train.service first, then relaunch to resume)"
    ssh "$NODE1" "rsync -a $NAS:$latest $NODE1_CKPT/"
    echo "done. Relaunch training on node1; it resumes from the highest-numbered ckpt."
    ;;
  *) echo "usage: $0 {to-nas|from-nas}"; exit 1 ;;
esac
