#!/usr/bin/env python3
"""Dense Llama-style pretraining on The Stack v2 (streaming zstd JSONL) on one GB10.

SMOKE=1  -> tiny model (~6M) + 50 steps + Python-only, to verify the whole pipeline fast.
default  -> ~5B dense run over ALL done languages, streamed, as long as you let it run.

Resumable: checkpoints model+optimizer+step to CKPT_DIR; reloads the latest on start.
Memory plan (5B on 121GB): bf16 weights + 8-bit AdamW (bitsandbytes) + gradient checkpointing.
"""
import os, sys, io, glob, json, time, math, random
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import zstandard as zstd
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
import bitsandbytes as bnb

SMOKE     = os.environ.get("SMOKE", "") == "1"
DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt/datasets/pretrain-data/the-stack-v2-content")
CKPT_DIR  = os.path.expanduser(os.environ.get("CKPT_DIR", "~/runs/stackv2-5b" if not SMOKE else "~/runs/smoke"))
TOKENIZER = os.environ.get("TOKENIZER", "bigcode/starcoder2-3b")
SEQ_LEN   = int(os.environ.get("SEQ_LEN", "2048"))
os.makedirs(CKPT_DIR, exist_ok=True)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

LANGS = ["Python"] if SMOKE else None  # None => every language subdir that's been fetched

def list_shards():
    fs = []
    if LANGS:
        for L in LANGS: fs += glob.glob(f"{DATA_ROOT}/{L}/*.zst")
    else:
        fs = glob.glob(f"{DATA_ROOT}/*/*.zst")
    fs.sort(); return fs

class StackStream(IterableDataset):
    """Stream zstd JSONL -> tokenize 'content' -> emit packed non-overlapping SEQ_LEN blocks."""
    def __init__(self, tok, seq_len): self.tok=tok; self.seq_len=seq_len; self.eos=tok.eos_token_id or 0
    def __iter__(self):
        wi = get_worker_info(); shards = list_shards()
        if wi: shards = shards[wi.id::wi.num_workers]
        random.shuffle(shards)
        dctx = zstd.ZstdDecompressor(); buf = []
        for path in shards:
            try:
                with open(path, "rb") as fh:
                    rdr = io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8", errors="ignore")
                    for line in rdr:
                        try: content = json.loads(line)["content"]
                        except Exception: continue
                        ids = self.tok(content, add_special_tokens=False)["input_ids"]; ids.append(self.eos)
                        buf.extend(ids)
                        while len(buf) >= self.seq_len:
                            yield torch.tensor(buf[:self.seq_len], dtype=torch.long); buf = buf[self.seq_len:]
            except Exception: continue

def main():
    torch.manual_seed(0); random.seed(0); dev="cuda"
    log(f"SMOKE={SMOKE}  tokenizer={TOKENIZER}  seq={SEQ_LEN}  rope={os.environ.get("ROPE_THETA","10000")}  ckpt={CKPT_DIR}")
    tok = AutoTokenizer.from_pretrained(TOKENIZER); vocab=len(tok)
    if SMOKE:
        cfg = LlamaConfig(vocab_size=vocab, hidden_size=256, intermediate_size=688,
                          num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
                          max_position_embeddings=SEQ_LEN)
        STEPS,BS,ACC,SAVE,LOGE,WARM = 50,2,1,25,5,5
    else:
        cfg = LlamaConfig(vocab_size=vocab, hidden_size=4096, intermediate_size=11008,
                          num_hidden_layers=26, num_attention_heads=32, num_key_value_heads=8,
                          max_position_embeddings=SEQ_LEN, rms_norm_eps=1e-5, rope_theta=float(os.environ.get("ROPE_THETA", "10000")))
        STEPS,BS,ACC,SAVE,LOGE,WARM = 500000, int(os.environ.get("BS","4")), int(os.environ.get("ACC","8")), int(os.environ.get("SAVE","1000")), 10, 2000
    cfg.use_cache=False
    cfg._attn_implementation = "sdpa"   # fast fused attention (no flash-attn needed)
    model = LlamaForCausalLM(cfg)
    log(f"model params: {sum(p.numel() for p in model.parameters())/1e9:.3f}B  vocab {vocab}  BS={BS} ACC={ACC} gckpt={os.environ.get('GRAD_CKPT','1')}")
    if os.environ.get("GRAD_CKPT","1") == "1": model.gradient_checkpointing_enable()
    model.to(dev, dtype=torch.bfloat16)
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=3e-4, betas=(0.9,0.95), weight_decay=0.1)
    LRMAX,LRMIN=3e-4,3e-5
    def lr_at(s):
        if s<WARM: return LRMAX*s/max(1,WARM)
        p=min(1,(s-WARM)/max(1,STEPS-WARM)); return LRMIN+0.5*(LRMAX-LRMIN)*(1+math.cos(math.pi*p))
    start=0; cks=sorted(glob.glob(f"{CKPT_DIR}/ckpt_*.pt"))
    if cks:
        sd=torch.load(cks[-1], map_location=dev)
        model.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"]); start=sd["step"]
        log(f"resumed {cks[-1]} @ step {start}")
    dl = DataLoader(StackStream(tok,SEQ_LEN), batch_size=BS, num_workers=(2 if SMOKE else int(os.environ.get("NW","8"))),
                    pin_memory=True, prefetch_factor=4, drop_last=True)
    it=iter(dl); model.train(); t0=time.time(); toks=0
    for step in range(start, STEPS):
        for g in opt.param_groups: g["lr"]=lr_at(step)
        opt.zero_grad(set_to_none=True); lacc=0.0
        for _ in range(ACC):
            try: b=next(it)
            except StopIteration: it=iter(dl); b=next(it)
            b=b.to(dev); out=model(input_ids=b, labels=b); (out.loss/ACC).backward()
            lacc+=out.loss.item()/ACC; toks+=b.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%LOGE==0:
            dt=time.time()-t0; tps=toks/dt if dt>0 else 0
            log(f"step {step}/{STEPS} loss {lacc:.4f} lr {lr_at(step):.2e} tok/s {tps:,.0f} peakmem {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
            t0=time.time(); toks=0
        if step>0 and step%SAVE==0:
            p=f"{CKPT_DIR}/ckpt_{step:08d}.pt"
            torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"step":step,"cfg":cfg.to_dict()},p); log(f"saved {p}")
            for o in sorted(glob.glob(f"{CKPT_DIR}/ckpt_*.pt"))[:-3]:
                try: os.remove(o)
                except Exception: pass
    torch.save({"model":model.state_dict(),"step":STEPS,"cfg":cfg.to_dict()}, f"{CKPT_DIR}/final.pt"); log("DONE")

if __name__=="__main__": main()
