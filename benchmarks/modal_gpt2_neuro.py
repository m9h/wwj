"""Train GPT-2-124M on neuroscience text on Modal, to add a domain-fine-tuning axis to
the WeightWatcher-replication study (base GPT-2 vs fine-tuned vs from-scratch, one
architecture). Faithful to braingpt-lovelab/matching_experts (arXiv 2405.09395):
gpt2 (124M), 1.3B-token BrainGPT/PMC-neuroscience subset (open, Apache-2.0), seq 1024,
per-device batch 16 x grad-accum 8 = 131072 tok/optimizer-step, AdamW lr 2e-5 cosine,
warmup 3%, wd 1e-3, 5 epochs.

Three checkpoints to analyse with wwjd afterward:
  base       -- stock gpt2 (no training here; load from HF for the spectral baseline)
  finetune   -- gpt2 continued on neuroscience (train_mode=finetune)
  scratch    -- gpt2 architecture trained from scratch on neuroscience (train_mode=scratch)

ONE-TIME:  modal volume create gpt2-neuro
PREP:      modal run modal_gpt2_neuro.py::prepare            # download + tokenize -> volume
TRAIN:     modal run --detach modal_gpt2_neuro.py::finetune  # full 5-epoch fine-tune
           modal run --detach modal_gpt2_neuro.py::scratch   # full 5-epoch from-scratch
           modal run --detach modal_gpt2_neuro.py::finetune --epochs 1   # cheap smoke
Spectral compare afterward with benchmarks/ww_replication.py / ww_gpt_detail.py on the
saved checkpoints (they are HF-format gpt2 dirs, drop straight into the existing pipeline).
"""
import modal

GPU = "A100-40GB"          # 124M fits easily; see header of the cost estimate. H100 ~2x faster.
VOL = "/vol"
app = modal.App("gpt2-neuro")
vol = modal.Volume.from_name("gpt2-neuro", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "transformers>=4.44",
        "datasets>=2.20",
        "accelerate>=0.33",
        "tokenizers>=0.19",
    )
)

DATA = "BrainGPT/train_valid_split_pmc_neuroscience_2002-2022_filtered_subset"
SEQ = 1024


@app.function(image=image, volumes={VOL: vol}, cpu=16, memory=65536, timeout=6 * 3600)
def prepare():
    """Download the neuroscience corpus, GPT2-tokenize, and pack into 1024-token blocks
    once; cache the tokenized arrays + the from-scratch neuro tokenizer to the volume."""
    import os
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(f"{VOL}/tok_gpt2", exist_ok=True)
    ds = load_dataset(DATA)
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    text_col = "text" if "text" in ds["train"].column_names else ds["train"].column_names[0]

    def tok_fn(b):
        return tok(b[text_col])

    def group(b):
        concat = sum(b["input_ids"], [])
        n = (len(concat) // SEQ) * SEQ
        ids = [concat[i:i + SEQ] for i in range(0, n, SEQ)]
        return {"input_ids": ids, "labels": [x[:] for x in ids]}

    for split in ds:
        t = ds[split].map(tok_fn, batched=True, remove_columns=ds[split].column_names, num_proc=16)
        # remove_columns drops the stale per-doc input_ids/attention_mask so only the new
        # fixed-length 1024-token blocks remain (else Arrow sees mismatched column lengths).
        t = t.map(group, batched=True, num_proc=16, remove_columns=t.column_names)
        t.save_to_disk(f"{VOL}/tok_gpt2/{split}")
        print(f"[prep] {split}: {len(t)} blocks of {SEQ} tokens", flush=True)

    # GPT2-Neuro tokenizer (BPE retrained on the corpus) for the from-scratch variant
    base = AutoTokenizer.from_pretrained("gpt2")
    def text_iter(bs=1000):
        for i in range(0, len(ds["train"]), bs):
            yield ds["train"][i:i + bs][text_col]
    neuro = base.train_new_from_iterator(text_iter(), vocab_size=base.vocab_size)
    neuro.save_pretrained(f"{VOL}/gpt2_neuro_tokenizer")
    vol.commit()
    print("[prep] done: tokenized blocks + neuro tokenizer committed to volume", flush=True)


def _train(mode: str, epochs: int, neuro_tokenizer: bool):
    import os
    import torch
    from datasets import load_from_disk
    from transformers import (GPT2LMHeadModel, GPT2Config, AutoTokenizer,
                              Trainer, TrainingArguments, default_data_collator)

    out = f"{VOL}/exp/{mode}{'_neurotok' if neuro_tokenizer else ''}_ep{epochs}"
    os.makedirs(out, exist_ok=True)
    train = load_from_disk(f"{VOL}/tok_gpt2/train")
    val = load_from_disk(f"{VOL}/tok_gpt2/validation") if os.path.exists(f"{VOL}/tok_gpt2/validation") else None

    if mode == "finetune":
        model = GPT2LMHeadModel.from_pretrained("gpt2")
    else:  # scratch: same architecture, fresh weights
        cfg = GPT2Config.from_pretrained("gpt2")
        if neuro_tokenizer:
            nt = AutoTokenizer.from_pretrained(f"{VOL}/gpt2_neuro_tokenizer")
            cfg.vocab_size = nt.vocab_size
        model = GPT2LMHeadModel(cfg)

    args = TrainingArguments(
        output_dir=out, overwrite_output_dir=True,
        per_device_train_batch_size=16, gradient_accumulation_steps=8,
        num_train_epochs=epochs, learning_rate=2e-5, weight_decay=1e-3,
        warmup_ratio=0.03, lr_scheduler_type="cosine",
        bf16=True, logging_steps=200, save_strategy="epoch",
        eval_strategy="epoch" if val is not None else "no",
        report_to=[],  # offline
    )
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                      data_collator=default_data_collator)
    print(f"[train] mode={mode} epochs={epochs} blocks={len(train)} "
          f"tokens/step={16*8*SEQ} effective", flush=True)
    trainer.train()
    trainer.save_model(f"{out}/final")
    vol.commit()
    print(f"[train] saved -> {out}/final (HF gpt2 dir; analyse with ww_replication.py)", flush=True)


@app.function(image=image, gpu=GPU, volumes={VOL: vol}, timeout=24 * 3600)
def _finetune(epochs: int = 5):
    _train("finetune", epochs, neuro_tokenizer=False)


@app.function(image=image, gpu=GPU, volumes={VOL: vol}, timeout=24 * 3600)
def _scratch(epochs: int = 5, neuro_tokenizer: bool = True):
    _train("scratch", epochs, neuro_tokenizer=neuro_tokenizer)


@app.local_entrypoint()
def finetune(epochs: int = 5):
    _finetune.remote(epochs)


@app.local_entrypoint()
def scratch(epochs: int = 5, neuro_tokenizer: bool = True):
    _scratch.remote(epochs, neuro_tokenizer)
