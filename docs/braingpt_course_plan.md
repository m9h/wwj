# BrainGPT-from-scratch: a teaching repo for the domain-LLM lifecycle

A standalone, student-facing repo that trains a domain language model (BrainGPT —
GPT-2 on neuroscience) end-to-end, embedding HuggingFace's open training literature as
the "learn more" scaffolding and using `wwj`/`wwjd` spectral analysis as a differentiated
"look inside the weights" capstone. Separate from the `wwj` research repo (different
audience); reuses `benchmarks/modal_gpt2_neuro.py` as the training engine and imports
`wwj` for the analysis module.

## Why this repo
Most open LLM courses stop at "it trains and it works." This one runs one domain model
through every stage a student needs to understand — data, tokenization, pretraining vs
fine-tuning, evaluation — and then *opens the trained weights* to show what training did
to them (heavy-tailed spectra, with calibrated uncertainty). The BrainGPT task
(predicting neuroscience results via BrainBench) is concrete and motivating, and the
1.3B-token corpus is open (Apache-2.0), so the whole thing is cheaply reproducible.

## Module map (each = a runnable notebook + a "learn more" box)

| Module | Student does | HF material embedded |
|---|---|---|
| 0. Why a domain LLM? | BrainGPT context, the BrainBench task | braingpt-lovelab papers (attribution) |
| **1. Data** | curate the PMC-neuroscience corpus: dedup, quality filter, splits | **FineWeb** (the anchor — see below) + `datasets` docs |
| 2. Tokenization | train a domain BPE, compare to GPT-2's (a real lesson in why tokenizers matter) | `tokenizers` "train a new tokenizer", the HF LLM Course tokenizer chapter |
| 3. Pretrain from scratch | GPT-2-124M from scratch; the Chinchilla/compute discussion | **Ultra-Scale Playbook**, **nanotron**, **picotron** (the teaching trainer) + `Trainer` docs |
| 4. Fine-tune vs scratch | the two configs side by side — adaptation vs pretraining | **smol-course** / LLM Course fine-tuning chapter, `trl` |
| 5. Evaluate | BrainBench, perplexity | `lighteval` + the eval ecosystem |
| **6. Look inside** ⭐ | `wwj`/`wwjd`: heavy-tailed spectra, α, power-law validity, the BMA correction | our paper — the unique capstone |

## FineWeb as the Data-module anchor
The **FineWeb blog** ("decanting the web for the finest text data at scale") is the
canonical open, *ablation-driven* walkthrough of an LLM-pretraining data pipeline — dedup,
quality filtering, heuristic choices, each justified by a downstream-eval ablation —
plus **FineWeb-Edu** (classifier-filtered educational subset). It heads the HF lineage
**FineWeb (data) → SmolLM (model) → nanotron/picotron (training) → Ultra-Scale Playbook
(scaling)**, which this repo mirrors with BrainGPT as the worked domain example.

The pedagogical move is a **scale/domain contrast**: FineWeb = web-scale, general; the
BrainGPT corpus = domain-specific (PMC neuroscience), smaller and cleaner. Students learn
the *transferable craft* — quality filtering and dedup are the same whether you decant 15T
tokens of web or 1.3B of neuroscience — and see the trade-offs side by side. Stronger than
teaching either alone.

**Novel tie-in (a small original contribution, not just reproduction):** FineWeb ablates
data choices by *downstream eval*. Module 1 + Module 6 add a second lens — **ablate by what
the data does to the weight spectrum**: train on different data filters, run `wwjd`, watch
α / power-law-validity shift. "Does cleaner data produce healthier heavy tails?" is a
question FineWeb's framework does not ask, and it is the Module-6 spectral capstone applied
to *data* rather than *training stage*.

## Repo structure (proposed)
```
braingpt-from-scratch/
  README.md                     # the lifecycle + the HF-lineage map
  notebooks/
    00_why_domain_llm.ipynb
    01_data_fineweb_and_domain.ipynb     # FineWeb principles -> PMC corpus; the spectral-ablation teaser
    02_tokenization.ipynb
    03_pretrain_from_scratch.ipynb
    04_finetune_vs_scratch.ipynb
    05_evaluate_brainbench.ipynb
    06_look_inside_wwjd.ipynb            # imports wwj; the capstone
  src/
    train.py                    # thin wrapper over modal_gpt2_neuro.py (local + Modal)
    data.py                     # corpus curation utilities (FineWeb-style filters, domain)
  modal_app.py                  # the cloud path (reuses modal_gpt2_neuro.py)
  CITATION / ATTRIBUTION.md
```

## Hosting + attribution
- Models / tokenizer / curated dataset → the **HF Hub**.
- Writeup → a **HF Cookbook recipe** or **community blog** (HF features community education).
- Mirror the **smol-course / LLM Course** lesson format.
- **Attribution**: the recipe + corpus are the braingpt-lovelab group's (Foresight-funded,
  Apache-2.0 code + open data). Reproduce with clear credit; co-branding / "based on" nod
  is ideal and (user can reach them) might get the repo more reach.

## Build order (after the BrainGPT fine-tune smoke lands)
1. Confirm the training engine (`modal_gpt2_neuro.py`) produces clean checkpoints + the
   `wwjd` spectral comparison works on them (the wwj study itself).
2. Scaffold the repo skeleton (notebooks + the HF-materials links + the FineWeb framing).
3. Module 6 notebook from the real checkpoints (base / fine-tune / from-scratch).
4. Optional original bit: the data-filter spectral-ablation mini-experiment.
