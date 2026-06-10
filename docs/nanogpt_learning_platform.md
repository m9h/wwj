# nanoGPT as a Learning Platform — Research & Resource Map

Research compiled 2026-06-09 to turn nanoGPT into a teaching platform for this project
(the HT-SR / `wwj`-`wwjd` spectral-analysis-and-steering angle, with BrainGPT as the
domain example). Links at the bottom.

## 0. The convergence — why nanoGPT is the *right* platform for this project

The community **modded-nanoGPT speedrun** trains a 124M GPT to **3.28 validation loss on
FineWeb** on 8×H100, and it is where the **Muon optimizer** was invented. That one
sentence unifies every thread we are already pulling on:

- **nanoGPT** — a hackable ~300-line training loop (unlike HF Trainer, you can edit the loss).
- **FineWeb** — the data (the HF project; already the speedrun's eval set).
- **Muon** — Martin's RG conjecture (Muon implicitly shapes weight spectra toward α=2) which
  our paper's Related Work *mentions but does not test*.
- **`wwj`/`wwjd`** — our spectral analysis + the differentiable `alpha_loss` steering.

So our two planned experiments drop straight into the speedrun arena: (A) sweep `wwjd`
across speedrun checkpoints to watch α emerge from scratch with credible intervals, and
(B) add `alpha_loss` to the loss and compare **AdamW vs Muon vs α-steering** on the *same*
FineWeb target. The platform and the research are the same artifact.

## 1. The Karpathy educational lineage (the spine to build on)

| Resource | What it teaches | Size | Role in our platform |
|---|---|---|---|
| **nn-zero-to-hero** | backprop → MLP → … → GPT, from scratch (video + notebooks + exercises) | course | the prerequisite scaffold; Module 0–1 framing |
| **ng-video-lecture** ("let's build GPT") | the minimal decoder-only GPT from an empty file | ~300 ln | the "build the model" lesson |
| **nanoGPT** | train/finetune GPT-2 cleanly; `transformer_sizing.ipynb` | ~300 ln | the **training engine** we instrument |
| **build-nanogpt** ("Let's reproduce GPT-2 124M", 4-hr video) | full GPT-2 reproduction incl. **`fineweb.py`** data prep | repo | the **spine** — already FineWeb-based |
| **llm.c** | the same in raw C/CUDA; GPT-2 124M on 10B FineWeb tokens, ~$20/90min on 8×A100 | ~4k ln C | sets the **3.28 FineWeb loss target**; "how it really runs" |
| **nanochat** | full ChatGPT pipeline (pretrain→SFT→RL→web UI), Muon, pedagogical | ~8k ln | the "all the way to a chat model" stretch module |

**Recommendation:** build on **build-nanogpt** (it already includes the FineWeb prep and is
the explicit educational reproduction) as the platform spine, and keep **nanochat** as the
optional "go further" capstone.

## 2. The speedrun ecosystem (optimization + the Muon connection)

- **KellerJordan/modded-nanogpt** — the speedrun (3.28 FineWeb val loss; record now <90 s on
  8×H100, down from llm.c's 45 min). The home of **Muon**.
- **Muon blog** (Keller Jordan) — the optimizer that orthogonalizes update matrices; 1.35×
  speedup. *Directly testable against our `alpha_loss`* (explicit vs implicit spectral shaping).
- **Field guides / worklogs** (great teaching material on *why* each optimization works):
  Tyler Romero's living worklog; Evan Conway's "Field Guide to NanoGPT Speedrun
  Optimizations"; the LessWrong "how the WR dropped 20%" writeup; **alexjc/nanogpt-speedrun**
  (a clean staged version); **Deveraux-Parker/nanoGPT_1GPU_SPEEDRUN** (single-4090 — the
  GPU-poor path, ideal for students); **PrimeIntellect auto-nanogpt** (autonomous AI research
  *on* the speedrun — ties to the autoresearch ethos).

## 3. Cloud + reproducibility (our Modal path is already paved)

- **AI Engineering Academy — "Training NanoGPT on Modal"** and **"Training Nanochat on
  Modal"** — step-by-step serverless tutorials; the exact pattern we use in
  `benchmarks/modal_gpt2_neuro.py`.
- **build-nanogpt/`fineweb.py`** — the canonical FineWeb(-Edu) shard-download + tokenize.
- **NVIDIA NeMo-AutoModel nanogpt-pretraining** guide — a "scale it up" reference.

## 4. Interpretability / "look inside the weights" — the gap that is our differentiator

- **The Annotated Transformer** (Harvard NLP) — the classic line-by-line transformer; the
  reference for "understand every line."
- **Prisma** — open mech-interp toolkit (SAEs, activation caching, circuit + viz tools, model
  checkpoints) — the model for an interpretability teaching toolkit, though vision-focused.
- General mech-interp teaching: attention-pattern visualization, training-dynamics /
  feature-emergence studies.

**The gap:** none of the nanoGPT educational ecosystem threads **heavy-tailed self-
regularization / weight-spectrum diagnostics** into teaching. Every course stops at loss
curves and attention maps. **A Module that opens the trained weights and shows the power-law
tail emerging (with calibrated Bayesian α, BMA-over-window, power-law validity) is genuinely
new** — it is `wwjd` applied to a from-scratch nanoGPT run, and it is the platform's unique
hook.

## 4b. What *other groups* have built around nanoGPT (third-party supplementary material)

Yes — a substantial ecosystem, led by a full university course:

- **Stanford CS336 "Language Modeling from Scratch"** (Liang, Hashimoto; 2025/2026) — the
  flagship academic course in exactly this lineage ("from scratch," OS-course-style). Five
  assignments mirror our module map: **A1 basics** (tokenizer + transformer + optimizer),
  **A2 systems** (FlashAttention-2 in Triton, distributed training), **A3 scaling** (fit a
  scaling law), **A4 data** (filter + dedup Common Crawl → the FineWeb-style data lesson),
  **A5 alignment** (SFT + RL + DPO). Public lectures, assignments, and code. **Notably it has
  no "look inside the weights"/spectral module** — confirming our `wwjd` angle is a genuine
  gap even against the best existing curriculum, and one that would slot naturally after A3
  (scaling: "what happens to α as you scale?") or A4 (data: "does cleaner data → healthier
  tails?").
- **beyond-nanoGPT** (tanishqkumar) — minimal, *annotated* from-scratch implementations of
  ~100 modern DL techniques, explicitly bridging nanoGPT → research-level work. The best
  "what's next after nanoGPT" supplement.
- **minGPT** (Karpathy) — the *educational* predecessor; nanoGPT is its functionality-first
  rewrite, so minGPT is still the cleanest "understand the model" reading.
- **Speedrun field guides** (Romero worklog, Conway "Field Guide", the LessWrong WR-drop
  writeup) — community-written explainers of *why* each optimization works; excellent
  teaching material.
- **Cloud tutorials** (AI Engineering Academy: nanoGPT-on-Modal, nanochat-on-Modal) and
  **NVIDIA NeMo-AutoModel**'s nanoGPT pretraining guide — institutional "run it" material.
- **The long tail**: `github.com/topics/nanogpt` (forks), and many written walkthroughs
  (Chumbar's deep-dive, the artinte walkthrough, Medium guides).

**Takeaway for us:** the ecosystem is mature on *build/train/scale/data/align* but empty on
*spectral diagnostics of the trained weights*. We do not need to re-teach the transformer —
CS336/beyond-nanoGPT/build-nanogpt already do it superbly. Our contribution is the **Module
that opens the weights** (`wwjd` α-trajectory + steering), designed to *plug into* a
CS336-style course rather than replace it.

## 5. Proposed platform = nanoGPT + the spectral lens

Spine: **build-nanogpt** (model + FineWeb). Instrumentation: **`wwj`/`wwjd`** as a
checkpoint-sweep + a 20-line `alpha_loss` hook in `train.py` (we already have
`backbone_alpha_loss` in torch from nanopath). Modules:

1. **Data** — FineWeb principles (`fineweb.py`) + the domain parallel (BrainGPT corpus).
2. **Build & train** — ng-video-lecture model → nanoGPT/build-nanogpt training to 3.28.
3. **Optimize** — AdamW vs **Muon** vs **`alpha_loss` steering**, same FineWeb target
   (the speedrun arena; tests Martin's Muon conjecture directly).
4. **Look inside** ⭐ — `wwjd` α-trajectory across checkpoints: when does the power law
   emerge, where is the BMA correction largest, does steering/Muon change it?
5. **Domain transfer** — fine-tune to BrainGPT; re-run the spectral analysis (ties to
   `docs/braingpt_course_plan.md`).

Reuse: `benchmarks/modal_gpt2_neuro.py` (engine), `wwj` (analysis), the open BrainGPT
corpus (domain). Cloud path via the AI Engineering Academy Modal pattern. The whole thing
is cheap (single-GPU speedrun forks exist) and fully reproducible.

## 6. Resource index

**Karpathy lineage**
- nn-zero-to-hero: https://github.com/karpathy/nn-zero-to-hero · https://karpathy.ai/zero-to-hero.html
- ng-video-lecture: https://github.com/karpathy/ng-video-lecture
- nanoGPT: https://github.com/karpathy/nanoGPT
- build-nanogpt (+ fineweb.py): https://github.com/karpathy/build-nanogpt · https://github.com/karpathy/build-nanogpt/blob/master/fineweb.py
- llm.c: https://github.com/karpathy/llm.c · https://github.com/karpathy/llm.c/discussions/481
- "Let's reproduce GPT-2 (124M)" video: https://archive.org/details/lets-reproduce-gpt-2-124-m

**Speedrun + Muon**
- modded-nanogpt: https://github.com/KellerJordan/modded-nanogpt
- Muon: https://kellerjordan.github.io/posts/muon/
- Speedrun worklog (Romero): https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/
- Field guide (Conway): https://evanjayconway.com/posts/2026/nanogpt-improvements/
- WR-drop writeup: https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/
- staged speedrun (alexjc): https://github.com/alexjc/nanogpt-speedrun
- 1-GPU/4090 speedrun: https://github.com/Deveraux-Parker/nanoGPT_1GPU_SPEEDRUN
- auto-nanogpt (PrimeIntellect): https://www.primeintellect.ai/auto-nanogpt

**Cloud / reproducibility**
- Train nanoGPT on Modal: https://aiengineering.academy/LLM/ServerLessFinetuning/TrainNanoGPTModalTutorial/
- Train nanochat on Modal: https://aiengineering.academy/LLM/ServerLessFinetuning/TrainNanochatModalTutorial/
- NeMo-AutoModel nanogpt pretraining: https://docs.nvidia.com/nemo/automodel/latest/guides/llm/nanogpt-pretraining.html

**Interpretability / inside-the-model**
- The Annotated Transformer: https://nlp.seas.harvard.edu/annotated-transformer/
- Prisma (mech-interp toolkit): https://arxiv.org/html/2504.19475v1
- nanoGPT deep-dive (Chumbar): https://medium.com/@shawn.chumbar/understanding-nanogpt-a-deep-dive-into-transformer-architecture-implementation-9a7167b7d58c

**Other groups' supplementary material (courses + annotated forks)**
- Stanford CS336 "Language Modeling from Scratch": https://cs336.stanford.edu/ · lectures: https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_ · org: https://github.com/stanford-cs336
- beyond-nanoGPT (annotated ~100 techniques): https://github.com/tanishqkumar/beyond-nanogpt
- minGPT (educational predecessor): https://github.com/karpathy/minGPT
- nanoGPT topic hub (forks): https://github.com/topics/nanogpt
