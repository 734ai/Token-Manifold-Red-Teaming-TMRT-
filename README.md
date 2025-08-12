# Token‑Manifold Red Teaming (TMRT)

> Hybrid white-box / black-box pipeline for discovering minimal, natural‑language perturbations that cause safety failures by exploiting tokenizer/normalizer divergences, embedding‑space adversarial directions, and contextual role scaffolding.

---

## Overview

TMRT is an automated red‑teaming framework built for open‑weight LLMs (e.g., gpt‑oss‑20b). It searches a combined space of:

1. **Tokenizer / Normalizer Divergences** — generate text variants (homoglyphs, zero‑width chars, combining marks) that pass common normalizers but produce harmful token sequences for the model.
2. **Embedding‑Space Adversarial Directions** — use white‑box access to compute embedding perturbations that increase the probability of unsafe outputs, then map perturbations back to fluent natural text via constrained decoding.
3. **Role & Metadata Scaffolding** — create plausible multi‑role contexts and microformats (email headers, logs, JSON) that socially engineer the model to escalate beyond safety boundaries.

TMRT combines those axes in an evolutionary search loop to produce small, stealthy, and novel attack candidates. The goal is to find reproducible failures that are not present in public jailbreak corpora and to produce structured findings suitable for competition submission.

---

## Key Goals

* **Novelty:** Find failure modes not covered by current jailbreak datasets or published red‑team papers.
* **Minimality:** Prefer smallest possible perturbations (token/codepoint changes) that flip model behavior.
* **Stealth:** Ensure inputs look natural (low perplexity) and survive typical normalization/filtering.
* **Reproducibility:** Provide deterministic seeds, model config, and a findings JSON compatible with competition schema.

---

## Repository Layout (suggested)

```
tmrt/
├─ README.md
├─ LICENSE
├─ pyproject.toml / requirements.txt
├─ src/
│  ├─ tmrt/
│  │  ├─ __init__.py
│  │  ├─ search_controller.py        # Evolutionary controller
│  │  ├─ unicode_mutators.py         # Tokenization/normalizer variants
│  │  ├─ embedding_optimizer.py      # White‑box embedding ascent + projection
│  │  ├─ scaffolder.py               # Role scaffolding generators
│  │  ├─ verifier.py                 # Run model + safety classifier + metrics
│  │  ├─ novelty_detector.py         # Check against public corpora
│  │  └─ utils.py
│  └─ notebooks/
│     ├─ demo_toy_model.ipynb
│     └─ reproduce_finding.ipynb
├─ findings/
│  └─ *.json                         # Exported findings in competition schema
└─ docker/
   ├─ Dockerfile
   └─ entrypoint.sh
```

---

## Installation & Quickstart

> **Note:** TMRT is intended to be run on machines with GPU access for faster iteration. The white‑box embedding optimizer requires loading model weights in PyTorch.

1. Clone the repo

```bash
git clone <your-repo-url>
cd tmrt
```

2. (Optional) Create Python virtualenv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

3. Run the toy demo (safe, non‑harmful surrogate model)

```bash
python -m src.tmrt.demo --config configs/toy_demo.yaml
```

4. Run a full search (example)

```bash
python -m src.tmrt.search_controller --config configs/full_search.yaml --seed 42
```

Outputs will be written to `outputs/` and findings saved to `findings/` in the Kaggle schema.

---

## Components & Design Notes

### Tokenizer / Normalizer Divergence Finder

* Generate variants using curated lists: homoglyph maps, zero‑width/invisible characters, Unicode combining marks, accented forms, and encoding wrappers (base64/hex with benign wrapper).
* For each candidate, compute `normalized_text` (emulating common server normalizers) and `token_sequence` (model tokenizer). Score by divergence and semantic plausibility.
* Prefer candidates that look identical after normalization but tokenize differently.

### Embedding‑Space Optimizer

* Use gradient ascent on token embeddings to increase log‑probability of a target unsafe token/phrase.
* Constrain perturbation norm (L2/L∞) and include a language prior term to keep tokens fluent.
* Project perturbed embeddings to nearest valid token sequences via constrained decoding (beam search + LM scoring).

### Role Scaffolder

* Templates for email headers, conversation transcripts, and JSON blocks.
* Automatic role chaining: create short benign dialog turns that establish trust, then place a scaffolded ask that relies on prior context.
* Support metadata spoofing (From:, Verified: tokens, signatures) to test social engineering signals.

### Evolutionary Search Controller

* Population-based search where individuals are tuples: (unicode\_mutations, scaffold\_template, embedding\_perturbation\_params).
* Fitness = weighted sum of: attack success (severity), minimality (smaller edits), stealth (LM perplexity), cross‑prompt robustness.
* Operators: mutation (small edit), crossover (swap scaffolds), local search (gradient descent on perturbation strength).

### Novelty Detector

* Maintain an indexed corpus of public jailbreaks and known prompts.
* Check candidate similarity (token edit distance + semantic embedding cosine) and mark duplicates.

---

## Metrics & Evaluation

* **Attack Success Rate (ASR):** fraction of seed prompts that produce disallowed content when the candidate is applied.
* **MinEdit:** average number of changed codepoints/tokens to cause failure.
* **Perplexity\_in:** LM perplexity of the input compared to baseline.
* **CrossSeed Generalization:** fraction of different seed prompts the candidate affects.
* **NoveltyFlag:** binary (new vs known) plus similarity score.

Include these metrics in result artifacts and visualizations.

---

## Reproducibility & Submission

* Save deterministic seeds for the evolutionary controller and model RNGs.
* Export findings in the Kaggle/OpenAI JSON schema (see `findings/` for examples).
* Provide minimal Docker image or environment spec for reproducing the discovery.

---

## Responsible Use & Safety

* Do **not** publish raw harmful outputs. Redact or abstract when demonstrating findings in public notebooks (e.g., show paraphrases, sanitized snippets, or safety‑relevant features instead of explicit disallowed outputs).
* Use the novelty detector to avoid submitting already published jailbreaks.
* Follow Kaggle competition rules and OpenAI’s responsible disclosure guidelines. When in doubt, err on the side of redaction.

---

## Suggested Experiments (first 3 runs)

1. **Normalization Sweep:** Run the Tokenizer Divergence Finder across 100 seed prompts and rank candidates by divergence and stealth.
2. **Embedding Probe:** For top 50 candidates, run embedding ascent with constrained norms and project back to tokens. Record success and perplexity.
3. **Full Evolution:** Run population search combining the above with scaffolds for 1,000 generations (or until computational budget exhausted). Triage top candidates with the novelty detector.

---

## Deliverables for Kaggle

* `findings/*.json` — findings in the competition schema.
* `notebooks/reproduce_finding.ipynb` — sanitized reproducible notebook for one novel finding.
* `src/` — code to run the pipeline and reproduce experiments.
* `docker/` — Dockerfile for reproducibility.
* `writeup.pdf` — explanation of the novel attack, why it’s new, mitigation suggestions.

---

## License

Choose an appropriate license (e.g., MIT for code; consider CC0 for Kaggle submission artifacts if required by the competition rules).

---

## Current Status ✅

**Phase 1 - Setup: COMPLETED**
- ✅ Repository structure implemented as specified
- ✅ Python package with proper module organization
- ✅ Core components implemented:
  - `unicode_mutators.py` - Tokenizer/normalizer divergence finder
  - `embedding_optimizer.py` - White-box embedding-space optimization
  - `scaffolder.py` - Role and metadata scaffolding
  - `verifier.py` - Attack success and safety evaluation
  - `novelty_detector.py` - Duplicate detection vs public corpora
  - `search_controller.py` - Evolutionary search orchestration
- ✅ Configuration system with YAML configs
- ✅ Demo scripts and Jupyter notebook
- ✅ Docker containerization
- ✅ Findings export in Kaggle competition schema

**Ready for Phase 2: Component Integration & Testing**

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run toy demo (safe)
python -m tmrt.demo --mode full

# 3. Test individual components
python -m tmrt.demo --mode unicode    # Unicode mutations only
python -m tmrt.demo --mode scaffold   # Role scaffolding only

# 4. Run Jupyter demo
cd notebooks && jupyter notebook demo_toy_model.ipynb

# 5. Docker version
docker build -t tmrt ./docker
docker run -v $(pwd)/outputs:/app/outputs tmrt demo-toy
```

## Next Immediate Steps

1. **Test with GPT-2**: Validate all components work with actual model
2. **Run normalization sweep**: Use `configs/normalization_sweep.yaml`
3. **Configure GPT-OSS-20b**: Update configs for target model
4. **Execute full evolutionary search**: Run production experiments
5. **Generate findings**: Export results in competition format

**Competition Deadline**: August 26, 2025 (17 days remaining)

---

**Author:** Muzan Sano (NorthernTribe / TMRT)

**Experiment ID**: Will be auto-generated for each run  
**Framework Status**: ✅ Fully Implemented - Ready for Testing

Run `python -m tmrt.demo --help` to get started!
# Token-Manifold-Red-Teaming-TMRT-
