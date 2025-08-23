TMRT — OpenAI Competition Dataset

This folder contains the TMRT submission package prepared for the OpenAI hackathon/Kaggle competition.

Contents
- `dataset-metadata.json` — Dataset metadata used for Kaggle publishing (includes long project description).
- `openai-competition00.ipynb` — Local placeholder notebook linking to the Kaggle notebook URL: https://www.kaggle.com/code/muzansano/openai-competition00
- `issues/` — Per-issue JSON files (tmrt_001.json ... tmrt_011.json)
- `kaggle_submission.json` — Primary competition manifest copied into the dataset folder for convenience.
- `WRITEUP.md` — Draft writeup for the competition (if present) and other supporting docs.

How to run the demo notebook (safe smoke-test):
1. Create and activate a Python virtualenv in the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Open and run the notebook `openai-competition00.ipynb` in Jupyter. The notebook contains a small, non-ML smoke-test that imports lightweight helper modules from `src/tmrt`. If you get import errors for packages like `ftfy` or `deap`, install missing dependencies listed in `requirements.txt`.

Notes
- The `openai-competition00.ipynb` file is a placeholder because the public Kaggle kernel page could not be fetched programmatically; if you have the original kernel notebook, replace this file with the full notebook JSON.
- Competition submission attempts previously hit a Kaggle platform block ("Evaluation system not configured"). If submission fails, wait for organizers or schedule retries.

Contact
- See repository root for the full TMRT source and submission package.
