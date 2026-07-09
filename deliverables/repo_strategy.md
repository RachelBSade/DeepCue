# DeepCue — Repository Organization & Polishing Strategy

**Goal:** one repository, two audiences. The professor finds the presentation, one-pager, and Kaggle notebooks (with cell outputs) in under 30 seconds from a link that never changes. The startup opens the repo and sees an engineering system, not an experiment dump.

**Strategy in one line:** restructure on an `academic-submission` branch → PR → merge to `main` → tag `v1.0-submission` → send the professor the tag URL. `main` keeps evolving; the tag never moves.

> All git commands here are for **you** to run. Nothing has been executed against your repo.

---

## 1. The Ultimate Directory Structure (`main` after merge)

Grounded in what exists today: `training_scripts/` mixes `.py` and `.ipynb`; academic files live in `deliverables/`; `scripts/` and `reports/` are empty; `venv/`, `checkpoints/`, `ignore/` are local-only.

```
DeepCue/
├── README.md                      ← dual-target (outline in §3)
├── LICENSE                        ← MIT; startups check for this first
├── .gitignore                     ← already good; add ~$*.pptx (Office lock files)
├── requirements.txt               ← inference/backend deps (pinned)
├── requirements-train.txt         ← GPU training deps (rename of requirements_kaggle.txt)
├── deepcue_workflow_checklist.md  ← stays at root (project source of truth)
│
├── docs/
│   ├── academic/                  ← 🎓 everything the professor needs, ONE folder
│   │   ├── deepcue_presentation.pptx
│   │   ├── deepcue_presentation.pdf
│   │   ├── one_page_summary.md    ← rename of project_summary.md
│   │   ├── DeepCue_OnePage.pdf
│   │   └── evaluation_graphs/     ← 3 PNGs + generate_evaluation_graphs.py
│   ├── architecture.md            ← system diagram + data-flow narrative
│   ├── DEPLOYMENT.md              ← moved from root
│   └── RESULTS.md                 ← moved from root; the run log IS research evidence
│
├── notebooks/                     ← 🎓 Kaggle notebooks, cell outputs PRESERVED
│   ├── README.md                  ← 5 lines: what each shows, run order
│   ├── 01_video_efficientnet_lstm.ipynb
│   ├── 02_audio_wav2vec2.ipynb
│   ├── 03_text_xlm_roberta.ipynb
│   ├── 04_fusion_transformer.ipynb   ← fixes the "deppcue" typo in transit
│   └── 05_export_and_evaluate.ipynb
│
├── training/                      ← 🏭 clean .py training/export scripts (Kaggle GPU)
│   ├── README.md                  ← moved from training_scripts/
│   ├── train_video_model.py
│   ├── train_audio_model.py
│   ├── finetune_xlm_roberta.py
│   ├── train_fusion_model.py
│   ├── evaluate_models.py
│   └── export_and_quantize.py
│
├── backend/                       ← 🏭 Django + Channels inference server (unchanged)
├── frontend/                      ← 🏭 HTML/JS SPA, no build step (unchanged)
└── models/                        ← README + .gitkeep only; .onnx gitignored,
                                     downloaded from the GitHub Release (§2.4)
```

Why it works: the professor's link lands on `docs/academic/` and `notebooks/` — zero code in the way. The engineer reads `training/`, `backend/`, and the README — zero `.pptx` in the way. Separating notebooks from scripts quietly signals you know the difference between exploration and production.

**Deliberate non-moves:** `backend/` and `frontend/` stay put — Django settings, `.env` model paths, and test imports depend on those paths; moving them buys polish nothing and risks breaking a working system. Delete the empty `scripts/` and `reports/` dirs.

**Cleanup flags found in the repo:**
- `deliverables/~$deepcue_presentation.pptx` is a PowerPoint lock file — remove it and add `~$*.pptx` to `.gitignore`.
- The `.gitignore` lines `.deepcue_workflow_checklist.md`, `.DEPLOYMENT.md`, `.RESULTS.md`, `.PROJECT.md` have a leading dot, so they match nothing. If ignoring them was the intent, drop the dots (and `git rm --cached`, since gitignore doesn't untrack); if they should be public, delete those lines.

---

## 2. Deep Learning Core Polish Strategy (4 steps)

**2.1 — Add CI that proves the tests pass (the #1 trust signal).**
Your test suite already runs with no live Redis/MongoDB/ONNX (`deepcue_backend.settings.test`, eager Celery, in-memory channel layer) — it's *made* for CI. Add `.github/workflows/ci.yml`: checkout → Python 3.12 → `pip install -r requirements.txt` → `cd backend && python -m pytest -q`. Put the green badge at the top of the README. A passing-CI badge is the difference between "claims to work" and "demonstrably works."

**2.2 — Split and pin requirements, with comments on the non-obvious pins.**
`requirements.txt` = inference/backend only; `requirements-train.txt` = GPU training. Pin exact versions and annotate the landmines — e.g. `redis==4.6.0  # 5.x sends CLIENT SETINFO, breaks channels-redis`. Annotated pins tell an engineer you debugged your dependency tree instead of copy-pasting it. State Python 3.12 explicitly (README + `requires-python` in a small `pyproject.toml` that also configures `ruff` — one lint config file signals a maintained codebase).

**2.3 — Write `docs/architecture.md` around your engineering constraints, not just boxes and arrows.**
Move the ASCII diagram there and frame the design as constraint-driven: <10s end-to-end latency on a weak Windows CPU (hence quantized ONNX, per-modality Celery queues, no inline inference in the consumer), macro F1 ≥ 0.50 on RAVDESS/CMU-MOSI (link `docs/RESULTS.md` + the evaluation graphs), and graceful degradation (`NEUTRAL_FALLBACK` instead of crashing a live session). Startups hire people who design *to* constraints; a document proving you did is worth more than any diagram.

**2.4 — Include the "trust files" and make the model weights reproducibly obtainable.**
`LICENSE` (MIT), `.env.example` (you have it — keep it in lockstep with `.env` keys), `models/README.md` explaining that `.onnx` weights are excluded from git and attached as **assets on the `v1.0-submission` GitHub Release** (better than Drive: versioned, permanent, same URL you send the professor), plus the four `VIDEO/AUDIO/TEXT/FUSION_MODEL_PATH` env vars that point at them. A stranger should get from `git clone` to a running session using only files in the repo.

---

## 3. Dual-Target README.md Outline

```markdown
# DeepCue 🎭
Real-time multimodal emotion recognition for Hebrew-speaking job interview candidates.

[CI badge] [Python 3.12] [License: MIT] [Release: v1.0-submission]

> One-paragraph pitch: 3 modalities → 8 emotions, live over WebSocket,
> <10s latency on CPU-only Windows, Hebrew RTL PDF report per session.

[30–60s demo GIF of a live session]

## 🎓 For Evaluators / Instructors
This project was submitted as **[v1.0-submission](…/tree/v1.0-submission)** —
a frozen snapshot unaffected by ongoing development.
| What | Where (in the tag) |
|---|---|
| Final presentation | docs/academic/deepcue_presentation.pptx |
| One-page summary | docs/academic/one_page_summary.md |
| Kaggle notebooks (EDA, training, outputs) | notebooks/ — read in order 01→05 |
| Results & evaluation graphs | docs/RESULTS.md · docs/academic/evaluation_graphs/ |

## 🏗️ System Architecture & Code
- Diagram (or link to docs/architecture.md) + 4-sentence data-flow narrative:
  browser landmarks/audio → WebSocket → Celery queues → per-modality ONNX
  pipelines → Redis → fusion → browser + MongoDB → PDF report
- Design constraints table: latency <10s CPU-only · macro F1 ≥ 0.50 · graceful
  degradation to neutral on pipeline failure
- Tech stack: Django Channels, Celery, Redis, MongoDB/GridFS, ONNX Runtime,
  MediaPipe, ReportLab

## 🧠 Models
| Modality | Model | Trained on | Macro F1 |
|---|---|---|---|
(video / audio / text / fusion rows — numbers from RESULTS.md)
Weights: download from the Release assets; set the four *_MODEL_PATH vars in .env.

## 🚀 Quickstart
Prereqs (Python 3.12, Memurai, MongoDB) → clone → venv → pip install →
.env from .env.example → migrate → daphne + celery workers + frontend serve.

## 🧪 Testing
cd backend && python -m pytest -q   # no live Redis/MongoDB/ONNX needed

## 📁 Repository Map
One line per top-level folder (the §1 tree, abbreviated).

## 🔬 Research & Training
Notebooks in notebooks/ (exploration, with outputs) vs training/ (clean
scripts). Trained on Kaggle GPUs; backend only consumes quantized ONNX.

## Roadmap · License
Phase 10 (bidirectional AI interviewer) planned; MIT.
```

The professor section comes **before** architecture — the professor is the deadline; engineers scroll, professors shouldn't have to.

---

## 4. Step-by-Step Git Execution Commands

All from repo root, Windows PowerShell. Review each block before running.

**Step 0 — Safety net**
```powershell
git checkout main
git pull origin main
git status          # must be clean before restructuring
```

**Step 1 — Branch**
```powershell
git checkout -b academic-submission
```

**Step 2 — Restructure with `git mv` (preserves history)**
```powershell
# Targets must exist before git mv
mkdir docs, docs\academic, notebooks, training

# Academic deliverables
git mv deliverables/deepcue_presentation.pptx docs/academic/
git mv deliverables/deepcue_presentation.pdf  docs/academic/
git mv deliverables/project_summary.md        docs/academic/one_page_summary.md
git mv deliverables/DeepCue_OnePage.pdf       docs/academic/
git mv deliverables/evaluation_graphs         docs/academic/evaluation_graphs
git mv deliverables/generate_evaluation_graphs.py docs/academic/evaluation_graphs/
git mv deliverables/presentation_outline.md   docs/academic/

# Docs
git mv RESULTS.md    docs/
git mv DEPLOYMENT.md docs/

# Notebooks — renamed to show research order, outputs untouched
git mv training_scripts/deepcue-video-1.ipynb           notebooks/01_video_efficientnet_lstm.ipynb
git mv training_scripts/deepcue-audio-2.ipynb           notebooks/02_audio_wav2vec2.ipynb
git mv training_scripts/deepcue-text-3.ipynb            notebooks/03_text_xlm_roberta.ipynb
git mv training_scripts/deppcue-fusion-4.ipynb          notebooks/04_fusion_transformer.ipynb
git mv training_scripts/deepcue-export-evaluate-5.ipynb notebooks/05_export_and_evaluate.ipynb

# Training scripts
git mv training_scripts/README.md               training/
git mv training_scripts/train_video_model.py    training/
git mv training_scripts/train_audio_model.py    training/
git mv training_scripts/finetune_xlm_roberta.py training/
git mv training_scripts/train_fusion_model.py   training/
git mv training_scripts/evaluate_models.py      training/
git mv training_scripts/export_and_quantize.py  training/

# Requirements rename
git mv requirements_kaggle.txt requirements-train.txt
```
`git mv` only errors on untracked files (e.g. `__pycache__` inside `training_scripts/`) — those aren't in git anyway; delete leftovers by hand afterward. If PowerShell chokes on a path, quote it.

**Step 3 — Cleanup**
```powershell
# Remove the tracked PowerPoint lock file, if tracked
git rm --cached "deliverables/~$deepcue_presentation.pptx"
# then add to .gitignore:  ~$*.pptx
# Delete now-empty dirs (git doesn't track empty dirs)
Remove-Item deliverables, training_scripts, scripts, reports -Recurse -Force -ErrorAction SilentlyContinue
```

**Step 4 — New content, then commit**
Add LICENSE, README rewrite, `docs/architecture.md`, `notebooks/README.md`, `models/README.md`, `.github/workflows/ci.yml`, `pyproject.toml`. Then:
```powershell
git add -A
git status                       # verify moves show as "renamed:" not delete+add
git commit -m "Restructure repo: separate academic deliverables (docs/academic, notebooks) from production code (training/, backend/); add LICENSE, CI, architecture docs"
git push -u origin academic-submission
```
If `git status` shows delete+add instead of `renamed:`, don't panic — git detects renames at diff/log time too (`git log --follow <file>` still works).

**Step 5 — PR and merge**
```powershell
gh pr create --base main --head academic-submission --title "Repository restructure for v1.0 academic submission" --body "Separates academic deliverables from production code. No code-behavior changes; backend/ and frontend/ untouched."
```
(or open the PR in the GitHub UI). Before merging: CI green and `python -m pytest -q` passes locally from `backend/`. Merge with **"Create a merge commit"** or **"Rebase"** — avoid *squash*, which collapses the renames into one blob and weakens `--follow` history.

**Step 6 — The frozen academic tag (the crucial part)**
```powershell
git checkout main
git pull origin main
git tag -a v1.0-submission -m "Frozen snapshot for academic submission - July 2026"
git push origin v1.0-submission
```
An **annotated** tag (`-a`) records author, date, and message — the correct choice for a milestone.

**Step 7 — GitHub Release on the tag (recommended)**
```powershell
gh release create v1.0-submission --title "DeepCue v1.0 — Academic Submission" --notes "Frozen submission snapshot. Academic materials: docs/academic/ and notebooks/. Quantized model weights attached below." models/video/efficientnet_lstm.onnx models/audio/wav2vec2_classifier.onnx models/text/xlm_roberta_sentiment.onnx models/fusion/cross_modal_transformer.onnx
```
This gives the weights a permanent, versioned home (no Drive links) and the professor a human-readable landing page.

**Links to send the professor (permanent — tags are immutable refs):**
- Snapshot root: `https://github.com/RachelBSade/DeepCue/tree/v1.0-submission`
- Academic folder: `https://github.com/RachelBSade/DeepCue/tree/v1.0-submission/docs/academic`
- Notebooks: `https://github.com/RachelBSade/DeepCue/tree/v1.0-submission/notebooks`
- Release page: `https://github.com/RachelBSade/DeepCue/releases/tag/v1.0-submission`

Future work on `main` never moves these. (Just never force-delete/re-push the tag — that's the one way to break the promise.)
