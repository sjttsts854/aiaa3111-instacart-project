# Shopping History and Candidate Selection

AIAA3111 data-mining final project using Instacart next-basket repeat purchases. The study separates the history used to score products from the historical products allowed to compete for recommendation slots.

## Final deliverables

The English closeout is in `final/`:

- [Single-column report PDF](final/Report_SingleColumn.pdf) and [editable LaTeX source](final/latex/main.tex): the latest paper-format report.
- [Overleaf-ready LaTeX ZIP](final/Report_SingleColumn_LaTeX.zip) and [figure bundle](final/Report_Figures.zip): downloadable source and illustrations.
- `Report_Draft.docx` and `Report_Draft.pdf`: earlier eight-page Word-format research report.
- `Poster_Draft.svg` and `Poster_Draft.pdf`: editable A1 landscape academic poster.
- `Defense_Guide.docx` and `Defense_Guide.pdf`: opening script, poster walkthrough, questions and answers, and team review checklist.
- [Portable project ZIP](final/Group_PENDING_Project_DRAFT.zip): source, evidence, demo, the earlier eight-page Report.pdf, README and data-download instructions.
- `START_HERE.html`: entry point for the deliverables and remaining team details.

An editable **single-column LaTeX paper** is also available in `final/latex/`.
Upload `final/Report_SingleColumn_LaTeX.zip` to Overleaf with `main.tex` as the
main document, or compile it with a local TeX distribution. The bundle includes
all sections, BibTeX references, and four vector PDF figures with high-resolution
PNG copies. `final/Report_Figures.zip` contains the figures separately, and
`final/Report_SingleColumn.pdf` is the locally compiled preview. This uses the
standard `article` class, not a conference-specific or two-column style.

On a GitHub ZIP file page, use **Download raw file** to save the archive. Extract the portable project ZIP to a separate folder before following its README; its extracted `final/package/` working copy is not stored in Git.

These are team review drafts. Group number, names, IDs and actual student contributions are pending. Review the separate course template if one is issued, check the AI-use disclosure, and finalize the team's own interpretation before submission. Repository sharing does not constitute course submission or peer-reviewed publication.

## Main finding

The value of old purchases depends on scoring, candidate eligibility and recommendation depth. On 2,526 long-history confirmation users, expanding candidates from the last five orders to full history changed Recall@10 by **−3.29 percentage points for cumulative frequency** and **+0.47 points for time-decayed frequency**, under fixed full-history scores and fixed targets.

A validation-selected policy retained products from the last five orders plus up to five older frequent products. On 8,000 new confirmation users it reduced total candidate volume by **44.79%**, from 64.92 to 35.85 candidates per user. Among 7,472 users with nonempty repeat targets, selected and full-candidate Recall@10 were **58.171% and 58.164%**. The simultaneous interval for the difference was **[−0.066, +0.071] percentage points**, meeting the prespecified −0.5-point noninferiority margin without establishing superiority.

The posthoc follow-up explains the boundary: 99.45% of eligible users had unchanged mean recall; only 0.26% of excluded true targets originally ranked in the full model's Top 10, while 95.54% ranked after 20. At K = 30 and 50, losses grew to 1.33 and 4.03 points. These diagnostics reuse the confirmation cohort and are not new independent confirmation.

Candidate filtering occurred after full model scoring. No runtime or storage saving was measured, no causal claim about consumer memory is made, and first-time product discovery is outside the repeat-target task.

## Evidence and source map

| Location | Contents |
| --- | --- |
| `basketlab/`, `research/` | Initial methods and the first confirmation study |
| `study3/` | TIFU-KNN adaptation, history diagnostics and quota reranking |
| `study4/` | Frozen scoring-history and candidate-policy study |
| `study4_followup/` | Posthoc user, rank and list-depth analysis |
| `reports/tables/`, `reports/study3/tables/` | Earlier aggregate evidence |
| `reports/study4/tables/` | Main confirmation results and observed-history diagnostics |
| `reports/study4_followup/tables/` | Posthoc evidence |
| `closeout/` | English artifact and portable package builders |
| `final/Group_PENDING_Project_DRAFT.zip` | Portable package, reproduction instructions and English evidence demo; extract before use |

Older research narrative files remain in their original language for traceability. The new report, poster, rehearsal guide, package README and evidence app are English.

## Quick evidence demo

Download and extract [the portable project ZIP](final/Group_PENDING_Project_DRAFT.zip), then use its README for the minimal demo installation, integrity check and launch commands. The app uses frozen Study 4 predictions and includes all 8,000 confirmation users. Individual cases show seed 42; report aggregates average per-user metrics across three seeds. It performs no live training.

The existing root `app.py` and `run_demo.ps1` are historical research interfaces, including older Study 2 cases. Use `evidence_app.py` in the extracted portable package for the final English demonstration.

## Reproduction and checks

The extracted package's `README.md`, `DATA.md` and `REPRODUCE.md` give the exact commands for clean data preparation and a research rerun. Raw CSV files, model caches, the original frozen run directory and the Python environment are excluded from both this repository and the portable archive. The data source is the [Instacart Kaggle mirror](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset). Extract its six original CSVs to `archive/` as described in the package.

On this original workspace, full data and frozen runs are available. Audit them without retraining:

```powershell
.\.venv\Scripts\python.exe -m study4.run audit
.\.venv\Scripts\python.exe -m study4_followup.run audit
.\.venv\Scripts\python.exe scripts/check_demo.py --require-study4 --require-followup
.\.venv\Scripts\python.exe -m pytest -q tests study3_tests study4_tests followup_tests
```

Do not reuse original hashes as a promise that a clean rerun will produce binary-identical model files. The portable package verifier checks shipped evidence and recomputes demo metrics; the original audit additionally needs raw data, models and complete caches.

## Research integrity and scope

All policies use the same full-history repeat target. Empty-target users retain their place in counts and precision/F1, while their recall is undefined. Global features use training users only. Study 4 uses 20,000 training, 5,000 validation and 8,000 confirmation users, excluding the 76,000 users used earlier. Three-seed results are averages of user metrics, not score ensembles. Nine primary contrasts use 6,000 paired bootstrap resamples with Bonferroni simultaneous intervals. Follow-up list-depth contrasts use a separate family of twelve and remain posthoc.

The independent TIFU-KNN implementation documents its source semantics and deviations in `study3/tifu_sources.md`; it is not a reproduction of the published benchmark score. Verified references for the final artifacts are stored in `closeout/references.json`.
