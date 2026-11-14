# Grocery Memory Study

Team review draft for AIAA 3111. Group number, names, student IDs and individual contributions are pending. The report and poster require team review and final approval before submission. No student review or authorship contribution is implied by these generated draft materials.

The project studies two uses of purchase history: information used for scoring and products allowed to compete for recommendation slots. The main experiment preserves the same users and complete repeat-target denominator when candidates are removed. A validation-selected policy keeps products from the last five orders plus up to five older frequently purchased products.

Study 4 confirmation found a 44.79% reduction in total eligible candidates with essentially unchanged Top10 repeat recall. The prespecified noninferiority comparison passed, but superiority over the full pool or a same-budget recency control was not established. Later posthoc analysis found that most users kept the same list and that longer lists exposed the lost target coverage. These are offline findings, not evidence of actual runtime speedup, consumer forgetting or increased sales.

## What is included

- `Report.pdf`: the English report draft, when supplied with the completed closeout package.
- `evidence_app.py`: portable English viewer with actual Study 4 confirmation cases.
- `data_demo/`: compact frozen records for all 8,000 confirmation users. Cases use model seed 42 only.
- `evidence/`: selected result tables, figures, protocols and audit summaries. Aggregate learned-model metrics average three models, not prediction scores.
- `code/`: unchanged research implementation, configurations and unit tests.
- `DATA.md`: source download instructions and exact source-file hashes.
- `REPRODUCE.md`: a clean-room reproduction route and its limitations.
- `verify_package.py`: local artifact-integrity and case-metric verification.
- `check_package_demo.py`: an integration check for mixed repeat/new-product baskets, changed recommendations, improved and harmed cases, and empty-target cases.
- `provenance/`: copied records of original completed audits and freezes, provided as evidence rather than as executable caches.

`Poster.pdf` is delivered separately, matching the course guide's separate poster submission. Editable report and poster files remain in the closeout folder outside this lean source-and-evidence package. Original data, environments, model binaries and large intermediate caches are intentionally excluded.

## Quick start without raw data

Use Python 3.11–3.13. The recorded research environment used Python 3.13.5 and the locked package versions supplied under `code/`. The portable demo uses only pandas, pyarrow and Streamlit.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv\Scripts\python.exe verify_package.py
.\.venv\Scripts\python.exe -m streamlit run evidence_app.py
```

On macOS/Linux, substitute `.venv/bin/python` for `.venv\Scripts\python.exe`. Run commands from the unpacked package root. The viewer reads packaged results; it does not need Kaggle access and never fits a model.

Run the optional UI check with the same environment:

```powershell
.\.venv\Scripts\python.exe check_package_demo.py
```

The case viewer offers every confirmation user. Suggested cases are recorded in `data_demo/metadata.json`: user 17 illustrates the repeat/new-product scope, 1872 has a changed list, 4350 gains a hit, 31433 loses a hit, and 490 has no repeat target. Each is the smallest eligible anonymous ID for its condition; these examples are illustrative, not a representative sample. Reveal the next basket only after explaining that the recommendations used prior history, not the revealed basket. Lists were computed during the original frozen run.

## Read the evidence correctly

1. The task is repeat-item ranking. Recall@10 is not classification accuracy or whole-basket recall. New products are not modeled.
2. The full repeat-target set remains the denominator even when a policy excludes true targets. Recall is undefined for 528 users with empty repeat targets; they remain in other applicable metrics and the demo.
3. Primary confirmation intervals use the prespecified family of nine comparisons. Posthoc length analyses use a different family of twelve comparisons. They are not interchangeable.
4. The top-level confirmation tables average metrics within users across seeds 42, 73 and 123. Individual cases use seed 42. Do not compare a selected case's recall with the macro table as if both were the same aggregation.
5. The original source audits passed. This package's verifier checks packaged records and their integrity; it does not rerun the original raw-data audit or certify retraining equivalence.

## Before submission

Fill team details and contributions, review the English report and poster, verify references, and reconcile the layout with the official template and detailed rubric if provided. The team's final writing and oral defense must reflect its own understanding. The course guide requests AIGC disclosure and warns against directly submitting AI-generated work or fabricated references.

Codex assisted research design, implementation, experiment execution, analysis, visualization, draft writing and packaging. Automated tests and independent computational checks were executed; they do not replace human review or establish original algorithmic novelty. The team must complete its own review and contribution statement.
