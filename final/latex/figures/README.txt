Paper figures for the single-column English LaTeX report

Use the PDF versions in LaTeX for vector text and graphics. Matching 400-dpi
PNG versions are supplied for slides, previews and other editors. These are
drawn directly from completed study tables; no model or policy is refitted.
All recall differences use percentage points (pp).

history_composition.pdf / .png
  Source: reports/study4/tables/history_sensitivity.csv
  Rows: candidate_only_min10, candidate_historical_basket_min10.
  Same-support raw and standardized differences with conditional marginal
  95% intervals. Dashed line: raw overall difference. Min-20 rows have the
  same support and estimates as min-10, not additional independent results.
  Inclusion is limited to nonempty-repeat-target users; this is not causal.

factorial_memory.pdf / .png
  Source: reports/study4/tables/factorial_metrics.csv
  Rows: subgroup=long_history; methods frequency and decay; scoring windows
  5/all crossed with candidate pools 5/all. Recall uses 2,526 nonempty users.
  The CSV users column counts all 2,590 long-history users, including 64
  empty-target users; these are not the recall denominator. Color scale is
  shared across panels (42 to 50 percent recall).

policy_contrasts.pdf / .png
  Source: reports/study4/tables/primary_contrasts.csv
  Rows: dual_vs_full, dual_vs_frequency_budget, dual_vs_recency_budget.
  Uses sim_low/sim_high from the ORIGINAL nine-contrast primary family.
  7,472 eligible users; metrics average seeds within user, not model scores.
  The -0.5 pp margin is a noninferiority criterion for selected versus full.

list_depth.pdf / .png
  Source: reports/study4_followup/tables/depth_contrasts.csv
  All 12 rows: selected versus full and versus recency budget at six K values.
  Uses sim_low/sim_high from the separate POSTHOC twelve-contrast family.
  Connected lines interpolate visually between tested K values only.
  The original -0.5 pp reference does not define a newly validated safe K.

Rebuild: .venv/Scripts/python.exe closeout/build_latex_figures.py
CSV SHA256 checksums are recorded in source_hashes.json.
