# Single-column LaTeX research report

This folder contains the English AIAA3111 report in single-column paper format, following the final formatting preference. It converts the completed study into an editable LaTeX article, with numbered equations, tables, figures and references. All results come from the saved experiments; no new training is required to compile the paper.

## Compile on Overleaf

1. Create a new Overleaf project using **Upload Project** and upload the supplied LaTeX ZIP. Keep the directory structure intact.
2. Set **Main document** to `main.tex` and **Compiler** to **pdfLaTeX**. A recent TeX Live version is recommended.
3. Select **Recompile**. Overleaf normally runs BibTeX and the necessary additional LaTeX passes automatically. If citations remain unresolved after an edit, use **Recompile from scratch**.
4. Replace the pending author and team information in `main.tex` and complete the team's review before removing draft wording.

The source uses a standard `article` document class and ordinary TeX Live packages. There is no custom conference style file to install. It needs no raw Instacart CSV files, Python installation, external model or shell-escape permission.

## Files

| File or folder | Purpose |
| --- | --- |
| `main.tex` | Main document, layout settings, title and author information |
| `sections/` | Editable paper sections, if split out from the main document |
| `references.bib` | Five references using BibTeX and the `plainnat` bibliography style |
| `figures/history_composition.pdf` | History-group comparison on common support |
| `figures/factorial_memory.pdf` | Scoring-history by candidate-eligibility experiment |
| `figures/policy_contrasts.pdf` | Selected-policy comparisons and noninferiority margin |
| `figures/list_depth.pdf` | Posthoc recall differences across recommendation lengths |
| `figures/*.png` | High-resolution previews of the corresponding vector PDF figures |
| `figures/README.txt` | Figure descriptions and source-data provenance |

The paper inserts the vector PDF figures directly, so plots remain sharp when printed or zoomed. The PNG copies are convenient for browsing or slides. Keep the original PDF filenames unless you also update the corresponding `\includegraphics` commands.

For example, a figure is inserted with:

```latex
\begin{figure}[tbp]
  \centering
  \includegraphics[width=\linewidth]{figures/factorial_memory.pdf}
  \caption{A caption describing the experiment and its population.}
  \label{fig:factorial}
\end{figure}
```

Refer to it in prose using `Figure~\ref{fig:factorial}`. Cite a source using, for example, `\citep{hu2020}`. The bibliography keys are `hu2020`, `li2023`, `ke2017`, `instacart2017` and `kaggle`.

## Compile locally

With TeX Live or MiKTeX and `latexmk` installed, run this command from the folder containing `main.tex`:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Alternatively, run the complete BibTeX sequence:

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

These commands produce `main.pdf`. If your local TeX distribution asks for a missing standard package, install it using that distribution's package manager. Overleaf normally supplies the required packages already.

## Editing and scientific scope

Main confirmation and later posthoc analyses remain explicitly separated. Preserve the distinctions among 8,000 confirmation users, 7,472 users with nonempty repeat targets, and the 2,526 long-history users used for the factorial experiment. Recall values are percentages; differences are reported in percentage points.

The original noninferiority interval at K = 10 belongs to the family of nine primary comparisons. The later K analysis uses twelve posthoc comparisons and therefore has a different interval at K = 10. Do not exchange these intervals or describe the follow-up as another independent confirmation.

The report makes an empirical contribution about scoring history, candidate competition and recommendation depth. Candidate-count reduction is not a measured speedup. Retain the limitations, AI-use disclosure and pending team review when adapting the draft. The team should supply actual names, IDs and contributions and check any separately issued course template before submission.

## Validation of this copy

The supplied source was compiled locally with Tectonic 0.17.0 (a XeTeX-based engine), including BibTeX and repeat passes. The resulting preview has 11 pages. Its equations, tables, figures and references were checked; there are no unresolved citations, missing characters or overfull boxes in the final log. A bibliography underfull-box warning does not affect content or readability. The PDF preview is delivered separately as `Report_SingleColumn.pdf`.

For Overleaf, use the pdfLaTeX setting above. The source uses packages supported by both pdfLaTeX and XeLaTeX; pagination can vary slightly with the TeX distribution and package versions. The local check was not an upload to Overleaf. The earlier Word report and source-and-evidence package remain separate deliverables.
