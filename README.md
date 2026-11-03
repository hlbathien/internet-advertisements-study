# Internet Advertisements: a grouped statistical study

Ridge logistic regression and a constrained tree on the UCI Internet Advertisements dataset. The study evaluates ranking, probabilities, threshold decisions, and conditional uncertainty while keeping identical predictor signatures together.

- [English report](report/report.pdf)
- [R analysis](analysis/run.R)
- [Dataset](https://archive.ics.uci.edu/dataset/51/internet+advertisements), DOI [10.24432/C5V011](https://doi.org/10.24432/C5V011)

## Reproduce

Run from the repository root. Requirements: R with `glmnet` and `rpart`; Python 3 with NumPy, pandas, scikit-learn and Matplotlib; PDFLaTeX with the packages declared in `report/report.tex`.

```sh
Rscript analysis/run.R
python3 analysis/verify.py
python3 analysis/critique_diagnostics.py
python3 analysis/plot_calibration.py
Rscript analysis/export_models.R
python3 analysis/render_tables.py
cd report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

`run.R` owns model fitting and metric exports; `plot_calibration.py` owns comparative ranking, development and reliability figures; `export_models.R` extracts the actual coefficients, imputation values, worked prediction and fitted tree; `render_tables.py` generates the manuscript's empirical tables, comparator equation and worked prediction. Generated outputs are not hand-edited. Narrative descriptions summarize this fixed experiment; changing its design requires revising the scientific interpretation, not just rebuilding tables.

The recorded experiment used R 4.1.2, glmnet 4.1-3, rpart 4.1.16 and Matrix 1.4-0. Session and Python version details are in `analysis/results/`. Fixed allocation and bootstrap seeds identify this experiment; they do not establish robustness across allocations. The independent verifier checks source identity, group separation and custom R metric calculations against scikit-learn.

## Findings and scope

Full ridge test AP is about .863 versus .463 for geometry/locality ridge. The small tree has higher observed test F1 at .5 than ridge at its validation-selected threshold. One repeated nonad signature accounts for 15 of 25 ridge false positives. These findings do not establish current-web performance or a deployment-ready blocking policy.

Intervals condition on fitted models and a working independent-signature-cluster population. They do not estimate retraining variability or remove unknown website dependence. The final test has been inspected: later tuning requires fresh evaluation data. Original annotation procedures, raw pages/images, and site identifiers are unavailable in this package. Cover identities and contributions remain for the submitting group to supply.

## Sources and attribution

Kushmerick, N. (1999). *Internet Advertisements*. UCI Machine Learning Repository. https://doi.org/10.24432/C5V011. UCI lists CC BY 4.0 for the dataset. `data/add.csv` is the supplied indexed/headered representation; it matches `analysis/source/ad.data` numerically, including missingness and labels. Original dictionary/documentation are retained. The actual files contain 459 ads; source prose lists 458.

Original study: Kushmerick, N. (1999). *Learning to remove Internet advertisements*. Proceedings of the Third Annual Conference on Autonomous Agents, 175–181. https://doi.org/10.1145/301136.301186. Collection is dated April–July 1998 in the accompanying documentation. The historical >97% C4.5rules result is not directly comparable to this grouped evaluation. Detailed acquisition and annotation claims have not been inferred from bibliographic metadata.

The public repository intentionally excludes supplied student-example reports, agent/session files, internal audit history, and unrelated source-fetch caches. No code license is assigned without an explicit author choice; the dataset's stated license is separate from code licensing.
