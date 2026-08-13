# Stratified Cross-Validation for Algorithm Comparison

Reproducible code and JMLR paper for the empirical study:

> **Do We Need Stratified Cross-Validation to Compare Classification Algorithms?**  
> A repeated 5-fold study across 13 OpenML datasets.

The experiment compares shuffled Regular K-Fold and Stratified K-Fold when
ranking Logistic Regression and Decision Tree classifiers with Macro-F1.

## Main result

- 13 OpenML classification datasets
- 2 algorithms, 2 split protocols, 10 repeats, and 5 folds
- 2,600 model fits in total
- 0 of 13 datasets changed the winning algorithm
- Wilcoxon test for the LR-DT gap change: `p = 0.946`
- Median reduction in fold class-composition variability: `93.4%`

Stratification made class proportions across folds much more predictable, but
it did not significantly improve mean Macro-F1 or change the LR-DT ranking in
this benchmark.

## Repository contents

| Path | Description |
| --- | --- |
| `reproduce_experiment.py` | Downloads the fixed OpenML datasets and runs the 2,600 fits. |
| `requirements.txt` | Python dependencies and recorded core versions. |
| `results/reported_gap_results.csv` | Dataset metadata and the LR-DT gaps reported in the paper. |
| `results/reported_summary.json` | Main reported statistical results and compute use. |
| `figures/generate_figures.py` | Regenerates both vector figures from reported values. |
| `paper.tex` | Complete paper in the supplied JMLR course template. |
| `references.bib` | Bibliography used by the paper. |
| `Stratified_CV_JMLR_Report.pdf` | Compiled submission. |

## Reproduce the experiment

Python 3.12 is recommended. Create an isolated environment and install the
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the complete benchmark with four parallel workers:

```bash
python reproduce_experiment.py --jobs 4 --output results/reproduced
```

OpenML access requires an internet connection. The output directory contains:

- one row per model fit and test fold in `fold_results.csv`;
- fold-composition measurements and dataset metadata;
- aggregate and repeat-level scores;
- LR-DT comparison gaps and ranking-flip indicators;
- statistical tests and a compute summary in JSON format.

The wall-clock time depends on the machine and OpenML cache. The reported run
used four workers and completed in approximately 106 seconds after data access.

## Regenerate figures and paper

```bash
python figures/generate_figures.py
latexmk -pdf paper.tex
```

The compiled document is `paper.pdf`. A TeX Live installation containing
`latexmk`, `pdflatex`, and BibTeX is required.

## Quick validation

The smoke tests use synthetic data and do not access OpenML:

```bash
python -m unittest discover -s tests -v
```

## Reproducibility notes

- Datasets are identified by fixed OpenML data IDs.
- Repeat `r` uses random seed `235975 + r`.
- Preprocessing is fitted inside each training fold to prevent leakage.
- The same fold is reused for both algorithms within a protocol.
- No class weighting, resampling, feature selection, or hyperparameter tuning
  is applied.

## License

The code is released under the MIT License. The paper remains attributed to
Ibrahim Abd Elhadi.
