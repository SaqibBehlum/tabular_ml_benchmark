# tabular_ml_benchmark

I wanted to check whether tree models really beat neural networks on small tabular datasets, so I compared four models on eight datasets from OpenML. The models are logistic regression, random forest, histogram gradient boosting (from sklearn) and an MLP. Each model gets the same tuning budget inside nested cross validation, and I score them with ROC AUC.

I ran three experiments: the plain datasets, the same datasets with 20 random noise features added, and the datasets cut down to 300 rows each.

## What I found

Random forest had the best average rank on the plain datasets, but the difference was not significant and most of it came from one dataset (phoneme). With the noise features the MLP got clearly worse than the tree models. With only 300 rows logistic regression did best. The full write up with tables and figures is in REPORT.md.

## Running it

Install the packages first:

```
pip install -r requirements.txt
```

Then run the experiments:

```
python benchmark.py --source openml --seeds 3 --budget 15
python benchmark.py --source openml --seeds 3 --budget 15 --noise 20
python benchmark.py --source openml --seeds 3 --budget 15 --subsample 300
python analyze.py --results results --figures figures
```

The full run takes a long time on a laptop. If you only want to try it, use `--seeds 1 --budget 5`, or pick a few datasets with `--datasets credit-g,diabetes`. There is also a quick offline test with `--source builtin`.

## Files

- `benchmark.py` runs the experiments
- `analyze.py` makes the tables, tests and figures from the results folder
- `results/` has my output files
- `figures/` has the plots
- `REPORT.md` is the report

## Notes

This is a small study: only binary classification, eight datasets and a small search budget, so the results are not a final answer. The report is not peer reviewed.
