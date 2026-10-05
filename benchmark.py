"""
Tabular benchmark: do tree ensembles still beat neural nets on small tabular data?

Nested cross-validation with an equal random-search budget for every model,
ROC-AUC as metric, optional experiments:
  --noise K      add K uninformative Gaussian features (robustness test)
  --subsample N  cap each dataset at N rows (dataset-size test)

Usage:
  python benchmark.py --source openml --seeds 3 --budget 15
  python benchmark.py --source builtin --seeds 1 --budget 3      # quick offline test
"""
import argparse, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, friedmanchisquare
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# OpenML data ids (binary classification, small). Check each id exists on openml.org.
OPENML_IDS = {"credit-g": 31, "diabetes": 37, "blood-transfusion": 1464,
              "banknote": 1462, "phoneme": 1489, "ilpd": 1480, "wdbc": 1510, "kc1": 1067}


def load_openml(name):
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=OPENML_IDS[name], as_frame=True, parser="auto")
    return d.data, pd.Series(LabelEncoder().fit_transform(d.target))


def load_builtin():
    from sklearn.datasets import load_breast_cancer, load_digits, load_wine
    out = {}
    d = load_breast_cancer(as_frame=True); out["breast-cancer"] = (d.data, d.target)
    d = load_wine(as_frame=True); out["wine(0-vs-rest)"] = (d.data, (d.target == 0).astype(int))
    d = load_digits(as_frame=True); out["digits(even-vs-odd)"] = (d.data, (d.target % 2).astype(int))
    return out


def add_noise(X, k, seed):
    rng = np.random.RandomState(seed)
    X = X.copy()
    for i in range(k):
        X[f"noise_{i}"] = rng.normal(size=len(X))
    return X


def preprocessor():
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older scikit-learn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)])
    return ColumnTransformer([
        ("num", num, make_column_selector(dtype_include=np.number)),
        ("cat", cat, make_column_selector(dtype_exclude=np.number)),
    ])


def models(seed):
    def pipe(clf):
        return Pipeline([("pre", preprocessor()), ("clf", clf)])
    return {
        "LogReg": (pipe(LogisticRegression(max_iter=2000)),
                   {"clf__C": loguniform(1e-3, 1e2)}),
        "RandomForest": (pipe(RandomForestClassifier(n_jobs=1, random_state=seed)),
                         {"clf__n_estimators": randint(100, 400),
                          "clf__max_depth": [None, 5, 10, 20],
                          "clf__min_samples_leaf": randint(1, 10),
                          "clf__max_features": ["sqrt", 0.5]}),
        "HistGB": (pipe(HistGradientBoostingClassifier(random_state=seed)),
                   {"clf__learning_rate": loguniform(0.01, 0.3),
                    "clf__max_leaf_nodes": randint(8, 64),
                    "clf__min_samples_leaf": randint(5, 50),
                    "clf__l2_regularization": loguniform(1e-4, 10)}),
        "MLP": (pipe(MLPClassifier(max_iter=500, early_stopping=True, random_state=seed)),
                {"clf__hidden_layer_sizes": [(64,), (128,), (64, 64), (128, 64)],
                 "clf__alpha": loguniform(1e-5, 1e-1),
                 "clf__learning_rate_init": loguniform(1e-4, 1e-2)}),
    }


def evaluate(name, X, y, seed, budget, n_jobs=-1):
    rows = []
    outer = StratifiedKFold(5, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(outer.split(X, y)):
        for mname, (est, space) in models(seed).items():
            t0 = time.time()
            search = RandomizedSearchCV(est, space, n_iter=budget, cv=3, scoring="roc_auc",
                                        random_state=seed, n_jobs=n_jobs)
            search.fit(X.iloc[tr], y.iloc[tr])
            auc = roc_auc_score(y.iloc[te], search.predict_proba(X.iloc[te])[:, 1])
            rows.append(dict(dataset=name, model=mname, seed=seed, fold=fold,
                             auc=auc, seconds=time.time() - t0))
        print(f"  {name} seed={seed} fold={fold} done", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["openml", "builtin"], default="openml")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--budget", type=int, default=15, help="random-search iterations per model")
    p.add_argument("--noise", type=int, default=0)
    p.add_argument("--subsample", type=int, default=0)
    p.add_argument("--datasets", default="", help="comma-separated subset, e.g. credit-g,diabetes")
    p.add_argument("--n-jobs", type=int, default=2, help="parallel workers (lower = less RAM/temp disk)")
    p.add_argument("--out", default="results")
    a = p.parse_args()

    pick = [d for d in a.datasets.split(",") if d]
    if a.source == "builtin":
        data = {n: v for n, v in load_builtin().items() if not pick or n in pick}
    else:
        data = {n: load_openml(n) for n in OPENML_IDS if not pick or n in pick}
    rows = []
    for name, (X, y) in data.items():
        X = X.reset_index(drop=True); y = y.reset_index(drop=True)
        if a.subsample and len(X) > a.subsample:
            idx = np.random.RandomState(0).choice(len(X), a.subsample, replace=False)
            X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
        if a.noise:
            X = add_noise(X, a.noise, 0)
        for seed in range(a.seeds):
            rows += evaluate(name, X, y, seed, a.budget, a.n_jobs)

    df = pd.DataFrame(rows)
    Path(a.out).mkdir(exist_ok=True)
    tag = f"noise{a.noise}_sub{a.subsample}"
    df.to_csv(f"{a.out}/raw_{tag}.csv", index=False)
    table = df.groupby(["dataset", "model"]).auc.mean().unstack()
    table.to_csv(f"{a.out}/mean_auc_{tag}.csv")
    print("\nMean ROC-AUC per dataset:\n", table.round(4))
    ranks = table.rank(axis=1, ascending=False).mean().sort_values()
    print("\nAverage rank (1 = best):\n", ranks.round(2))
    if table.shape[0] >= 3:
        stat, pval = friedmanchisquare(*[table[m] for m in table.columns])
        print(f"\nFriedman test: chi2={stat:.3f}, p={pval:.4f}  (few datasets -> low power)")
    print("\nMean fit time (s):\n", df.groupby("model").seconds.mean().round(1))


if __name__ == "__main__":
    main()
