"""Load the feature tables, train the classifiers and score them on the test set."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from .feature_groups import BOOLEAN_FEATURES, GROUPS

TARGET = "silence_label"
SEED = 42
C_VALUES = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
META = ["film", "gap_sec", TARGET]


def load_features(data_dir: Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Train, validation and test tables. Continuous features are standardised with the training set's mean and deviation."""
    tables = {name: pd.read_csv(Path(data_dir) / f"features_{name}.csv.gz", dtype={"film": str})
              for name in ("train", "validation", "test")}
    features = [c for c in tables["train"].columns if c not in META]
    assert set(features) == set(sum(GROUPS.values(), []))
    continuous = [c for c in features if c not in BOOLEAN_FEATURES]
    scaler = StandardScaler().fit(tables["train"][continuous])
    for table in tables.values():
        table[continuous] = scaler.transform(table[continuous])
    return tables, features


def scores(y_true, y_pred, y_score) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred), "roc_auc": roc_auc_score(y_true, y_score),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "precision_0": precision_score(y_true, y_pred, pos_label=0), "recall_0": recall_score(y_true, y_pred, pos_label=0),
        "precision_1": precision_score(y_true, y_pred), "recall_1": recall_score(y_true, y_pred),
    }


def lasso(X, y, search: bool = False):
    """L1 logistic regression; the features it leaves at zero are dropped.

    The thesis chose C = 0.0336 from 20 values by 5-fold cross-validation. That search takes hours, so by default only the
    final model is trained, with that C.
    """
    settings = dict(penalty="l1", solver="saga", max_iter=10000, random_state=SEED, class_weight="balanced")
    if search:
        return LogisticRegressionCV(Cs=20, cv=5, scoring="accuracy", n_jobs=-1, **settings).fit(X, y)
    return LogisticRegression(C=np.logspace(-4, 4, 20)[6], **settings).fit(X, y)


def linear_svm(C: float) -> LinearSVC:
    return LinearSVC(C=C, class_weight="balanced", max_iter=10000, random_state=SEED)


def tune_linear_svm(X, y, wrap=lambda svm: svm):
    """Choose C by 5-fold cross-validation, then train on everything. `wrap` can put a vectoriser in front of the SVM."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best = max(C_VALUES, key=lambda C: cross_val_score(wrap(linear_svm(C)), X, y, cv=folds, scoring="accuracy").mean())
    return wrap(linear_svm(best)).fit(X, y), best


def with_tfidf(svm):
    return make_pipeline(TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True), svm)


def embedding_svm(kernel: str):
    """The two classifiers used on transformer embeddings (no search over settings)."""
    svm = LinearSVC(C=1.0, max_iter=5000, random_state=SEED) if kernel == "linear" else SVC(kernel="rbf", C=1.0, gamma="scale")
    return make_pipeline(StandardScaler(), svm)
