"""Model registry (owner: P2 - Aditya).

make_model(cfg, seed) turns one entry of configs/models.yaml into an sklearn Pipeline:

    SimpleImputer(median) -> [StandardScaler] -> estimator

Everything that is *learned* from data (imputation medians, scaling means/stds, the model)
lives inside the Pipeline, so it is fitted on the training folds only and is saved together
with the model. The API then calls pipeline.predict_proba(X)[:, 1] - nothing else to apply.
"""
from pathlib import Path

import yaml
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _lightgbm(**p):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**p)


def _mlp(**p):
    if "hidden_layer_sizes" in p:
        p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
    return MLPClassifier(**p)


# estimator key -> (constructor, does it accept random_state?)
ESTIMATORS = {
    "logistic_regression": (LogisticRegression, True),
    "gaussian_nb": (GaussianNB, False),
    "knn": (KNeighborsClassifier, False),
    "extra_trees": (ExtraTreesClassifier, True),
    "random_forest": (RandomForestClassifier, True),
    "hist_gradient_boosting": (HistGradientBoostingClassifier, True),
    "lightgbm": (_lightgbm, True),
    "mlp": (_mlp, True),
}


def load_model_configs(path="configs/models.yaml"):
    with open(Path(path)) as f:
        return yaml.safe_load(f)


def make_model(cfg, seed=42, params_override=None):
    """Build an unfitted Pipeline from one model config dict."""
    ctor, takes_seed = ESTIMATORS[cfg["estimator"]]
    params = dict(cfg.get("params") or {})
    params.update(params_override or {})
    if takes_seed:
        params.setdefault("random_state", seed)
    steps = [("impute", SimpleImputer(strategy="median"))]
    if cfg.get("scale", False):
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", ctor(**params)))
    return Pipeline(steps)
