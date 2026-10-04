"""
train.py
--------
Trains and compares machine-learning models that classify URLs as phishing
or legitimate, using the lexical and domain-based features in features.py.

Usage:
    python train.py                      # uses the CSV inside data/
    python train.py --sample 50000       # quicker run on a random subset
    python train.py --data path/to/file.csv
    python train.py --phishing-label 1   # if numeric labels use 1 = phishing

Outputs (written to models/):
    model.joblib          the best model
    metrics.json          scores, confusion matrices, ROC curves, analytics
    analytics_sample.csv  sample of URLs with features, for the dashboard
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import FEATURE_NAMES, extract_features_df, get_tld

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RANDOM_STATE = 42
DASHBOARD_SAMPLE_SIZE = 20000

URL_COLUMNS = ("url", "urls", "website", "link")
LABEL_COLUMNS = ("label", "labels", "status", "type", "result", "class", "phishing")
PHISHING_WORDS = {"bad", "phishing", "phish", "malicious", "fraud"}
LEGIT_WORDS = {"good", "legitimate", "legit", "benign", "safe"}


def find_dataset(path_arg):
    """Return the dataset path: the --data argument, or the largest CSV in data/."""
    if path_arg:
        path = Path(path_arg)
        if not path.exists():
            raise SystemExit(f"Dataset not found: {path}")
        return path
    candidates = sorted(
        DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True
    )
    if not candidates:
        raise SystemExit(
            "No CSV file found in the data/ folder.\n"
            "Download the dataset, unzip it and put the CSV inside data/."
        )
    return candidates[0]


def load_dataset(path, phishing_label):
    """Load a CSV and return a DataFrame with columns: url, is_phishing (1/0)."""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)

    columns = {str(c).strip().lower(): c for c in df.columns}
    url_col = next((columns[k] for k in URL_COLUMNS if k in columns), None)
    label_col = next((columns[k] for k in LABEL_COLUMNS if k in columns), None)
    if url_col is None or label_col is None:
        raise SystemExit(
            "Could not find the URL and label columns.\n"
            f"Columns in the file: {list(df.columns)[:20]}"
        )

    raw = df[label_col]
    if pd.api.types.is_numeric_dtype(raw):
        if phishing_label is None:
            # The PhiUSIIL dataset uses 1 = legitimate and 0 = phishing.
            phishing_label = 0 if "urlsimilarityindex" in columns else 1
        print(f"      numeric labels: treating {phishing_label} as phishing")
        target = (raw == phishing_label).astype(float)
    else:
        text = raw.astype(str).str.strip().str.lower()
        target = pd.Series(np.nan, index=df.index)
        target[text.isin(PHISHING_WORDS)] = 1.0
        target[text.isin(LEGIT_WORDS)] = 0.0

    data = pd.DataFrame(
        {"url": df[url_col].astype(str).str.strip(), "is_phishing": target}
    )
    data = data.dropna().drop_duplicates(subset="url")
    data = data[data["url"].str.len() > 0]
    data["is_phishing"] = data["is_phishing"].astype(int)
    if data["is_phishing"].nunique() < 2:
        raise SystemExit(
            "Only one class was found after reading the labels. "
            "Try the --phishing-label option."
        )
    return data.reset_index(drop=True)


def build_models():
    """Return the models to compare, keyed by display name."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200, random_state=RANDOM_STATE
        ),
    }


def evaluate(name, model, X_train, X_test, y_train, y_test):
    """Fit one model and return its test-set metrics as a dict."""
    start = time.time()
    model.fit(X_train, y_train)
    seconds = time.time() - start

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 200)).astype(int))

    result = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "train_seconds": round(seconds, 2),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "roc": {"fpr": fpr[keep].tolist(), "tpr": tpr[keep].tolist()},
    }
    print(
        f"      {name:22s} accuracy={result['accuracy']:.4f}  "
        f"f1={result['f1']:.4f}  ({seconds:.1f}s)"
    )
    return result


def to_native(value):
    """Convert numpy values so json.dump can write them."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialise {type(value)}")


def main():
    """Run the full pipeline: load, extract features, train, evaluate, save."""
    parser = argparse.ArgumentParser(description="Train phishing URL classifiers")
    parser.add_argument("--data", help="path to the dataset CSV")
    parser.add_argument("--sample", type=int, default=0,
                        help="use only this many random URLs (0 = all)")
    parser.add_argument("--phishing-label", type=int, default=None,
                        dest="phishing_label",
                        help="numeric label value that means phishing")
    args = parser.parse_args()

    path = find_dataset(args.data)
    print(f"[1/5] Loading {path.name}")
    data = load_dataset(path, args.phishing_label)
    if args.sample and args.sample < len(data):
        data = data.sample(args.sample, random_state=RANDOM_STATE).reset_index(drop=True)
    total = len(data)
    n_phishing = int(data["is_phishing"].sum())
    print(f"      {total:,} URLs  ({n_phishing:,} phishing, "
          f"{total - n_phishing:,} legitimate)")

    print("[2/5] Extracting features (this can take a minute)")
    start = time.time()
    X = extract_features_df(data["url"])
    y = data["is_phishing"].to_numpy()
    data["tld"] = [get_tld(u) for u in data["url"]]
    print(f"      {len(FEATURE_NAMES)} features per URL, "
          f"done in {time.time() - start:.1f}s")

    print("[3/5] Training and evaluating models (80% train / 20% test)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    models = build_models()
    results = {}
    for name, model in models.items():
        results[name] = evaluate(name, model, X_train, X_test, y_train, y_test)
    best_name = max(results, key=lambda n: results[n]["f1"])

    print("[4/5] Building analytics")
    forest = models["Random Forest"]
    importance = dict(
        sorted(
            zip(FEATURE_NAMES, (float(v) for v in forest.feature_importances_)),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )
    means = X.groupby(y).mean()
    feature_means = {
        f: {"legitimate": float(means.loc[0, f]), "phishing": float(means.loc[1, f])}
        for f in FEATURE_NAMES
    }
    tld_table = (
        data.groupby("tld")["is_phishing"]
        .agg(total="count", phishing="sum")
        .sort_values("total", ascending=False)
        .head(15)
    )
    tld_table["legitimate"] = tld_table["total"] - tld_table["phishing"]
    tld_table["phishing_rate"] = tld_table["phishing"] / tld_table["total"]
    tld_stats = tld_table.reset_index().to_dict("records")

    sample_idx = data.sample(
        min(DASHBOARD_SAMPLE_SIZE, total), random_state=RANDOM_STATE
    ).index
    sample = pd.concat([data.loc[sample_idx], X.loc[sample_idx]], axis=1)

    print("[5/5] Saving outputs to models/")
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"model": models[best_name], "name": best_name, "feature_names": FEATURE_NAMES},
        MODEL_DIR / "model.joblib",
        compress=3,
    )
    sample.to_csv(MODEL_DIR / "analytics_sample.csv", index=False)
    metrics = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": {
            "file": path.name,
            "total": total,
            "phishing": n_phishing,
            "legitimate": total - n_phishing,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "num_features": len(FEATURE_NAMES),
        },
        "best_model": best_name,
        "models": results,
        "feature_importance": importance,
        "feature_means": feature_means,
        "tld_stats": tld_stats,
    }
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=to_native)

    print("\nResults on the test set")
    print(f"{'Model':22s} {'Accuracy':>9s} {'Precision':>10s} {'Recall':>8s} "
          f"{'F1':>8s} {'ROC-AUC':>8s}")
    for name, r in results.items():
        marker = "  <- best" if name == best_name else ""
        print(f"{name:22s} {r['accuracy']:9.4f} {r['precision']:10.4f} "
              f"{r['recall']:8.4f} {r['f1']:8.4f} {r['roc_auc']:8.4f}{marker}")
    print("\nDone. Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
