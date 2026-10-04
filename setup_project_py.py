"""
setup_project.py
----------------
Run this once, from the project folder that already contains features.py:

    python setup_project.py

It creates: train.py, app.py, README.md, .gitignore and the data/ folder.
"""

from pathlib import Path

FILES = {}

# ===========================================================================
# train.py
# ===========================================================================
FILES["train.py"] = r'''
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
'''

# ===========================================================================
# app.py
# ===========================================================================
FILES["app.py"] = r'''
"""
app.py
------
Streamlit dashboard for "Phishing URL Analytics and Classification".

Run:  streamlit run app.py      (after python train.py)

Pages:
    Overview           dataset summary and top-level-domain analytics
    Feature Analytics  how each feature differs between phishing and legitimate
    Model Performance  model comparison, confusion matrix, ROC, importance
    URL Checker        classify one URL and explain the warning signs
    Batch Check        classify many URLs and download the results
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from features import (
    DOMAIN_FEATURES,
    FEATURE_NAMES,
    LEXICAL_FEATURES,
    extract_features,
    extract_features_df,
    risk_flags,
)

MODEL_DIR = Path(__file__).resolve().parent / "models"
COLORS = {"Legitimate": "#2E9E6B", "Phishing": "#D9534F"}
MAX_BATCH = 5000

st.set_page_config(page_title="Phishing URL Analytics", layout="wide")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load the trained model bundle saved by train.py."""
    return joblib.load(MODEL_DIR / "model.joblib")


@st.cache_data
def load_metrics():
    """Load evaluation metrics and analytics saved by train.py."""
    with open(MODEL_DIR / "metrics.json", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data
def load_sample():
    """Load the sample of URLs with features used for the charts."""
    df = pd.read_csv(MODEL_DIR / "analytics_sample.csv")
    df["Class"] = df["is_phishing"].map({0: "Legitimate", 1: "Phishing"})
    return df


def predict(bundle, urls):
    """Return the phishing probability (0 to 1) for each URL."""
    X = extract_features_df(urls)[bundle["feature_names"]]
    return bundle["model"].predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview(metrics):
    """Dataset summary and top-level-domain analytics."""
    ds = metrics["dataset"]
    best = metrics["best_model"]

    st.title("Phishing URL Analytics and Classification")
    st.caption("Classifying URLs using lexical and domain-based features")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("URLs analysed", f"{ds['total']:,}")
    c2.metric("Phishing", f"{ds['phishing']:,}")
    c3.metric("Legitimate", f"{ds['legitimate']:,}")
    c4.metric("Best model accuracy",
              f"{metrics['models'][best]['accuracy'] * 100:.2f}%")
    st.caption(
        f"Dataset: {ds['file']} | {ds['num_features']} features per URL | "
        f"best model: {best} | trained {metrics['trained_at']}"
    )

    left, right = st.columns([1, 2])
    with left:
        fig = px.pie(
            names=["Legitimate", "Phishing"],
            values=[ds["legitimate"], ds["phishing"]],
            color=["Legitimate", "Phishing"],
            color_discrete_map=COLORS,
            hole=0.5,
            title="Class distribution",
        )
        st.plotly_chart(fig)
    with right:
        tld = pd.DataFrame(metrics["tld_stats"])
        fig = px.bar(
            tld,
            x="tld",
            y=["legitimate", "phishing"],
            color_discrete_map={
                "legitimate": COLORS["Legitimate"],
                "phishing": COLORS["Phishing"],
            },
            labels={"tld": "Top-level domain", "value": "URLs", "variable": "Class"},
            title="Most common top-level domains",
        )
        st.plotly_chart(fig)

    st.subheader("Phishing rate by top-level domain")
    table = tld.rename(columns={
        "tld": "TLD", "total": "Total", "phishing": "Phishing",
        "legitimate": "Legitimate", "phishing_rate": "Phishing rate (%)",
    })
    table["Phishing rate (%)"] = (table["Phishing rate (%)"] * 100).round(1)
    st.dataframe(
        table[["TLD", "Total", "Legitimate", "Phishing", "Phishing rate (%)"]],
        hide_index=True,
    )


def page_features(metrics, sample):
    """Compare feature values between phishing and legitimate URLs."""
    st.title("Feature Analytics")
    st.caption(f"Charts use a random sample of {len(sample):,} URLs.")

    group = st.radio("Feature group", ["Lexical", "Domain-based"], horizontal=True)
    options = LEXICAL_FEATURES if group == "Lexical" else DOMAIN_FEATURES
    feature = st.selectbox("Feature", options)

    means = metrics["feature_means"][feature]
    c1, c2 = st.columns(2)
    c1.metric("Average for legitimate URLs", f"{means['legitimate']:.3f}")
    c2.metric("Average for phishing URLs", f"{means['phishing']:.3f}")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(
            sample, x=feature, color="Class", barmode="overlay",
            histnorm="percent", nbins=40, opacity=0.65,
            color_discrete_map=COLORS, title=f"Distribution of {feature}",
        )
        fig.update_yaxes(title="% of URLs in class")
        st.plotly_chart(fig)
    with right:
        fig = px.box(
            sample, x="Class", y=feature, color="Class",
            color_discrete_map=COLORS, title=f"Spread of {feature}",
        )
        st.plotly_chart(fig)

    st.subheader("Which features move with the phishing label?")
    corr = (
        sample[FEATURE_NAMES]
        .corrwith(sample["is_phishing"])
        .fillna(0)
        .sort_values()
    )
    fig = px.bar(
        x=corr.values, y=corr.index, orientation="h",
        labels={"x": "Correlation with phishing (-1 to +1)", "y": ""},
        height=900,
    )
    st.plotly_chart(fig)
    st.caption(
        "Positive values are more common in phishing URLs, "
        "negative values in legitimate URLs."
    )

    st.subheader("Average value of every feature")
    rows = [
        {
            "Feature": f,
            "Group": "Lexical" if f in LEXICAL_FEATURES else "Domain-based",
            "Legitimate": round(metrics["feature_means"][f]["legitimate"], 3),
            "Phishing": round(metrics["feature_means"][f]["phishing"], 3),
        }
        for f in FEATURE_NAMES
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def page_models(metrics):
    """Model comparison, confusion matrix, ROC curves and feature importance."""
    st.title("Model Performance")
    results = metrics["models"]
    best = metrics["best_model"]
    ds = metrics["dataset"]
    st.caption(
        f"Trained on {ds['train_size']:,} URLs, tested on {ds['test_size']:,} "
        f"unseen URLs. Best model by F1 score: {best}."
    )

    table = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1 score": r["f1"],
            "ROC-AUC": r["roc_auc"],
            "Train time (s)": r["train_seconds"],
        }
        for name, r in results.items()
    ])
    st.dataframe(table.round(4), hide_index=True)

    score_cols = ["Accuracy", "Precision", "Recall", "F1 score", "ROC-AUC"]
    long = table.melt(id_vars="Model", value_vars=score_cols,
                      var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group",
                 title="Model comparison")
    fig.update_yaxes(range=[max(0.0, float(long["Score"].min()) - 0.05), 1.0])
    st.plotly_chart(fig)

    left, right = st.columns(2)
    with left:
        names = list(results)
        chosen = st.selectbox("Confusion matrix for", names, index=names.index(best))
        fig = px.imshow(
            results[chosen]["confusion_matrix"],
            x=["Predicted legitimate", "Predicted phishing"],
            y=["Actual legitimate", "Actual phishing"],
            text_auto=True,
            color_continuous_scale="Blues",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig)
    with right:
        fig = go.Figure()
        for name, r in results.items():
            fig.add_trace(go.Scatter(
                x=r["roc"]["fpr"], y=r["roc"]["tpr"], mode="lines",
                name=f"{name} (AUC {r['roc_auc']:.3f})",
            ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
            line=dict(dash="dash", color="gray"),
        ))
        fig.update_layout(
            title="ROC curves",
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
        )
        st.plotly_chart(fig)

    st.subheader("Most important features (Random Forest)")
    imp = pd.DataFrame(
        list(metrics["feature_importance"].items()),
        columns=["Feature", "Importance"],
    ).head(15)
    imp["Group"] = imp["Feature"].map(
        lambda f: "Lexical" if f in LEXICAL_FEATURES else "Domain-based"
    )
    fig = px.bar(imp, x="Importance", y="Feature", color="Group",
                 orientation="h", height=520)
    fig.update_yaxes(categoryorder="total ascending", title="")
    st.plotly_chart(fig)


def page_checker(bundle):
    """Classify a single URL and explain the result."""
    st.title("URL Checker")
    st.caption(
        "The prediction uses only the text of the URL. The page is never "
        "opened, so treat the result as a risk estimate, not a guarantee."
    )

    url = st.text_input("Enter a URL", placeholder="https://www.example.com")
    if not st.button("Analyse", type="primary"):
        return
    url = url.strip()
    if not url:
        st.warning("Please enter a URL first.")
        return

    probability = float(predict(bundle, [url])[0])
    if probability >= 0.5:
        st.error(f"Likely PHISHING  ({probability * 100:.1f}% phishing probability)")
    else:
        st.success(f"Likely LEGITIMATE  ({probability * 100:.1f}% phishing probability)")
    st.progress(min(max(probability, 0.0), 1.0))

    flags = risk_flags(url)
    st.subheader("Warning signs")
    if flags:
        for flag in flags:
            st.write(f"- {flag}")
    else:
        st.write("No common warning signs found in this URL.")

    def as_table(names, values):
        shown = [f"{values[n]:.3f}" if isinstance(values[n], float) else str(values[n])
                 for n in names]
        return pd.DataFrame({"Feature": names, "Value": shown})

    feats = extract_features(url)
    st.subheader("Extracted features")
    left, right = st.columns(2)
    with left:
        st.write("Lexical features")
        st.dataframe(as_table(LEXICAL_FEATURES, feats), hide_index=True)
    with right:
        st.write("Domain-based features")
        st.dataframe(as_table(DOMAIN_FEATURES, feats), hide_index=True)


def page_batch(bundle):
    """Classify many URLs at once and offer the results as a CSV download."""
    st.title("Batch Check")
    text = st.text_area("Paste URLs, one per line", height=200)
    uploaded = st.file_uploader("or upload a CSV with a 'url' column", type="csv")
    if not st.button("Classify", type="primary"):
        return

    urls = [line.strip() for line in text.splitlines() if line.strip()]
    if uploaded is not None:
        frame = pd.read_csv(uploaded)
        column = next(
            (c for c in frame.columns if str(c).strip().lower() == "url"), None
        )
        if column is None:
            st.error("The uploaded file has no column named 'url'.")
            return
        urls += frame[column].dropna().astype(str).tolist()
    if not urls:
        st.warning("Please paste or upload some URLs first.")
        return
    if len(urls) > MAX_BATCH:
        st.info(f"Only the first {MAX_BATCH:,} URLs were processed.")
        urls = urls[:MAX_BATCH]

    probabilities = predict(bundle, urls)
    out = pd.DataFrame({
        "URL": urls,
        "Phishing probability (%)": [round(float(p) * 100, 1) for p in probabilities],
        "Verdict": ["Phishing" if p >= 0.5 else "Legitimate" for p in probabilities],
        "Warning signs": [len(risk_flags(u)) for u in urls],
    })
    flagged = int((out["Verdict"] == "Phishing").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("URLs checked", f"{len(out):,}")
    c2.metric("Flagged as phishing", f"{flagged:,}")
    c3.metric("Looks legitimate", f"{len(out) - flagged:,}")
    st.dataframe(out, hide_index=True)
    st.download_button(
        "Download results as CSV",
        out.to_csv(index=False).encode("utf-8"),
        file_name="url_check_results.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    """Sidebar navigation and page routing."""
    needed = ["model.joblib", "metrics.json", "analytics_sample.csv"]
    if not all((MODEL_DIR / name).exists() for name in needed):
        st.error(
            "Trained model not found. Run `python train.py` first, "
            "then reload this page."
        )
        st.stop()

    bundle = load_model()
    metrics = load_metrics()
    sample = load_sample()

    st.sidebar.title("Phishing URL Analytics")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Feature Analytics", "Model Performance",
         "URL Checker", "Batch Check"],
    )
    st.sidebar.caption(f"Model in use: {metrics['best_model']}")

    if page == "Overview":
        page_overview(metrics)
    elif page == "Feature Analytics":
        page_features(metrics, sample)
    elif page == "Model Performance":
        page_models(metrics)
    elif page == "URL Checker":
        page_checker(bundle)
    else:
        page_batch(bundle)


main()
'''

# ===========================================================================
# README.md
# ===========================================================================
FILES["README.md"] = r'''
# Phishing URL Analytics and Classification

Big Data Analytics mini project (topic 66): classify URLs as phishing or
legitimate using lexical and domain-based features.

- **Name:** YOUR NAME
- **Roll number:** YOUR ROLL NUMBER
- **College:** YOUR COLLEGE

## What it does

1. Reads a dataset of about 235,000 labelled URLs.
2. Converts every URL into 37 numeric features, using only the URL text
   (no page download, no WHOIS lookup).
3. Trains and compares four machine-learning models.
4. Shows the analytics and results in an interactive dashboard, where you
   can also check any URL yourself.

## Architecture

    Dataset (CSV of URLs + labels)
            |
    Data processing      features.py   URL -> 37 features
            |
    Machine learning     train.py      4 models, evaluation, best model saved
            |
    Dashboard            app.py        Streamlit + Plotly

## Features

**Lexical (22):** URL length, counts of dots, hyphens, underscores, slashes,
question marks, equals signs, @ symbols, ampersands, percent signs and
digits, digit ratio, letter ratio, special-character count, URL entropy,
path length, path depth, query length, number of query parameters, number
of suspicious keywords (login, verify, secure, ...), double slash in the
path, and longest word length.

**Domain-based (15):** HTTPS used, host length, domain length, number of
subdomains, IP address used instead of a domain, non-standard port, TLD
length, frequently abused TLD, common TLD, digits in the host, hyphens in
the host, host entropy, URL shortener, punycode, and a brand name appearing
outside the main domain.

## Models

Logistic Regression, Decision Tree, Random Forest and Gradient Boosting
(scikit-learn). The data is split 80% for training and 20% for testing.
The best model is chosen by F1 score. Accuracy, precision, recall, F1 and
ROC-AUC are reported for each model.

## Setup

Requires Python 3.10 or newer.

    pip install -r requirements.txt

Download the PhiUSIIL Phishing URL Dataset from
https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
then unzip it and put the CSV file inside the `data/` folder.

## Run

    python train.py
    streamlit run app.py

`train.py` takes a few minutes on the full dataset. For a quick run use
`python train.py --sample 50000`. If the `streamlit` command is not found,
use `python -m streamlit run app.py`.

## Project structure

    features.py        feature extraction (lexical + domain-based)
    train.py           training, evaluation and analytics
    app.py             Streamlit dashboard
    requirements.txt   Python libraries
    data/              dataset CSV (not uploaded to GitHub)
    models/            created by train.py (not uploaded to GitHub)

## Dashboard pages

- **Overview:** dataset size, class balance, top-level-domain analytics
- **Feature Analytics:** how each feature differs between the two classes
- **Model Performance:** comparison table, confusion matrix, ROC curves,
  feature importance
- **URL Checker:** classify one URL and list its warning signs
- **Batch Check:** classify many URLs and download the results

## Results

Run `python train.py` and copy the table it prints into this section.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | | | | | |
| Decision Tree | | | | | |
| Random Forest | | | | | |
| Gradient Boosting | | | | | |

## Limitations

- The model sees only the URL text. It cannot know what is on the page, how
  old the domain is, or whether the site is on a blocklist.
- In this dataset most legitimate URLs are plain homepages, while phishing
  URLs often have long paths. Test scores are therefore very high, but the
  model can wrongly flag legitimate deep links (long paths, many query
  parameters) as phishing.
- Attackers change their tricks over time, so a model trained on old URLs
  slowly becomes less accurate.

## Future scope

- Add WHOIS-based features such as domain age and registrar.
- Add page-content features (forms, external links, title match).
- Process URLs with PySpark for larger datasets, or score a live stream of
  URLs with Kafka.
- Retrain regularly on fresh phishing feeds.

## Dataset credit

Prasad, A. and Chandra, S. (2024). PhiUSIIL: A diverse security profile
empowered phishing URL detection framework based on similarity index and
incremental learning. Computers & Security. Dataset from the UCI Machine
Learning Repository, licensed CC BY 4.0.
'''

# ===========================================================================
# .gitignore
# ===========================================================================
FILES[".gitignore"] = r'''
data/*.csv
models/
__pycache__/
*.pyc
.venv/
venv/
setup_project.py
'''


def main():
    """Write every file in FILES next to this script and create data/."""
    base = Path(__file__).resolve().parent
    (base / "data").mkdir(exist_ok=True)
    for name, content in FILES.items():
        (base / name).write_text(content.lstrip("\n"), encoding="utf-8")
        print("created", name)
    if not (base / "features.py").exists():
        print("\nWARNING: features.py is missing from this folder.")
    if not list((base / "data").glob("*.csv")):
        print("\nNext: put the dataset CSV inside the data/ folder.")
    print("\nThen run:")
    print("  python train.py")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
