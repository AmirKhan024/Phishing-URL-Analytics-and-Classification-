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
