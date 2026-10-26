"""FraudShield dashboard. Read-only UI over existing training artifacts."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve

from src.features import MODEL_FEATURES
from src.train import evaluate

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
TEAL, RED, NAVY = "#087f8c", "#c44b58", "#172b4d"
LABELS = {
    "TX_AMOUNT": "Transaction amount",
    "TX_HOUR": "Hour of day",
    "IS_WEEKEND": "Weekend (1 = yes)",
    "AMOUNT_VS_AVG": "Amount / previous average",
    "TX_COUNT_24H": "Previous 24h transactions",
}

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")
st.markdown("""<style>
.block-container {max-width:1280px;padding-top:2rem;padding-bottom:3rem;}
h1,h2,h3 {letter-spacing:-.035em;}
[data-testid="stMetric"] {background:white;border:1px solid #e2e8f0;
border-radius:14px;padding:18px 20px;min-height:112px;}
[data-testid="stMetricLabel"] {color:#52647c;}
[data-testid="stMetricValue"] {color:#172b4d;font-weight:650;}
.eyebrow {font-size:11px;letter-spacing:.17em;font-weight:700;color:#087f8c;}
.hero {background:#172b4d;color:white;border-radius:18px;padding:26px 30px;margin:14px 0 24px;}
.hero h2 {color:white;margin:0 0 8px;font-size:26px;}
.hero p {color:#d5e2ee;margin:0;max-width:800px;}
.tag {display:inline-block;padding:6px 12px;border-radius:20px;font-size:12px;
font-weight:650;background:#e4f3f2;color:#12656c;margin-bottom:12px;}
.flag {background:#fbe9ec;color:#9d3243;}
</style>""", unsafe_allow_html=True)


def read_json(name):
    return json.loads((ART / name).read_text(encoding="utf-8"))


@st.cache_resource
def model():
    return joblib.load(ART / "xgboost.joblib")


@st.cache_data
def validation():
    df = pd.read_csv(ROOT / "data/processed/val.csv.gz")
    df["score"] = model().predict_proba(df[MODEL_FEATURES])[:, 1]
    return df


@st.cache_data
def policies():
    original = read_json("threshold_selection.json")
    if original["selected_model"] != "xgboost":
        raise ValueError("This dashboard requires the selected XGBoost model.")
    df = validation()
    # Group tied scores before selecting a strict validation operating point.
    groups = df.groupby("score")["TX_FRAUD"].agg(["sum", "count"]).sort_index(ascending=False)
    tp = groups["sum"].cumsum()
    fp = (groups["count"] - groups["sum"]).cumsum()
    candidates = pd.DataFrame({"threshold": groups.index, "tp": tp.values, "fp": fp.values})
    candidates = candidates[candidates.fp <= .001 * (df.TX_FRAUD == 0).sum()]
    strict = float(np.nextafter(df.score.max(), np.inf))
    if not candidates.empty:
        strict = float(candidates.sort_values(["tp", "fp", "threshold"], ascending=[False, True, False]).iloc[0].threshold)
    return {"Fewer false alarms": strict, "Original policy": original["selected_threshold"]}


def chart(fig, height=340):
    fig.update_layout(template="plotly_white", height=height,
                      paper_bgcolor="rgba(0,0,0,0)", font_color=NAVY,
                      margin=dict(l=15, r=20, t=25, b=30),
                      legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def metrics(m):
    for col, label, value in zip(st.columns(4),
        ["Precision", "Fraud recall", "False-positive rate", "Fraud caught"],
        [f"{m['precision']:.2%}", f"{m['recall']:.2%}", f"{m['false_positive_rate']:.2%}", f"{m['true_positives']:,}"]):
        col.metric(label, value)


def heading(title, description, scope):
    st.markdown(f'<div class="eyebrow">{scope}</div>', unsafe_allow_html=True)
    st.title(title)
    st.write(description)


def overview():
    report = read_json("test_metrics.json")
    m = report["metrics"]
    heading("A clear view of model performance.",
            "Frozen results from unseen transactions at the original operating threshold.", "OVERVIEW / HELD-OUT TEST")
    st.markdown('<div class="hero"><h2>Every alert has a trade-off.</h2><p>Explore what the model catches, what it misses, and the legitimate transactions it flags. All results use synthetic data.</p></div>', unsafe_allow_html=True)
    metrics(m)
    st.caption(f"{report['transactions']:,} transactions · {report['fraud_cases']:,} fraud cases · {report['model']} · threshold {report['threshold']:.8f}")
    left, right = st.columns([1.3, 1])
    with left, st.container(border=True):
        st.subheader("Where predictions land")
        matrix = [[m["true_negatives"], m["false_positives"]], [m["false_negatives"], m["true_positives"]]]
        fig = go.Figure(go.Heatmap(z=np.log1p(matrix), text=np.array(matrix),
            texttemplate="%{text:,}", x=["Not flagged", "Flagged"],
            y=["Actually legitimate", "Actually fraud"], colorscale="Blues",
            showscale=False, hovertemplate="%{y}<br>%{x}: %{text:,}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        chart(fig)
        st.caption("Cell labels are exact counts. Color uses a logarithmic scale to keep rare outcomes visible.")
    with right, st.container(border=True):
        st.subheader("Read the results")
        st.metric("Average Precision", f"{m['average_precision']:.4f}")
        st.write(f"**{m['false_negatives']:,} fraud cases were missed.** Recall measures how much fraud we catch.")
        st.write(f"**{m['false_positives']:,} legitimate transactions were flagged.** Precision measures how trustworthy the alert queue is.")
        st.info("The stricter dashboard preset is a validation experiment. Its results do not replace this test report.")


@st.cache_data
def explain_row(values):
    import shap
    inputs = pd.DataFrame([values], columns=MODEL_FEATURES)
    explanation = shap.TreeExplainer(model())(inputs)
    return np.asarray(explanation.values).reshape(-1)


def investigate():
    heading("Look inside a decision.", "Inspect a transaction and the features behind its score.", "INVESTIGATOR / VALIDATION")
    df, presets = validation(), policies()
    a, b = st.columns(2)
    selected_policy = a.selectbox("Decision setting", list(presets))
    threshold = presets[selected_policy]
    flagged, fraud = df.score >= threshold, df.TX_FRAUD == 1
    masks = {"Caught fraud": flagged & fraud, "False alarms": flagged & ~fraud,
             "Missed fraud": ~flagged & fraud, "Correctly accepted": ~flagged & ~fraud}
    group = b.selectbox("Outcome to inspect", list(masks))
    examples = df.loc[masks[group]].sort_values("score", ascending=False)
    st.caption(f"{len(examples):,} matching transactions · threshold {threshold:.8f} · known labels used only for inspection")
    search = st.text_input("Find a transaction ID within this group", placeholder="Leave blank to browse")
    if search.strip():
        examples = examples[examples.TRANSACTION_ID.astype(str) == search.strip()]
    if examples.empty:
        st.info("No matching transaction. Try another ID or outcome group.")
        return
    indices = examples.head(200).index.tolist()
    idx = st.selectbox("Transaction", indices, format_func=lambda i: f"#{df.at[i, 'TRANSACTION_ID']} · {df.at[i, 'TX_DATETIME']}")
    st.caption("Browsing up to 200 highest-scoring matches. Search an exact ID to inspect other matches.")
    row = df.loc[idx]
    decision = row.score >= threshold
    st.markdown(f'<span class="tag {"flag" if decision else ""}">{"FLAG FOR REVIEW" if decision else "NOT FLAGGED"}</span>', unsafe_allow_html=True)
    cols = st.columns(3)
    cols[0].metric("Transaction amount", f"{row.TX_AMOUNT:,.2f}")
    cols[1].metric("Model score · 0 to 1", f"{row.score:.4f}")
    cols[2].metric("Known outcome", "Fraud" if row.TX_FRAUD else "Legitimate")
    st.caption("Amount uses dataset units. The model score is not a calibrated probability.")
    left, right = st.columns([1, 1.4])
    with left, st.container(border=True):
        st.subheader("Transaction context")
        st.write(f"Customer **{row.CUSTOMER_ID}** · Terminal **{row.TERMINAL_ID}**")
        st.dataframe(pd.DataFrame({"Signal": [LABELS[f] for f in MODEL_FEATURES],
            "Value": [float(row[f]) for f in MODEL_FEATURES]}), hide_index=True, width="stretch")
        with st.expander("Technical feature names"):
            st.json(LABELS)
    with right, st.container(border=True):
        st.subheader("What moved the score?")
        st.caption("Red raises the score · Teal lowers the score")
        try:
            values = explain_row(tuple(float(row[f]) for f in MODEL_FEATURES))
            order = np.argsort(np.abs(values))
            fig = go.Figure(go.Bar(x=values[order], y=[LABELS[MODEL_FEATURES[i]] for i in order],
                orientation="h", marker_color=[RED if v > 0 else TEAL for v in values[order]],
                hovertemplate="%{y}<br>Contribution: %{x:.4f}<extra></extra>"))
            fig.update_layout(xaxis_title="SHAP contribution · log-odds")
            chart(fig)
            st.caption("Explains this model's behavior, not the causes of fraud. Contributions are not percentage points.")
        except Exception as exc:
            st.warning("Prediction is available, but its SHAP explanation could not load.")
            with st.expander("Error details"):
                st.code(str(exc))


def simulator():
    heading("Choose the trade-off.", "See how stricter decisions change alert quality and fraud coverage.", "THRESHOLD LAB / VALIDATION ONLY")
    df, presets = validation(), policies()
    mode = st.radio("Decision setting", [*presets, "Custom"], horizontal=True)
    threshold = st.slider("Score cutoff", 0.0, 1.0, .95, .001, format="%.3f") if mode == "Custom" else presets[mode]
    st.caption(f"Active threshold {threshold:.8f} · Flags scores at or above this value · Does not save policy changes")
    m = evaluate(df.TX_FRAUD, df.score, threshold)
    metrics(m)
    a, b = st.columns(2)
    a.metric("False alarms", f"{m['false_positives']:,}")
    b.metric("Fraud missed", f"{m['false_negatives']:,}")
    left, right = st.columns([1.2, 1])
    with left, st.container(border=True):
        st.subheader("Precision vs. recall")
        precision, recall, _ = precision_recall_curve(df.TX_FRAUD, df.score)
        # Decimate only the display; metrics above use every validation row.
        take = np.unique(np.linspace(0, len(precision)-1, min(2000, len(precision))).astype(int))
        fig = go.Figure(go.Scatter(x=recall[take], y=precision[take], mode="lines", name="XGBoost", line_color=TEAL))
        fig.add_trace(go.Scatter(x=[m["recall"]], y=[m["precision"]], mode="markers", name="Selected cutoff", marker=dict(size=13, color=RED)))
        fig.update_xaxes(title="Recall", tickformat=".0%", range=[0,1])
        fig.update_yaxes(title="Precision", tickformat=".0%", range=[0,1])
        chart(fig)
    with right, st.container(border=True):
        st.subheader("Compare the presets")
        rows = []
        for name, cutoff in presets.items():
            r = evaluate(df.TX_FRAUD, df.score, cutoff)
            rows.append({"Policy": name, "Precision": f"{r['precision']:.2%}", "Recall": f"{r['recall']:.2%}", "False alarms": r['false_positives']})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.write("**Precision:** how many alerts are actually fraud.")
        st.write("**Recall:** how much of all fraud we catch.")
        st.info("Preset constraints were measured on validation data. They are not guarantees for future transactions.")


with st.sidebar:
    st.markdown("## 🛡️ FraudShield")
    st.caption("TRANSACTION INTELLIGENCE")
    st.divider()
    page = st.radio("Workspace", ["Overview", "Investigator", "Threshold lab"])
    st.divider()
    st.markdown("**XGBoost · Five features**")
    st.caption("Synthetic data demo\n\nScores are not calibrated probabilities.")
    if st.button("Reload local artifacts"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

try:
    {"Overview": overview, "Investigator": investigate, "Threshold lab": simulator}[page]()
except FileNotFoundError as exc:
    st.error("A required project file is missing. Keep this app in your project root alongside src, data and artifacts.")
    st.code(str(exc))

st.divider()
st.caption("FraudShield AI · Portfolio demonstration · Validation experiments and held-out test results are reported separately.")
