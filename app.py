import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE    = r"C:\Users\ThanhT\OneDrive - Keyloop\Desktop\Thạc sĩ\Nghiên cứu dữ liệu lớn và Ứng dụng trong kinh doanh\Quá trình & Final"
RESULTS = os.path.join(BASE, "Results")

st.set_page_config(page_title="Olist E-Commerce Analytics", layout="wide")

# ── helpers ──────────────────────────────────
def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def img(name):
    path = os.path.join(RESULTS, name)
    return path if os.path.exists(path) else None

SEGMENT_COLORS = {0: "#4C72B0", 1: "#55A868", 2: "#C44E52", 3: "#8172B2"}

# ── sidebar ───────────────────────────────────
st.sidebar.title("Olist Analytics")
st.sidebar.markdown("""
**Dataset**: Brazilian E-Commerce (Olist)
**Pipeline**: PySpark MLlib
**Steps**: EDA → RFM → Segmentation → Churn → Benchmark → XAI
""")
st.sidebar.markdown("---")
eda = load("eda_summary.csv")
if eda is not None:
    for _, row in eda.iterrows():
        st.sidebar.metric(row["Metric"], row["Value"])

# ── tabs ──────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA Overview",
    "👥 Customer Segmentation",
    "⚠️ Churn Prediction",
    "⚡ Benchmark & XAI",
])

# ═══════════════════════════════════════════════
# TAB 1 — EDA Overview
# ═══════════════════════════════════════════════
with tab1:
    st.header("Exploratory Data Analysis")
    st.markdown("Dataset: **Olist Brazilian E-Commerce** — 100K+ orders (2016–2018)")

    # KPI cards
    if eda is not None:
        cols = st.columns(4)
        for i, (_, row) in enumerate(eda.iterrows()):
            cols[i].metric(row["Metric"], row["Value"])

    st.divider()

    # Monthly trend
    p = img("eda_monthly_trend.png")
    if p:
        st.subheader("Monthly Orders Trend")
        st.image(p, use_container_width=True)

    st.divider()

    # Top categories
    top_cat = load("eda_top_categories.csv")
    if top_cat is not None:
        st.subheader("Top 10 Product Categories")
        top_cat = top_cat.dropna(subset=["product_category_name_english"])
        fig = px.bar(
            top_cat.sort_values("count"),
            x="count", y="product_category_name_english",
            orientation="h",
            labels={"count": "Number of Items", "product_category_name_english": "Category"},
            color="count", color_continuous_scale="Blues",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 2 — Customer Segmentation
# ═══════════════════════════════════════════════
with tab2:
    st.header("Customer Segmentation — RFM + K-Means (PySpark MLlib)")

    seg_profile = load("segment_profile.csv")
    segmented   = load("segmented.csv")

    if seg_profile is None or segmented is None:
        st.warning("Run pipeline.py first to generate results.")
        st.stop()

    # Segment summary table
    st.subheader("Segment Profiles")
    display_cols = ["segment", "Segment_Label", "Count", "Avg_Recency", "Avg_Frequency", "Avg_Monetary"] \
        if "Segment_Label" in seg_profile.columns \
        else ["segment", "Count", "Avg_Recency", "Avg_Frequency", "Avg_Monetary"]

    styled = (seg_profile[display_cols]
              .rename(columns={"segment": "Cluster", "Segment_Label": "Label",
                               "Avg_Recency": "Avg Recency (days)",
                               "Avg_Frequency": "Avg Frequency",
                               "Avg_Monetary": "Avg Monetary (R$)"})
              .style
              .format({"Avg Recency (days)": "{:.0f}",
                       "Avg Frequency":      "{:.2f}",
                       "Avg Monetary (R$)":  "{:.2f}",
                       "Count":              "{:,}"})
              .background_gradient(subset=["Avg Monetary (R$)"], cmap="Greens"))
    st.dataframe(styled, use_container_width=True)

    st.divider()

    # Customer count per segment
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(
            seg_profile, values="Count",
            names=seg_profile["Segment_Label"] if "Segment_Label" in seg_profile.columns else seg_profile["segment"].astype(str),
            title="Customer Distribution by Segment",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        p_elbow = img("kmeans_elbow.png")
        if p_elbow:
            st.subheader("Elbow Method — Optimal K")
            st.image(p_elbow, use_container_width=True)

    st.divider()

    # 3D scatter RFM
    st.subheader("3D RFM Scatter by Segment")
    sample = segmented.sample(min(3000, len(segmented)), random_state=42)
    fig3d = px.scatter_3d(
        sample, x="Recency", y="Frequency", z="Monetary",
        color=sample["segment"].astype(str),
        labels={"color": "Segment"},
        opacity=0.6, height=550,
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig3d.update_traces(marker=dict(size=2))
    st.plotly_chart(fig3d, use_container_width=True)

    # RFM distribution
    st.subheader("RFM Distribution")
    c1, c2, c3 = st.columns(3)
    for col_widget, metric in zip([c1, c2, c3], ["Recency", "Frequency", "Monetary"]):
        fig_hist = px.histogram(segmented, x=metric, nbins=40,
                                title=f"{metric} Distribution",
                                color_discrete_sequence=["steelblue"])
        fig_hist.update_layout(showlegend=False, height=300)
        col_widget.plotly_chart(fig_hist, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 3 — Churn Prediction
# ═══════════════════════════════════════════════
with tab3:
    st.header("Churn Prediction — Logistic Regression & Random Forest (PySpark MLlib)")
    st.markdown("""
    **Churn Definition**: Customer with `Frequency = 1` AND `Recency > 180 days`
    — i.e., bought only once and hasn't returned within 6 months.
    """)

    metrics      = load("churn_metrics.csv")
    churn_pred   = load("churn_predictions.csv")

    if metrics is None:
        st.warning("Run pipeline.py first to generate results.")
        st.stop()

    # Metrics table
    st.subheader("Model Performance")
    styled_m = (metrics.style
                .format({"AUC-ROC": "{:.4f}", "Accuracy": "{:.4f}", "F1-Score": "{:.4f}"})
                .highlight_max(subset=["AUC-ROC", "Accuracy", "F1-Score"],
                               color="#d4edda"))
    st.dataframe(styled_m, use_container_width=True)

    st.divider()

    # Metrics bar chart
    st.subheader("Model Comparison")
    metrics_long = metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig_bar = px.bar(
        metrics_long, x="Metric", y="Score", color="Model",
        barmode="group", range_y=[0, 1],
        color_discrete_sequence=["#4C72B0", "#DD8452"],
        text_auto=".4f"
    )
    fig_bar.update_layout(height=380)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # Churn rate stats
    if churn_pred is not None:
        churn_rate = churn_pred["churn"].mean()
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Churn Rate", f"{churn_rate:.1%}")
        col2.metric("Churned Customers", f"{int(churn_pred['churn'].sum()):,}")
        col3.metric("Non-Churned Customers", f"{int((churn_pred['churn'] == 0).sum()):,}")

        st.subheader("Churn Distribution")
        fig_churn = px.pie(
            names=["Churned", "Not Churned"],
            values=[churn_pred["churn"].sum(), (churn_pred["churn"] == 0).sum()],
            color_discrete_sequence=["#C44E52", "#55A868"]
        )
        st.plotly_chart(fig_churn, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 4 — Benchmark & XAI
# ═══════════════════════════════════════════════
with tab4:
    st.header("Tool Benchmark & Explainable AI (SHAP)")

    # ── Benchmark ──
    st.subheader("⚡ PySpark vs Pandas — Execution Time")
    benchmark = load("benchmark.csv")

    if benchmark is not None:
        col1, col2 = st.columns([3, 2])
        with col1:
            fig_bm = go.Figure()
            fig_bm.add_bar(name="Pandas",  x=benchmark["Task"], y=benchmark["Pandas (s)"],
                           marker_color="#4C72B0",
                           text=benchmark["Pandas (s)"].apply(lambda x: f"{x}s"),
                           textposition="outside")
            fig_bm.add_bar(name="PySpark", x=benchmark["Task"], y=benchmark["PySpark (s)"],
                           marker_color="#DD8452",
                           text=benchmark["PySpark (s)"].apply(lambda x: f"{x}s"),
                           textposition="outside")
            fig_bm.update_layout(barmode="group", height=380,
                                 title="Execution Time (seconds) — local mode, 100K records",
                                 yaxis_title="Seconds")
            st.plotly_chart(fig_bm, use_container_width=True)
        with col2:
            st.markdown("**Results Table**")
            st.dataframe(benchmark, use_container_width=True)
            st.info("""
**Why PySpark on local mode?**

On 100K records, Pandas is often faster due to JVM overhead. However, PySpark is the correct
choice for production-scale data (millions of records on a cluster), where distributed
processing delivers significant speedups. This benchmark establishes the baseline.
""")
    else:
        st.warning("Run pipeline.py first.")

    st.divider()

    # ── SHAP ──
    st.subheader("🔍 Explainable AI — SHAP Feature Importance")
    st.markdown("SHAP values show **which features drive churn prediction** for each customer.")

    shap_imp = load("shap_importance.csv")
    if shap_imp is not None:
        col1, col2 = st.columns(2)
        with col1:
            fig_shap = px.bar(
                shap_imp.sort_values("Mean |SHAP|"),
                x="Mean |SHAP|", y="Feature", orientation="h",
                title="Mean |SHAP| Value (Feature Importance)",
                color="Mean |SHAP|", color_continuous_scale="Reds"
            )
            fig_shap.update_layout(coloraxis_showscale=False, height=350)
            st.plotly_chart(fig_shap, use_container_width=True)
        with col2:
            p_bee = img("shap_beeswarm.png")
            if p_bee:
                st.image(p_bee, caption="SHAP Beeswarm Plot", use_container_width=True)

        st.markdown("""
**Interpretation**:
- **Recency**: Customers who haven't bought recently are the strongest churn signal
- **Frequency**: Low purchase frequency strongly predicts churn
- **Monetary**: Spending level has secondary influence on churn likelihood
""")
    else:
        p_bar = img("shap_bar.png")
        if p_bar:
            st.image(p_bar, use_container_width=True)
        else:
            st.warning("SHAP results not found. Ensure `shap` is installed and pipeline ran Step 6.")
