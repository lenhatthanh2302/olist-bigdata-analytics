import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

BASE    = os.path.dirname(os.path.abspath(__file__))
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
MODEL_COLORS   = {"Logistic Regression": "#4C72B0", "Random Forest": "#DD8452",
                  "Gradient Boosted Trees": "#55A868"}

def ci95(p, n):
    # Khoảng tin cậy 95% của một tỷ lệ (xấp xỉ chuẩn): 1.96 * sqrt(p(1-p)/n)
    return 1.96 * np.sqrt(p * (1 - p) / n)

def show_image_fallback(name, caption=None):
    # Dùng ảnh tĩnh khi chưa có dữ liệu gốc để vẽ biểu đồ tương tác (chạy lại pipeline sẽ có)
    p = img(name)
    if p:
        st.image(p, caption=caption, use_container_width=True)

def roc_figure(curves, title, x="FPR", y="TPR", auc=None):
    fig = go.Figure()
    for name, d in curves.groupby("Model", sort=False):
        label = f"{name} (AUC = {auc[name]:.3f})" if auc is not None and name in auc else name
        fig.add_scatter(x=d[x], y=d[y], mode="lines", name=label,
                        line=dict(color=MODEL_COLORS.get(name), width=2))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Đoán ngẫu nhiên",
                    line=dict(color="grey", dash="dash"))
    fig.update_layout(title=title, height=400, xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate", legend=dict(orientation="h", y=-0.25))
    return fig

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 EDA Overview",
    "👥 Customer Segmentation",
    "⚠️ Churn Prediction",
    "⚡ Benchmark & XAI",
    "🔬 Phân tích chuyên sâu (RQ)",
    "🎯 Quyết định & Hạn chế",
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
    monthly = load("eda_monthly_trend.csv")
    st.subheader("Monthly Orders Trend")
    if monthly is not None:
        fig_m = px.line(monthly, x="month", y="count", markers=True,
                        labels={"month": "Tháng", "count": "Số đơn"})
        fig_m.update_layout(height=380)
        st.plotly_chart(fig_m, use_container_width=True)
    else:
        show_image_fallback("eda_monthly_trend.png")

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
        ksel = load("kmeans_selection.csv")
        st.subheader("Chọn K: Elbow + Silhouette")
        if ksel is not None:
            fig_k = go.Figure()
            fig_k.add_scatter(x=ksel["k"], y=ksel["WSSSE"], mode="lines+markers", name="WSSSE (Elbow)",
                              line=dict(color="#4C72B0"))
            fig_k.add_scatter(x=ksel["k"], y=ksel["Silhouette"], mode="lines+markers", name="Silhouette",
                              line=dict(color="#C44E52"), yaxis="y2")
            fig_k.add_vline(x=4, line_dash="dash", line_color="grey", annotation_text="k = 4 (được chọn)")
            fig_k.update_layout(height=340, xaxis=dict(title="k", dtick=1),
                                yaxis=dict(title="WSSSE"),
                                yaxis2=dict(title="Silhouette", overlaying="y", side="right"),
                                legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(fig_k, use_container_width=True)
            st.dataframe(ksel.style.format({"WSSSE": "{:,.0f}", "Silhouette": "{:.3f}"}),
                         use_container_width=True)
            st.caption("Silhouette cao nhất ở k=2, nhưng k=2 chỉ tách ~3% khách mua lại (Frequency ≥ 2) "
                       "khỏi ~97% còn lại, quá thô để làm CRM. K=4 cho Recent / At-Risk / Lost / "
                       "Repeat Buyers có hành động khác nhau. Đây là lựa chọn kinh doanh có chủ đích, "
                       "đánh đổi một phần chất lượng cụm về mặt toán học, không phải tối ưu thống kê.")

    st.divider()

    # 3D scatter RFM
    st.subheader("3D RFM Scatter by Segment")
    sample = segmented.sample(min(3000, len(segmented)), random_state=42)
    fig3d = px.scatter_3d(
        sample, x="Recency", y="Frequency", z="Monetary",
        color=sample["Segment_Label"],
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
    st.header("Churn Prediction — LR, Random Forest, GBT (PySpark MLlib)")
    st.markdown("""
    **Định nghĩa churn**: khách `Frequency = 1` VÀ `Recency > 180 ngày` — mua đúng một lần
    và không quay lại trong 6 tháng.

    Vì churn được định nghĩa từ Recency và Frequency, hai biến này (và các biến suy ra từ
    chúng như segment) **không được dùng làm đặc trưng**. Mô hình chỉ dùng thông tin có
    ngay tại đơn hàng đầu tiên: giá trị đơn, phí vận chuyển, giao hàng, đánh giá, trả góp, vùng.
    """)

    metrics    = load("churn_metrics.csv")
    ablation   = load("churn_leakage_ablation.csv")
    churn_pred = load("churn_predictions.csv")

    if metrics is None:
        st.warning("Run pipeline.py first to generate results.")
        st.stop()

    st.subheader("Model Performance")
    num_cols = [c for c in metrics.columns if c != "Model"]
    st.dataframe(metrics.style.format({c: "{:.4f}" for c in num_cols})
                 .highlight_max(subset=num_cols, color="#d4edda"),
                 use_container_width=True)

    best = metrics.sort_values("AUC-ROC", ascending=False).iloc[0]
    if churn_pred is not None:
        base = churn_pred["churn"].mean()
        st.markdown(
            f"**Đánh đổi Precision – Recall ({best['Model']})**: Recall {best['Recall (Churn)']:.2f} "
            f"nghĩa là bắt được khoảng {best['Recall (Churn)']:.0%} khách thực sự churn, nhưng "
            f"Precision chỉ {best['Precision (Churn)']:.2f}, tức khoảng "
            f"{1 - best['Precision (Churn)']:.0%} khách bị cảnh báo là báo nhầm. Tỷ lệ churn nền "
            f"đã là {base:.1%}, nên Precision này chỉ cao hơn mức đoán ngẫu nhiên một khoảng vừa phải. "
            "Mô hình phù hợp để xếp hạng/ưu tiên khách cần chăm sóc hơn là để kết luận từng cá nhân."
        )

    metrics_long = metrics[["Model", "AUC-ROC", "PR-AUC", "Accuracy", "F1-Score"]] \
        .melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig_bar = px.bar(metrics_long, x="Metric", y="Score", color="Model",
                     barmode="group", range_y=[0, 1], text_auto=".3f",
                     color_discrete_sequence=["#4C72B0", "#DD8452", "#55A868"])
    fig_bar.update_layout(height=380)
    st.plotly_chart(fig_bar, use_container_width=True)

    roc_df = load("churn_roc_curves.csv")
    cm_df  = load("churn_confusion.csv")
    c1, c2 = st.columns(2)
    with c1:
        if roc_df is not None:
            auc_map = metrics.set_index("Model")["AUC-ROC"].to_dict()
            st.plotly_chart(roc_figure(roc_df, "ROC — dự đoán churn (đặc trưng sạch)", auc=auc_map),
                            use_container_width=True)
        else:
            show_image_fallback("churn_roc.png", "ROC curve của 3 mô hình")
    with c2:
        if cm_df is not None:
            model_cm = st.selectbox("Confusion matrix của mô hình", cm_df["Model"].tolist(),
                                    index=cm_df["Model"].tolist().index(best["Model"]))
            r = cm_df[cm_df["Model"] == model_cm].iloc[0]
            z = [[r["TN"], r["FP"]], [r["FN"], r["TP"]]]
            fig_cm = go.Figure(go.Heatmap(
                z=z, x=["Dự đoán: Giữ", "Dự đoán: Churn"], y=["Thực tế: Giữ", "Thực tế: Churn"],
                text=[[f"{v:,}" for v in row] for row in z], texttemplate="%{text}",
                colorscale="Blues", showscale=False, hovertemplate="%{y} / %{x}: %{z:,}<extra></extra>"))
            fig_cm.update_yaxes(autorange="reversed")
            fig_cm.update_layout(height=360, margin=dict(t=30))
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            show_image_fallback("churn_confusion.png", "Confusion matrix (mô hình AUC cao nhất)")

    st.divider()

    st.subheader("Thí nghiệm rò rỉ dữ liệu (Data Leakage)")
    if ablation is not None:
        fig_ab = px.bar(ablation, x="AUC-ROC", y="Feature set", orientation="h",
                        text_auto=".3f", range_x=[0, 1.05],
                        color="AUC-ROC", color_continuous_scale="RdYlGn_r")
        fig_ab.update_layout(coloraxis_showscale=False, height=320, yaxis_title="")
        st.plotly_chart(fig_ab, use_container_width=True)
        st.markdown("""
        Nếu đưa Recency/Frequency vào mô hình, AUC gần như tuyệt đối (~0.99), nhưng đó chỉ là
        mô hình "đọc lại" định nghĩa churn. Segment và M_score cũng rò rỉ gián tiếp (~0.96) vì
        K-Means gom nhóm theo Recency.

        Điểm đáng chú ý nhất là dòng thứ ba: **chỉ cần biết ngày mua đầu tiên** đã cho AUC ~0.97.
        Lý do là hiệu ứng *ngày cắt dữ liệu* (right-censoring): khách mua trong 180 ngày cuối của
        dataset chưa có đủ thời gian để "quay lại", nên theo định nghĩa họ không thể bị gán churn.
        Vì vậy AUC 0.70–0.75 của mô hình đặc trưng sạch phía trên **vẫn còn bị thổi phồng một phần**
        do tín hiệu thời gian lọt vào qua các đặc trưng như giá trị đơn hay phí vận chuyển (các biến
        này thay đổi theo thời kỳ). Phần dưới đây loại hẳn thiên lệch này.
        """)

    st.divider()
    st.subheader("Mô hình sửa thiên lệch: dự đoán khách mua lại trong 180 ngày (nhóm khách đủ tuổi)")
    rep_m  = load("repeat_metrics.csv")
    rep_d  = load("repeat_deciles.csv")
    if rep_m is not None:
        base_r = rep_m["Base rate"].iloc[0]
        st.markdown(
            "Chỉ giữ khách có đơn đầu tiên cách ngày cuối dataset ít nhất 180 ngày (tức ai cũng có đủ "
            f"thời gian để mua lại), rồi dự đoán họ có mua đơn thứ hai trong 180 ngày hay không. "
            f"Tỷ lệ mua lại nền chỉ **{base_r:.1%}**, nên dùng class weight và đánh giá bằng "
            "PR-AUC, Lift và Cumulative Gains thay vì Accuracy."
        )
        num_r = [c for c in rep_m.columns if c != "Model"]
        st.dataframe(rep_m.style.format({c: "{:.4f}" for c in num_r}), use_container_width=True)
        best_r = rep_m.sort_values("AUC-ROC", ascending=False).iloc[0]
        top_dec = rep_d.iloc[0] if rep_d is not None else None
        msg = (f"AUC tốt nhất chỉ ~{best_r['AUC-ROC']:.2f} (mức 0.5 là đoán ngẫu nhiên): khi loại hiệu ứng "
               "thời gian, thông tin trong đơn hàng đầu tiên dự báo được rất ít việc khách có quay lại.")
        if top_dec is not None:
            n_top = int(top_dec["Customers"]); k_top = round(n_top * top_dec["Repeat_Rate"])
            se_top = (top_dec["Repeat_Rate"] * (1 - top_dec["Repeat_Rate"]) / n_top) ** 0.5
            lo_l = (top_dec["Repeat_Rate"] - 1.96 * se_top) / base_r
            hi_l = (top_dec["Repeat_Rate"] + 1.96 * se_top) / base_r
            msg += (f" Trên tập kiểm tra, nhóm 10% khách có điểm dự báo cao nhất ghi nhận tỷ lệ mua lại "
                    f"{top_dec['Repeat_Rate']:.1%}, cao hơn {top_dec['Lift']:.2f} lần so với mức nền. Đây là mô tả "
                    f"dữ liệu, không phải hiệu quả của một chiến dịch; nhóm này chỉ có {k_top} khách mua lại nên "
                    f"khoảng tin cậy 95% của lift khá rộng (≈ {lo_l:.1f}–{hi_l:.1f}). Mô hình hữu ích để xếp hạng "
                    "ưu tiên, không đủ mạnh để dự báo từng cá nhân.")
        st.info(msg)
        rep_c = load("repeat_curves.csv")
        if rep_c is not None:
            auc_r = rep_m.set_index("Model")["AUC-ROC"].to_dict()
            g1, g2 = st.columns(2)
            g1.plotly_chart(roc_figure(rep_c[rep_c["Curve"] == "ROC"], "ROC — mua lại trong 180 ngày",
                                       x="X", y="Y", auc=auc_r), use_container_width=True)
            fig_g = go.Figure()
            for name, d in rep_c[rep_c["Curve"] == "Gains"].groupby("Model", sort=False):
                fig_g.add_scatter(x=d["X"], y=d["Y"], mode="lines", name=name,
                                  line=dict(color=MODEL_COLORS.get(name), width=2))
            fig_g.add_scatter(x=[0, 100], y=[0, 100], mode="lines", name="Chọn ngẫu nhiên",
                              line=dict(color="grey", dash="dash"))
            fig_g.update_layout(title="Cumulative Gains", height=400,
                                xaxis_title="% khách được chọn (xếp theo điểm giảm dần)",
                                yaxis_title="% khách mua lại bắt được", legend=dict(orientation="h", y=-0.25))
            g2.plotly_chart(fig_g, use_container_width=True)
        else:
            show_image_fallback("repeat_gains.png",
                                "ROC và Cumulative Gains của mô hình mua lại (nhóm khách đủ tuổi)")
        if rep_d is not None:
            fig_dec = px.bar(rep_d, x="Decile", y="Repeat_Rate", text_auto=".1%",
                             title="Tỷ lệ mua lại theo decile điểm dự báo (1 = điểm cao nhất)")
            fig_dec.add_hline(y=base_r, line_dash="dash", annotation_text="mức nền")
            fig_dec.update_layout(height=340, yaxis_tickformat=".0%")
            st.plotly_chart(fig_dec, use_container_width=True)

    if churn_pred is not None:
        st.divider()
        rate = churn_pred["churn"].mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Churn Rate", f"{rate:.1%}")
        c2.metric("Churned Customers", f"{int(churn_pred['churn'].sum()):,}")
        c3.metric("Non-Churned Customers", f"{int((churn_pred['churn'] == 0).sum()):,}")

# ═══════════════════════════════════════════════
# TAB 4 — Benchmark & XAI
# ═══════════════════════════════════════════════
with tab4:
    st.header("Tool Benchmark & Explainable AI (SHAP)")

    st.subheader("⚡ PySpark vs Pandas — theo quy mô dữ liệu")
    benchmark = load("benchmark.csv")

    if benchmark is not None:
        benchmark["Scale_n"] = benchmark["Scale"].str.replace("×", "").astype(int)
        tasks = benchmark["Task"].unique()
        cols = st.columns(len(tasks))
        for col_w, task in zip(cols, tasks):
            d = benchmark[benchmark["Task"] == task].sort_values("Scale_n")
            fig = go.Figure()
            fig.add_scatter(x=d["Orders"], y=d["Pandas (s)"], mode="lines+markers",
                            name="Pandas", line=dict(color="#4C72B0"))
            fig.add_scatter(x=d["Orders"], y=d["PySpark (s)"], mode="lines+markers",
                            name="PySpark", line=dict(color="#DD8452"))
            fig.update_layout(title=task, height=340, xaxis_title="Số đơn hàng",
                              yaxis_title="Giây (median 3 lần)",
                              legend=dict(orientation="h", y=-0.3))
            col_w.plotly_chart(fig, use_container_width=True)

        st.dataframe(benchmark.drop(columns="Scale_n"), use_container_width=True)

        wins = benchmark[benchmark["Speedup (Pandas/PySpark)"] > 1]
        st.info(
            "**Đọc kết quả**: ở quy mô ~100K–500K dòng, Pandas nhanh hơn vì PySpark phải trả "
            "chi phí khởi tạo JVM, serialization và shuffle. Khi quy mô tiến gần 1 triệu bản ghi "
            "(×10, dữ liệu nhân bản), lợi thế xử lý phân tán bắt đầu bù đắp chi phí này ở tác vụ "
            f"Join + RFM ({len(wins)} trường hợp PySpark nhanh hơn); bài chưa thử quy mô lớn hơn "
            "nên chưa xác định được điểm hòa vốn chính xác. K-Means vẫn chậm hơn nhiều vì mỗi vòng "
            "lặp là một job riêng, còn scikit-learn chạy rất hiệu quả trên một máy. "
            "Lưu ý: chế độ local[*] chỉ dùng nhiều core của một máy, chưa phải cluster; lợi thế "
            "thực sự của Spark xuất hiện khi dữ liệu vượt quá RAM của một máy."
        )
    else:
        st.warning("Run pipeline.py first.")

    st.divider()

    st.subheader("🔍 Explainable AI — SHAP")
    st.markdown("SHAP cho biết đặc trưng nào đẩy xác suất **mua lại trong 180 ngày** của từng khách lên hoặc xuống "
                "(Random Forest scikit-learn, huấn luyện trên nhóm khách đủ tuổi để tránh thiên lệch ngày cắt dữ liệu).")

    shap_imp = load("shap_importance.csv")
    if shap_imp is not None:
        col1, col2 = st.columns(2)
        with col1:
            fig_shap = px.bar(shap_imp.sort_values("Mean |SHAP|"),
                              x="Mean |SHAP|", y="Feature", orientation="h",
                              title="Mean |SHAP| (độ quan trọng đặc trưng)",
                              color="Mean |SHAP|", color_continuous_scale="Reds")
            fig_shap.update_layout(coloraxis_showscale=False, height=450)
            st.plotly_chart(fig_shap, use_container_width=True)
        with col2:
            shap_v = load("shap_values_top.csv")
            if shap_v is not None:
                order = (shap_v.groupby("Feature")["SHAP"].apply(lambda s: s.abs().mean())
                         .sort_values().index.tolist())
                pos = {f: i for i, f in enumerate(order)}
                rng = np.random.default_rng(42)
                shap_v["y"] = shap_v["Feature"].map(pos) + rng.uniform(-0.3, 0.3, len(shap_v))
                fig_bee = go.Figure(go.Scattergl(
                    x=shap_v["SHAP"], y=shap_v["y"], mode="markers",
                    marker=dict(size=4, color=shap_v["Value_pct"], colorscale="Bluered", opacity=0.6,
                                colorbar=dict(title="Giá trị đặc trưng", tickvals=[0, 1], ticktext=["thấp", "cao"])),
                    customdata=np.stack([shap_v["Feature"], shap_v["Value"]], axis=-1),
                    hovertemplate="%{customdata[0]} = %{customdata[1]}<br>SHAP = %{x:.4f}<extra></extra>"))
                fig_bee.update_layout(
                    title="SHAP beeswarm (top 10 đặc trưng, 3.000 khách)", height=450,
                    xaxis_title="SHAP (>0: đẩy lên khả năng mua lại)",
                    yaxis=dict(tickmode="array", tickvals=list(pos.values()), ticktext=list(pos.keys())))
                st.plotly_chart(fig_bee, use_container_width=True)
            else:
                show_image_fallback("shap_beeswarm.png", "SHAP Beeswarm")

        ranked = shap_imp.sort_values("Mean |SHAP|", ascending=False)
        top = ranked.head(3)["Feature"].tolist()
        ops = ["delivery_days", "delay_days", "avg_review_score"]
        ops_share = ranked[ranked["Feature"].isin(ops)]["Mean |SHAP|"].sum() / ranked["Mean |SHAP|"].sum()
        st.markdown(f"""
**Diễn giải**: ba đặc trưng ảnh hưởng nhiều nhất là `{top[0]}`, `{top[1]}`, `{top[2]}`; nhóm
đặc trưng giao hàng và đánh giá (`delivery_days`, `delay_days`, `avg_review_score`) chiếm khoảng
{ops_share:.0%} tổng mức ảnh hưởng. Nghĩa là khả năng khách mua lại gắn nhiều hơn với **cách họ thanh
toán và giá trị giỏ hàng đầu tiên** so với trải nghiệm giao hàng. Cần đọc thận trọng: AUC của mô hình chỉ
~0.6, SHAP mô tả cách mô hình dùng đặc trưng chứ không chứng minh quan hệ nhân quả.
""")
    else:
        st.warning("SHAP results not found.")

# ═══════════════════════════════════════════════
# TAB 5 — Phân tích chuyên sâu (RQ1–RQ4)
# ═══════════════════════════════════════════════
with tab5:
    st.header("Phân tích chuyên sâu — từ mô tả đến ra quyết định")
    st.markdown("""
    Mỗi câu hỏi nghiên cứu (RQ) được soi ở nhiều cấp (phân khúc, bang, danh mục, đơn hàng) và có bảng
    kiểm tra nhất quán ở cuối: cộng các cấp lại phải ra đúng tổng.

    **Lưu ý phương pháp**: RQ2–RQ4 đo bằng *tỷ lệ mua đơn thứ hai trong 180 ngày trên nhóm khách đủ tuổi*,
    không dùng churn thô, vì churn thô phụ thuộc mạnh vào ngày mua đầu tiên (xem tab Churn).
    """)

    seg_val = load("segment_value.csv")
    pareto  = load("pareto_summary.csv")
    d_state = load("drill_state.csv")
    d_cat   = load("drill_category.csv")
    d_del   = load("drill_delivery.csv")
    assoc   = load("association_tests.csv")
    checks  = load("consistency_checks.csv")

    if seg_val is None:
        st.warning("Run pipeline.py first to generate results.")
        st.stop()

    # ── RQ1 ──
    st.subheader("RQ1 — Nhóm khách nào tạo ra doanh thu?")
    seg_colors = {"Recent Buyers": "#55A868", "At-Risk Buyers": "#DD8452",
                  "Lost Customers": "#C44E52", "Repeat Buyers": "#4C72B0"}
    c1, c2 = st.columns(2)
    with c1:
        long = seg_val.melt(id_vars="Segment_Label", value_vars=["Customer_Share_%", "Revenue_Share_%"],
                            var_name="Metric", value_name="Percent")
        fig = px.bar(long, x="Segment_Label", y="Percent", color="Metric", barmode="group",
                     text_auto=".1f", color_discrete_sequence=["#8c9bb5", "#C44E52"],
                     title="Tỷ trọng khách vs tỷ trọng doanh thu (%)")
        fig.update_layout(height=380, xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig_box = go.Figure()
        for _, r in seg_val.iterrows():
            fig_box.add_trace(go.Box(
                name=r["Segment_Label"], q1=[r["P25"]], median=[r["Median"]], q3=[r["P75"]],
                lowerfence=[r["P25"]], upperfence=[r["P95"]], mean=[r["Mean_Monetary"]],
                marker_color=seg_colors.get(r["Segment_Label"])))
        fig_box.update_layout(title="Phân phối chi tiêu theo phân khúc (R$, thang log)", height=380,
                              yaxis_type="log", showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption("Hộp = P25–P75, vạch giữa = trung vị, râu trên = P95, chấm = trung bình. "
                   "Trung bình luôn cao hơn trung vị: vài khách chi rất lớn kéo đuôi phải.")
    st.dataframe(seg_val, use_container_width=True)

    top_seg = seg_val.sort_values("Revenue_Share_%", ascending=False).iloc[0]
    rep_seg = seg_val[seg_val["Segment_Label"] == "Repeat Buyers"].iloc[0]
    if pareto is not None:
        g = dict(zip(pareto["Top_%_khách"], pareto["Doanh_thu_%"]))
        st.markdown(
            f"**Đọc kết quả**: `{top_seg['Segment_Label']}` chiếm {top_seg['Customer_Share_%']:.1f}% khách nhưng "
            f"{top_seg['Revenue_Share_%']:.1f}% doanh thu. `Repeat Buyers` chỉ {rep_seg['Customer_Share_%']:.1f}% khách "
            f"({rep_seg['Revenue_Share_%']:.1f}% doanh thu) nhưng chi trung bình R${rep_seg['Mean_Monetary']:.0f}, cao nhất. "
            f"Doanh thu cũng tập trung: top 10% khách tạo {g.get(10, float('nan')):.1f}% doanh thu, top 20% tạo "
            f"{g.get(20, float('nan')):.1f}%."
        )
        fig_p = px.line(pareto, x="Top_%_khách", y="Doanh_thu_%", markers=True,
                        title="Mức tập trung doanh thu (Pareto)")
        fig_p.add_scatter(x=[0, 100], y=[0, 100], mode="lines", name="Phân phối đều",
                          line=dict(dash="dash", color="grey"))
        fig_p.update_layout(height=340)
        st.plotly_chart(fig_p, use_container_width=True)
    st.divider()

    # ── RQ2 ──
    st.subheader("RQ2 — Mua lại khác nhau thế nào giữa các bang?")
    if d_state is not None:
        ds = d_state[d_state["Mature_Customers"] >= 500].sort_values("Repeat180_Mature", ascending=False)
        c1, c2 = st.columns([3, 2])
        with c1:
            overall = (d_state["Repeat180_Mature"] * d_state["Mature_Customers"]).sum() / d_state["Mature_Customers"].sum()
            fig_s = px.bar(ds, x="customer_state", y="Repeat180_Mature", error_y=ci95(ds["Repeat180_Mature"], ds["Mature_Customers"]),
                           hover_data={"Mature_Customers": ":,"}, color_discrete_sequence=["#4C72B0"],
                           title="Tỷ lệ mua lại trong 180 ngày theo bang (thanh sai số = CI 95%)",
                           labels={"customer_state": "Bang", "Repeat180_Mature": "Tỷ lệ mua lại"})
            fig_s.add_hline(y=overall, line_dash="dash", annotation_text=f"toàn bộ {overall:.1%}")
            fig_s.update_layout(height=380, yaxis_tickformat=".0%")
            st.plotly_chart(fig_s, use_container_width=True)
        with c2:
            st.dataframe(ds[["customer_state", "Customers", "Mature_Customers", "Repeat180_Mature",
                             "Churn_Rate", "Avg_Delivery_Days", "Late_Rate"]]
                         .style.format({"Repeat180_Mature": "{:.2%}", "Churn_Rate": "{:.1%}",
                                        "Avg_Delivery_Days": "{:.1f}", "Late_Rate": "{:.1%}"}),
                         use_container_width=True, height=360)
        st.caption("Churn_Rate là nhãn thô (bị lệch theo ngày cắt dữ liệu), để đối chiếu; "
                   "kết luận dựa trên cột Repeat180_Mature.")

    # ── RQ3 ──
    st.subheader("RQ3 — Mua lại khác nhau thế nào giữa các danh mục của đơn đầu tiên?")
    if d_cat is not None:
        min_n = st.slider("Số khách đủ tuổi tối thiểu mỗi danh mục", 100, 1000, 300, 50,
                          help="Danh mục quá nhỏ cho tỷ lệ không ổn định (khoảng tin cậy rất rộng)")
        dc = d_cat[d_cat["Mature_Customers"] >= min_n].sort_values("Repeat180_Mature", ascending=False)
        fig_c = px.bar(dc.sort_values("Repeat180_Mature"), x="Repeat180_Mature", y="category", orientation="h",
                       error_x=ci95(dc.sort_values("Repeat180_Mature")["Repeat180_Mature"],
                                    dc.sort_values("Repeat180_Mature")["Mature_Customers"]),
                       hover_data={"Mature_Customers": ":,"}, color_discrete_sequence=["#55A868"],
                       title=f"Tỷ lệ mua lại theo danh mục đơn đầu (≥{min_n} khách đủ tuổi, CI 95%)",
                       labels={"Repeat180_Mature": "Tỷ lệ mua lại", "category": ""})
        fig_c.update_layout(height=max(380, 24 * len(dc)), xaxis_tickformat=".0%")
        st.plotly_chart(fig_c, use_container_width=True)
        if len(dc) >= 2:
            hi, lo = dc.iloc[0], dc.iloc[-1]
            st.markdown(
                f"Cao nhất là `{hi['category']}` ({hi['Repeat180_Mature']:.1%}), thấp nhất là "
                f"`{lo['category']}` ({lo['Repeat180_Mature']:.1%}): chênh khoảng "
                f"{hi['Repeat180_Mature'] / lo['Repeat180_Mature']:.1f} lần. Ba danh mục cao nhất: "
                + ", ".join(f"`{x}`" for x in dc.head(3)["category"]) + "; ba danh mục thấp nhất: "
                + ", ".join(f"`{x}`" for x in dc.tail(3)["category"])
                + ". Nguyên nhân (ví dụ tính chất mua một lần của một số mặt hàng) mới là giả thuyết chưa kiểm chứng."
            )
        with st.expander("Bảng chi tiết theo danh mục"):
            st.dataframe(dc, use_container_width=True)

    # ── RQ4 ──
    st.subheader("RQ4 — Trải nghiệm giao hàng có liên quan đến việc quay lại không?")
    if d_del is not None:
        dims = d_del["Dimension"].unique().tolist()
        base_del = (d_del["Repeat180"] * d_del["Customers"]).sum() / d_del["Customers"].sum()
        fig_d = make_subplots(rows=1, cols=len(dims), subplot_titles=[d.replace("_", " ") for d in dims],
                              shared_yaxes=True)
        for i, dim in enumerate(dims, start=1):
            d = d_del[d_del["Dimension"] == dim]
            fig_d.add_trace(go.Bar(
                x=d["Bucket"], y=d["Repeat180"], error_y=dict(type="data", array=ci95(d["Repeat180"], d["Customers"])),
                marker_color="#4C72B0", customdata=d["Customers"],
                hovertemplate="%{x}<br>Mua lại: %{y:.2%}<br>Khách: %{customdata:,}<extra></extra>"), row=1, col=i)
            fig_d.add_hline(y=base_del, line_dash="dash", line_color="grey", row=1, col=i)
        fig_d.update_layout(title="Tỷ lệ mua lại theo trải nghiệm giao hàng (nhóm khách đủ tuổi, CI 95%; đường đứt = mức chung)",
                            height=400, showlegend=False, yaxis_tickformat=".1%")
        st.plotly_chart(fig_d, use_container_width=True)
        with st.expander("Bảng chi tiết"):
            st.dataframe(d_del, use_container_width=True)

    if assoc is not None:
        st.markdown("**Kiểm định mức độ liên hệ (chi-square + Cramér's V)** — nhóm khách đủ tuổi")
        st.dataframe(assoc.style.format({"Chi2": "{:.1f}", "p_value": "{:.3g}", "Cramers_V": "{:.3f}"}),
                     use_container_width=True)
        sig = assoc[assoc["p_value"] < 0.05]["Dimension"].tolist()
        st.info(
            "**Kết luận thận trọng**: mọi yếu tố đều có hiệu ứng yếu (Cramér's V < 0.1). "
            f"Chỉ {', '.join(sig) if sig else 'không yếu tố nào'} có ý nghĩa thống kê; thời gian giao, giao trễ và điểm đánh giá "
            "không khác biệt có ý nghĩa với việc mua lại trên dữ liệu Olist. Xu hướng giảm nhẹ khi giao chậm có "
            "tồn tại nhưng nằm trong sai số lấy mẫu. Vì vậy 'cải thiện giao hàng sẽ giữ chân khách' là giả thuyết "
            "chưa được dữ liệu ủng hộ mạnh, cần A/B test trước khi đầu tư."
        )

    st.divider()
    st.subheader("Kiểm tra nhất quán giữa các cấp tổng hợp")
    if checks is not None:
        ok = (checks["Kết luận"] == "Khớp").sum()
        st.success(f"{ok}/{len(checks)} phép kiểm tra khớp (cộng các cấp phân khúc / bang / danh mục lại ra đúng tổng).")
        with st.expander("Xem chi tiết"):
            st.dataframe(checks, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 6 — Quyết định & Hạn chế
# ═══════════════════════════════════════════════
with tab6:
    st.header("Từ phân tích đến quyết định")
    st.markdown("""
    Bảng dưới chuyển các phát hiện thành hành động theo từng phân khúc. Dữ liệu Olist là dữ liệu quan sát,
    nên cột *Tác động* là **kịch bản** ("nếu uplift đạt X thì giá trị là Y"), không phải dự báo.
    Con số đáng chú ý nhất là **điểm hòa vốn**: chi phí tối đa cho mỗi khách được nhắm để chiến dịch không lỗ.
    """)
    act   = load("action_table.csv")
    scen  = load("impact_scenarios.csv")
    inst  = load("installments_repeat.csv")
    prof  = load("segment_profile.csv")

    if act is None:
        st.warning("Run pipeline.py first to generate results.")
        st.stop()

    st.subheader("Bảng hành động: Phân khúc → Hành động → KPI → Tác động → Cách kiểm chứng")
    st.dataframe(act, use_container_width=True, hide_index=True)

    st.subheader("Kịch bản tác động và điểm hòa vốn")
    if scen is not None:
        be_col = [c for c in scen.columns if c.startswith("Chi phí hòa vốn")][0]
        fig_sc = px.bar(scen, x="Phân khúc", y="Doanh thu thêm (R$)", color="Giả định uplift", barmode="group",
                        text_auto=",.0f", hover_data={be_col: ":.2f", "% doanh thu hiện tại": True},
                        title="Doanh thu thêm theo kịch bản uplift (rê chuột để xem chi phí hòa vốn)",
                        color_discrete_sequence=["#9ecae1", "#4C72B0", "#08306b"])
        fig_sc.update_layout(height=400, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_sc, use_container_width=True)
    st.markdown(
        "Ngay cả khi chiến dịch làm tăng 1 điểm % khách mua đơn thứ hai ở nhóm lớn nhất, giá trị chỉ vào khoảng "
        "0.4% doanh thu, và ngân sách hòa vốn chỉ vài R$ mỗi khách. Vì vậy hành động nên **rẻ và tự động**, "
        "chiết khấu sâu gần như chắc chắn làm lỗ."
    )

    st.subheader("Máy tính hòa vốn (tự thử giả định)")
    if prof is not None:
        prof = prof.copy()
        prof["AOV"] = prof["Avg_Monetary"] / prof["Avg_Frequency"]
        prof = prof.set_index("Segment_Label")
        c1, c2, c3, c4 = st.columns(4)
        seg = c1.selectbox("Phân khúc nhắm tới", list(prof.index), index=list(prof.index).index("Recent Buyers"))
        upl = c2.slider("Uplift (điểm % khách mua đơn 2)", 0.0, 5.0, 1.0, 0.1)
        cost = c3.slider("Chi phí mỗi khách được nhắm (R$)", 0.0, 10.0, 1.0, 0.1)
        mar = c4.slider("Biên lợi nhuận giả định (%)", 5, 100, 20, 5)
        n = int(prof.loc[seg, "Count"]); aov = float(prof.loc[seg, "AOV"])
        extra_rev = n * upl / 100 * aov
        profit = extra_rev * mar / 100
        spend = n * cost
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Doanh thu thêm", f"R${extra_rev:,.0f}")
        m2.metric("Lợi nhuận thêm", f"R${profit:,.0f}")
        m3.metric("Chi phí chiến dịch", f"R${spend:,.0f}")
        m4.metric("Lãi/lỗ ròng", f"R${profit - spend:,.0f}",
                  delta="có lãi" if profit - spend >= 0 else "lỗ", delta_color="normal")
        be_cost = upl / 100 * aov * mar / 100
        st.caption(f"Với giả định này, chi phí tối đa để không lỗ là R${be_cost:.2f} mỗi khách "
                   f"(giá trị đơn trung bình R${aov:.0f}). Biên lợi nhuận không có trong dữ liệu Olist; "
                   "đây là tham số để người dùng tự đặt.")

    if inst is not None:
        st.subheader("Bằng chứng cho giả thuyết trả góp")
        fig_i = px.bar(inst, x="Số kỳ trả góp", y="Repeat180", text_auto=".2%",
                       title="Tỷ lệ mua lại trong 180 ngày theo số kỳ trả góp của đơn đầu (nhóm khách đủ tuổi)")
        fig_i.update_layout(height=340, yaxis_tickformat=".1%", yaxis_title="Tỷ lệ mua lại")
        st.plotly_chart(fig_i, use_container_width=True)
        st.caption("Xu hướng tăng rõ và có ý nghĩa thống kê, nhưng hiệu ứng yếu (Cramér's V ≈ 0.02). "
                   "Có thể là chọn lọc: khách chọn trả góp vốn đã khác nhóm còn lại. Cần A/B test.")

    st.divider()
    st.header("Thảo luận")
    st.markdown("""
    **Điều dữ liệu ủng hộ.** (1) Doanh thu tập trung: top 10% khách tạo khoảng 38% doanh thu, và nhóm
    Recent Buyers đóng góp nhiều nhất. (2) Danh mục của đơn đầu tiên và số kỳ trả góp là hai yếu tố có liên hệ
    thống kê với việc mua lại, dù yếu. (3) Nhóm Repeat Buyers rất nhỏ nhưng chi tiêu cao nhất, nên đáng bảo vệ.

    **Điều dữ liệu chưa ủng hộ.** Bang, thời gian giao, giao trễ và điểm đánh giá không khác biệt có ý nghĩa với việc
    quay lại. Giả thuyết phổ biến "giao nhanh hơn thì khách quay lại" vì vậy chưa được xác nhận ở đây. Khách đánh giá
    1 sao vẫn mua lại gần như bằng khách đánh giá 5 sao, điều này gợi ý Olist là marketplace nơi khách chủ yếu mua một lần theo nhu cầu.

    **Bài học phương pháp.** Hai kết quả đầu tiên trông rất đẹp (AUC 0.99 rồi 0.70–0.75) nhưng đều do rò rỉ dữ liệu:
    trực tiếp (Recency, Frequency), gián tiếp (segment) và theo lịch (ngày mua đầu tiên, vì khách mua gần cuối dữ liệu
    chưa kịp quay lại). Chỉ khi giới hạn vào nhóm khách đủ tuổi mới thấy mức dự báo thực: yếu (AUC ≈ 0.55–0.59).

    **Mô hình dùng để làm gì.** Trên tập kiểm tra, nhóm 10% khách có điểm dự báo cao nhất ghi nhận tỷ lệ mua lại cao
    hơn khoảng 1.8 lần so với mức nền (mô tả dữ liệu, với khoảng tin cậy rộng vì chỉ có vài chục khách mua lại trong nhóm).
    Điều này đủ để xếp hạng ưu tiên nguồn lực, không đủ để kết luận từng cá nhân. Quan trọng hơn: điểm dự báo cho biết ai
    *có khả năng* mua lại, chưa cho biết ai *bị tác động* bởi chiến dịch (dự báo ≠ uplift); người vốn sẵn sàng mua lại vẫn
    sẽ mua dù không được nhắn tin. Vì vậy cần nhóm đối chứng (holdout) khi triển khai.
    """)

    st.header("Hạn chế")
    st.markdown("""
    1. **Quan sát, không nhân quả.** Mọi liên hệ (trả góp, danh mục, giao hàng) chỉ là tương quan; chọn lọc và biến nhiễu chưa được kiểm soát. Hành động đề xuất đều kèm A/B test.
    2. **Nhãn mua lại hiếm (3.1%).** Mẫu 56,035 khách đủ tuổi cho ít khách dương tính, nên mô hình yếu và các so sánh nhóm nhỏ có khoảng tin cậy rộng.
    3. **Thiếu chi phí và biên lợi nhuận.** Kịch bản chỉ tính trên doanh thu; điểm hòa vốn phải nhân thêm biên lợi nhuận thực của doanh nghiệp.
    4. **Phạm vi dữ liệu.** Một marketplace, giai đoạn 2016–2018, không có dữ liệu quảng cáo, kênh tiếp cận hay hành vi duyệt web.
    5. **Chọn k = 4 theo ý nghĩa kinh doanh.** Silhouette cao nhất ở k = 2, nhưng k = 2 chỉ tách khoảng 3% khách mua lại khỏi phần còn lại.
    6. **Benchmark chạy trên một máy.** `local[*]` dùng nhiều core nhưng chưa phải cluster, dữ liệu ×10 là nhân bản; kết quả cho thấy xu hướng, không phải bằng chứng về hiệu năng cluster.
    7. **Điểm đánh giá chỉ có ở khách chịu đánh giá**, nên phân tích theo sao có thể lệch về nhóm khách hay phản hồi.
    """)
