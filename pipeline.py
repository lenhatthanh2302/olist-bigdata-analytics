import os, time, warnings, shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

warnings.filterwarnings("ignore")

BASE    = r"C:\Users\ThanhT\OneDrive - Keyloop\Desktop\Thạc sĩ\Nghiên cứu dữ liệu lớn và Ứng dụng trong kinh doanh\Quá trình & Final"
DATA    = os.path.join(BASE, "Data")
RESULTS = os.path.join(BASE, "Results")
os.makedirs(RESULTS, exist_ok=True)

# Xóa thư mục parquet cũ do PySpark tạo (nếu có) để pandas ghi file bình thường
for _p in ["master.parquet", "rfm.parquet", "segmented.parquet"]:
    _full = os.path.join(RESULTS, _p)
    if os.path.isdir(_full):
        shutil.rmtree(_full)

spark = (SparkSession.builder
         .appName("Olist_Analytics")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# ─────────────────────────────────────────────
# STEP 1 — Load & EDA
# ─────────────────────────────────────────────
#%% Step 1 - Load & EDA
print("=" * 55)
print("STEP 1 — Load & EDA")
print("=" * 55)

orders   = spark.read.csv(f"{DATA}/olist_orders_dataset.csv",         header=True, inferSchema=True)
customers= spark.read.csv(f"{DATA}/olist_customers_dataset.csv",      header=True, inferSchema=True)
items    = spark.read.csv(f"{DATA}/olist_order_items_dataset.csv",    header=True, inferSchema=True)
payments = spark.read.csv(f"{DATA}/olist_order_payments_dataset.csv", header=True, inferSchema=True)
reviews  = spark.read.csv(f"{DATA}/olist_order_reviews_dataset.csv",  header=True, inferSchema=True)
products = spark.read.csv(f"{DATA}/olist_products_dataset.csv",       header=True, inferSchema=True)
category = spark.read.csv(f"{DATA}/product_category_name_translation.csv", header=True, inferSchema=True)

items_agg = items.groupBy("order_id").agg(
    F.sum("price").alias("total_price"),
    F.sum("freight_value").alias("total_freight"),
    F.count("*").alias("item_count")
)
pay_agg = payments.groupBy("order_id").agg(
    F.sum("payment_value").alias("payment_value")
)
rev_agg = reviews.groupBy("order_id").agg(
    F.avg("review_score").alias("avg_review_score")
)

master = (orders
          .join(customers, "customer_id", "left")
          .join(items_agg, "order_id",    "left")
          .join(pay_agg,   "order_id",    "left")
          .join(rev_agg,   "order_id",    "left")
          .withColumn("order_purchase_timestamp",
                      F.to_timestamp("order_purchase_timestamp"))
          .filter(F.col("order_status") == "delivered")
          .filter(F.col("order_purchase_timestamp").isNotNull())
          .filter(F.col("payment_value").isNotNull()))

total_orders    = master.count()
total_customers = master.select("customer_unique_id").distinct().count()
total_revenue   = master.agg(F.sum("payment_value")).collect()[0][0]
avg_order_value = master.agg(F.avg("payment_value")).collect()[0][0]

print(f"  Total Delivered Orders : {total_orders:,}")
print(f"  Unique Customers       : {total_customers:,}")
print(f"  Total Revenue (R$)     : {total_revenue:,.2f}")
print(f"  Avg Order Value (R$)   : {avg_order_value:.2f}")

pd.DataFrame({
    "Metric": ["Total Orders", "Unique Customers", "Total Revenue (R$)", "Avg Order Value (R$)"],
    "Value":  [f"{total_orders:,}", f"{total_customers:,}",
               f"{total_revenue:,.2f}", f"{avg_order_value:.2f}"]
}).to_csv(f"{RESULTS}/eda_summary.csv", index=False)

# Monthly trend
monthly = (master
           .withColumn("month", F.date_format("order_purchase_timestamp", "yyyy-MM"))
           .groupBy("month").count()
           .orderBy("month")
           .toPandas())
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(monthly["month"], monthly["count"], marker="o", color="steelblue", linewidth=1.5, label="Monthly Orders")
ax.set_title("Monthly Orders Trend (Delivered)", fontsize=13)
ax.set_xlabel("Month"); ax.set_ylabel("Orders")
ax.legend()
plt.xticks(rotation=45, ha="right"); plt.tight_layout()
plt.savefig(f"{RESULTS}/eda_monthly_trend.png", dpi=150); plt.show(); plt.close()

# Top 10 product categories (English)
items_cat = (items
             .join(products.select("product_id", "product_category_name"), "product_id", "left")
             .join(category, "product_category_name", "left"))
top_cat = (items_cat
           .groupBy("product_category_name_english")
           .count()
           .orderBy(F.desc("count"))
           .limit(10)
           .toPandas())
top_cat.to_csv(f"{RESULTS}/eda_top_categories.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(top_cat["product_category_name_english"][::-1],
        top_cat["count"][::-1], color="steelblue", label="Item Count")
ax.set_title("Top 10 Product Categories"); ax.set_xlabel("Number of Items")
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS}/eda_top_categories.png", dpi=150); plt.show(); plt.close()

master.toPandas().to_parquet(f"{RESULTS}/master.parquet", index=False, coerce_timestamps="us", allow_truncated_timestamps=True)
print("  STEP 1 done ✓\n")

# ─────────────────────────────────────────────
# STEP 2 — RFM
# ─────────────────────────────────────────────
#%% Step 2 - RFM
print("=" * 55)
print("STEP 2 — RFM Computation")
print("=" * 55)

master = spark.read.parquet(f"{RESULTS}/master.parquet")

max_ts  = master.agg(F.max("order_purchase_timestamp")).collect()[0][0]
ref_ts  = max_ts + pd.Timedelta(days=1)
ref_lit = F.lit(ref_ts.strftime("%Y-%m-%d %H:%M:%S")).cast("timestamp")

rfm = (master
       .groupBy("customer_unique_id")
       .agg(
           F.datediff(ref_lit, F.max("order_purchase_timestamp")).alias("Recency"),
           F.count("order_id").alias("Frequency"),
           F.sum("payment_value").alias("Monetary")
       )
       .filter(F.col("Monetary") > 0))

# Score 1–5 via ntile (Recency: lower = better → reverse)
w_asc  = Window.orderBy(F.col("Recency").desc())   # high recency score for LOW recency
w_freq = Window.orderBy("Frequency")
w_mon  = Window.orderBy("Monetary")

rfm = (rfm
       .withColumn("R_score", F.ntile(5).over(w_asc))
       .withColumn("F_score", F.ntile(5).over(w_freq))
       .withColumn("M_score", F.ntile(5).over(w_mon))
       .withColumn("RFM_Score", F.col("R_score") + F.col("F_score") + F.col("M_score")))

rfm.toPandas().to_parquet(f"{RESULTS}/rfm.parquet", index=False)
rfm.toPandas().to_csv(f"{RESULTS}/rfm.csv", index=False)
print(f"  RFM records : {rfm.count():,}")
print("  STEP 2 done ✓\n")

# ─────────────────────────────────────────────
# STEP 3 — Customer Segmentation (K-Means)
# ─────────────────────────────────────────────
#%% Step 3 - Segmentation
print("=" * 55)
print("STEP 3 — Customer Segmentation")
print("=" * 55)

rfm = spark.read.parquet(f"{RESULTS}/rfm.parquet")

assembler = VectorAssembler(inputCols=["Recency", "Frequency", "Monetary"], outputCol="features_raw")
scaler    = StandardScaler(inputCol="features_raw", outputCol="features",
                           withStd=True, withMean=True)

assembled    = assembler.transform(rfm)
scaler_model = scaler.fit(assembled)
scaled       = scaler_model.transform(assembled)

# Elbow method k=2..6
wssse = []
for k in range(2, 7):
    km = KMeans(k=k, seed=42, featuresCol="features")
    wssse.append((k, km.fit(scaled).summary.trainingCost))

ks, costs = zip(*wssse)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ks, costs, marker="o", color="coral", linewidth=1.5, label="WSSSE")
ax.set_title("Elbow Method — Optimal K"); ax.set_xlabel("K"); ax.set_ylabel("WSSSE")
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS}/kmeans_elbow.png", dpi=150); plt.show(); plt.close()

# Final model k=4
km_model  = KMeans(k=4, seed=42, featuresCol="features", predictionCol="segment").fit(scaled)
segmented = km_model.transform(scaled)

seg_profile = (segmented
               .groupBy("segment")
               .agg(
                   F.avg("Recency").alias("Avg_Recency"),
                   F.avg("Frequency").alias("Avg_Frequency"),
                   F.avg("Monetary").alias("Avg_Monetary"),
                   F.count("*").alias("Count")
               )
               .orderBy("segment")
               .toPandas())

# Label segments by business meaning
def label_segment(row):
    if row["Avg_Recency"] < 100 and row["Avg_Frequency"] > 1.5:
        return "Champions"
    elif row["Avg_Recency"] < 150 and row["Avg_Monetary"] > 200:
        return "Loyal Customers"
    elif row["Avg_Recency"] > 300:
        return "Lost Customers"
    else:
        return "At-Risk Customers"

seg_profile["Segment_Label"] = seg_profile.apply(label_segment, axis=1)
seg_profile.to_csv(f"{RESULTS}/segment_profile.csv", index=False)
print(seg_profile[["segment", "Segment_Label", "Avg_Recency", "Avg_Frequency", "Avg_Monetary", "Count"]])

(segmented
 .select("customer_unique_id", "Recency", "Frequency", "Monetary",
         "R_score", "F_score", "M_score", "RFM_Score", "segment")
 .toPandas()
 .to_parquet(f"{RESULTS}/segmented.parquet", index=False))
(segmented
 .select("customer_unique_id", "Recency", "Frequency", "Monetary", "RFM_Score", "segment")
 .toPandas()
 .to_csv(f"{RESULTS}/segmented.csv", index=False))
print("  STEP 3 done ✓\n")

# ─────────────────────────────────────────────
# STEP 4 — Churn Prediction (LR + RF)
# ─────────────────────────────────────────────
#%% Step 4 - Churn Prediction
print("=" * 55)
print("STEP 4 — Churn Prediction")
print("=" * 55)

segmented = spark.read.parquet(f"{RESULTS}/segmented.parquet")

# Churn: bought only once AND last purchase > 180 days ago
churn_df = segmented.withColumn(
    "churn",
    F.when((F.col("Frequency") == 1) & (F.col("Recency") > 180), 1)
     .otherwise(0)
     .cast("int")
)
churn_rate = churn_df.filter(F.col("churn") == 1).count() / churn_df.count()
print(f"  Churn rate : {churn_rate:.1%}")

feat_assembler = VectorAssembler(
    inputCols=["Monetary", "M_score", "segment"],
    outputCol="features"
)
churn_df = feat_assembler.transform(churn_df)
train, test = churn_df.randomSplit([0.8, 0.2], seed=42)

lr = LogisticRegression(featuresCol="features", labelCol="churn", maxIter=100)
lr_model = lr.fit(train)
lr_pred  = lr_model.transform(test)

rf = RandomForestClassifier(featuresCol="features", labelCol="churn",
                            numTrees=100, seed=42)
rf_model = rf.fit(train)
rf_pred  = rf_model.transform(test)

ev_auc = BinaryClassificationEvaluator(labelCol="churn", metricName="areaUnderROC")
ev_acc = MulticlassClassificationEvaluator(labelCol="churn", predictionCol="prediction",
                                           metricName="accuracy")
ev_f1  = MulticlassClassificationEvaluator(labelCol="churn", predictionCol="prediction",
                                           metricName="f1")

metrics = pd.DataFrame({
    "Model":    ["Logistic Regression", "Random Forest"],
    "AUC-ROC":  [round(ev_auc.evaluate(lr_pred), 4), round(ev_auc.evaluate(rf_pred), 4)],
    "Accuracy": [round(ev_acc.evaluate(lr_pred), 4), round(ev_acc.evaluate(rf_pred), 4)],
    "F1-Score": [round(ev_f1.evaluate(lr_pred),  4), round(ev_f1.evaluate(rf_pred),  4)],
})
metrics.to_csv(f"{RESULTS}/churn_metrics.csv", index=False)
print(metrics.to_string(index=False))

(rf_pred
 .select("customer_unique_id", "Recency", "Frequency", "Monetary", "M_score", "segment", "churn", "prediction")
 .toPandas()
 .to_csv(f"{RESULTS}/churn_predictions.csv", index=False))
print("  STEP 4 done ✓\n")

# ─────────────────────────────────────────────
# STEP 5 — Tool Benchmark (PySpark vs Pandas)
# ─────────────────────────────────────────────
#%% Step 5 - Benchmark
print("=" * 55)
print("STEP 5 — Tool Benchmark: PySpark vs Pandas")
print("=" * 55)

from sklearn.preprocessing import StandardScaler as SKScaler
from sklearn.cluster import KMeans as SKMeans

# --- Pandas RFM ---
t0 = time.time()
df_ord  = pd.read_csv(f"{DATA}/olist_orders_dataset.csv")
df_cus  = pd.read_csv(f"{DATA}/olist_customers_dataset.csv")
df_itm  = pd.read_csv(f"{DATA}/olist_order_items_dataset.csv")
df_pay  = pd.read_csv(f"{DATA}/olist_order_payments_dataset.csv")

df = (df_ord.merge(df_cus, on="customer_id")
            .merge(df_pay.groupby("order_id")["payment_value"].sum().reset_index(),
                   on="order_id", how="left"))
df = df[df["order_status"] == "delivered"].copy()
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
ref_pd = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
rfm_pd = (df.groupby("customer_unique_id")
            .agg(Recency=("order_purchase_timestamp", lambda x: (ref_pd - x.max()).days),
                 Frequency=("order_id", "count"),
                 Monetary=("payment_value", "sum"))
            .reset_index())
rfm_pd = rfm_pd[rfm_pd["Monetary"] > 0]
_ = rfm_pd.shape   # force eval
pandas_rfm_time = round(time.time() - t0, 3)

# --- Pandas K-Means ---
t0 = time.time()
X_pd = SKScaler().fit_transform(rfm_pd[["Recency", "Frequency", "Monetary"]])
SKMeans(n_clusters=4, random_state=42, n_init=10).fit(X_pd)
pandas_km_time = round(time.time() - t0, 3)

# --- PySpark RFM ---
t0 = time.time()
sp_orders = spark.read.csv(f"{DATA}/olist_orders_dataset.csv", header=True, inferSchema=True)
sp_cust   = spark.read.csv(f"{DATA}/olist_customers_dataset.csv", header=True, inferSchema=True)
sp_pay    = spark.read.csv(f"{DATA}/olist_order_payments_dataset.csv", header=True, inferSchema=True)
sp_pay_agg = sp_pay.groupBy("order_id").agg(F.sum("payment_value").alias("payment_value"))
sp_master = (sp_orders.join(sp_cust, "customer_id", "left")
                      .join(sp_pay_agg, "order_id", "left")
                      .filter(F.col("order_status") == "delivered")
                      .withColumn("order_purchase_timestamp",
                                  F.to_timestamp("order_purchase_timestamp")))
ref_sp = F.lit((sp_master.agg(F.max("order_purchase_timestamp")).collect()[0][0]
                + pd.Timedelta(days=1)).strftime("%Y-%m-%d")).cast("timestamp")
rfm_sp = (sp_master
          .groupBy("customer_unique_id")
          .agg(F.datediff(ref_sp, F.max("order_purchase_timestamp")).alias("Recency"),
               F.count("order_id").alias("Frequency"),
               F.sum("payment_value").alias("Monetary"))
          .filter(F.col("Monetary") > 0))
rfm_sp.count()   # trigger action
spark_rfm_time = round(time.time() - t0, 3)

# --- PySpark K-Means ---
t0 = time.time()
asm_b = VectorAssembler(inputCols=["Recency","Frequency","Monetary"], outputCol="fr").transform(rfm_sp)
scl_b = StandardScaler(inputCol="fr", outputCol="fs", withStd=True, withMean=True).fit(asm_b).transform(asm_b)
KMeans(k=4, seed=42, featuresCol="fs").fit(scl_b)
spark_km_time = round(time.time() - t0, 3)

benchmark = pd.DataFrame({
    "Task":         ["RFM Computation", "K-Means Clustering"],
    "Pandas (s)":   [pandas_rfm_time, pandas_km_time],
    "PySpark (s)":  [spark_rfm_time,  spark_km_time],
})
benchmark["Speedup (Pandas/PySpark)"] = (benchmark["Pandas (s)"] / benchmark["PySpark (s)"]).round(2)
benchmark.to_csv(f"{RESULTS}/benchmark.csv", index=False)
print(benchmark.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(benchmark))
w = 0.35
ax.bar(x - w/2, benchmark["Pandas (s)"],  w, label="Pandas",  color="#4C72B0")
ax.bar(x + w/2, benchmark["PySpark (s)"], w, label="PySpark", color="#DD8452")
ax.set_xticks(x); ax.set_xticklabels(benchmark["Task"])
ax.set_ylabel("Time (seconds)")
ax.set_title("Execution Time: Pandas vs PySpark (local mode, 100K records)")
ax.legend(); plt.tight_layout()
plt.savefig(f"{RESULTS}/benchmark.png", dpi=150); plt.show(); plt.close()
print("  STEP 5 done ✓\n")

# ─────────────────────────────────────────────
# STEP 6 — XAI (SHAP)
# ─────────────────────────────────────────────
#%% Step 6 - SHAP
print("=" * 55)
print("STEP 6 — XAI: SHAP Feature Importance")
print("=" * 55)

try:
    import shap
    from sklearn.ensemble import RandomForestClassifier as SKRf

    churn_pdf = pd.read_csv(f"{RESULTS}/churn_predictions.csv")
    feature_cols = ["Monetary", "M_score", "segment"]
    X = churn_pdf[feature_cols]
    y = churn_pdf["churn"].astype(int)

    sk_rf = SKRf(n_estimators=100, random_state=42)
    sk_rf.fit(X, y)

    explainer   = shap.TreeExplainer(sk_rf)
    shap_values = explainer.shap_values(X)
    # Normalize across SHAP API versions for binary classification (class 1):
    #   list  → old API: [class0_arr, class1_arr]
    #   3-D   → new API: (n_samples, n_features, n_classes)
    #   2-D   → some mid versions: (n_samples, n_features) already for positive class
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # Bar importance
    plt.figure(figsize=(7, 4))
    shap.summary_plot(sv, X, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Churn Prediction)")
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/shap_bar.png", dpi=150, bbox_inches="tight"); plt.show(); plt.close()

    # Beeswarm
    plt.figure(figsize=(8, 5))
    shap.summary_plot(sv, X, show=False)
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.show(); plt.close()

    # Save mean |SHAP| values
    shap_importance = pd.DataFrame({
        "Feature":          feature_cols,
        "Mean |SHAP|":      np.abs(sv).mean(axis=0)
    }).sort_values("Mean |SHAP|", ascending=False)
    shap_importance.to_csv(f"{RESULTS}/shap_importance.csv", index=False)
    print(shap_importance.to_string(index=False))
    print("  STEP 6 done ✓\n")

except ImportError:
    print("  [SKIP] shap not installed. Run: pip install shap")
    print("  STEP 6 skipped.\n")

spark.stop()
print("=" * 55)
print("Pipeline complete. Results saved to:", RESULTS)
print("=" * 55)
