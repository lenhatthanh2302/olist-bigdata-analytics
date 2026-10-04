# =============================================================================
# BÀI CÁ NHÂN — MÔN: NGHIÊN CỨU DỮ LIỆU LỚN VÀ ỨNG DỤNG TRONG KINH DOANH
# Đề tài  : Phân tích hành vi khách hàng thương mại điện tử Brazil bằng PySpark
#           (EDA → RFM → K-Means → Churn → Benchmark → SHAP → RQ → Hành động)
# Dataset : Brazilian E-Commerce Public Dataset by Olist (Kaggle) — 9 file CSV
# =============================================================================

#%%
# =====================================================================
# 0. KHAI BÁO THƯ VIỆN & CẤU HÌNH TOÀN CỤC
# =====================================================================
import os
import time
import shutil
import warnings
from functools import reduce

import numpy as np
import pandas as pd
import matplotlib

# Chạy trong PyCharm thì để TkAgg cho plt.show() hiện cửa sổ.
# Chạy tự động (không có màn hình) thì đặt OLIST_HEADLESS=1 để chuyển sang Agg.
matplotlib.use("Agg" if os.environ.get("OLIST_HEADLESS") == "1" else "TkAgg")
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import (VectorAssembler, StandardScaler, Imputer,
                                StringIndexer, OneHotEncoder)
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import (LogisticRegression, RandomForestClassifier,
                                       GBTClassifier)
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator,
                                   ClusteringEvaluator)

warnings.filterwarnings("ignore")

# --- Đường dẫn: dùng đường dẫn tương đối để chạy được cả local lẫn trên máy khác ---
BASE    = os.path.dirname(os.path.abspath(__file__))
DATA    = os.path.join(BASE, "Data")
RESULTS = os.path.join(BASE, "Results")
os.makedirs(RESULTS, exist_ok=True)

# --- Tham số chung ---
SEED        = 42     # cố định seed để kết quả tái lập được
CHURN_DAYS  = 180    # ngưỡng Recency để coi là đã rời bỏ
# Silhouette cao nhất ở k=2, nhưng k=2 chỉ tách ~3% khách mua lại khỏi ~97% còn lại
# (ranh giới Frequency hiển nhiên), không đủ để làm CRM. Chọn k=4 là lựa chọn kinh doanh
# có chủ đích, đánh đổi một phần chất lượng cụm về mặt toán học.
K_FINAL     = 4

# Benchmark: nhân bản dữ liệu gốc (~100K đơn) lên nhiều lần để quan sát xu hướng
BENCH_SCALES  = [1, 2, 5, 10]
BENCH_REPEATS = 3

# --- Phong cách biểu đồ thống nhất toàn pipeline ---
plt.rcParams.update({"axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.6,
                     "font.family": "sans-serif"})
COLOR_MAIN   = "steelblue"
COLOR_PANDAS = "#4C72B0"
COLOR_SPARK  = "#DD8452"
COLOR_GBT    = "#55A868"

# Spark đôi khi để lại thư mục .parquet cũ trong Results -> xóa trước để pandas ghi file đơn
for _p in ["master.parquet", "rfm.parquet", "segmented.parquet", "churn_features.parquet"]:
    _full = os.path.join(RESULTS, _p)
    if os.path.isdir(_full):
        shutil.rmtree(_full)

spark = (SparkSession.builder
         .appName("Olist_Analytics")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8")   # máy local nên không cần 200 partition mặc định
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

print("=" * 70)
print("PIPELINE PHÂN TÍCH HÀNH VI KHÁCH HÀNG — OLIST E-COMMERCE (PySpark)")
print("=" * 70)
print(f"  Thư mục dữ liệu : {DATA}")
print(f"  Thư mục kết quả : {RESULTS}")
print(f"  Spark version   : {spark.version}")


#%%
# =====================================================================
# 1. NẠP DỮ LIỆU & PHÂN TÍCH KHÁM PHÁ (EDA)
# =====================================================================
# Mục tiêu: nối 9 file CSV rời rạc thành 1 bảng master ở cấp đơn hàng,
# làm sạch (chỉ giữ đơn đã giao có thanh toán) và nắm bức tranh chung
# về doanh thu, xu hướng theo tháng và danh mục sản phẩm.
# =====================================================================
print("\n" + "=" * 70)
print("BƯỚC 1 — NẠP DỮ LIỆU & EDA")
print("=" * 70)

orders    = spark.read.csv(f"{DATA}/olist_orders_dataset.csv",              header=True, inferSchema=True)
customers = spark.read.csv(f"{DATA}/olist_customers_dataset.csv",           header=True, inferSchema=True)
items     = spark.read.csv(f"{DATA}/olist_order_items_dataset.csv",         header=True, inferSchema=True)
payments  = spark.read.csv(f"{DATA}/olist_order_payments_dataset.csv",      header=True, inferSchema=True)
reviews   = spark.read.csv(f"{DATA}/olist_order_reviews_dataset.csv",       header=True, inferSchema=True)
products  = spark.read.csv(f"{DATA}/olist_products_dataset.csv",            header=True, inferSchema=True)
category  = spark.read.csv(f"{DATA}/product_category_name_translation.csv", header=True, inferSchema=True)

# ── Kiểm tra chất lượng dữ liệu trước khi lọc ───────────────────────
print("\n[1/4] Kiểm tra trạng thái đơn hàng ...")
n_orders_raw = orders.count()
status_dist  = (orders.groupBy("order_status").count()
                .orderBy(F.desc("count")).toPandas())
status_dist["pct"] = (status_dist["count"] / n_orders_raw * 100).round(2)
print(f"      Tổng đơn gốc: {n_orders_raw:,}")
print(status_dist.to_string(index=False))

# ── Gom các bảng chi tiết về cấp đơn hàng (1 dòng = 1 đơn) ─────────
# Một đơn có thể có nhiều sản phẩm / nhiều lần thanh toán nên phải gộp trước khi join,
# nếu không số dòng sẽ bị nhân lên và doanh thu bị tính trùng.
items_agg = items.groupBy("order_id").agg(
    F.sum("price").alias("total_price"),
    F.sum("freight_value").alias("total_freight"),
    F.count("*").alias("item_count")
)
pay_agg = payments.groupBy("order_id").agg(
    F.sum("payment_value").alias("payment_value"),
    F.max("payment_installments").alias("max_installments"),
    F.max(F.when(F.col("payment_type") == "voucher", 1).otherwise(0)).alias("used_voucher")
)
rev_agg = reviews.groupBy("order_id").agg(
    F.avg("review_score").alias("avg_review_score")
)

print("\n[2/4] Đang join các bảng thành master ...")
master = (orders
          .join(customers, "customer_id", "left")
          .join(items_agg, "order_id", "left")
          .join(pay_agg,   "order_id", "left")
          .join(rev_agg,   "order_id", "left")
          .withColumn("order_purchase_timestamp", F.to_timestamp("order_purchase_timestamp"))
          # Chỉ phân tích đơn đã giao: đơn hủy / chưa giao chưa phản ánh hành vi mua thật
          .filter(F.col("order_status") == "delivered")
          .filter(F.col("order_purchase_timestamp").isNotNull())
          .filter(F.col("payment_value").isNotNull()))

total_orders    = master.count()
total_customers = master.select("customer_unique_id").distinct().count()
total_revenue   = master.agg(F.sum("payment_value")).collect()[0][0]
avg_order_value = master.agg(F.avg("payment_value")).collect()[0][0]

print(f"      Đơn đã giao (sau làm sạch) : {total_orders:,}  "
      f"(loại {n_orders_raw - total_orders:,} đơn)")
print(f"      Khách hàng duy nhất        : {total_customers:,}")
print(f"      Tổng doanh thu (R$)        : {total_revenue:,.2f}")
print(f"      Giá trị đơn trung bình (R$): {avg_order_value:.2f}")

pd.DataFrame({
    "Metric": ["Total Orders", "Unique Customers", "Total Revenue (R$)", "Avg Order Value (R$)"],
    "Value":  [f"{total_orders:,}", f"{total_customers:,}",
               f"{total_revenue:,.2f}", f"{avg_order_value:.2f}"]
}).to_csv(f"{RESULTS}/eda_summary.csv", index=False)
status_dist.to_csv(f"{RESULTS}/eda_order_status.csv", index=False)

# ── Xu hướng đơn hàng theo tháng ────────────────────────────────────
print("\n[3/4] Đang vẽ xu hướng đơn hàng theo tháng ...")
monthly = (master
           .withColumn("month", F.date_format("order_purchase_timestamp", "yyyy-MM"))
           .groupBy("month").count()
           .orderBy("month")
           .toPandas())
monthly.to_csv(f"{RESULTS}/eda_monthly_trend.csv", index=False)   # dashboard vẽ lại bằng Plotly

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(monthly["month"], monthly["count"], marker="o", color=COLOR_MAIN,
        linewidth=1.5, label="Số đơn mỗi tháng")
ax.set_title("Xu hướng đơn hàng theo tháng (đơn đã giao)", fontsize=13)
ax.set_xlabel("Tháng"); ax.set_ylabel("Số đơn")
ax.legend()
plt.xticks(rotation=45, ha="right"); plt.tight_layout()
plt.savefig(f"{RESULTS}/eda_monthly_trend.png", dpi=150); plt.show(); plt.close()

# ── Top 10 danh mục sản phẩm ────────────────────────────────────────
print("\n[4/4] Đang tổng hợp Top 10 danh mục ...")
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
        top_cat["count"][::-1], color=COLOR_MAIN, label="Số sản phẩm bán ra")
ax.set_title("Top 10 danh mục sản phẩm"); ax.set_xlabel("Số sản phẩm")
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS}/eda_top_categories.png", dpi=150); plt.show(); plt.close()

# Lưu master ra parquet qua pandas — ghi trực tiếp bằng Spark trên Windows hay lỗi winutils
master.toPandas().to_parquet(f"{RESULTS}/master.parquet", index=False,
                             coerce_timestamps="us", allow_truncated_timestamps=True)
print("\n[✓] BƯỚC 1 hoàn tất\n")


#%%
# =====================================================================
# 2. TÍNH CHỈ SỐ RFM BẰNG WINDOW FUNCTIONS
# =====================================================================
# Mục tiêu: chấm điểm mỗi khách hàng theo 3 chiều
#   Recency   — lần mua gần nhất cách đây bao nhiêu ngày
#   Frequency — đã mua bao nhiêu đơn
#   Monetary  — tổng tiền đã chi
# rồi quy về thang 1–5 bằng ntile để các chiều so sánh được với nhau.
# =====================================================================
print("=" * 70)
print("BƯỚC 2 — TÍNH RFM")
print("=" * 70)

master = spark.read.parquet(f"{RESULTS}/master.parquet")

# Mốc tham chiếu = 1 ngày sau đơn cuối cùng trong dữ liệu (tránh Recency = 0)
max_ts  = master.agg(F.max("order_purchase_timestamp")).collect()[0][0]
ref_ts  = max_ts + pd.Timedelta(days=1)
ref_lit = F.lit(ref_ts.strftime("%Y-%m-%d %H:%M:%S")).cast("timestamp")
print(f"  Mốc tham chiếu : {ref_ts:%Y-%m-%d}")

rfm = (master
       .groupBy("customer_unique_id")
       .agg(F.datediff(ref_lit, F.max("order_purchase_timestamp")).alias("Recency"),
            F.count("order_id").alias("Frequency"),
            F.sum("payment_value").alias("Monetary"))
       .filter(F.col("Monetary") > 0))

# Recency càng NHỎ càng tốt nên sắp giảm dần để khách mua gần đây nhận điểm cao
# (Window không có partitionBy => Spark gom về 1 partition, chấp nhận được ở quy mô 100K)
w_rec  = Window.orderBy(F.col("Recency").desc())
w_freq = Window.orderBy("Frequency")
w_mon  = Window.orderBy("Monetary")

rfm = (rfm
       .withColumn("R_score", F.ntile(5).over(w_rec))
       .withColumn("F_score", F.ntile(5).over(w_freq))
       .withColumn("M_score", F.ntile(5).over(w_mon))
       .withColumn("RFM_Score", F.col("R_score") + F.col("F_score") + F.col("M_score")))

rfm_pdf = rfm.toPandas()
rfm_pdf.to_parquet(f"{RESULTS}/rfm.parquet", index=False)
print(f"  Số khách hàng có RFM : {len(rfm_pdf):,}")
print(f"  Khách mua đúng 1 lần : {(rfm_pdf['Frequency'] == 1).mean():.1%}")
print("\n[✓] BƯỚC 2 hoàn tất\n")


#%%
# =====================================================================
# 3. PHÂN KHÚC KHÁCH HÀNG BẰNG K-MEANS (MLlib)
# =====================================================================
# Mục tiêu: gom khách hàng thành các nhóm có hành vi tương đồng.
# Chọn k dựa trên 2 tiêu chí: Elbow (WSSSE) và Silhouette, sau đó
# đối chiếu với khả năng diễn giải về mặt kinh doanh.
# =====================================================================
print("=" * 70)
print("BƯỚC 3 — PHÂN KHÚC KHÁCH HÀNG (K-MEANS)")
print("=" * 70)

rfm = spark.read.parquet(f"{RESULTS}/rfm.parquet")

assembler = VectorAssembler(inputCols=["Recency", "Frequency", "Monetary"], outputCol="features_raw")
assembled = assembler.transform(rfm)

# Chuẩn hóa vì Monetary (hàng trăm) lệch thang rất xa Frequency (1–2); không chuẩn hóa
# thì K-Means gần như chỉ nhìn vào Monetary và Recency
scaler_model = StandardScaler(inputCol="features_raw", outputCol="features",
                              withStd=True, withMean=True).fit(assembled)
scaled = scaler_model.transform(assembled).cache()
scaled.count()   # ép Spark nạp vào cache để vòng lặp bên dưới không phải tính lại

# ── Chọn k: Elbow + Silhouette ───────────────────────────────────────
print("\n[1/3] Đang thử k = 2..6 ...")
sil_eval = ClusteringEvaluator(featuresCol="features", predictionCol="prediction",
                               metricName="silhouette", distanceMeasure="squaredEuclidean")
k_rows = []
for k in range(2, 7):
    km_k = KMeans(k=k, seed=SEED, featuresCol="features").fit(scaled)
    sil  = sil_eval.evaluate(km_k.transform(scaled))
    k_rows.append({"k": k, "WSSSE": km_k.summary.trainingCost, "Silhouette": sil})
    print(f"      k={k}  WSSSE={km_k.summary.trainingCost:>12,.0f}  Silhouette={sil:.4f}")

k_sel = pd.DataFrame(k_rows)
k_sel.to_csv(f"{RESULTS}/kmeans_selection.csv", index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(k_sel["k"], k_sel["WSSSE"], marker="o", color="coral", linewidth=1.5, label="WSSSE")
ax1.axvline(K_FINAL, color="gray", linestyle="--", label=f"k chọn = {K_FINAL}")
ax1.set_title("Elbow Method"); ax1.set_xlabel("k"); ax1.set_ylabel("WSSSE"); ax1.legend()
ax2.plot(k_sel["k"], k_sel["Silhouette"], marker="s", color="teal", linewidth=1.5, label="Silhouette")
ax2.axvline(K_FINAL, color="gray", linestyle="--", label=f"k chọn = {K_FINAL}")
ax2.set_title("Silhouette Score"); ax2.set_xlabel("k"); ax2.set_ylabel("Silhouette"); ax2.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS}/kmeans_elbow.png", dpi=150); plt.show(); plt.close()

# ── Mô hình chính thức ───────────────────────────────────────────────
print(f"\n[2/3] Huấn luyện K-Means chính thức với k = {K_FINAL} ...")
km_model  = KMeans(k=K_FINAL, seed=SEED, featuresCol="features", predictionCol="segment").fit(scaled)
segmented = km_model.transform(scaled)

seg_profile = (segmented
               .groupBy("segment")
               .agg(F.avg("Recency").alias("Avg_Recency"),
                    F.avg("Frequency").alias("Avg_Frequency"),
                    F.avg("Monetary").alias("Avg_Monetary"),
                    F.count("*").alias("Count"))
               .orderBy("segment")
               .toPandas())

# Đặt tên cụm SAU khi xem số liệu thực tế, theo thứ tự ưu tiên của luật.
# Lưu ý: gần như mọi cụm đều có Frequency = 1 nên không thể gọi là "Loyal";
# cụm mua gần đây chỉ là khách mới chứ chưa chứng minh được sự trung thành.
def label_segment(row):
    if row["Avg_Frequency"] > 1.5:
        return "Repeat Buyers"          # nhóm duy nhất có hành vi mua lặp lại
    elif row["Avg_Recency"] > 300:
        return "Lost Customers"         # im lặng hơn 10 tháng
    elif row["Avg_Recency"] < 150:
        return "Recent Buyers"          # mới mua, còn trong cửa sổ có thể kéo quay lại
    else:
        return "At-Risk Buyers"         # mua 1 lần và đang nguội dần

seg_profile["Segment_Label"] = seg_profile.apply(label_segment, axis=1)
seg_profile["Share_%"] = (seg_profile["Count"] / seg_profile["Count"].sum() * 100).round(1)
seg_profile.to_csv(f"{RESULTS}/segment_profile.csv", index=False)

print("\n[3/3] Hồ sơ các phân khúc:")
print(seg_profile[["segment", "Segment_Label", "Avg_Recency", "Avg_Frequency",
                   "Avg_Monetary", "Count", "Share_%"]].round(2).to_string(index=False))

label_map = dict(zip(seg_profile["segment"], seg_profile["Segment_Label"]))
seg_pdf = (segmented
           .select("customer_unique_id", "Recency", "Frequency", "Monetary",
                   "R_score", "F_score", "M_score", "RFM_Score", "segment")
           .toPandas())
seg_pdf["Segment_Label"] = seg_pdf["segment"].map(label_map)
seg_pdf.to_parquet(f"{RESULTS}/segmented.parquet", index=False)
seg_pdf[["customer_unique_id", "Recency", "Frequency", "Monetary",
         "RFM_Score", "segment", "Segment_Label"]].to_csv(f"{RESULTS}/segmented.csv", index=False)
scaled.unpersist()
print("\n[✓] BƯỚC 3 hoàn tất\n")


#%%
# =====================================================================
# 4. DỰ ĐOÁN CHURN — LR vs RF vs GBT
# =====================================================================
# Mục tiêu: dự đoán khách mua 1 lần có quay lại hay không.
#
# Định nghĩa churn: Frequency == 1 VÀ Recency > 180 ngày.
# Vì nhãn được tạo từ Recency và Frequency nên mô hình chỉ dùng thông tin từ
# ĐƠN HÀNG ĐẦU TIÊN (giá trị, phí ship, đánh giá, giao trễ, trả góp, vùng miền);
# Recency, Frequency, phân khúc và ngày mua bị loại khỏi đặc trưng.
# =====================================================================
print("=" * 70)
print("BƯỚC 4 — DỰ ĐOÁN CHURN")
print("=" * 70)

master    = spark.read.parquet(f"{RESULTS}/master.parquet")
segmented = spark.read.parquet(f"{RESULTS}/segmented.parquet")

# ── 4.1 Gắn nhãn churn ──────────────────────────────────────────────
churn_df = segmented.withColumn(
    "churn",
    F.when((F.col("Frequency") == 1) & (F.col("Recency") > CHURN_DAYS), 1.0).otherwise(0.0)
)
n_cust     = churn_df.count()
n_churn    = churn_df.filter(F.col("churn") == 1).count()
churn_rate = n_churn / n_cust
print(f"\n[1/4] Churn rate: {churn_rate:.1%}  ({n_churn:,} / {n_cust:,} khách)")
pd.DataFrame({"Metric": ["Total Customers", "Churned", "Churn Rate"],
              "Value":  [n_cust, n_churn, round(churn_rate, 4)]}
             ).to_csv(f"{RESULTS}/churn_summary.csv", index=False)

# ── 4.2 Đặc trưng từ đơn hàng đầu tiên của mỗi khách ────────────────
print("\n[2/4] Đang tạo đặc trưng từ đơn hàng đầu tiên ...")
REGION_MAP = {
    "North":      ["AC", "AP", "AM", "PA", "RO", "RR", "TO"],
    "Northeast":  ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "Central-West": ["DF", "GO", "MT", "MS"],
    "Southeast":  ["ES", "MG", "RJ", "SP"],
    "South":      ["PR", "RS", "SC"],
}
region_expr = F.lit("Other")
for _name, _states in REGION_MAP.items():
    region_expr = F.when(F.col("customer_state").isin(_states), _name).otherwise(region_expr)

w_first = Window.partitionBy("customer_unique_id").orderBy("order_purchase_timestamp")

# Hai cột phục vụ kiểm tra thiên lệch thời gian:
#   first_day      — đơn đầu tiên cách ngày đầu dataset bao nhiêu ngày
#   days_to_second — sau bao nhiêu ngày khách mua đơn thứ hai (null nếu chưa mua lại)
min_date   = master.agg(F.min("order_purchase_timestamp")).collect()[0][0].strftime("%Y-%m-%d")
max_first_day = master.agg(F.datediff(F.max("order_purchase_timestamp"), F.lit(min_date))).collect()[0][0]

first_order = (master
               .withColumn("rn", F.row_number().over(w_first))
               .withColumn("next_ts", F.lead("order_purchase_timestamp", 1).over(w_first))
               .filter(F.col("rn") == 1)
               .select(
                   "customer_unique_id",
                   F.datediff("order_purchase_timestamp", F.lit(min_date)).cast("double").alias("first_day"),
                   F.datediff("next_ts", "order_purchase_timestamp").cast("double").alias("days_to_second"),
                   F.log1p("payment_value").alias("log_order_value"),
                   F.col("item_count").cast("double").alias("item_count"),
                   # Giới hạn tỉ lệ ship/giá ở mức 2 để vài đơn giá rẻ không kéo lệch cả phân phối
                   F.least(F.col("total_freight") / F.col("total_price"), F.lit(2.0)).alias("freight_ratio"),
                   F.col("avg_review_score").alias("avg_review_score"),
                   F.col("avg_review_score").isNotNull().cast("double").alias("has_review"),
                   F.datediff("order_delivered_customer_date", "order_purchase_timestamp")
                    .cast("double").alias("delivery_days"),
                   # Dương = giao trễ so với ngày hứa, âm = giao sớm
                   F.datediff("order_delivered_customer_date", "order_estimated_delivery_date")
                    .cast("double").alias("delay_days"),
                   F.col("max_installments").cast("double").alias("max_installments"),
                   F.col("used_voucher").cast("double").alias("used_voucher"),
                   region_expr.alias("region")))

churn_df = churn_df.join(first_order, "customer_unique_id", "inner")

NUM_FEATURES = ["log_order_value", "item_count", "freight_ratio", "avg_review_score",
                "has_review", "delivery_days", "delay_days", "max_installments", "used_voucher"]

# Chia train/test 1 lần duy nhất rồi dùng chung cho cả ba mô hình để so sánh công bằng
train_raw, test_raw = churn_df.randomSplit([0.8, 0.2], seed=SEED)
train_raw.cache(); test_raw.cache()
print(f"      Train: {train_raw.count():,} | Test: {test_raw.count():,}")

ev_auc = BinaryClassificationEvaluator(labelCol="churn", metricName="areaUnderROC")
ev_pr  = BinaryClassificationEvaluator(labelCol="churn", metricName="areaUnderPR")
ev_acc = MulticlassClassificationEvaluator(labelCol="churn", predictionCol="prediction", metricName="accuracy")
ev_f1  = MulticlassClassificationEvaluator(labelCol="churn", predictionCol="prediction", metricName="f1")

def confusion(pred_df):
    cm = np.zeros((2, 2), dtype=int)
    for r in pred_df.groupBy("churn", "prediction").count().collect():
        cm[int(r["churn"]), int(r["prediction"])] = r["count"]
    return cm

# ── 4.3 Mô hình churn: đặc trưng đơn hàng đầu tiên ───────────────────
print("\n[3/4] Huấn luyện 3 mô hình trên đặc trưng đơn hàng đầu tiên ...")

# Imputer điền median cho giá trị thiếu (đơn chưa có review, thiếu ngày giao ...).
# Vùng miền là biến phân loại nên phải qua StringIndexer + OneHotEncoder;
# nếu đưa mã số thẳng vào Logistic Regression thì nó coi "vùng 3" gấp 3 lần "vùng 1".
imp_cols = ["avg_review_score", "delivery_days", "delay_days", "freight_ratio", "max_installments"]
prep = Pipeline(stages=[
    Imputer(inputCols=imp_cols, outputCols=[f"{c}_imp" for c in imp_cols], strategy="median"),
    StringIndexer(inputCol="region", outputCol="region_idx", handleInvalid="keep"),
    OneHotEncoder(inputCols=["region_idx"], outputCols=["region_vec"]),
    VectorAssembler(
        inputCols=["log_order_value", "item_count", "has_review", "used_voucher"]
                  + [f"{c}_imp" for c in imp_cols] + ["region_vec"],
        outputCol="features_raw"),
    # withMean=False vì vector có phần one-hot dạng thưa, trừ mean sẽ phá cấu trúc thưa
    StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=False),
])
prep_model = prep.fit(train_raw)
train = prep_model.transform(train_raw).cache()
test  = prep_model.transform(test_raw).cache()

models = {
    "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="churn", maxIter=100),
    "Random Forest":       RandomForestClassifier(featuresCol="features", labelCol="churn",
                                                  numTrees=100, maxDepth=8, seed=SEED),
    "Gradient Boosted Trees": GBTClassifier(featuresCol="features", labelCol="churn",
                                            maxIter=50, maxDepth=5, seed=SEED),
}

metric_rows, preds, cms = [], {}, {}
for name, est in models.items():
    t0   = time.time()
    pred = est.fit(train).transform(test).cache()
    cm   = confusion(pred)
    prec = cm[1, 1] / max(cm[1, 1] + cm[0, 1], 1)
    rec  = cm[1, 1] / max(cm[1, 1] + cm[1, 0], 1)
    metric_rows.append({"Model": name,
                        "AUC-ROC":   round(ev_auc.evaluate(pred), 4),
                        "PR-AUC":    round(ev_pr.evaluate(pred), 4),
                        "Accuracy":  round(ev_acc.evaluate(pred), 4),
                        "F1-Score":  round(ev_f1.evaluate(pred), 4),
                        "Precision (Churn)": round(prec, 4),
                        "Recall (Churn)":    round(rec, 4)})
    preds[name], cms[name] = pred, cm
    print(f"      {name:<24} AUC={metric_rows[-1]['AUC-ROC']:.4f}  "
          f"Acc={metric_rows[-1]['Accuracy']:.4f}  ({time.time() - t0:.1f}s)")

# Mốc so sánh tối thiểu: luôn đoán "churn" thì Accuracy bằng đúng tỉ lệ lớp đa số
baseline_acc = max(churn_rate, 1 - churn_rate)
print(f"      Baseline (luôn đoán lớp đa số): Accuracy = {baseline_acc:.4f}")

metrics = pd.DataFrame(metric_rows)
metrics.to_csv(f"{RESULTS}/churn_metrics.csv", index=False)
print("\n" + metrics.to_string(index=False))

# ── 4.4 ROC + Confusion matrix ──────────────────────────────────────
print("\n[4/4] Đang vẽ ROC và Confusion matrix ...")
from sklearn.metrics import roc_curve

def proba_pdf(pred):
    return pred.select("churn", vector_to_array("probability")[1].alias("p")).toPandas()

def curve_points(x, y, n=300):
    # Giữ tối đa n điểm đều nhau trên đường cong để file CSV nhỏ mà hình dạng không đổi
    idx = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
    return np.asarray(x)[idx], np.asarray(y)[idx]

roc_rows = []
fig, ax = plt.subplots(figsize=(6, 5))
for (name, pred), col in zip(preds.items(), [COLOR_PANDAS, COLOR_SPARK, COLOR_GBT]):
    pdf = proba_pdf(pred)
    fpr, tpr, _ = roc_curve(pdf["churn"], pdf["p"])
    fx, ty = curve_points(fpr, tpr)
    roc_rows += [{"Model": name, "FPR": a, "TPR": b} for a, b in zip(fx, ty)]
    ax.plot(fpr, tpr, color=col, linewidth=2,
            label=f"{name} (AUC = {metrics.set_index('Model').loc[name, 'AUC-ROC']:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Đoán ngẫu nhiên")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC — dự đoán churn (đặc trưng sạch)"); ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{RESULTS}/churn_roc.png", dpi=150); plt.show(); plt.close()
pd.DataFrame(roc_rows).to_csv(f"{RESULTS}/churn_roc_curves.csv", index=False)
pd.DataFrame([{"Model": n, "TN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0]), "TP": int(cm[1, 1])}
              for n, cm in cms.items()]).to_csv(f"{RESULTS}/churn_confusion.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (name, cm) in zip(axes, cms.items()):
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Dự đoán: Giữ", "Dự đoán: Churn"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Thực tế: Giữ", "Thực tế: Churn"])
    ax.set_title(name, fontsize=11)
plt.tight_layout()
plt.savefig(f"{RESULTS}/churn_confusion.png", dpi=150); plt.show(); plt.close()

# ── 4.6 Kiểm tra thiên lệch thời gian: mô hình "mua lại trong 180 ngày" ──
# Thí nghiệm ở 4.3 cho thấy chỉ riêng ngày mua đầu tiên đã đoán được nhãn churn gần như
# hoàn hảo. Lý do: khách mua trong 180 ngày cuối dataset chưa kịp "rời bỏ" nên luôn có
# churn = 0. Mô hình trên nhãn này vì vậy một phần chỉ học lại lịch.
# Cách xử lý: chỉ giữ khách có đơn đầu cách ngày cuối dataset ít nhất 180 ngày (ai cũng có
# đủ thời gian để quay lại) và dự đoán "có mua đơn thứ hai trong 180 ngày hay không".
print("\n[Bổ sung] Mô hình mua lại trên nhóm khách đủ tuổi (loại thiên lệch ngày cắt) ...")
mature_df = (churn_df
             .filter(F.col("first_day") <= max_first_day - CHURN_DAYS)
             .withColumn("repeat_180",
                         F.when(F.col("days_to_second") <= CHURN_DAYS, 1.0).otherwise(0.0)))
n_mat      = mature_df.count()
n_rep      = mature_df.filter(F.col("repeat_180") == 1).count()
base_rate  = n_rep / n_mat
print(f"      Khách đủ tuổi: {n_mat:,} | mua lại trong 180 ngày: {n_rep:,} ({base_rate:.1%})")

tr_m, te_m = mature_df.randomSplit([0.8, 0.2], seed=SEED)
# Lớp "mua lại" chỉ ~4% nên gán trọng số để mô hình không đoán toàn "không mua lại"
neg_pos = tr_m.filter(F.col("repeat_180") == 0).count() / max(tr_m.filter(F.col("repeat_180") == 1).count(), 1)
tr_m = tr_m.withColumn("w", F.when(F.col("repeat_180") == 1.0, neg_pos).otherwise(1.0))
prep_m = prep.fit(tr_m)
tr_f = prep_m.transform(tr_m).cache()
te_f = prep_m.transform(te_m).cache()

ev_auc_m = BinaryClassificationEvaluator(labelCol="repeat_180", metricName="areaUnderROC")
ev_pr_m  = BinaryClassificationEvaluator(labelCol="repeat_180", metricName="areaUnderPR")
models_m = {
    "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="repeat_180",
                                              weightCol="w", maxIter=100),
    "Random Forest":       RandomForestClassifier(featuresCol="features", labelCol="repeat_180",
                                                  weightCol="w", numTrees=100, maxDepth=8, seed=SEED),
    "Gradient Boosted Trees": GBTClassifier(featuresCol="features", labelCol="repeat_180",
                                            weightCol="w", maxIter=50, maxDepth=4, seed=SEED),
}

from sklearn.metrics import roc_curve
rep_rows, rep_probs = [], {}
for name, est in models_m.items():
    pred = est.fit(tr_f).transform(te_f)
    pdf  = pred.select(F.col("repeat_180").alias("y"),
                       vector_to_array("probability")[1].alias("p")).toPandas()
    pdf  = pdf.sort_values("p", ascending=False).reset_index(drop=True)
    def capture(frac):
        return pdf["y"].head(int(len(pdf) * frac)).sum() / pdf["y"].sum()
    rep_rows.append({"Model": name,
                     "AUC-ROC": round(ev_auc_m.evaluate(pred), 4),
                     "PR-AUC":  round(ev_pr_m.evaluate(pred), 4),
                     "Base rate": round(pdf["y"].mean(), 4),
                     "Recall@top20%": round(capture(0.2), 4),
                     "Lift@top10%":   round(capture(0.1) / 0.1, 2),
                     "Lift@top20%":   round(capture(0.2) / 0.2, 2)})
    rep_probs[name] = pdf
    print(f"      {name:<24} AUC={rep_rows[-1]['AUC-ROC']:.4f}  PR-AUC={rep_rows[-1]['PR-AUC']:.4f}  "
          f"Lift@20%={rep_rows[-1]['Lift@top20%']:.2f}")

repeat_metrics = pd.DataFrame(rep_rows)
repeat_metrics.to_csv(f"{RESULTS}/repeat_metrics.csv", index=False)
best_rep = repeat_metrics.sort_values("AUC-ROC", ascending=False).iloc[0]["Model"]

# Bảng theo thập phân vị điểm: khách điểm cao có thật sự mua lại nhiều hơn không?
dec = rep_probs[best_rep].copy()
dec["Decile"] = pd.qcut(dec["p"].rank(method="first", ascending=False), 10, labels=range(1, 11))
dec_tab = dec.groupby("Decile").agg(Customers=("y", "size"), Repeat_Rate=("y", "mean")).reset_index()
dec_tab["Lift"] = (dec_tab["Repeat_Rate"] / dec["y"].mean()).round(2)
dec_tab["Repeat_Rate"] = dec_tab["Repeat_Rate"].round(4)
dec_tab.to_csv(f"{RESULTS}/repeat_deciles.csv", index=False)

rep_curve_rows = []
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for (name, pdf), col in zip(rep_probs.items(), [COLOR_PANDAS, COLOR_SPARK, COLOR_GBT]):
    fpr, tpr, _ = roc_curve(pdf["y"], pdf["p"])
    auc = repeat_metrics.set_index("Model").loc[name, "AUC-ROC"]
    ax1.plot(fpr, tpr, color=col, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
    gain = pdf["y"].cumsum() / pdf["y"].sum()
    fx, ty = curve_points(fpr, tpr)
    rep_curve_rows += [{"Model": name, "Curve": "ROC", "X": a, "Y": b} for a, b in zip(fx, ty)]
    pct = np.arange(1, len(pdf) + 1) / len(pdf) * 100
    gx, gy = curve_points(pct, gain.values * 100)
    rep_curve_rows += [{"Model": name, "Curve": "Gains", "X": a, "Y": b} for a, b in zip(gx, gy)]
    ax2.plot(np.arange(1, len(pdf) + 1) / len(pdf) * 100, gain * 100, color=col, linewidth=2, label=name)
ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Đoán ngẫu nhiên")
ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC — mua lại trong 180 ngày (khách đủ tuổi)"); ax1.legend(loc="lower right")
ax2.plot([0, 100], [0, 100], "k--", linewidth=1, label="Chọn ngẫu nhiên")
ax2.set_xlabel("% khách được chọn (xếp theo điểm giảm dần)")
ax2.set_ylabel("% khách mua lại thực tế bắt được")
ax2.set_title("Đường cộng dồn (Cumulative Gains)"); ax2.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{RESULTS}/repeat_gains.png", dpi=150); plt.show(); plt.close()
pd.DataFrame(rep_curve_rows).to_csv(f"{RESULTS}/repeat_curves.csv", index=False)
tr_f.unpersist(); te_f.unpersist()

# ── Lưu kết quả cho dashboard & bước SHAP ────────────────────────────
best_name =metrics.sort_values("AUC-ROC", ascending=False).iloc[0]["Model"]
print(f"\n  Mô hình tốt nhất theo AUC: {best_name}")
(preds[best_name]
 .select("customer_unique_id", "Segment_Label" if "Segment_Label" in preds[best_name].columns else "segment",
         "log_order_value", "item_count", "freight_ratio", "avg_review_score", "delivery_days",
         "delay_days", "max_installments", "used_voucher", "region", "churn",
         vector_to_array("probability")[1].alias("churn_prob"), "prediction")
 .toPandas()
 .to_csv(f"{RESULTS}/churn_predictions.csv", index=False))

(churn_df.select("customer_unique_id", "Recency", "Frequency", "Monetary", "segment", "churn",
                 "first_day", "days_to_second", *NUM_FEATURES, "region")
 .toPandas()
 .to_parquet(f"{RESULTS}/churn_features.parquet", index=False))

train_raw.unpersist(); test_raw.unpersist(); train.unpersist(); test.unpersist()
print("\n[✓] BƯỚC 4 hoàn tất\n")


#%%
# =====================================================================
# 5. BENCHMARK PySpark vs Pandas Ở NHIỀU QUY MÔ DỮ LIỆU
# =====================================================================
# Mục tiêu: không chỉ so 1 con số ở 100K dòng, mà quan sát thời gian
# thay đổi thế nào khi dữ liệu tăng 1× → 2× → 5× → 10×
# (nhân bản bảng gốc, gắn hậu tố vào ID để các bản sao độc lập nhau).
#
# Quy ước để so sánh công bằng:
#   - Chỉ đo thời gian XỬ LÝ; dữ liệu đã nằm sẵn trong bộ nhớ (Spark .cache(), Pandas DataFrame).
#   - Mỗi tác vụ chạy BENCH_REPEATS lần, lấy trung vị; Spark có 1 lượt chạy khởi động riêng
#     để loại chi phí nạp JVM/MLlib lần đầu.
#   - K-Means ở hai bên dùng cùng cấu hình: 1 lần khởi tạo, tối đa 20 vòng lặp.
#   - Spark chạy local[*] trên 1 máy nên KHÔNG có lợi thế phân tán thật; kết quả thể
#     hiện chi phí overhead và điểm bắt đầu có lợi, chưa phải hiệu năng trên cluster.
# =====================================================================
print("=" * 70)
print("BƯỚC 5 — BENCHMARK PySpark vs Pandas")
print("=" * 70)

from sklearn.preprocessing import StandardScaler as SKScaler
from sklearn.cluster import KMeans as SKMeans

# ── Nạp 3 bảng cần dùng cho RFM ──────────────────────────────────────
pd_orders = pd.read_csv(f"{DATA}/olist_orders_dataset.csv",
                        usecols=["order_id", "customer_id", "order_status", "order_purchase_timestamp"],
                        parse_dates=["order_purchase_timestamp"])
pd_cust   = pd.read_csv(f"{DATA}/olist_customers_dataset.csv",
                        usecols=["customer_id", "customer_unique_id"])
pd_pay    = pd.read_csv(f"{DATA}/olist_order_payments_dataset.csv",
                        usecols=["order_id", "payment_value"])

sp_orders = (spark.read.csv(f"{DATA}/olist_orders_dataset.csv", header=True, inferSchema=True)
             .select("order_id", "customer_id", "order_status", "order_purchase_timestamp"))
sp_cust   = (spark.read.csv(f"{DATA}/olist_customers_dataset.csv", header=True, inferSchema=True)
             .select("customer_id", "customer_unique_id"))
sp_pay    = (spark.read.csv(f"{DATA}/olist_order_payments_dataset.csv", header=True, inferSchema=True)
             .select("order_id", "payment_value"))

# ── Hàm nhân bản dữ liệu ─────────────────────────────────────────────
def scale_pandas(df, s, id_cols):
    parts = []
    for i in range(s):
        part = df.copy()
        for c in id_cols:
            part[c] = part[c] + f"_{i}"
        parts.append(part)
    return pd.concat(parts, ignore_index=True)

def scale_spark(df, s, id_cols):
    parts = []
    for i in range(s):
        part = df
        for c in id_cols:
            part = part.withColumn(c, F.concat(F.col(c), F.lit(f"_{i}")))
        parts.append(part)
    return reduce(DataFrame.unionByName, parts)

# ── Các tác vụ đo: mỗi tác vụ có bản Pandas và bản PySpark tương đương ──
def pandas_rfm(orders_, cust_, pay_):
    pay_agg = pay_.groupby("order_id", as_index=False)["payment_value"].sum()
    d = orders_.merge(cust_, on="customer_id").merge(pay_agg, on="order_id", how="left")
    d = d[(d["order_status"] == "delivered") & d["payment_value"].notna()]
    ref = d["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    g = d.groupby("customer_unique_id").agg(
        last_ts=("order_purchase_timestamp", "max"),
        Frequency=("order_id", "count"),
        Monetary=("payment_value", "sum")).reset_index()
    g["Recency"] = (ref - g["last_ts"]).dt.days
    return g[g["Monetary"] > 0]

def spark_rfm(orders_, cust_, pay_):
    pay_agg = pay_.groupBy("order_id").agg(F.sum("payment_value").alias("payment_value"))
    d = (orders_.join(cust_, "customer_id").join(pay_agg, "order_id", "left")
         .filter((F.col("order_status") == "delivered") & F.col("payment_value").isNotNull()))
    ref = F.date_add(F.max("order_purchase_timestamp").over(Window.partitionBy()), 1)
    return (d.groupBy("customer_unique_id")
            .agg(F.max("order_purchase_timestamp").alias("last_ts"),
                 F.count("order_id").alias("Frequency"),
                 F.sum("payment_value").alias("Monetary"))
            .withColumn("Recency", F.datediff(F.lit(max_ts_bench), F.col("last_ts")))
            .filter(F.col("Monetary") > 0))

def pandas_ntile(rfm_):
    out = rfm_.copy()
    for col in ["Recency", "Frequency", "Monetary"]:
        out[f"{col}_score"] = pd.qcut(out[col].rank(method="first"), 5, labels=False)
    return out

def spark_ntile(rfm_):
    out = rfm_
    for col in ["Recency", "Frequency", "Monetary"]:
        out = out.withColumn(f"{col}_score", F.ntile(5).over(Window.orderBy(col)))
    return out

def pandas_kmeans(rfm_):
    X = SKScaler().fit_transform(rfm_[["Recency", "Frequency", "Monetary"]])
    SKMeans(n_clusters=K_FINAL, n_init=1, max_iter=20, random_state=SEED).fit(X)

def spark_kmeans(rfm_):
    a = VectorAssembler(inputCols=["Recency", "Frequency", "Monetary"], outputCol="fr").transform(rfm_)
    s_ = StandardScaler(inputCol="fr", outputCol="fs", withStd=True, withMean=True).fit(a).transform(a)
    KMeans(k=K_FINAL, maxIter=20, seed=SEED, featuresCol="fs").fit(s_)

def median_time(fn, repeats=BENCH_REPEATS):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return round(float(np.median(ts)), 3)

# Mốc ngày dùng chung cho Recency (bản sao nhân bản không đổi ngày nên mốc giữ nguyên)
max_ts_bench = pd_orders["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

# ── Chạy khởi động cho Spark (không tính vào kết quả) ────────────────
print("\n[0] Chạy khởi động Spark (loại chi phí nạp JVM/MLlib lần đầu) ...")
_w = spark_rfm(sp_orders, sp_cust, sp_pay); _w.count()
spark_ntile(_w).count(); spark_kmeans(_w)

bench_rows = []
for s in BENCH_SCALES:
    print(f"\n[×{s}] Chuẩn bị dữ liệu nhân bản ...")
    # --- Pandas ---
    po = scale_pandas(pd_orders, s, ["order_id", "customer_id"])
    pc = scale_pandas(pd_cust,   s, ["customer_id", "customer_unique_id"])
    pp = scale_pandas(pd_pay,    s, ["order_id"])
    # --- PySpark: cache + count để dữ liệu thật sự nằm trong bộ nhớ trước khi bấm giờ ---
    so = scale_spark(sp_orders, s, ["order_id", "customer_id"]).cache()
    sc = scale_spark(sp_cust,   s, ["customer_id", "customer_unique_id"]).cache()
    sp = scale_spark(sp_pay,    s, ["order_id"]).cache()
    n_orders_s = so.count(); sc.count(); sp.count()
    print(f"      Số đơn hàng: {n_orders_s:,}")

    # Tác vụ 1: join 3 bảng + gom RFM (đây là phép tính phân tán điển hình)
    t_pd = median_time(lambda: pandas_rfm(po, pc, pp))
    t_sp = median_time(lambda: spark_rfm(so, sc, sp).count())
    bench_rows.append({"Task": "Join + RFM", "Scale": f"×{s}", "Orders": n_orders_s,
                       "Pandas (s)": t_pd, "PySpark (s)": t_sp})
    print(f"      Join + RFM   : Pandas {t_pd:>7.3f}s | PySpark {t_sp:>7.3f}s")

    # Chuẩn bị đầu vào cho tác vụ 2 và 3 (bảng RFM đã tính xong, nằm sẵn trong bộ nhớ)
    rfm_p = pandas_rfm(po, pc, pp)
    rfm_s = spark_rfm(so, sc, sp).cache(); n_cust_s = rfm_s.count()

    # Tác vụ 2: chấm điểm 1–5 bằng Window ntile (cần sắp xếp toàn cục nên tốn shuffle)
    t_pd = median_time(lambda: pandas_ntile(rfm_p))
    t_sp = median_time(lambda: spark_ntile(rfm_s).count())
    bench_rows.append({"Task": "Window ntile", "Scale": f"×{s}", "Orders": n_orders_s,
                       "Pandas (s)": t_pd, "PySpark (s)": t_sp})
    print(f"      Window ntile : Pandas {t_pd:>7.3f}s | PySpark {t_sp:>7.3f}s")

    # Tác vụ 3: K-Means (thuật toán lặp, mỗi vòng phải quét lại toàn bộ dữ liệu)
    t_pd = median_time(lambda: pandas_kmeans(rfm_p))
    t_sp = median_time(lambda: spark_kmeans(rfm_s))
    bench_rows.append({"Task": "K-Means (k=4)", "Scale": f"×{s}", "Orders": n_orders_s,
                       "Pandas (s)": t_pd, "PySpark (s)": t_sp})
    print(f"      K-Means      : Pandas {t_pd:>7.3f}s | PySpark {t_sp:>7.3f}s")

    for df_ in (so, sc, sp, rfm_s):
        df_.unpersist()

benchmark = pd.DataFrame(bench_rows)
benchmark["Speedup (Pandas/PySpark)"] = (benchmark["Pandas (s)"] / benchmark["PySpark (s)"]).round(2)
benchmark.to_csv(f"{RESULTS}/benchmark.csv", index=False)
print("\n" + benchmark.to_string(index=False))

# Tìm điểm giao cắt nếu có (PySpark bắt đầu nhanh hơn Pandas)
wins = benchmark[benchmark["Speedup (Pandas/PySpark)"] > 1]
print("\n  Các trường hợp PySpark nhanh hơn Pandas:")
print(wins[["Task", "Scale", "Orders", "Speedup (Pandas/PySpark)"]].to_string(index=False)
      if not wins.empty else "  (không có — Pandas nhanh hơn ở mọi quy mô đã thử)")

# ── Biểu đồ: thời gian theo quy mô, mỗi tác vụ 1 khung ──────────────
tasks = benchmark["Task"].unique()
fig, axes = plt.subplots(1, len(tasks), figsize=(14, 4))
for ax, task in zip(axes, tasks):
    sub = benchmark[benchmark["Task"] == task]
    ax.plot(sub["Orders"] / 1e3, sub["Pandas (s)"],  marker="o", color=COLOR_PANDAS, linewidth=2, label="Pandas")
    ax.plot(sub["Orders"] / 1e3, sub["PySpark (s)"], marker="s", color=COLOR_SPARK,  linewidth=2, label="PySpark")
    ax.set_title(task, fontsize=12)
    ax.set_xlabel("Số đơn hàng (nghìn)"); ax.set_ylabel("Thời gian (giây)")
    ax.legend()
fig.suptitle("Thời gian xử lý theo quy mô dữ liệu — PySpark local[*] vs Pandas", fontsize=13)
plt.tight_layout()
plt.savefig(f"{RESULTS}/benchmark.png", dpi=150, bbox_inches="tight"); plt.show(); plt.close()
print("\n[✓] BƯỚC 5 hoàn tất\n")


#%%
# =====================================================================
# 6. GIẢI THÍCH MÔ HÌNH BẰNG SHAP (EXPLAINABLE AI)
# =====================================================================
# Mục tiêu: chỉ ra đặc trưng nào đẩy xác suất mua lại lên / xuống.
# SHAP không đọc trực tiếp được mô hình Spark, nên huấn luyện lại 1 Random Forest
# bản scikit-learn trên cùng đặc trưng sạch rồi dùng TreeExplainer.
# =====================================================================
print("=" * 70)
print("BƯỚC 6 — SHAP")
print("=" * 70)

try:
    import shap
    from sklearn.ensemble import RandomForestClassifier as SKRf
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(f"{RESULTS}/churn_features.parquet")
    df[["avg_review_score", "delivery_days", "delay_days", "freight_ratio", "max_installments"]] = (
        df[["avg_review_score", "delivery_days", "delay_days", "freight_ratio", "max_installments"]]
        .fillna(df[["avg_review_score", "delivery_days", "delay_days", "freight_ratio", "max_installments"]].median()))
    # Giải thích mô hình "mua lại trong 180 ngày" trên khách đủ tuổi,
    # vì mô hình trên nhãn churn gốc bị lẫn hiệu ứng ngày cắt dữ liệu
    df = df[df["first_day"] <= max_first_day - CHURN_DAYS].copy()
    df["repeat_180"] = (df["days_to_second"] <= CHURN_DAYS).astype(int)
    X = pd.get_dummies(df[NUM_FEATURES + ["region"]], columns=["region"], dtype=int)
    y = df["repeat_180"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    from sklearn.metrics import roc_auc_score
    sk_rf = SKRf(n_estimators=100, max_depth=8, min_samples_leaf=20, random_state=SEED,
                 n_jobs=-1, class_weight="balanced_subsample")
    sk_rf.fit(X_tr, y_tr)
    print(f"  AUC trên tập test (sklearn RF, nhóm đủ tuổi): "
          f"{roc_auc_score(y_te, sk_rf.predict_proba(X_te)[:, 1]):.4f}")

    # Chỉ giải thích trên mẫu 3.000 khách của tập test để SHAP chạy nhanh mà vẫn đại diện
    X_shap      = X_te.sample(3000, random_state=SEED)
    explainer   = shap.TreeExplainer(sk_rf)
    shap_values = explainer.shap_values(X_shap)

    # Các phiên bản SHAP trả về định dạng khác nhau; chuẩn hóa về mảng của lớp churn (=1)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    plt.figure(figsize=(8, 5))
    shap.summary_plot(sv, X_shap, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance — dự đoán mua lại trong 180 ngày")
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/shap_bar.png", dpi=150, bbox_inches="tight"); plt.show(); plt.close()

    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, X_shap, show=False)
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.show(); plt.close()

    shap_importance = (pd.DataFrame({"Feature": X_shap.columns,
                                     "Mean |SHAP|": np.abs(sv).mean(axis=0)})
                       .sort_values("Mean |SHAP|", ascending=False))
    shap_importance.to_csv(f"{RESULTS}/shap_importance.csv", index=False)

    # Lưu SHAP của 10 đặc trưng quan trọng nhất (dạng dài) để dashboard vẽ beeswarm tương tác;
    # Value_pct là phân vị của giá trị đặc trưng (0 = thấp, 1 = cao) dùng để tô màu như SHAP gốc
    top_feats = shap_importance["Feature"].head(10).tolist()
    sv_df = pd.DataFrame(sv, columns=X_shap.columns, index=X_shap.index)
    long_rows = []
    for f_name in top_feats:
        long_rows.append(pd.DataFrame({
            "Feature": f_name, "SHAP": sv_df[f_name].values, "Value": X_shap[f_name].values,
            "Value_pct": X_shap[f_name].rank(pct=True).values}))
    pd.concat(long_rows).round(5).to_csv(f"{RESULTS}/shap_values_top.csv", index=False)
    print("\n" + shap_importance.round(4).to_string(index=False))
    print("\n[✓] BƯỚC 6 hoàn tất\n")

except ImportError:
    print("  [BỎ QUA] Chưa cài shap. Chạy: pip install shap")

#%%
# =====================================================================
# 7. PHÂN TÍCH CHUYÊN SÂU — TRẢ LỜI CÁC CÂU HỎI NGHIÊN CỨU
# =====================================================================
# Mục tiêu: đi từ "mô tả" sang "giải thích để ra quyết định". Bước này trả lời
# 4 câu hỏi nghiên cứu (RQ) bằng số liệu, mỗi câu đều được soi ở nhiều cấp:
#   RQ1 — Nhóm khách nào tạo ra doanh thu, và giá trị chi tiêu phân bố ra sao?
#   RQ2 — Khả năng mua lại khác nhau thế nào giữa các bang (cấp địa lý)?
#   RQ3 — Khả năng mua lại khác nhau thế nào giữa các danh mục của đơn đầu tiên?
#   RQ4 — Trải nghiệm giao hàng (thời gian, giao trễ, điểm đánh giá) liên quan
#         thế nào đến việc khách có quay lại không?
# Lưu ý quan trọng: nhãn churn gốc phụ thuộc mạnh vào ngày mua đầu tiên,
# nên RQ2–RQ4 đo bằng "mua đơn thứ hai trong 180 ngày" trên khách đủ tuổi. Nhãn churn
# thô vẫn được giữ trong bảng để đối chiếu và cho thấy mức độ sai lệch nếu dùng nó.
# Cuối bước có bảng kiểm tra nhất quán: cộng các cấp lại phải ra đúng tổng.
# =====================================================================
from scipy.stats import chi2_contingency

print("=" * 70)
print("BƯỚC 7 — PHÂN TÍCH CHUYÊN SÂU (RQ1–RQ4) & KIỂM TRA NHẤT QUÁN")
print("=" * 70)

SEG_ORDER  = ["Recent Buyers", "At-Risk Buyers", "Lost Customers", "Repeat Buyers"]
SEG_COLORS = {"Recent Buyers": "#55A868", "At-Risk Buyers": "#DD8452",
              "Lost Customers": "#C44E52", "Repeat Buyers": "#4C72B0"}

master    = spark.read.parquet(f"{RESULTS}/master.parquet")
segmented = spark.read.parquet(f"{RESULTS}/segmented.parquet")
cf        = spark.read.parquet(f"{RESULTS}/churn_features.parquet")

# Đọc lại 3 bảng sản phẩm để bước này chạy độc lập được
items_7    = spark.read.csv(f"{DATA}/olist_order_items_dataset.csv",         header=True, inferSchema=True)
products_7 = spark.read.csv(f"{DATA}/olist_products_dataset.csv",            header=True, inferSchema=True)
category_7 = spark.read.csv(f"{DATA}/product_category_name_translation.csv", header=True, inferSchema=True)

def cramers_v(ct):
    # Với n ~ 93K thì p-value gần như luôn rất nhỏ, nên cần Cramér's V để biết
    # mức độ liên hệ có đáng kể về mặt thực tế hay không
    chi2, p, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    return chi2, p, float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))

# ── 7.1 RQ1: doanh thu & phân phối Monetary theo phân khúc ──────────
print("\n[1/5] RQ1 — Doanh thu và phân phối chi tiêu theo phân khúc ...")
seg_value = (segmented.groupBy("Segment_Label")
             .agg(F.count("*").alias("Customers"),
                  F.sum("Monetary").alias("Revenue"),
                  F.avg("Monetary").alias("Mean_Monetary"),
                  F.percentile_approx("Monetary", [0.25, 0.5, 0.75, 0.95]).alias("q"))
             .toPandas())
for i, qn in enumerate(["P25", "Median", "P75", "P95"]):
    seg_value[qn] = seg_value["q"].apply(lambda v: float(v[i]))
seg_value = seg_value.drop(columns="q")
seg_value["Customer_Share_%"] = (seg_value["Customers"] / seg_value["Customers"].sum() * 100).round(1)
seg_value["Revenue_Share_%"]  = (seg_value["Revenue"] / seg_value["Revenue"].sum() * 100).round(1)
seg_value["Segment_Label"] = pd.Categorical(seg_value["Segment_Label"], SEG_ORDER, ordered=True)
seg_value = seg_value.sort_values("Segment_Label").reset_index(drop=True)
seg_value.round(2).to_csv(f"{RESULTS}/segment_value.csv", index=False)
print(seg_value[["Segment_Label", "Customers", "Customer_Share_%", "Revenue_Share_%",
                 "Mean_Monetary", "Median", "P95"]].round(1).to_string(index=False))

seg_pdf = segmented.select("Segment_Label", "Monetary").toPandas()

# Mức độ tập trung doanh thu: top x% khách chi tiêu nhiều nhất chiếm bao nhiêu % doanh thu
mon_sorted = np.sort(seg_pdf["Monetary"].values)[::-1]
cum_share  = mon_sorted.cumsum() / mon_sorted.sum()
pareto = pd.DataFrame({
    "Top_%_khách": [1, 5, 10, 20, 50],
    "Doanh_thu_%": [round(cum_share[int(len(mon_sorted) * p / 100) - 1] * 100, 1) for p in [1, 5, 10, 20, 50]]})
pareto.to_csv(f"{RESULTS}/pareto_summary.csv", index=False)
print("\n  Mức tập trung doanh thu:\n" + pareto.to_string(index=False))

labels_present = [l for l in SEG_ORDER if l in set(seg_pdf["Segment_Label"])]
fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot([seg_pdf.loc[seg_pdf["Segment_Label"] == l, "Monetary"] for l in labels_present],
                patch_artist=True, showfliers=True,
                flierprops=dict(marker=".", markersize=2, alpha=0.25),
                medianprops=dict(color="black", linewidth=1.5))
for patch, l in zip(bp["boxes"], labels_present):
    patch.set_facecolor(SEG_COLORS[l]); patch.set_alpha(0.7)
ax.set_xticklabels(labels_present)
for i, l in enumerate(labels_present, start=1):
    med = seg_pdf.loc[seg_pdf["Segment_Label"] == l, "Monetary"].median()
    ax.text(i + 0.32, med, f"{med:,.0f}", va="center", fontsize=9)
# Thang log vì Monetary lệch phải rất mạnh (max ~ R$13.664 trong khi median ~ R$108)
ax.set_yscale("log")
ax.set_ylabel("Monetary (R$, thang log)")
ax.set_title("Phân phối chi tiêu theo phân khúc (số trên hộp = median)")
plt.tight_layout()
plt.savefig(f"{RESULTS}/segment_monetary_boxplot.png", dpi=150); plt.show(); plt.close()

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(seg_value)); w = 0.38
ax.bar(x - w / 2, seg_value["Customer_Share_%"], w, color="#9DB4CE", label="% số khách")
ax.bar(x + w / 2, seg_value["Revenue_Share_%"],  w, color=COLOR_SPARK, label="% doanh thu")
for xi, a, b in zip(x, seg_value["Customer_Share_%"], seg_value["Revenue_Share_%"]):
    ax.text(xi - w / 2, a + 0.5, f"{a:.1f}", ha="center", fontsize=9)
    ax.text(xi + w / 2, b + 0.5, f"{b:.1f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(seg_value["Segment_Label"].astype(str))
ax.set_ylabel("%"); ax.set_title("Tỉ trọng khách hàng so với tỉ trọng doanh thu theo phân khúc")
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS}/segment_revenue_share.png", dpi=150); plt.show(); plt.close()

# ── Bảng khách hàng ở cấp 1 dòng / khách, dùng chung cho RQ2–RQ4 ────
# Gắn bang, ngày mua và danh mục chính của ĐƠN ĐẦU TIÊN để các đặc trưng
# luôn có sẵn tại thời điểm khách mua lần đầu (khớp với mô hình churn).
w_first = Window.partitionBy("customer_unique_id").orderBy("order_purchase_timestamp")
first_ord = (master.withColumn("rn", F.row_number().over(w_first))
             .filter(F.col("rn") == 1)
             .select("customer_unique_id", "order_id", "customer_state",
                     F.col("order_purchase_timestamp").alias("first_ts")))

# Danh mục chính của một đơn = danh mục của sản phẩm có giá cao nhất trong đơn
w_item = Window.partitionBy("order_id").orderBy(F.desc("price"), "order_item_id")
main_cat = (items_7
            .join(products_7.select("product_id", "product_category_name"), "product_id", "left")
            .join(category_7, "product_category_name", "left")
            .withColumn("rn", F.row_number().over(w_item))
            .filter(F.col("rn") == 1)
            .select("order_id",
                    F.coalesce("product_category_name_english", F.lit("unknown")).alias("category")))

# Khách có đơn đầu trong 180 ngày cuối dữ liệu chưa đủ thời gian để "quay lại",
# nên nhãn churn của họ bị thiên lệch xuống. Cột `mature` đánh dấu nhóm đủ tuổi,
# dùng đúng quy tắc theo ngày như phần mô hình mua lại để hai bước cho cùng một quy mô mẫu.

cust = (cf.select("customer_unique_id", "Frequency", "Monetary", "churn", "delivery_days",
                  "delay_days", "avg_review_score", "freight_ratio", "days_to_second", "first_day")
        .join(first_ord, "customer_unique_id", "inner")
        .join(main_cat, "order_id", "left")
        .fillna({"category": "unknown"})
        .join(segmented.select("customer_unique_id", "Segment_Label"), "customer_unique_id", "inner")
        .withColumn("mature", F.col("first_day") <= max_first_day - CHURN_DAYS)
        .withColumn("repeat_180", F.when(F.col("days_to_second") <= CHURN_DAYS, 1.0).otherwise(0.0)))
cust.cache()
n_cust_7    = cust.count()
overall_ch  = cust.agg(F.avg("churn")).collect()[0][0]
overall_rep = cust.filter("mature").agg(F.avg("repeat_180")).collect()[0][0]
n_mature    = cust.filter("mature").count()
print(f"\n  Tổng khách: {n_cust_7:,} | churn thô tổng thể: {overall_ch:.1%} | "
      f"nhóm đủ tuổi: {n_mature:,} khách, mua lại trong 180 ngày: {overall_rep:.1%}")

def group_table(by):
    return (cust.groupBy(by)
            .agg(F.count("*").alias("Customers"),
                 F.sum("churn").alias("Churned"),
                 F.avg("churn").alias("Churn_Rate"),
                 F.sum(F.col("mature").cast("int")).alias("Mature_Customers"),
                 F.avg(F.when(F.col("mature"), F.col("repeat_180"))).alias("Repeat180_Mature"),
                 F.sum("Monetary").alias("Revenue"),
                 F.avg("Monetary").alias("Avg_Monetary"),
                 F.avg("avg_review_score").alias("Avg_Review"),
                 F.avg("delivery_days").alias("Avg_Delivery_Days"),
                 F.avg(F.when(F.col("delay_days").isNotNull(), (F.col("delay_days") > 0).cast("double")))
                  .alias("Late_Rate"))
            .toPandas())

# ── 7.2 RQ2: drill-down theo bang ────────────────────────────────────
print("\n[2/5] RQ2 — Churn theo bang ...")
state_full = group_table("customer_state").sort_values("Customers", ascending=False)
state_tab  = state_full.copy()
state_tab.round(4).to_csv(f"{RESULTS}/drill_state.csv", index=False)
big_states = state_tab[state_tab["Mature_Customers"] >= 500]
rho_state  = big_states["Avg_Delivery_Days"].corr(big_states["Repeat180_Mature"], method="spearman")
print(state_tab.head(10)[["customer_state", "Customers", "Churn_Rate", "Repeat180_Mature",
                          "Avg_Delivery_Days", "Late_Rate"]].round(3).to_string(index=False))
print(f"  Spearman(thời gian giao TB, tỉ lệ mua lại) giữa {len(big_states)} bang (≥500 khách đủ tuổi): {rho_state:.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.scatter(big_states["Avg_Delivery_Days"], big_states["Repeat180_Mature"] * 100,
            s=big_states["Mature_Customers"] / 20, alpha=0.6, color=COLOR_MAIN)
for _, r in big_states[big_states["Mature_Customers"] >= 1500].iterrows():
    ax1.annotate(r["customer_state"], (r["Avg_Delivery_Days"], r["Repeat180_Mature"] * 100),
                 fontsize=8, xytext=(3, 3), textcoords="offset points")
ax1.axhline(overall_rep * 100, color="gray", linestyle="--", linewidth=1, label=f"Trung bình {overall_rep:.1%}")
ax1.set_xlabel("Thời gian giao trung bình của đơn đầu (ngày)"); ax1.set_ylabel("Tỉ lệ mua lại trong 180 ngày (%)")
ax1.set_title(f"Mỗi bang: thời gian giao vs mua lại (Spearman ρ = {rho_state:.2f})")
ax1.legend()
top10 = state_tab.head(10).sort_values("Repeat180_Mature")
se10  = np.sqrt(top10["Repeat180_Mature"] * (1 - top10["Repeat180_Mature"]) / top10["Mature_Customers"])
ax2.barh(top10["customer_state"], top10["Repeat180_Mature"] * 100, xerr=1.96 * se10 * 100,
         color=COLOR_MAIN, capsize=3, label="Tỉ lệ mua lại (%) ± 95% CI")
ax2.axvline(overall_rep * 100, color="gray", linestyle="--", linewidth=1, label="Trung bình")
ax2.set_title("Mua lại ở 10 bang đông khách nhất (nhóm đủ tuổi)"); ax2.set_xlabel("%"); ax2.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS}/drill_state.png", dpi=150); plt.show(); plt.close()

# ── 7.3 RQ3: drill-down theo danh mục đơn đầu tiên ───────────────────
print("\n[3/5] RQ3 — Churn theo danh mục của đơn đầu tiên ...")
cat_full = group_table("category").sort_values("Customers", ascending=False)
cat_tab  = cat_full[cat_full["Mature_Customers"] >= 300]
cat_tab.round(4).to_csv(f"{RESULTS}/drill_category.csv", index=False)
cat15 = cat_tab.head(15).sort_values("Repeat180_Mature")
print(cat15[["category", "Mature_Customers", "Repeat180_Mature", "Churn_Rate", "Avg_Review"]]
      .round(3).to_string(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
se15 = np.sqrt(cat15["Repeat180_Mature"] * (1 - cat15["Repeat180_Mature"]) / cat15["Mature_Customers"])
ax1.barh(cat15["category"], cat15["Repeat180_Mature"] * 100, xerr=1.96 * se15 * 100, capsize=3,
         color=[COLOR_GBT if v > overall_rep else COLOR_MAIN for v in cat15["Repeat180_Mature"]],
         label="Tỉ lệ mua lại (%) ± 95% CI")
ax1.axvline(overall_rep * 100, color="gray", linestyle="--", linewidth=1, label="Trung bình")
ax1.set_title("Mua lại trong 180 ngày (nhóm đủ tuổi)"); ax1.set_xlabel("%"); ax1.legend()
ax2.barh(cat15["category"], cat15["Churn_Rate"] * 100, color="#C9C9C9", label="Churn thô (%)")
ax2.axvline(overall_ch * 100, color="gray", linestyle="--", linewidth=1)
ax2.set_title("Đối chiếu: churn thô (bị lẫn hiệu ứng thời điểm)"); ax2.set_xlabel("%"); ax2.legend()
fig.suptitle("15 danh mục đông khách nhất (theo sản phẩm đắt nhất của đơn đầu tiên)")
plt.tight_layout()
plt.savefig(f"{RESULTS}/drill_category.png", dpi=150); plt.show(); plt.close()

# ── 7.4 RQ4: trải nghiệm giao hàng → đánh giá → quay lại ────────────
print("\n[4/5] RQ4 — Trải nghiệm giao hàng và churn ...")
cust_pdf = (cust.filter(F.col("delivery_days").isNotNull() & F.col("mature"))
            .select("customer_unique_id", "churn", "repeat_180", "delivery_days", "delay_days",
                    "avg_review_score", "customer_state", "category")
            .toPandas())

def bucket_delivery(d):
    return "≤7 ngày" if d <= 7 else "8–14 ngày" if d <= 14 else "15–21 ngày" if d <= 21 else ">21 ngày"
def bucket_delay(d):
    if pd.isna(d): return None
    return "Đúng hạn / sớm" if d <= 0 else "Trễ 1–7 ngày" if d <= 7 else "Trễ >7 ngày"
def bucket_review(s):
    return "Không đánh giá" if pd.isna(s) else f"{int(round(s))} sao"

cust_pdf["Giao_hàng"]   = cust_pdf["delivery_days"].apply(bucket_delivery)
cust_pdf["Giao_trễ"]    = cust_pdf["delay_days"].apply(bucket_delay)
cust_pdf["Đánh_giá"]    = cust_pdf["avg_review_score"].apply(bucket_review)
ORD_DELIVERY = ["≤7 ngày", "8–14 ngày", "15–21 ngày", ">21 ngày"]
ORD_DELAY    = ["Đúng hạn / sớm", "Trễ 1–7 ngày", "Trễ >7 ngày"]
ORD_REVIEW   = ["1 sao", "2 sao", "3 sao", "4 sao", "5 sao", "Không đánh giá"]

drill_rows = []
for col, order in [("Giao_hàng", ORD_DELIVERY), ("Giao_trễ", ORD_DELAY), ("Đánh_giá", ORD_REVIEW)]:
    g = (cust_pdf.dropna(subset=[col]).groupby(col)
         .agg(Customers=("repeat_180", "size"), Repeat180=("repeat_180", "mean"),
              Churn_Rate_Raw=("churn", "mean"),
              Avg_Review=("avg_review_score", "mean")).reindex(order).reset_index()
         .rename(columns={col: "Bucket"}))
    g.insert(0, "Dimension", col)
    drill_rows.append(g)
drill_delivery = pd.concat(drill_rows, ignore_index=True)
drill_delivery.round(4).to_csv(f"{RESULTS}/drill_delivery.csv", index=False)
print(drill_delivery.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, (col, order), title in zip(axes, [("Giao_hàng", ORD_DELIVERY), ("Giao_trễ", ORD_DELAY),
                                          ("Đánh_giá", ORD_REVIEW)],
                                   ["Theo thời gian giao", "Theo mức giao trễ", "Theo điểm đánh giá"]):
    d = drill_delivery[drill_delivery["Dimension"] == col]
    se = np.sqrt(d["Repeat180"] * (1 - d["Repeat180"]) / d["Customers"])
    ax.bar(d["Bucket"], d["Repeat180"] * 100, yerr=1.96 * se * 100, capsize=4,
           color=COLOR_MAIN, label="Tỉ lệ mua lại (%) ± 95% CI")
    for xi, (v, n) in enumerate(zip(d["Repeat180"], d["Customers"])):
        ax.text(xi, v * 100 + 0.9, f"{v:.1%}\n(n={n:,})", ha="center", fontsize=8)
    ax.axhline(overall_rep * 100, color="gray", linestyle="--", linewidth=1)
    ax.set_title(title); ax.tick_params(axis="x", rotation=25)
axes[0].set_ylabel("Tỉ lệ mua lại trong 180 ngày (%)"); axes[0].legend()
axes[0].set_ylim(0, 8)
plt.tight_layout()
plt.savefig(f"{RESULTS}/drill_delivery.png", dpi=150); plt.show(); plt.close()

# Kiểm định độc lập chi-bình phương + Cramér's V cho từng chiều phân tích.
# Chỉ giữ nhóm đủ lớn (≥100 khách đủ tuổi) vì tỉ lệ mua lại chỉ vài % nên ô kỳ vọng dễ quá nhỏ.
tests = []
for name, col in [("Bang", "customer_state"), ("Danh mục đơn đầu", "category"),
                  ("Thời gian giao", "Giao_hàng"), ("Mức giao trễ", "Giao_trễ"),
                  ("Điểm đánh giá", "Đánh_giá")]:
    sub = cust_pdf.dropna(subset=[col])
    sizes = sub[col].value_counts()
    sub = sub[sub[col].isin(sizes[sizes >= 100].index)]
    chi2, p, v = cramers_v(pd.crosstab(sub[col], sub["repeat_180"]))
    tests.append({"Dimension": name, "Groups": sub[col].nunique(), "N": len(sub),
                  "Chi2": round(chi2, 1), "p_value": p, "Cramers_V": round(v, 3),
                  "Effect": "yếu" if v < 0.1 else "vừa" if v < 0.3 else "mạnh"})
tests_df = pd.DataFrame(tests)
tests_df.to_csv(f"{RESULTS}/association_tests.csv", index=False)
print("\n  Kiểm định liên hệ với việc mua lại (nhóm đủ tuổi):\n" + tests_df.to_string(index=False))

# ── 7.5 Kiểm tra nhất quán đa cấp ───────────────────────────────────
print("\n[5/5] Kiểm tra nhất quán giữa các cấp tổng hợp ...")
# Cộng các cấp nhỏ lại phải bằng tổng ở cấp lớn; nếu lệch thì có lỗi join/lọc ở đâu đó.
checks = []
def add_check(metric, a_name, a, b_name, b, tol=0.0, note=""):
    diff = a - b
    checks.append({"Chỉ số": metric, "Nguồn A": a_name, "Giá trị A": round(a, 2),
                   "Nguồn B": b_name, "Giá trị B": round(b, 2), "Chênh lệch": round(diff, 2),
                   "Kết luận": "Khớp" if abs(diff) <= tol else "Lệch", "Ghi chú": note})

tot_orders  = master.count()
tot_rev     = master.agg(F.sum("payment_value")).collect()[0][0]
seg_cust    = segmented.count()
seg_orders  = segmented.agg(F.sum("Frequency")).collect()[0][0]
seg_rev     = segmented.agg(F.sum("Monetary")).collect()[0][0]
tot_churned = cust.agg(F.sum("churn")).collect()[0][0]
seg_churn_pdf = cust.groupBy("Segment_Label").agg(F.sum("churn").alias("c")).toPandas()

add_check("Số khách", "segmented", seg_cust, "bảng khách (cust)", n_cust_7)
add_check("Số khách", "segmented", seg_cust, "Σ theo phân khúc", seg_value["Customers"].sum())
add_check("Số khách", "segmented", seg_cust, "Σ theo bang", state_full["Customers"].sum())
add_check("Số khách", "segmented", seg_cust, "Σ theo danh mục", cat_full["Customers"].sum())
add_check("Số khách đủ tuổi", "bảng khách", n_mature, "Σ theo bang", state_full["Mature_Customers"].sum())
add_check("Số khách đủ tuổi", "bảng khách", n_mature, "Σ theo danh mục", cat_full["Mature_Customers"].sum())
add_check("Số đơn", "master (đơn đã giao)", tot_orders, "Σ Frequency", seg_orders)
add_check("Doanh thu (R$)", "master", tot_rev, "Σ Monetary", seg_rev, tol=0.5)
add_check("Doanh thu (R$)", "master", tot_rev, "Σ theo phân khúc", seg_value["Revenue"].sum(), tol=0.5)
add_check("Doanh thu (R$)", "master", tot_rev, "Σ theo bang", state_full["Revenue"].sum(), tol=0.5)
add_check("Doanh thu (R$)", "master", tot_rev, "Σ theo danh mục", cat_full["Revenue"].sum(), tol=0.5)
add_check("Số khách churn", "bảng khách", tot_churned, "Σ theo phân khúc", seg_churn_pdf["c"].sum())
add_check("Số khách churn", "bảng khách", tot_churned, "Σ theo bang", state_full["Churned"].sum())
add_check("Số khách churn", "bảng khách", tot_churned, "Σ theo danh mục", cat_full["Churned"].sum())

# Đối chiếu với bảng items: doanh thu theo giá + phí ship của các đơn đã giao.
# Không kỳ vọng khớp tuyệt đối vì payment_value gồm cả lãi trả góp, voucher và làm tròn.
items_rev = (items_7.join(master.select("order_id"), "order_id", "inner")
             .agg(F.sum(F.col("price") + F.col("freight_value"))).collect()[0][0])
gap_pct = (tot_rev - items_rev) / items_rev * 100
add_check("Doanh thu (R$)", "master (payment_value)", tot_rev, "bảng items (giá + ship)", items_rev,
          tol=0.02 * items_rev,
          note=f"Lệch {gap_pct:+.2f}%: payment_value gồm lãi trả góp/làm tròn, voucher")

checks_df = pd.DataFrame(checks)
checks_df.to_csv(f"{RESULTS}/consistency_checks.csv", index=False)
print(checks_df[["Chỉ số", "Nguồn A", "Giá trị A", "Nguồn B", "Giá trị B", "Chênh lệch", "Kết luận"]]
      .to_string(index=False))
n_bad = (checks_df["Kết luận"] == "Lệch").sum()
print(f"\n  {len(checks_df) - n_bad}/{len(checks_df)} phép kiểm tra khớp"
      + ("" if n_bad == 0 else f" — {n_bad} phép lệch, cần xem lại"))

# Nếu dùng nhãn churn thô thì thứ hạng giữa các bang / danh mục có giống thứ hạng "mua lại" không?
# Tương quan thấp hoặc ÂM nghĩa là nhãn thô đang đo thời điểm mua chứ không phải hành vi.
for nm, tab in [("bang", big_states), ("danh mục", cat_tab)]:
    rho = tab["Churn_Rate"].corr(-tab["Repeat180_Mature"], method="spearman")
    print(f"  Thứ hạng theo {nm}: Spearman(churn thô, 1 − mua lại) = {rho:.2f}")

cust.unpersist()
print("\n[✓] BƯỚC 7 hoàn tất\n")

#%%
# =====================================================================
# 8. TỪ PHÂN TÍCH ĐẾN QUYẾT ĐỊNH — BẢNG HÀNH ĐỘNG & KỊCH BẢN TÁC ĐỘNG
# =====================================================================
# Mục tiêu: chuyển kết quả bước 2–7 thành các hành động cụ thể cho từng phân khúc,
# mỗi hành động có KPI đo được và một ước lượng tác động. Nguyên tắc:
#   - Dữ liệu Olist là quan sát, KHÔNG có thí nghiệm, nên các con số tác động ở đây là
#     KỊCH BẢN "nếu uplift đạt X thì giá trị là Y", không phải dự báo.
#   - Điều đáng tính nhất là điểm hòa vốn: chi phí tối đa cho mỗi khách được nhắm.
#   - Hành động nào dữ liệu chưa ủng hộ (ví dụ giao hàng nhanh hơn để giữ chân) thì ghi rõ
#     là cần A/B test trước khi đầu tư.
# =====================================================================
print("=" * 70)
print("BƯỚC 8 — BẢNG HÀNH ĐỘNG & KỊCH BẢN TÁC ĐỘNG")
print("=" * 70)

# ── 8.1 Dữ kiện nền: quy mô và giá trị mỗi phân khúc (tính bằng Spark) ──
seg8 = (spark.read.parquet(f"{RESULTS}/segmented.parquet")
        .groupBy("Segment_Label")
        .agg(F.count("*").alias("Customers"),
             F.sum("Monetary").alias("Revenue"),
             F.sum("Frequency").alias("Orders"))
        .toPandas().set_index("Segment_Label"))
seg8["AOV"] = seg8["Revenue"] / seg8["Orders"]        # giá trị trung bình mỗi đơn
total_rev   = seg8["Revenue"].sum()
print(seg8.round(1).to_string())
print(f"  Doanh thu toàn bộ: R${total_rev:,.0f}")

# ── 8.2 Thanh toán trả góp: tín hiệu duy nhất đi theo một chiều rõ ràng ──
# SHAP chỉ nói đặc trưng nào quan trọng, không nói chiều tác động, nên kiểm tra riêng.
cf8 = pd.read_parquet(f"{RESULTS}/churn_features.parquet")
mat = cf8[cf8["first_day"] <= max_first_day - CHURN_DAYS].copy()
mat["repeat_180"] = (mat["days_to_second"] <= CHURN_DAYS).astype(int)
base_rep = mat["repeat_180"].mean()   # tỷ lệ mua lại nền trên toàn nhóm đủ tuổi
print(f"\n  Nhóm đủ tuổi: {len(mat):,} khách | tỷ lệ mua lại nền: {base_rep:.1%}")
mat["Số kỳ trả góp"] = pd.cut(mat["max_installments"], [0, 1, 3, 6, 10, 100],
                              labels=["1 kỳ", "2–3 kỳ", "4–6 kỳ", "7–10 kỳ", ">10 kỳ"])
inst = (mat.groupby("Số kỳ trả góp", observed=True)["repeat_180"]
        .agg(Customers="size", Repeat180="mean").reset_index())
inst.to_csv(f"{RESULTS}/installments_repeat.csv", index=False)
ct = pd.crosstab(mat["Số kỳ trả góp"], mat["repeat_180"])
chi2_i, p_i, _, _ = chi2_contingency(ct)
v_i = np.sqrt(chi2_i / len(mat) / (min(ct.shape) - 1))
print("\n  Mua lại trong 180 ngày theo số kỳ trả góp của đơn đầu:")
print(inst.assign(Repeat180=lambda d: (d["Repeat180"] * 100).round(2)).to_string(index=False))
print(f"  Chi-square p = {p_i:.2g} | Cramér's V = {v_i:.3f}")

# ── 8.3 Kịch bản tác động và điểm hòa vốn ──
# Với khách chỉ mua một lần: nếu một chiến dịch làm tăng thêm `u` điểm phần trăm khách mua
# đơn thứ hai thì doanh thu tăng = Customers × u × AOV. Chi phí tối đa trên MỖI khách được nhắm
# để không lỗ (tính trên doanh thu, chưa nhân biên lợi nhuận) = u × AOV.
# Với Repeat Buyers: kịch bản là tăng tương đối giá trị chi tiêu của nhóm này.
BE_COL = "Chi phí hòa vốn / khách (R$, trước biên LN)"
rows = []
for seg in ["Recent Buyers", "At-Risk Buyers", "Lost Customers"]:
    r = seg8.loc[seg]
    for u in (0.005, 0.01, 0.02):
        rows.append({"Phân khúc": seg, "Giả định uplift": f"+{u*100:.1f} điểm % mua đơn 2",
                     "Khách nhắm tới": int(r["Customers"]),
                     "Đơn thêm": round(r["Customers"] * u),
                     "Doanh thu thêm (R$)": round(r["Customers"] * u * r["AOV"]),
                     "% doanh thu hiện tại": round(100 * r["Customers"] * u * r["AOV"] / total_rev, 2),
                     BE_COL: round(u * r["AOV"], 2)})
r = seg8.loc["Repeat Buyers"]
for u in (0.05, 0.10, 0.20):
    rows.append({"Phân khúc": "Repeat Buyers", "Giả định uplift": f"+{u*100:.0f}% giá trị chi tiêu",
                 "Khách nhắm tới": int(r["Customers"]),
                 "Đơn thêm": round(r["Orders"] * u),
                 "Doanh thu thêm (R$)": round(r["Revenue"] * u),
                 "% doanh thu hiện tại": round(100 * r["Revenue"] * u / total_rev, 2),
                 BE_COL: round(r["Revenue"] * u / r["Customers"], 2)})
scen = pd.DataFrame(rows)
scen.to_csv(f"{RESULTS}/impact_scenarios.csv", index=False)
print("\n" + scen.to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for seg in ["Recent Buyers", "At-Risk Buyers", "Lost Customers"]:
    dd = scen[scen["Phân khúc"] == seg]
    ax[0].plot([0.5, 1, 2], dd["Doanh thu thêm (R$)"] / 1e3, marker="o", label=seg, color=SEG_COLORS[seg])
ax[0].set_xticks([0.5, 1, 2]); ax[0].set_xlabel("Uplift giả định (điểm % khách mua đơn 2)")
ax[0].set_ylabel("Doanh thu thêm (nghìn R$)")
ax[0].set_title("Giá trị kịch bản theo phân khúc"); ax[0].legend(); ax[0].grid(alpha=.3)
be = scen[scen["Giả định uplift"].str.startswith("+1.0")].set_index("Phân khúc")
ax[1].bar(be.index, be[BE_COL], color=[SEG_COLORS[s] for s in be.index])
for i, v in enumerate(be[BE_COL]):
    ax[1].text(i, v, f"R${v:.2f}", ha="center", va="bottom", fontsize=9)
ax[1].set_title("Chi phí tối đa / khách được nhắm nếu uplift = +1 điểm %")
ax[1].set_ylabel("R$ (tính trên doanh thu)"); ax[1].grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(f"{RESULTS}/impact_scenarios.png", dpi=150); plt.close()

# ── 8.4 Bảng hành động: Phân khúc → Hành động → KPI → Tác động → Cách kiểm chứng ──
def sc(seg, tag):
    return scen[(scen["Phân khúc"] == seg) & scen["Giả định uplift"].str.startswith(tag)].iloc[0]

dl  = pd.read_csv(f"{RESULTS}/drill_delivery.csv")
asc = pd.read_csv(f"{RESULTS}/association_tests.csv").set_index("Dimension")
dc8 = pd.read_csv(f"{RESULTS}/drill_category.csv")
dc8 = dc8[dc8["Mature_Customers"] >= 300].sort_values("Repeat180_Mature", ascending=False)
fast = dl[dl["Bucket"] == "≤7 ngày"]["Repeat180"].iloc[0]
slow = dl[dl["Bucket"] == ">21 ngày"]["Repeat180"].iloc[0]
rv_1 = dl[(dl["Dimension"] == "Đánh_giá") & (dl["Bucket"] == "1 sao")]["Repeat180"].iloc[0]
i_lo, i_hi = inst["Repeat180"].iloc[0], inst["Repeat180"].iloc[3]
n_all = seg8["Customers"].sum()
a, b, c = sc("Recent Buyers", "+1.0"), sc("At-Risk Buyers", "+1.0"), sc("Lost Customers", "+1.0")
d = sc("Repeat Buyers", "+10")

actions = [
 {"Phân khúc / chủ đề": "Recent Buyers",
  "Bằng chứng": f"{seg8.loc['Recent Buyers','Customers']/n_all:.1%} khách, "
                f"{seg8.loc['Recent Buyers','Revenue']/total_rev:.1%} doanh thu; mua gần đây, vẫn trong cửa sổ 180 ngày",
  "Hành động": "Chuỗi tin nhắn sau mua (khoảng ngày 30 và 60) gợi ý sản phẩm bổ sung ở danh mục có tỷ lệ mua lại cao "
               f"({dc8.iloc[0]['category']}, {dc8.iloc[1]['category']}); ưu đãi nhỏ, không chiết khấu sâu",
  "KPI": f"Tỷ lệ mua đơn 2 trong 180 ngày (nền {base_rep:.1%}); chi phí trên mỗi khách tái mua",
  "Tác động (kịch bản)": f"+1 điểm % ≈ +R${a['Doanh thu thêm (R$)']:,.0f} ({a['% doanh thu hiện tại']:.2f}% doanh thu); "
                         f"hòa vốn ≤ R${a[BE_COL]:.2f}/khách",
  "Cách kiểm chứng": "A/B test, nhóm đối chứng (holdout) 50%, đo sau 180 ngày"},
 {"Phân khúc / chủ đề": "At-Risk Buyers",
  "Bằng chứng": f"{seg8.loc['At-Risk Buyers','Customers']/n_all:.1%} khách nhưng chỉ "
                f"{seg8.loc['At-Risk Buyers','Revenue']/total_rev:.1%} doanh thu; giá trị đơn thấp nhất (R${seg8.loc['At-Risk Buyers','AOV']:.0f})",
  "Hành động": "Chỉ dùng kênh chi phí thấp, tự động (email/push); tránh giảm giá sâu vì biên hòa vốn rất mỏng",
  "KPI": "Tỷ lệ kích hoạt lại; chi phí trên mỗi khách kích hoạt lại",
  "Tác động (kịch bản)": f"+1 điểm % ≈ +R${b['Doanh thu thêm (R$)']:,.0f}; hòa vốn chỉ ≤ R${b[BE_COL]:.2f}/khách",
  "Cách kiểm chứng": "A/B test trên một mẫu nhỏ trước khi mở rộng"},
 {"Phân khúc / chủ đề": "Lost Customers",
  "Bằng chứng": f"{seg8.loc['Lost Customers','Customers']/n_all:.1%} khách, "
                f"{seg8.loc['Lost Customers','Revenue']/total_rev:.1%} doanh thu; giá trị đơn cao hơn At-Risk "
                f"(R${seg8.loc['Lost Customers','AOV']:.0f}) nhưng đã vắng hơn một năm",
  "Hành động": "Một đợt win-back có ngân sách trần, sau đó dừng; không đầu tư lặp lại nếu tỷ lệ phản hồi thấp",
  "KPI": "Tỷ lệ phản hồi; ROI của đợt win-back",
  "Tác động (kịch bản)": f"+1 điểm % ≈ +R${c['Doanh thu thêm (R$)']:,.0f}; hòa vốn ≤ R${c[BE_COL]:.2f}/khách "
                         "(uplift thực tế nhiều khả năng thấp hơn)",
  "Cách kiểm chứng": "A/B test; dừng nếu uplift không vượt chi phí hòa vốn"},
 {"Phân khúc / chủ đề": "Repeat Buyers",
  "Bằng chứng": f"Chỉ {seg8.loc['Repeat Buyers','Customers']/n_all:.1%} khách nhưng chi tiêu trung bình "
                f"R${seg8.loc['Repeat Buyers','Revenue']/seg8.loc['Repeat Buyers','Customers']:.0f}/khách, cao nhất",
  "Hành động": "Bảo vệ nhóm nhỏ nhưng giá trị cao: ưu tiên xử lý đơn/hỗ trợ, ưu đãi sớm, tránh để một trải nghiệm xấu làm mất khách",
  "KPI": "Số đơn trung bình mỗi khách; tỷ lệ giữ chân 180 ngày",
  "Tác động (kịch bản)": f"+10% chi tiêu ≈ +R${d['Doanh thu thêm (R$)']:,.0f} ({d['% doanh thu hiện tại']:.2f}% doanh thu)",
  "Cách kiểm chứng": "So sánh nhóm được ưu tiên với nhóm đối chứng cùng kỳ"},
 {"Phân khúc / chủ đề": "Trả góp ở đơn đầu (toàn hệ thống)",
  "Bằng chứng": f"Mua lại tăng theo số kỳ trả góp: {i_lo:.1%} (1 kỳ) → {i_hi:.1%} (7–10 kỳ); p = {p_i:.1e} "
                f"nhưng Cramér's V chỉ {v_i:.3f} (yếu); đồng thời là đặc trưng SHAP số 1",
  "Hành động": "Thử nghiệm làm nổi bật tùy chọn trả góp cho giỏ hàng giá trị cao ở đơn đầu tiên",
  "KPI": "Tỷ lệ mua đơn 2 trong 180 ngày theo nhánh A/B; tỷ lệ bỏ giỏ",
  "Tác động (kịch bản)": "Không ước lượng: có thể do chọn lọc (khách chọn trả góp vốn đã khác nhóm còn lại), chưa chứng minh nhân quả",
  "Cách kiểm chứng": "A/B test ngẫu nhiên; chỉ nhân rộng nếu uplift vượt mức nhiễu"},
 {"Phân khúc / chủ đề": "Giao hàng nhanh hơn (toàn hệ thống)",
  "Bằng chứng": f"Mua lại giảm nhẹ khi giao chậm ({fast:.2%} → {slow:.2%}) nhưng không có ý nghĩa thống kê "
                f"(p = {asc.loc['Thời gian giao','p_value']:.2f}, V = {asc.loc['Thời gian giao','Cramers_V']:.3f}); "
                f"khách đánh giá 1 sao vẫn mua lại {rv_1:.2%}",
  "Hành động": "Giữ chuẩn dịch vụ để bảo vệ điểm đánh giá, nhưng không dùng 'giữ chân khách' làm lý do chính để đầu tư lớn",
  "KPI": "Điểm đánh giá, tỷ lệ giao trễ; tỷ lệ mua đơn 2 nếu có thử nghiệm",
  "Tác động (kịch bản)": "Không ước lượng: dữ liệu chưa ủng hộ mối liên hệ với việc quay lại",
  "Cách kiểm chứng": "A/B test (ví dụ giao nhanh cho một nhóm bang) trước khi đầu tư"},
]
act = pd.DataFrame(actions)
act.to_csv(f"{RESULTS}/action_table.csv", index=False)
print("\n  Bảng hành động đã lưu (action_table.csv):")
print(act[["Phân khúc / chủ đề", "Tác động (kịch bản)"]].to_string(index=False))
print("\n[✓] BƯỚC 8 hoàn tất\n")

spark.stop()
print("=" * 70)
print("PIPELINE HOÀN TẤT — kết quả lưu tại:", RESULTS)
print("=" * 70)
