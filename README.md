# Olist E-Commerce Big Data Analytics

Phân tích dữ liệu lớn trên Brazil E-Commerce Public Dataset (Olist) sử dụng PySpark, với pipeline 6 bước: EDA → RFM → Customer Segmentation → Churn Prediction → Benchmark → SHAP Explainability, kèm Streamlit Dashboard 4 tabs.

## Key Findings

| Metric | Value |
|---|---|
| Total Delivered Orders | 96,477 |
| Unique Customers | 93,357 |
| Total Revenue | R$ 15,422,461.77 |
| Avg Order Value | R$ 159.86 |
| Churn Rate | 57.6% |
| LR Churn AUC-ROC | 1.0000 |
| RF Churn AUC-ROC | 0.9997 |
| Top Churn Feature (SHAP) | Recency (0.468) |
| PySpark RFM Speedup vs Pandas | 3.09× faster |

## Pipeline

| Bước | Nội dung |
|---|---|
| 1. Load & EDA | Load 9 CSV, join master DataFrame, tổng quan đơn hàng/doanh thu/danh mục/khu vực |
| 2. RFM | Tính Recency / Frequency / Monetary bằng PySpark Window Functions |
| 3. Customer Segmentation | K-Means (k=4) clustering trên RFM (PySpark MLlib) |
| 4. Churn Prediction | Logistic Regression + Random Forest (PySpark MLlib) |
| 5. Tool Benchmark | So sánh PySpark vs Pandas (thời gian xử lý) |
| 6. SHAP XAI | Explain top features của churn prediction (scikit-learn + SHAP) |

## Customer Segments

| Segment | Label | Count | Avg Recency | Avg Monetary |
|---|---|---|---|---|
| 0 | Lost Customers | 23,007 | 450 days | R$ 175 |
| 1 | Loyal Customers | 32,301 | 86 days | R$ 206 |
| 2 | At-Risk Customers | 35,244 | 241 days | R$ 109 |
| 3 | At-Risk Customers (repeat) | 2,805 | 221 days | R$ 320 |

## Cài đặt

```bash
pip install -r requirements.txt
```

## Dataset

Dataset đã được đưa vào repo. Nếu cần tải lại, lấy từ Kaggle:

> https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Đặt 9 file CSV vào thư mục `Data/`.

## Chạy

```bash
# Bước 1: Chạy pipeline PySpark (tạo thư mục Results/)
python pipeline.py

# Bước 2: Khởi động Streamlit dashboard
streamlit run app.py
```

## Cấu trúc project

```
├── Data/               # 9 Olist CSV files
├── Results/            # Kết quả pipeline (tự generate khi chạy pipeline.py)
├── pipeline.py         # PySpark pipeline 6 bước
├── app.py              # Streamlit dashboard 4 tabs
├── requirements.txt
└── test.py
```
