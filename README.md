# Olist E-Commerce Big Data Analytics

Phân tích dữ liệu lớn trên Brazil E-Commerce Public Dataset (Olist) sử dụng PySpark, với pipeline 7 bước: EDA → RFM → Customer Segmentation → Churn Prediction → Benchmark → SHAP Explainability → Streamlit Dashboard.

## Pipeline

| Bước | Nội dung |
|---|---|
| 1. Load & EDA | Load 9 CSV, join master DataFrame, tổng quan đơn hàng/doanh thu/danh mục/khu vực |
| 2. RFM | Tính Recency / Frequency / Monetary bằng PySpark Window Functions |
| 3. Customer Segmentation | K-Means clustering trên RFM (PySpark MLlib) |
| 4. Churn Prediction | Logistic Regression + Random Forest (PySpark MLlib) |
| 5. Tool Benchmark | So sánh PySpark vs Pandas (thời gian xử lý) |
| 6. SHAP XAI | Explain top features của churn prediction |
| 7. Streamlit Dashboard | 4 tabs: EDA / Segmentation / Churn / Benchmark+XAI |

## Cài đặt

```bash
pip install -r requirements.txt
```

## Dataset

Tải **Brazil E-Commerce Public Dataset by Olist** từ Kaggle:

> https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Giải nén và đặt 9 file CSV vào thư mục `Data/`:

```
Data/
├── olist_customers_dataset.csv
├── olist_geolocation_dataset.csv
├── olist_orders_dataset.csv
├── olist_order_items_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_products_dataset.csv
├── olist_sellers_dataset.csv
└── product_category_name_translation.csv
```

## Chạy

```bash
# Bước 1: Chạy pipeline PySpark (tạo thư mục Results/)
python pipeline.py

# Bước 2: Khởi động Streamlit dashboard
streamlit run app.py
```

## Cấu trúc project

```
├── Data/               # Dataset (không commit - tải từ Kaggle)
├── Results/            # Kết quả pipeline (tự generate)
├── pipeline.py         # PySpark pipeline 7 bước
├── app.py              # Streamlit dashboard 4 tabs
├── requirements.txt
└── test.py
```
