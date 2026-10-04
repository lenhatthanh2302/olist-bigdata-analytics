# Olist E-Commerce Big Data Analytics

Phân tích dữ liệu lớn trên Brazil E-Commerce Public Dataset (Olist) bằng PySpark. Pipeline gồm 8 bước: EDA → RFM → K-Means → Churn → Benchmark → SHAP → Câu hỏi nghiên cứu (RQ1–4) → Bảng hành động, kèm Streamlit dashboard 6 tab với biểu đồ Plotly tương tác.

Bài nghiên cứu môn *Nghiên cứu dữ liệu lớn và Ứng dụng trong kinh doanh* (253MIE400801).

## Kết quả chính

| Chỉ số | Giá trị |
|---|---|
| Đơn hàng đã giao | 96,477 |
| Khách hàng duy nhất | 93,357 |
| Doanh thu | R$ 15,422,461.77 |
| Giá trị đơn trung bình | R$ 159.86 |
| Tỷ lệ churn (mua 1 lần, không quay lại sau 180 ngày) | 57.6% |
| AUC churn (đặc trưng đơn hàng đầu tiên) — LR / RF / GBT | 0.638 / 0.701 / 0.746 |
| AUC mua lại 180 ngày (nhóm khách đủ tuổi, 56,035 khách) | 0.54–0.56 |
| Tỷ lệ mua lại nền (nhóm đủ tuổi) | 3.1% |
| Top 10% khách tạo ra | 38.2% doanh thu |
| PySpark vs Pandas (Join + RFM, ×10 ≈ 1 triệu đơn) | 1.85× nhanh hơn; ở ×1–×2 Pandas nhanh hơn |

**Về nhãn churn.** Churn được định nghĩa bằng quy ước (mua một lần, quá 180 ngày kể từ đơn cuối) nên nhãn phụ thuộc vào ngày cắt dữ liệu: khách mua trong 180 ngày cuối chưa đủ thời gian để quay lại. Vì vậy pipeline có thêm mô hình trên **nhóm khách đủ tuổi** để loại thiên lệch này. Kết quả là thông tin đơn hàng đầu tiên chỉ dự báo yếu việc khách quay lại; mô hình phù hợp để xếp hạng ưu tiên, không phải dự báo từng cá nhân.

## Pipeline

| Bước | Nội dung |
|---|---|
| 1. Load & EDA | Load 9 CSV, join master DataFrame, tổng quan đơn hàng / doanh thu / danh mục |
| 2. RFM | Recency / Frequency / Monetary bằng PySpark |
| 3. Phân khúc | K-Means (k=4, chọn bằng Elbow + Silhouette) trên RFM, PySpark MLlib |
| 4. Churn | LR, Random Forest, GBT; mô hình mua lại trên nhóm đủ tuổi (class weight, Lift, Cumulative Gains) |
| 5. Benchmark | PySpark vs Pandas ở quy mô ×1/×2/×5/×10 (median 3 lần chạy, `local[*]`) |
| 6. SHAP | Giải thích mô hình mua lại (Random Forest scikit-learn) |
| 7. RQ1–4 | Doanh thu theo phân khúc / Pareto; mua lại theo bang, danh mục, trải nghiệm giao hàng; chi-square + Cramér's V; 15 phép kiểm tra nhất quán |
| 8. Quyết định | Bảng hành động theo phân khúc, kịch bản tác động và điểm hòa vốn, bằng chứng trả góp |

## Phân khúc khách hàng

| Phân khúc | Số khách | Recency TB | Chi tiêu TB |
|---|---|---|---|
| Recent Buyers | 32,301 (34.6%) | 86 ngày | R$ 206 |
| At-Risk Buyers | 35,244 (37.8%) | 241 ngày | R$ 109 |
| Lost Customers | 23,007 (24.6%) | 450 ngày | R$ 175 |
| Repeat Buyers | 2,805 (3.0%) | 221 ngày | R$ 320 |

K=4 là lựa chọn kinh doanh có chủ đích (Silhouette cao nhất ở k=2 nhưng chỉ tách ~3% khách mua lại khỏi phần còn lại).

## Dashboard (6 tab)

1. **EDA Overview** — KPI, xu hướng đơn theo tháng, top danh mục
2. **Customer Segmentation** — phân bố phân khúc, chọn K, 3D RFM scatter
3. **Churn Prediction** — so sánh mô hình, ROC, confusion matrix, mô hình mua lại
4. **Benchmark & XAI** — PySpark vs Pandas theo quy mô, SHAP
5. **Phân tích chuyên sâu (RQ)** — RQ1–4 với khoảng tin cậy 95% và kiểm tra nhất quán
6. **Quyết định & Hạn chế** — bảng hành động, máy tính hòa vốn tự đặt giả định, thảo luận

Dữ liệu Olist là dữ liệu quan sát nên các con số tác động là kịch bản giả định, không phải quan hệ nhân quả.

## Cài đặt

Dashboard (Streamlit Cloud chỉ cần file này, đọc kết quả có sẵn trong `Results/`):

```bash
pip install -r requirements.txt
```

Chạy lại pipeline (cần Java 8/11/17 cho PySpark):

```bash
pip install -r requirements_pipeline.txt
```

## Dataset

Dataset đã có sẵn trong repo. Nếu cần tải lại: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
Đặt 9 file CSV vào thư mục `Data/`.

## Chạy

```bash
# Bước 1: pipeline PySpark (tạo lại Results/, mất khoảng 10–15 phút)
# Đặt OLIST_HEADLESS=1 nếu chạy không có màn hình để không mở cửa sổ biểu đồ
python pipeline.py

# Bước 2: dashboard
streamlit run app.py
```

Các file `Results/*.parquet` (~26 MB) không được commit; pipeline tự tạo lại khi chạy.

## Cấu trúc project

```
├── Data/                      # 9 Olist CSV
├── Results/                   # CSV + PNG kết quả (dashboard đọc từ đây)
├── pipeline.py                # PySpark pipeline 8 bước
├── app.py                     # Streamlit dashboard 6 tab (Plotly)
├── requirements.txt           # Cho dashboard / Streamlit Cloud
└── requirements_pipeline.txt  # Cho pipeline (PySpark, scikit-learn, SHAP, ...)
```
