# Bài Tập Lớn Môn Học Máy — Travel Insurance Prediction Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VietAnh1803/HCMUT_ML_Travel_Insurance_Pipeline/blob/main/notebooks/tabular_data_pipeline.ipynb)

## Thông tin chung

| Mục | Chi tiết |
|---|---|
| **Môn học** | Nhập môn Học Máy |
| **Mã môn học** | *(Vui lòng cập nhật mã môn)* |
| **Trường** | Đại học Bách Khoa TP.HCM (HCMUT) |
| **Học kỳ** | HK2 - Năm học 2025-2026 |
| **GVHD** | *(Vui lòng cập nhật tên GVHD)* |

## Thành viên nhóm

| STT | Họ và Tên | MSSV | Email |
|---|---|---|---|
| 1 | *(Cập nhật)* | *(Cập nhật)* | *(Cập nhật)* |
| 2 | *(Cập nhật)* | *(Cập nhật)* | *(Cập nhật)* |

## Mục tiêu

Xây dựng pipeline học máy toàn diện cho bài toán **dự đoán mua bảo hiểm du lịch** (Travel Insurance Prediction), bao gồm:

- **EDA** (Exploratory Data Analysis): Khám phá dữ liệu, phát hiện outliers, phân tích tương quan
- **Tiền xử lý**: Imputation (KNNImputer), Encoding (OneHotEncoder), Scaling (StandardScaler/MinMaxScaler)
- **Giảm chiều**: PCA với phân tích explained variance (90%/95%)
- **Huấn luyện mô hình**: Logistic Regression, SVM, Random Forest
- **So sánh cấu hình**: 6 cấu hình (3 mô hình × 2 scaler), bảng tổng hợp kết quả
- **Tối ưu siêu tham số**: RandomizedSearchCV cho Random Forest
- **Đánh giá**: Confusion Matrix, ROC Curve, AUC, Cross-Validation
- **Deep Learning**: Mạng MLP (Multi-Layer Perceptron) với PyTorch
- **Lưu đặc trưng**: File `.npy` và mô hình `.pkl`, `.pth`

## Dataset

- **Tên**: Travel Insurance Prediction Data
- **Nguồn**: [Kaggle - tejashvi14/travel-insurance-prediction-data](https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data)
- **Kích thước**: 1987 mẫu, 10 đặc trưng
- Dataset được tải tự động bằng `kagglehub` khi chạy notebook

## Hướng dẫn chạy Notebook

### Trên Google Colab (Khuyến nghị)

1. Mở [Google Colab](https://colab.research.google.com/)
2. Chọn **File → Open notebook → GitHub**
3. Nhập URL: `https://github.com/VietAnh1803/HCMUT_ML_Travel_Insurance_Pipeline`
4. Chọn file `notebooks/tabular_data_pipeline.ipynb`
5. Chọn **Runtime → Run all**

### Trên máy tính cá nhân

```bash
git clone https://github.com/VietAnh1803/HCMUT_ML_Travel_Insurance_Pipeline.git
cd HCMUT_ML_Travel_Insurance_Pipeline
pip install kagglehub torch matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn pandas numpy joblib
jupyter notebook notebooks/tabular_data_pipeline.ipynb
```

### Yêu cầu thư viện

- `kagglehub` — Tải dataset từ Kaggle
- `pandas`, `numpy` — Xử lý dữ liệu
- `scikit-learn`, `imbalanced-learn` — Pipeline ML, SMOTE, đánh giá
- `xgboost`, `lightgbm` — Gradient Boosting Models
- `torch` — Deep Learning (PyTorch)
- `matplotlib`, `seaborn` — Trực quan hóa
- `joblib` — Lưu mô hình

## Cấu trúc thư mục

```
HCMUT_ML_Travel_Insurance_Pipeline/
├── notebooks/
│   └── tabular_data_pipeline.ipynb   # Notebook chính
├── modules/                           # Module hỗ trợ (nếu có)
├── reports/                           # Báo cáo PDF
├── features/                          # Đặc trưng & Mô hình đã lưu
│   ├── X_train_features.npy
│   ├── X_test_features.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   ├── travel_insurance_pipeline.pkl
│   └── travel_insurance_mlp.pth
├── mlAssignments_v1.1.pdf             # Đề bài
├── README.md
└── .gitignore
```

## Liên kết

- **GitHub**: [https://github.com/VietAnh1803/HCMUT_ML_Travel_Insurance_Pipeline](https://github.com/VietAnh1803/HCMUT_ML_Travel_Insurance_Pipeline)
- **Colab Notebook**: Mở trực tiếp từ GitHub qua Colab (xem hướng dẫn ở trên)
