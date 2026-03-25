# Bài Tập Lớn Môn Học Máy — Adult Census Income Prediction Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VietAnh1803/HCMUT_ML_Travel_Insurance_Pipeline/blob/main/notebooks/tabular_data_pipeline.ipynb)

## Thông tin chung

| Mục | Chi tiết |
|---|---|
| **Môn học** | Nhập môn Học Máy |
| **Mã môn học** | *(Vui lòng cập nhật mã môn)* |
| **Trường** | Đại học Bách Khoa TP.HCM (HCMUT) |
| **Học kỳ** | HK2 - Năm học 2025-2026 |
| **GVHD** | *(Vui lòng cập nhật tên GVHD)* |

## Mục tiêu Báo cáo

Xây dựng hệ thống học máy (Machine Learning Pipeline) toàn diện và mạnh mẽ để giải quyết bài toán **Dự đoán thu nhập cá nhân >50K$** (Adult Census Income). Dự án đáp ứng 100% các tiêu chí nghiêm ngặt từ đồ án học phần:

- **Khám phá dữ liệu (EDA):** Tự động truy xuất Boxplot, Heatmap Tương quan, CountPlot.
- **Tiền xử lý (Imputation & Encoding):** Xử lý quy mô lớn các giá trị khuyết mã `?` bằng `SimpleImputer`, tự động nhúng `OneHotEncoder` và `StandardScaler`.
- **Xử lý Dữ liệu Mất cân bằng (Imbalanced):** Can thiệp tỷ lệ với **SMOTE** (Synthetic Minority Over-sampling Technique).
- **So sánh Đấu trường Mô hình (Model Comparison):** Đánh giá song song giữa `Logistic Regression`, `Random Forest`, và `LightGBM`.
- **Nghiên cứu Nâng cao (Deep Learning):** Áp dụng Mạng Nơ-ron `MLP PyTorch` kết hợp `Class Weights`.
- **Trực quan hóa Hoàn hảo:** Tự động xuất biểu đồ Bar Chart so sánh F1/AUC/Accuracy và Heatmap ma trận nhầm lẫn (Confusion Matrix).
- **Xuất đặc trưng (Features Exporting):** Rút trích Vector dữ liệu đã xử lý ra định dạng `.npy` (Numpy Array).

## Dataset

- **Tên dataset**: Adult Census Income
- **Nguồn Kaggle**: [uciml/adult-census-income](https://www.kaggle.com/datasets/uciml/adult-census-income)
- **Kích thước**: Trọng tải hơn 48.000 mẫu với 15 cấu trúc đặc trưng phong phú.
- **Phương thức tải**: Tự động thông qua thư viện `kagglehub` (Không cần Mount Drive thủ công).

## Hướng dẫn chạy Notebook

### Trên Google Colab (Khuyến nghị để tận dụng GPU/Tốc độ)

1. Mở [Google Colab](https://colab.research.google.com/)
2. Chọn hộp thoại **File → Open notebook → GitHub**
3. Nhập đường dẫn Repository: `https://github.com/VietAnh1803/HCMUT_ML_Travel_Insurance_Pipeline`
4. Mở trực tiếp file `notebooks/tabular_data_pipeline.ipynb`
5. Bấm chọn **Runtime → Run all (Chạy tất cả)** để mô hình tự động kết xuất báo cáo cuối cùng.

### Yêu cầu thư viện (Requirements)

- `kagglehub`, `imbalanced-learn`, `lightgbm`, `torch`
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

## Cấu trúc thư mục Nộp Bài

```
HCMUT_ML_Travel_Insurance_Pipeline/
├── notebooks/
│   └── tabular_data_pipeline.ipynb    # Code chạy chính toàn bộ quy trình
├── features/                          # Thư mục lưu Model & Vector 特徵
│   ├── X_train_processed.npy
│   ├── y_train_resampled.npy
│   ├── X_test_processed.npy
│   ├── y_test.npy
│   ├── best_classic_pipeline.pkl
│   └── mlp_weights.pth
├── mlAssignments_v1.1.pdf             # Yêu cầu đề bài
└── README.md                          # Hướng dẫn chi tiết
```
