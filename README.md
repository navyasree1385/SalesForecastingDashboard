# 📊 Sales Forecasting Dashboard

An interactive Sales Forecasting Dashboard developed using **Streamlit** that provides sales analysis, demand forecasting, anomaly detection, and product demand segmentation. The dashboard helps businesses analyze historical sales trends, forecast future sales, identify unusual sales patterns, and make inventory decisions using machine learning techniques.

---

## 🚀 Live Demo

🔗 **Streamlit App:** https://salesforecastingdashboard-kiyrzgwof6mz3tf3moglo2.streamlit.app/

---

## 📌 Project Overview

This dashboard was developed as part of a sales analytics and forecasting project. It integrates statistical analysis, machine learning, and interactive visualization into a single web application.

The dashboard enables users to:

- Analyze yearly and monthly sales trends
- Forecast future sales using XGBoost
- Detect anomalous sales periods
- Segment product demand using K-Means clustering
- Support inventory and stocking decisions through data-driven insights

---

## ✨ Features

### 📈 Page 1 – Sales Overview

- Total Sales KPI
- Total Orders KPI
- Average Order Value KPI
- Yearly Sales Bar Chart
- Monthly Sales Trend Line Chart
- Interactive Region Filter
- Interactive Category Filter

---

### 🔮 Page 2 – Forecast Explorer

- Forecast by Category or Region
- Forecast Horizon Slider (1–3 Months)
- XGBoost-based Sales Forecasting
- Forecast Table
- Forecast Visualization
- Model Performance Metrics (MAE & RMSE)

---

### 🚨 Page 3 – Anomaly Report

- Weekly Sales Trend
- Isolation Forest Anomaly Detection
- Highlighted Anomaly Points
- Anomaly Report Table

---

### 📦 Page 4 – Product Demand Segments

- Product Demand Clustering using K-Means
- PCA-based Cluster Visualization
- Cluster-wise Product Information
- Inventory & Stocking Strategy Recommendations

---

## 🧠 Machine Learning Models Used

### Sales Forecasting

- SARIMA
- Facebook Prophet
- XGBoost Regressor (Selected as Best Model)

### Anomaly Detection

- Isolation Forest
- Z-Score Method

### Product Segmentation

- K-Means Clustering
- Principal Component Analysis (PCA)

---

## 📊 Evaluation Metrics

The forecasting model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Matplotlib
- Scikit-learn
- XGBoost
- Statsmodels
- Prophet

---

## 📂 Project Structure

```
SalesForecastingDashboard/
│
├── app.py
├── train.csv
├── anomaly_data.csv
├── cluster_data.csv
├── requirements.txt
├── README.md
```

---

## ▶️ Running the Project Locally

### Clone the repository

```bash
git clone https://github.com/navyasree1385/SalesForecastingDashboard.git
```

### Navigate to the project folder

```bash
cd SalesForecastingDashboard
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📷 Dashboard Pages

- 📊 Sales Overview
- 🔮 Forecast Explorer
- 🚨 Anomaly Report
- 📦 Product Demand Segments

---

## 📈 Business Value

This dashboard helps organizations:

- Monitor sales performance
- Forecast future demand
- Detect unusual sales behavior
- Optimize inventory planning
- Support data-driven decision making

---

## 👩‍💻 Developed By

**Navyasree M**

GitHub: https://github.com/navyasree1385

---

## 📄 License

This project is developed for educational and academic purposes.
