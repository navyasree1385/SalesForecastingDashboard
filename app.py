# ===========================
# Sidebar Navigation
# ===========================
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)
st.title("📊 Sales Forecast Dashboard")
st.markdown(
    "Interactive dashboard for sales analysis, forecasting, anomaly detection, and product demand segmentation."
)
@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="%d/%m/%Y"
    )

    return df


df = load_data()


def build_features(series):

    data = pd.DataFrame({"Sales": series})

    data["lag1"] = data["Sales"].shift(1)
    data["lag2"] = data["Sales"].shift(2)
    data["lag3"] = data["Sales"].shift(3)

    data["rolling_mean_3"] = (
        data["Sales"]
        .rolling(3)
        .mean()
    )

    data["month"] = data.index.month
    data["quarter"] = data.index.quarter
    data["season"] = ((data.index.month % 12 + 3) // 3)

    return data.dropna()
page = st.sidebar.radio(
    "Navigation",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Demand Segments"
    ]
)

# ============================================================
# PAGE 1 : SALES OVERVIEW
# ============================================================

if page == "Sales Overview":

    st.header("📊 Sales Overview")

    # KPI Cards
    total_sales = df["Sales"].sum()
    total_orders = len(df)
    avg_sales = df["Sales"].mean()

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Sales", f"${total_sales:,.0f}")
    c2.metric("Orders", total_orders)
    c3.metric("Average Order Value", f"${avg_sales:,.2f}")

    # -------------------------
    # Filters
    # -------------------------

    region = st.sidebar.multiselect(
        "Select Region",
        options=sorted(df["Region"].unique()),
        default=sorted(df["Region"].unique())
    )

    category = st.sidebar.multiselect(
        "Select Category",
        options=sorted(df["Category"].unique()),
        default=sorted(df["Category"].unique())
    )

    filtered = df[
        (df["Region"].isin(region))
        &
        (df["Category"].isin(category))
    ]

    # -------------------------
    # Yearly Sales
    # -------------------------

    year_sales = (
        filtered
        .groupby(filtered["Order Date"].dt.year)["Sales"]
        .sum()
        .reset_index()
    )

    year_sales.columns = ["Year", "Sales"]

    fig = px.bar(
        year_sales,
        x="Year",
        y="Sales",
        title="Total Sales by Year",
        color="Sales",
        color_continuous_scale="Blues",
        text_auto=".2s"
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title="Year",
        yaxis_title="Sales ($)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Monthly Trend
    # -------------------------

    monthly = (
        filtered
        .groupby(pd.Grouper(
            key="Order Date",
            freq="MS"
        ))["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="Order Date",
        y="Sales",
        title="Monthly Sales Trend",
        markers=True
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Sales ($)"
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 2 : FORECAST EXPLORER
# ============================================================

elif page == "Forecast Explorer":

    st.header("📈 Forecast Explorer")

    forecast_type = st.selectbox(
        "Forecast By",
        ["Category", "Region"]
    )

    if forecast_type == "Category":

        item = st.selectbox(
            "Select Category",
            sorted(df["Category"].unique())
        )

        temp = df[df["Category"] == item]

    else:

        item = st.selectbox(
            "Select Region",
            sorted(df["Region"].unique())
        )

        temp = df[df["Region"] == item]

    months = st.slider(
        "Forecast Horizon (Months)",
        1,
        3,
        3
    )
    # -------------------------
    # Monthly Sales
    # -------------------------

    monthly = (
        temp
        .groupby(pd.Grouper(
            key="Order Date",
            freq="MS"
        ))["Sales"]
        .sum()
    )

    data = build_features(monthly)

    train = data.iloc[:-3]
    test = data.iloc[-3:]

    X_train = train.drop("Sales", axis=1)
    y_train = train["Sales"]

    X_test = test.drop("Sales", axis=1)
    y_test = test["Sales"]

    # -------------------------
    # Train XGBoost
    # -------------------------

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )

    # -------------------------
    # Recursive Forecast
    # -------------------------

    history = monthly.copy()

    future = []

    for i in range(months):

        lag1 = history.iloc[-1]
        lag2 = history.iloc[-2]
        lag3 = history.iloc[-3]

        rolling = history.iloc[-3:].mean()

        next_date = history.index[-1] + pd.DateOffset(months=1)

        row = pd.DataFrame({

            "lag1": [lag1],
            "lag2": [lag2],
            "lag3": [lag3],
            "rolling_mean_3": [rolling],
            "month": [next_date.month],
            "quarter": [next_date.quarter],
            "season": [((next_date.month % 12 + 3)//3)]

        })

        forecast = model.predict(row)[0]

        future.append(forecast)

        history.loc[next_date] = forecast

    # -------------------------
    # Metrics
    # -------------------------

    c1, c2 = st.columns(2)

    c1.metric("MAE", f"{mae:,.2f}")

    c2.metric("RMSE", f"{rmse:,.2f}")

    # -------------------------
    # Forecast Table
    # -------------------------
    forecast_df = pd.DataFrame({
        "Month": [f"Month {i+1}" for i in range(months)],
        "Forecast Sales ($)": np.round(future, 2)
    })

    st.dataframe(
        forecast_df,
        use_container_width=True,
        hide_index=True
    )

    # -------------------------
    # Forecast Chart
    # -------------------------

    fig = px.line(
        forecast_df,
        x="Month",
        y="Forecast Sales ($)",
        markers=True,
        title=f"{item} Sales Forecast"
    )

    fig.update_traces(
        line=dict(width=4),
        marker=dict(size=10)
    )

    fig.update_layout(
        xaxis_title="Forecast Month",
        yaxis_title="Predicted Sales ($)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Forecast Summary
    # -------------------------

    st.subheader("Forecast Summary")

    st.success(
        f"""
The XGBoost model predicts sales for **{item}** over the next **{months} month(s)**.

**Model Performance**
- MAE: {mae:,.2f}
- RMSE: {rmse:,.2f}
        """
    )

# ============================================================
# PAGE 3 : ANOMALY REPORT
# ============================================================

elif page == "Anomaly Report":

    st.header("🚨 Sales Anomaly Report")

    anomaly_df = pd.read_csv("anomaly_data.csv")

    anomaly_df["Order Date"] = pd.to_datetime(
        anomaly_df["Order Date"]
    )

    st.subheader("Weekly Sales with Detected Anomalies")

    fig = px.line(
        anomaly_df,
        x="Order Date",
        y="Sales",
        title="Weekly Sales"
    )

    anomalies = anomaly_df[
        anomaly_df["Anomaly"] == 1
    ]

    fig.add_scatter(
        x=anomalies["Order Date"],
        y=anomalies["Sales"],
        mode="markers",
        marker=dict(
            color="red",
            size=10
        ),
        name="Anomaly"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Detected Anomalies")

    st.dataframe(

        anomalies[
            [
                "Order Date",
                "Sales"
            ]
        ],

        use_container_width=True,
        hide_index=True

    )

    st.info(
        """
Isolation Forest identified these weeks as unusual sales periods.
These anomalies may correspond to promotions,
holiday sales, bulk orders, or unexpected demand fluctuations.
"""
    )
    # ============================================================
# PAGE 4 : PRODUCT DEMAND SEGMENTS
# ============================================================

elif page == "Demand Segments":

    st.header("📦 Product Demand Segments")

    cluster_df = pd.read_csv(
        "cluster_data.csv",
        index_col=0
    )

    st.subheader(
        "Demand Clusters"
    )

    fig = px.scatter(

        cluster_df,

        x="PC1",

        y="PC2",

        color="Cluster Name",

        hover_name=cluster_df.index,

        size="TotalSales",

        title="Product Demand Clusters"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Cluster Members"
    )

    st.dataframe(

        cluster_df[
            [
                "Cluster Name",
                "TotalSales",
                "GrowthRate",
                "Volatility",
                "AverageOrderValue"
            ]
        ],

        use_container_width=True

    )

    st.subheader(
        "Recommended Stocking Strategy"
    )

    st.markdown("""

### 🟢 High Volume, Stable Demand

Maintain high inventory levels and replenish stock regularly.
These products generate consistent sales and should always be available.

---

### 🔵 Growing Demand

Increase inventory gradually to support rising customer demand.
Monitor trends closely to avoid stock shortages.

---

### 🟠 Low Volume, High Volatility

Maintain limited inventory and adopt demand-driven replenishment.
Avoid overstocking due to unpredictable sales.

---

### 🔴 Declining Demand

Reduce inventory levels and clear excess stock through promotions.
Review demand periodically before reordering.

""")