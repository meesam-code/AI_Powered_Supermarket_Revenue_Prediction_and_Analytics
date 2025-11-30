import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import time
from calendar import monthrange    


 
# PAGE CONFIGURE
st.set_page_config(
    page_title="Smart Supermarket Revenue Predictor",
    page_icon="📊",
    layout="wide"
)

# SIDEBAR – INPUTS

with st.sidebar:
    st.markdown("<h2 style='color:black;'>⚙️ Prediction Parameters</h2>", unsafe_allow_html=True)

    item_category = st.selectbox("Item Category", ["Dairy","Snacks","Meat","Beverages","Household","Frozen"])
    store_type = st.selectbox("Store Type", ["Grocery","Supermarket Type1","Supermarket Type2"])
    outlet_location = st.selectbox("Outlet Location", ["Tier 1","Tier 2","Tier 3"])
    outlet_size = st.selectbox("Outlet Size", ["Small","Medium","Large"])

    item_weight = st.number_input("Item Weight (kg)", min_value=1.0, max_value=100.0, value=10.0)
    item_mrp = st.number_input("Item Price (MRP)", min_value=1.0, max_value=500.0, value=50.0)
    outlet_sales = st.number_input("Outlet Sales", min_value=1.0, max_value=1000.0, value=100.0)

    st.divider()
    st.markdown("<h2 style='color:black;'>🗓 Time Selection</h2>", unsafe_allow_html=True)

    year = st.number_input("Year", min_value=2000, max_value=2100, value=2025)
    month = st.selectbox(
        "Month",
        range(1,13),
        format_func=lambda x: pd.Timestamp(2025,x,1).strftime("%B")
    )

st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#f0f8ff,#e0f4ff,#d6ecff);
    font-family: Segoe UI, sans-serif;
}
h1 {
    text-align:center;
    background: linear-gradient(90deg,#0d6efd,#00d2ff);
    -webkit-background-clip:text;
    color: transparent;
    font-size:64px;
    font-weight:800;
}
h3 {
    text-align:center;
    color:#001529;
}

/* SIDEBAR */
[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#010c1f,#041836);
    color:black;
    padding:25px;
    border-radius:15px;
    box-shadow:4px 0 12px rgba(0,0,0,0.8);
}
[data-testid="stSidebar"] * {
    color:black;
}
            
}

/* SIDEBAR INPUTS */
[data-testid="stSidebar"] input, [data-testid="stSidebar"] .stSelectbox {
    background:#081f40 !important;
    color:white !important;
    border-radius:10px;
    border:1px solid #00d2ff;
}

/* DROPDOWNS */
div[role="listbox"]{
    background:#041836;
    color:white;
}

/* KPI CARDS */
.kpi {
    background: linear-gradient(135deg,#dff7ff,#bdf0ff,#9bdcff);
    padding:25px;
    border-radius:24px;
    text-align:center;
    box-shadow:0px 6px 20px rgba(0,0,0,0.15);
    font-size:22px;
    color:#001529;
}

/* BUTTON */
button[kind="primary"]{
    width:100%;
    border-radius:35px;
    padding:17px;
    font-size:24px;
    background: linear-gradient(145deg,#0d6efd,#00d2ff);
    border:none;
    color:black;
    box-shadow: 0 0 12px #00d2ff;
    transition:.3s ease;
}
button[kind="primary"]:hover{
    transform: translateY(-4px);
    box-shadow:0 0 25px #00d2ff;
}

/* PANELS */
.stDataFrame, .stPlotlyChart, .element-container{
    background: linear-gradient(135deg,#ffffff,#e0f4ff);
    padding:15px;
    border-radius:20px;
    box-shadow:0 5px 15px rgba(0,0,0,0.15);
}

/* FOOTER */
.footer {
    background:#d6ecff;
    color:#001529;
    padding:15px;
    border-radius:10px;
    text-align:center;
    margin-top:15px;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("<h1>🛒 Smart Supermarket Revenue Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3>AI Revenue Forecast & Sales Intelligence</h3><br>", unsafe_allow_html=True)


# MAIN BUTTON

simulate = st.button("🚀 Predict Monthly Revenue", type="primary")
API = "http://127.0.0.1:8000/predict"

def predict(payload):
    try:
        r = requests.post(API, json=payload, timeout=8)
        if r.status_code == 200:
            return float(r.json()["Predicted_Revenue"])
    except:
        return 0

# MONTHLY PREDICTION
if simulate:
    days = monthrange(year,month)[1]
    all_dates = pd.date_range(f"{year}-{month}-01", periods=days)
    weekdays = all_dates

    progress_bar = st.progress(0)
    progress_text = st.empty()

    results=[]
    for i,day in enumerate(weekdays):
        progress_bar.progress((i+1)/len(weekdays))
        progress_text.markdown(f"⏳ Day {i+1} of {len(weekdays)}")

        outlet_sales_day=max(outlet_sales*(1+np.random.uniform(-0.2,0.2)),15)
        payload={
            "Item_Category":item_category,
            "Store_Type":store_type,
            "Outlet_Location_Type":outlet_location,
            "Outlet_Size":outlet_size,
            "Item_Weight":float(item_weight),
            "Item_MRP":float(item_mrp),
            "Outlet_Sales":float(outlet_sales_day)
        }
        revenue=predict(payload)

        results.append({
            "Date":day,
            "Day":day.strftime("%A"),
            "Outlet_Sales":round(outlet_sales_day,2),
            "Predicted_Revenue":round(revenue,2)
        })

    progress_text.success("✅ Prediction completed")

    df=pd.DataFrame(results)
    df["Cumulative_Revenue"]=df["Predicted_Revenue"].cumsum()

    st.divider()

    # KPI CARDS
    col1,col2,col3=st.columns(3)
    col1.markdown(f"<div class='kpi'>💰 Total Revenue<br><h2>${df['Predicted_Revenue'].sum():,.2f}</h2></div>",unsafe_allow_html=True)
    col2.markdown(f"<div class='kpi'>📊 Average Revenue<br><h2>${df['Predicted_Revenue'].mean():,.2f}</h2></div>",unsafe_allow_html=True)
    peak_day = df.loc[df["Predicted_Revenue"].idxmax(),'Date'].strftime("%d %b")
    col3.markdown(f"<div class='kpi'>🔥 Top Day<br><h2>{peak_day}</h2></div>",unsafe_allow_html=True)

    # GRAPHS

    c1,c2 = st.columns(2)
    c1.plotly_chart(px.line(df,x="Date",y="Predicted_Revenue",markers=True,color_discrete_sequence=px.colors.qualitative.Vivid,title="📈 Daily Revenue Trend"),use_container_width=True)
    c2.plotly_chart(px.scatter(df,x="Outlet_Sales",y="Predicted_Revenue",color="Day",size="Predicted_Revenue",color_discrete_sequence=px.colors.qualitative.Set2,title="🛒 Sales vs Revenue"),use_container_width=True)
    st.plotly_chart(px.area(df,x="Date",y="Cumulative_Revenue",color_discrete_sequence=px.colors.qualitative.Pastel1,title="💹 Cumulative Revenue Growth"),use_container_width=True)

    # Weekday revenue bar graph with unique colors
    df['Weekday'] = df['Date'].dt.day_name()
    weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    weekday_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    st.plotly_chart(
        px.bar(
            df,
            x="Weekday",
            y="Predicted_Revenue",
            color="Weekday",
            category_orders={"Weekday": weekday_order},
            color_discrete_sequence=weekday_colors,
            title="📊 Revenue by Weekday"
        ),
        use_container_width=True
    )

    # DATA TABLE & DOWNLOAD
    
    st.subheader("📋 Daily Predictions")
    st.dataframe(df)
    csv=df.to_csv(index=False).encode()
    st.download_button("📥 Download CSV",csv,f"Revenue_{month}_{year}.csv")

# BATCH PREDICTION
st.subheader("📂 Batch CSV Prediction")
file = st.file_uploader("Upload CSV with required columns", type="csv")
if file:
    batch=pd.read_csv(file)
    st.dataframe(batch)
    if st.button("🚀 Predict Batch Revenue"):
        results=[]
        for _,row in batch.iterrows():
            payload={
                "Item_Category":row["Item_Category"],
                "Store_Type":row["Store_Type"],
                "Outlet_Location_Type":row["Outlet_Location_Type"],
                "Outlet_Size":row["Outlet_Size"],
                "Item_Weight":float(row["Item_Weight"]),
                "Item_MRP":float(row["Item_MRP"]),
                "Outlet_Sales":float(row["Outlet_Sales"])
            }
            row["Predicted_Revenue"]=predict(payload)
            results.append(row)
        output=pd.DataFrame(results)
        st.success("✅ Batch predictions ready")
        st.dataframe(output)
        out_csv=output.to_csv(index=False).encode()
        st.download_button("📥 Download Batch Results", out_csv,"batch_predictions.csv",mime="text/csv")
