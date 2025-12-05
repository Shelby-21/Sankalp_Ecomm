import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import datetime

# --- Configuration ---
st.set_page_config(page_title="Hustler Inventory & Sales Dashboard", layout="wide")

# --- Title and Intro ---
st.title("📊 The Hustler: Sales Forecasting & Inventory Optimization Dashboard")
st.markdown("""
This dashboard automates the sales analysis and inventory planning process. 
It uses **Machine Learning (Random Forest)** to forecast future demand and **Inventory Theory** to optimize reorder points, replacing manual Excel Solver tasks.
""")

# --- 1. Data Loading ---
@st.cache_data
def load_data():
    # Attempt to load the file directly if it exists, otherwise ask for upload
    try:
        # Tring to read the specific filename provided in your prompt context
        # In a real deployment, you might rename this file to 'daily_aggregated.csv'
        df = pd.read_excel("C:/Users/sinha/Downloads/E_Comm_File.xlsx", sheet_name='daily_aggregated')
    except FileNotFoundError:
        st.warning("Default file not found. Please upload 'daily_aggregated.csv' below.")
        uploaded_file = st.file_uploader("Upload Daily Aggregated excel", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
        else:
            return None
    
    # Preprocessing
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

if df is not None:
    # --- Sidebar: Filters ---
    st.sidebar.header("Filter Data")
    
    # City Filter
    all_cities = ['All'] + list(df['city'].unique())
    selected_city = st.sidebar.selectbox("Select City", all_cities)
    
    # SKU Filter
    all_products = list(df['product_name'].unique())
    selected_product = st.sidebar.selectbox("Select Product", all_products)

    # Filter Logic
    df_filtered = df[df['product_name'] == selected_product].copy()
    if selected_city != 'All':
        df_filtered = df_filtered[df_filtered['city'] == selected_city]
    
    # Aggregate data (if City is 'All', sum across cities)
    df_grouped = df_filtered.groupby('date')['sales_qty'].sum().reset_index()
    df_grouped = df_grouped.sort_values('date')

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📈 Historical Analysis", "🤖 ML Forecasting", "📦 Inventory Optimization"])

    # --- TAB 1: Historical Analysis ---
    with tab1:
        st.subheader(f"Historical Sales: {selected_product}")
        
        # Key Metrics
        total_sales = df_grouped['sales_qty'].sum()
        avg_daily_sales = df_grouped['sales_qty'].mean()
        peak_sales = df_grouped['sales_qty'].max()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Units Sold", f"{total_sales:,.0f}")
        col2.metric("Avg Daily Sales", f"{avg_daily_sales:.2f}")
        col3.metric("Peak Single Day Sales", f"{peak_sales:,.0f}")

        # Plot
        fig = px.line(df_grouped, x='date', y='sales_qty', title='Daily Sales Trend')
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonality (Day of Week)
        df_grouped['day_of_week'] = df_grouped['date'].dt.day_name()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_sales = df_grouped.groupby('day_of_week')['sales_qty'].mean().reindex(dow_order)
        
        fig_dow = px.bar(dow_sales, x=dow_sales.index, y='sales_qty', title="Average Sales by Day of Week")
        st.plotly_chart(fig_dow, use_container_width=True)

    # --- TAB 2: ML Forecasting ---
    with tab2:
        st.subheader("Machine Learning Forecast (Random Forest)")
        
        # Feature Engineering
        df_ml = df_grouped.copy()
        df_ml['day_of_year'] = df_ml['date'].dt.dayofyear
        df_ml['day_of_week'] = df_ml['date'].dt.dayofweek
        df_ml['month'] = df_ml['date'].dt.month
        df_ml['year'] = df_ml['date'].dt.year
        df_ml['days_since_start'] = (df_ml['date'] - df_ml['date'].min()).dt.days

        # Train/Test Split
        X = df_ml[['day_of_year', 'day_of_week', 'month', 'year', 'days_since_start']]
        y = df_ml['sales_qty']
        
        # Using a time-based split (Train on past, Test on recent)
        split_point = int(len(df_ml) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        # Model Training
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluation
        preds_test = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds_test)
        r2 = r2_score(y_test, preds_test)
        
        col1, col2 = st.columns(2)
        col1.info(f"Model MAE (Mean Absolute Error): {mae:.2f} units")
        col2.info(f"Model R² Score: {r2:.2f}")

        # Future Forecasting
        forecast_days = st.slider("Select Forecast Horizon (Days)", 7, 90, 30)
        
        last_date = df_ml['date'].max()
        future_dates = [last_date + datetime.timedelta(days=x) for x in range(1, forecast_days + 1)]
        
        future_df = pd.DataFrame({'date': future_dates})
        future_df['day_of_year'] = future_df['date'].dt.dayofyear
        future_df['day_of_week'] = future_df['date'].dt.dayofweek
        future_df['month'] = future_df['date'].dt.month
        future_df['year'] = future_df['date'].dt.year
        future_df['days_since_start'] = (future_df['date'] - df_ml['date'].min()).dt.days
        
        future_preds = model.predict(future_df[['day_of_year', 'day_of_week', 'month', 'year', 'days_since_start']])
        
        # Visualization: History + Forecast
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=df_grouped['date'], y=df_grouped['sales_qty'], name='Historical Sales', line=dict(color='blue')))
        fig_forecast.add_trace(go.Scatter(x=future_dates, y=future_preds, name='Forecast', line=dict(color='orange', dash='dash')))
        fig_forecast.update_layout(title="Sales Forecast vs History", xaxis_title="Date", yaxis_title="Sales Quantity")
        st.plotly_chart(fig_forecast, use_container_width=True)

    # --- TAB 3: Inventory Optimization (Solver Logic) ---
    with tab3:
        st.subheader("Inventory Optimization & Stockout Reduction")
        st.markdown("This section calculates optimal inventory levels to minimize stockouts and holding costs, replacing the manual Solver method.")

        col_input1, col_input2, col_input3 = st.columns(3)
        lead_time = col_input1.number_input("Lead Time (Days)", min_value=1, value=7, help="Time to receive goods from supplier.")
        service_level = col_input2.selectbox("Target Service Level", [0.90, 0.95, 0.99], index=1, help="Probability of NOT having a stockout.")
        holding_cost = col_input3.number_input("Holding Cost per Unit ($)", min_value=0.1, value=2.0)

        # Calculations based on Demand during Lead Time
        # Demand during Lead Time follows a distribution with Mean = Avg_Daily * LT and StdDev = Std_Daily * sqrt(LT)
        
        forecast_mean = np.mean(future_preds) # Using the FORECASTED demand, not just historical
        forecast_std = np.std(future_preds) if np.std(future_preds) > 0 else np.std(df_grouped['sales_qty'])
        
        # Z-score for service level
        z_scores = {0.90: 1.28, 0.95: 1.645, 0.99: 2.33}
        z = z_scores[service_level]
        
        # Safety Stock Calculation
        safety_stock = z * forecast_std * np.sqrt(lead_time)
        
        # Reorder Point (ROP)
        reorder_point = (forecast_mean * lead_time) + safety_stock
        
        # Optimal Order Quantity (EOQ) logic (Simplified)
        # Assuming Setup Cost ($50 per order - arbitrary default for calculation demo)
        setup_cost = 50 
        annual_demand = forecast_mean * 365
        eoq = np.sqrt((2 * annual_demand * setup_cost) / holding_cost)

        # Display Results
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("🛡️ Safety Stock", f"{int(np.ceil(safety_stock))} units", help="Buffer stock to keep to prevent stockouts.")
        c2.metric("🔄 Reorder Point", f"{int(np.ceil(reorder_point))} units", help="Place a new order when inventory hits this level.")
        c3.metric("📦 Optimal Order Qty (EOQ)", f"{int(np.ceil(eoq))} units", help="Most cost-effective quantity to order.")

        st.success(f"""
        **Managerial Insight:** To maintain a **{service_level*100}%** service level for **{selected_product}** (in {selected_city}), 
        you should keep a buffer of **{int(np.ceil(safety_stock))}** units. 
        Place a new order of **{int(np.ceil(eoq))}** units whenever your stock drops to **{int(np.ceil(reorder_point))}**.
        """)
        
        # Stockout Risk Visualization
        st.write("### Projected Inventory Simulation")
        current_stock = st.number_input("Current Stock Level", min_value=0, value=int(reorder_point * 1.5))
        
        inventory_levels = []
        stock = current_stock
        dates = future_dates
        
        for pred in future_preds:
            stock -= pred
            if stock <= reorder_point:
                stock += eoq # Simulate reorder arriving immediately (simplified) or mark as reorder trigger
            inventory_levels.append(stock)
            
        fig_inv = px.line(x=dates, y=inventory_levels, title="Simulated Inventory Depletion & Reordering")
        fig_inv.add_hline(y=reorder_point, line_dash="dot", annotation_text="Reorder Point", line_color="red")
        fig_inv.add_hline(y=0, line_color="black", annotation_text="Stockout Line")
        st.plotly_chart(fig_inv, use_container_width=True)

else:
    st.info("Awaiting data upload...")