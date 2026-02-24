import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.api_client import run_forecast, get_forecast_results, get_forecast_compare, get_products

st.set_page_config(page_title="Forecasting - PharmaIQ", page_icon="📈", layout="wide")

# Sidebar Branding
st.sidebar.title("PharmaIQ")
st.sidebar.caption("Sales Intelligence Platform")

st.markdown("# 📈 Sales Forecasting")

# Initialize session state
if "last_forecast" not in st.session_state:
    st.session_state["last_forecast"] = None

# Sidebar Filters
st.sidebar.header("Forecast Settings")

# Product Dropdown (cached)
products_data = get_products()

if not products_data:
    st.sidebar.warning("Connect Database & Upload Data First")
    product_options = []
    category_options = []
else:
    df_prods = pd.DataFrame(products_data)
    product_options = sorted(df_prods["product_name"].unique().tolist())
    category_options = sorted(df_prods["product_category"].astype(str).unique().tolist()) if "product_category" in df_prods.columns else []

if product_options:
    selected_product = st.sidebar.selectbox("Select Product", product_options, index=0)
    
    # Auto-select category based on product if possible, otherwise offer dropdown
    default_cat_index = 0
    if selected_product and not df_prods.empty:
        prod_cats = df_prods[df_prods["product_name"] == selected_product]["product_category"].unique()
        if len(prod_cats) > 0 and prod_cats[0] in category_options:
             default_cat_index = category_options.index(prod_cats[0])

    selected_category = st.sidebar.selectbox("Category", category_options, index=default_cat_index)

    model_options = ["Auto", "prophet", "xgboost", "sarimax", "hybrid"]
    selected_model = st.sidebar.selectbox("Model", model_options, index=0)

    forecast_year = st.sidebar.number_input("Forecast Year", min_value=2024, max_value=2030, value=2025)

    if st.sidebar.button("Run Forecast", type="primary"):
        with st.spinner(f"Running {selected_model} forecast for {selected_product}..."):
            result = run_forecast(selected_product, selected_category, selected_model, forecast_year)
            
            if result and result.get("status") == "success":
                st.session_state["last_forecast"] = result.get("result", {})
            else:
                st.error("Forecast failed. Check logs.")

# --- Main Content: Render Last Forecast ---
if st.session_state["last_forecast"]:
    forecast_data = st.session_state["last_forecast"]
    
    st.subheader(f"Forecast Result: {forecast_data.get('product', 'Unknown')} ({forecast_year})")
    
    # --- Metrics ---
    metrics = forecast_data.get("metrics", {})
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{metrics.get('MAE', 0):.2f}")
        col2.metric("RMSE", f"{metrics.get('RMSE', 0):.2f}")
        
        mape_val = metrics.get('MAPE', "N/A")
        col3.metric("MAPE", mape_val)
        col4.metric("Model Used", forecast_data.get("model_used", "Unknown"))
        
        # Warning for high MAPE
        try:
            if isinstance(mape_val, (int, float)) and mape_val > 30:
                st.warning(f"⚠️ High Error Warning: MAPE > 30% ({mape_val}%)")
            elif isinstance(mape_val, str) and "%" in mape_val:
                val = float(mape_val.strip("%"))
                if val > 30:
                    st.warning(f"⚠️ High Error Warning: MAPE > 30% ({val}%)")
        except ValueError:
            pass
    
    # --- Plot ---
    months = forecast_data.get("months", [])
    actual = forecast_data.get("actual", [])
    predicted = forecast_data.get("predicted", [])
    
    # Interactive Plotly Chart (Dual Line as requested in Fix 1 for consistency, or bar/line here?)
    # "Change forecast chart from bar+line combo to dual-line chart" was for pages/5_Plots.py.
    # But for consistency, doing it here too is better.
    
    fig = go.Figure()
    
    # Actual Sales
    fig.add_trace(go.Scatter(
        x=months, y=actual, name="Actual Sales",
        mode='lines+markers', line=dict(color='#636EFA', width=3), marker=dict(size=8)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=months, y=predicted, name="Predicted Forecast",
        mode='lines+markers', line=dict(color='#EF553B', width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond'),
        fill='tonexty', fillcolor='rgba(239, 85, 59, 0.08)'
    ))
    
    fig.update_layout(
        title=f"Actual vs Predicted Sales", 
        xaxis_title="Month", 
        yaxis_title="Units Sold",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- JSON ---
    with st.expander("View Raw JSON Result"):
        st.json(forecast_data)

st.markdown("---")

# 📋 Results History
st.markdown("## Forecast History")
# Only fetch history for selected product if available
prod_for_history = selected_product if product_options else None
history_data = get_forecast_results(prod_for_history)

if history_data:
    df_history = pd.DataFrame(history_data)
    
    # Extract metrics from dict column
    if "metrics" in df_history.columns:
         # simple Extraction
         df_history["MAE"] = df_history["metrics"].apply(lambda x: x.get("MAE") if isinstance(x, dict) else None)
         df_history["MAPE"] = df_history["metrics"].apply(lambda x: x.get("MAPE") if isinstance(x, dict) else None)
         df_history["RMSE"] = df_history["metrics"].apply(lambda x: x.get("RMSE") if isinstance(x, dict) else None)
    
    cols_to_show = ["product", "category", "model_used", "MAE", "RMSE", "MAPE"]
    # Filter available cols
    cols = [c for c in cols_to_show if c in df_history.columns]
    
    st.dataframe(df_history[cols], use_container_width=True)
else:
    st.info("No forecast history available.")
