import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.api_client import get_forecast_plot, get_products

st.set_page_config(page_title="Plots - PharmaIQ", page_icon="📊", layout="wide")

# Sidebar Branding
st.sidebar.title("PharmaIQ")
st.sidebar.caption("Sales Intelligence Platform")

st.markdown("# 📊 Forecast Plots")

# Initialize session state for plot persistence
if "last_plot" not in st.session_state:
    st.session_state["last_plot"] = None

# Product Selection
# get_products is now cached in api_client
products_data = get_products()

if not products_data:
    st.warning("Connect Database & Upload Data to view products.")
    product_options = []
else:
    df_prods = pd.DataFrame(products_data)
    product_options = sorted(df_prods["product_name"].unique().tolist())

if product_options:
    selected_product = st.selectbox("Select Product", product_options, index=0)

    if st.button("Generate Plot", type="primary"):
        with st.spinner("Fetching plot data..."):
            plot_data = get_forecast_plot(selected_product)
            if plot_data:
                st.session_state["last_plot"] = plot_data
            else:
                st.error("Failed to fetch plot data.")

# Render Plot if exists in session state
if st.session_state["last_plot"]:
    plot_data = st.session_state["last_plot"]
    
    # Prepare Data
    months = plot_data.get("months", [])
    actual = plot_data.get("actual", [])
    predicted = plot_data.get("predicted", [])
    metrics = plot_data.get("metrics", {})
    model_used = plot_data.get("model_used", "Unknown")
    product_name = plot_data.get("product", "Unknown Product")
    
    # Create Plotly Figure
    fig = go.Figure()
    
    # Actual Sales - Solid Line with Markers
    # Handle None in actuals for gaps
    fig.add_trace(go.Scatter(
        x=months, 
        y=actual, 
        name="Actual Sales", 
        mode='lines+markers',
        line=dict(color='#636EFA', width=3),
        marker=dict(size=8)
    ))
    
    # Predicted Sales - Dashed Line with Markers + Shaded Error Area
    fig.add_trace(go.Scatter(
        x=months, 
        y=predicted, 
        name="Forecast", 
        mode='lines+markers', 
        line=dict(color='#EF553B', width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond'),
        fill='tonexty', # Fill area between this trace and the previous one (Actual)
        fillcolor='rgba(239, 85, 59, 0.08)' # Light red with low opacity
    ))
    
    fig.update_layout(
        title=f"Sales Forecast for {product_name} (Model: {model_used})",
        xaxis_title="Month",
        yaxis_title="Units Sold",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{metrics.get('MAE', 0):.2f}")
    col2.metric("RMSE", f"{metrics.get('RMSE', 0):.2f}")
    col3.metric("MAPE", metrics.get('MAPE', "N/A"))
    col4.metric("Model", model_used)
    
    st.caption("ℹ️ To download the chart as PNG, click the camera icon in the chart toolbar.")

elif product_options:
    st.info("Select a product and click 'Generate Plot' to visualize forecasts.")
