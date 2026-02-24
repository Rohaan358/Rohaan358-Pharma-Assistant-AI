import streamlit as st

st.set_page_config(
    page_title="PharmaIQ",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("PharmaIQ")
st.sidebar.caption("Sales Intelligence Platform")

st.title("💊 PharmaIQ - Sales Forecasting Agent")
st.markdown("---")

st.markdown("""
### Welcome to PharmaIQ

This application provides an intelligent interface for pharmaceutical sales forecasting, powered by advanced machine learning models (Prophet, XGBoost, SARIMAX, Hybrid) and AI agents (Llama 3.3 70B).

**Use the sidebar to navigate:**

- **Dashboard**: Overview of current database status and connected products.
- **Upload Data**: Upload new sales data (CSV/Excel).
- **Forecasting**: Run ML models to predict future sales performance.
- **AI Agent**: Chat with the AI agent to analyze trends and get business insights.
- **Plots**: Visualize forecast results and download charts.

### Getting Started

1.  **Automatic Connection**: The backend automatically connects to MongoDB on startup using your secure `.env` configuration. Check the **Dashboard** to verify status.
2.  **Upload Data**: If the database is empty, use the **Upload Data** page to load your pharma sales records.
3.  **Run Forecasts**: Select products and models on the **Forecasting** page to generate 2025 predictions.
4.  **Analyze & Chat**: Use the **AI Agent** for natural language analysis or the **Plots** page for visual trends.

---
*Backend API: http://localhost:8000*
""")

# Sidebar info
with st.sidebar:
    st.info("💡 **Tip:** Ensure the backend server is running on localhost:8000 before using this app.")
