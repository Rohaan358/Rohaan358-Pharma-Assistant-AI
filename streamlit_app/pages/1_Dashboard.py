import streamlit as st
import pandas as pd
import plotly.express as px
from utils.api_client import get_data_summary, get_products

st.set_page_config(page_title="Dashboard - PharmaIQ", page_icon="📊", layout="wide")

# Sidebar Branding
st.sidebar.title("PharmaIQ")
st.sidebar.caption("Sales Intelligence Platform")

st.markdown("# 📊 Dashboard / Overview")

# Initialize session state for dashboard data
if "dashboard_data" not in st.session_state:
    st.session_state["dashboard_data"] = {"summary": None, "products": None}

# 🔗 MongoDB Connection Status
st.markdown("## Connection Status")

# Fetch initial data
summary = get_data_summary()

if summary:
    st.success("✅ Database Connected")
    st.session_state["dashboard_data"]["summary"] = summary
    st.session_state["dashboard_data"]["products"] = get_products()
else:
    st.error("❌ Backend not reachable or Database not connected. Contact admin.")

st.markdown("---")

# 📊 Data Summary
st.markdown("## Data Overview")

# Render Summary
summary = st.session_state["dashboard_data"]["summary"]

if summary:
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", summary.get("total_records", 0))
    product_count = len(summary.get("products", []))
    col2.metric("Products", product_count)
    col3.metric("Categories", len(summary.get("categories", [])))
    
    date_range = summary.get("date_range", {})
    col4.metric("Date Range", f"{date_range.get('min', 'N/A')} - {date_range.get('max', 'N/A')}")
    
    # 📋 Product List Table
    st.markdown("## Product Database")
    
    products_data = st.session_state["dashboard_data"]["products"]
    if products_data:
        df_products = pd.DataFrame(products_data)
        
        # Search / Filter
        search_query = st.text_input("Search Products", "")
        
        if search_query:
            # Case insensitive search on name or category
            mask = (
                df_products["product_name"].astype(str).str.contains(search_query, case=False, na=False) |
                df_products["product_category"].astype(str).str.contains(search_query, case=False, na=False)
            )
            df_filtered = df_products[mask]
        else:
            df_filtered = df_products
        
        st.dataframe(
            df_filtered,
            use_container_width=True,
            column_config={
                "product_name": "Product",
                "product_category": "Category",
                "record_count": st.column_config.NumberColumn("Records"),
                "date_min": "Start Date",
                "date_max": "End Date",
            },
            hide_index=True,
        )
        
        # Category Chart using full product data
        if not df_filtered.empty and "product_category" in df_filtered.columns:
            st.subheader("Category Distribution")
            cat_counts = df_filtered["product_category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Product Count"]
            
            fig = px.bar(
                cat_counts, 
                x="Category", 
                y="Product Count", 
                # title="Products per Category",
                color="Category",
                text="Product Count"
            )
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.info("No products found in database.")

else:
    st.warning("Could not fetch data summary. Please connect to MongoDB first.")
