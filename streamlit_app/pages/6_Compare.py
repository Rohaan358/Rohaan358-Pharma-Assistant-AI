import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.api_client import get_comparison, get_products

st.set_page_config(page_title="Compare - PharmaIQ", page_icon="📊", layout="wide")

# Sidebar Branding
st.sidebar.title("PharmaIQ")
st.sidebar.caption("Sales Intelligence Platform")

st.markdown("# 📊 Product Year-on-Year Comparison")

# Initialize session state
if "last_comparison" not in st.session_state:
    st.session_state["last_comparison"] = None

# Product Selection
products_data = get_products()
if not products_data:
    st.warning("Connect Database & Upload Data to view products.")
    product_options = []
else:
    df_prods = pd.DataFrame(products_data)
    product_options = sorted(df_prods["product_name"].unique().tolist())

# Filters Row
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    selected_product = st.selectbox("Select Product", product_options, index=0 if product_options else None)

with col2:
    selected_years = st.multiselect("Select Years", [2023, 2024, 2025], default=[2023, 2024, 2025])

with col3:
    st.write("") # Spacer
    st.write("")
    if st.button("Generate Table", type="primary"):
        if not selected_product:
            st.error("Please select a product.")
        elif not selected_years:
            st.error("Please select at least one year.")
        else:
            res = get_comparison(selected_product, selected_years)
            if res:
                st.session_state["last_comparison"] = res
            else:
                st.error("Failed to fetch comparison data.")

    if st.button("Reset"):
        st.session_state["last_comparison"] = None
        st.rerun()

st.markdown("---")

# Render Tables and Charts
if st.session_state["last_comparison"]:
    comp_data = st.session_state["last_comparison"]
    product_name = comp_data.get("product")
    data = comp_data.get("data", {}) # structured as {year: {month: val}}

    # Convert to DataFrame for Table 1
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    rows = []
    for year in sorted(data.keys()):
        row = {"Year": year}
        year_total = 0
        for m in months:
            val = data[year].get(m, 0)
            row[m] = val
            year_total += val
        row["Total"] = year_total
        rows.append(row)

    df_monthly = pd.DataFrame(rows)

    # Calculate YoY % (Compare last two years in the selection if at least 2 years)
    sorted_years = sorted(data.keys())
    if len(sorted_years) >= 2:
        prev_year = sorted_years[-2]
        curr_year = sorted_years[-1]
        
        yoy_row = {"Year": f"YoY % ({prev_year}v{curr_year})"}
        prev_vals = data[prev_year]
        curr_vals = data[curr_year]
        
        total_prev = 0
        total_curr = 0
        
        for m in months:
            p = prev_vals.get(m, 0)
            c = curr_vals.get(m, 0)
            total_prev += p
            total_curr += c
            
            if p > 0:
                yoy = ((c - p) / p) * 100
                yoy_row[m] = f"{yoy:+.1f}%"
            else:
                yoy_row[m] = "—"
        
        if total_prev > 0:
            total_yoy = ((total_curr - total_prev) / total_prev) * 100
            yoy_row["Total"] = f"{total_yoy:+.1f}%"
        else:
            yoy_row["Total"] = "—"
        
        df_monthly = pd.concat([df_monthly, pd.DataFrame([yoy_row])], ignore_index=True)

    st.subheader("Monthly Comparison")
    
    # Custom display for color coding YoY
    def color_yoy(val):
        if isinstance(val, str) and "%" in val:
            if val.startswith("+"):
                return "color: green"
            elif val.startswith("-"):
                return "color: red"
        return ""

    st.dataframe(
        df_monthly.style.applymap(color_yoy),
        use_container_width=True,
        hide_index=True
    )

    # TABLE 2 — Annual Summary
    st.markdown("### Annual Summary")
    summary_rows = []
    
    prev_total = None
    for year in sorted(data.keys()):
        year_data = data[year]
        total_units = sum(year_data.values())
        avg_units = total_units / 12
        
        # Peak Month
        peak_month = "N/A"
        peak_val = -1
        for m, v in year_data.items():
            if v > peak_val:
                peak_val = v
                peak_month = f"{m}-{year}"
        
        # YoY Change
        yoy_change = "—"
        if prev_total is not None and prev_total > 0:
            change = ((total_units - prev_total) / prev_total) * 100
            yoy_change = f"{change:+.1f}%"
        
        summary_rows.append({
            "Year": year,
            "Total Units": total_units,
            "Avg Monthly": round(avg_units, 1),
            "Peak Month": peak_month,
            "Peak Units": peak_val,
            "YoY Change": yoy_change
        })
        prev_total = total_units

    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(
        df_summary.style.applymap(color_yoy, subset=["YoY Change"]),
        use_container_width=True,
        hide_index=True
    )

    # CHART
    st.markdown("### Visualization")
    fig = go.Figure()
    
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    for i, year in enumerate(sorted(data.keys())):
        y_vals = [data[year].get(m, 0) for m in months]
        fig.add_trace(go.Scatter(
            x=months,
            y=y_vals,
            name=str(year),
            mode='lines+markers',
            line=dict(width=3, color=colors[i % len(colors)]),
            marker=dict(size=8)
        ))

    fig.update_layout(
        title=f"{product_name} — Year on Year Comparison",
        xaxis_title="Month",
        yaxis_title="Units Sold",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Select a product and years, then click 'Generate Table' to compare performance.")
