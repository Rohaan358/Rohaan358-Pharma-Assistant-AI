import streamlit as st
import pandas as pd
from utils.api_client import upload_file

st.set_page_config(page_title="Upload Data - PharmaIQ", page_icon="⬆️", layout="wide")

# Sidebar Branding
st.sidebar.title("PharmaIQ")
st.sidebar.caption("Sales Intelligence Platform")

st.markdown("# ⬆️ Upload Sales Data")

if "last_upload_res" not in st.session_state:
    st.session_state["last_upload_res"] = None

st.info("Upload CSV or Excel files to populate the database.")
st.markdown("Ensure your file follows the format: `date, product_name, product_category, units_sold`")

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file", 
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=False, 
    help="Max file size 200MB"
)

if uploaded_file is not None:
    # Preview
    st.markdown("### Preview")
    try:
        # We read it once for preview
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.dataframe(df.head(20), use_container_width=True)
        
        # Upload Button
        if st.button("Upload to Database", type="primary"):
            # Reset pointer for the API call
            uploaded_file.seek(0)
            
            success, msg = upload_file(uploaded_file)
            
            if success:
                st.session_state["last_upload_res"] = msg
                st.success(f"Upload Successful!")
                # Clear cached metadata since data changed
                st.cache_data.clear()
            else:
                st.error(msg)
                
    except Exception as e:
        st.error(f"Error previewing file: {str(e)}")

if st.session_state["last_upload_res"]:
    with st.expander("Last Upload Details"):
        st.json(st.session_state["last_upload_res"])
