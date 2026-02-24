import requests
import streamlit as st
import sys
import os

# Add parent directory to path to allow importing streamlit_config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from streamlit_config import FASTAPI_BASE_URL

# Connection pool
session = requests.Session()

def get_base_url():
    return FASTAPI_BASE_URL

@st.cache_data(ttl=300)
def get_data_summary():
    """Call GET /data/summary"""
    url = f"{get_base_url()}/data/summary"
    try:
        # Don't use spinner inside cached function to avoid UI glitches
        response = session.get(url, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        return None

@st.cache_data(ttl=300)
def get_products():
    """Call GET /data/products"""
    url = f"{get_base_url()}/data/products"
    try:
        response = session.get(url, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception as e:
        return []

def upload_file(files):
    """Call POST /data/upload"""
    url = f"{get_base_url()}/data/upload"
    try:
        if not files:
            return False, "No file selected."
        
        file_payload = {"file": (files.name, files, files.type)}
        
        with st.spinner("Uploading file..."):
            response = session.post(url, files=file_payload, timeout=300)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, f"Upload failed: {response.text}"
    except Exception as e:
        return False, f"Upload error: {str(e)}"

def run_forecast(product, category, model, year=2025):
    """Call POST /forecast/run"""
    url = f"{get_base_url()}/forecast/run"
    payload = {
        "product": product,
        "category": category,
        "model": model,
        "year": year
    }
    try:
        with st.spinner(f"Running forecast for {product}..."):
            response = session.post(url, json=payload, timeout=300)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Forecast failed: {response.text}")
                return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def get_forecast_results(product=None):
    """Call GET /forecast/results"""
    url = f"{get_base_url()}/forecast/results"
    params = {}
    if product:
        params["product"] = product
    try:
        with st.spinner("Fetching results..."):
            response = session.get(url, params=params, timeout=60)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error: {response.text}")
                return []
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return []

def get_forecast_compare(product=None):
    """Call GET /forecast/compare"""
    url = f"{get_base_url()}/forecast/compare"
    params = {}
    if product:
        params["product"] = product
    try:
        response = session.get(url, params=params, timeout=60)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return []
        
def get_forecast_plot(product):
    """Call GET /forecast/plot"""
    url = f"{get_base_url()}/forecast/plot"
    params = {"product": product}
    try:
        with st.spinner("Generating plot data..."):
            response = session.get(url, params=params, timeout=60)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error fetching plot: {response.text}")
                return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def agent_query(query, session_id=None, include_context=False, product=None, category=None):
    """Call POST /agent/query"""
    url = f"{get_base_url()}/agent/query"
    payload = {
        "query": query, 
        "session_id": session_id,
        "include_forecast_context": include_context,
        "product": product,
        "category": category
    }
    try:
        response = session.post(url, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json()
        else:
            return {"response": f"Error: {response.text}"}
    except Exception as e:
        return {"response": f"API Error: {str(e)}"}

def get_history(session_id: str):
    """Call GET /agent/history/{session_id}"""
    url = f"{get_base_url()}/agent/history/{session_id}"
    try:
        response = session.get(url, timeout=60)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception:
        return []

def get_all_sessions():
    """Call GET /agent/history/all"""
    url = f"{get_base_url()}/agent/history/all"
    try:
        response = session.get(url, timeout=60)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception:
        return []

def delete_session(session_id: str):
    """Call DELETE /agent/history/{session_id}"""
    url = f"{get_base_url()}/agent/history/{session_id}"
    try:
        response = session.delete(url, timeout=60)
        return response.status_code == 200
    except Exception:
        return False

def agent_insights():
    """Call GET /agent/insights"""
    url = f"{get_base_url()}/agent/insights"
    try:
        with st.spinner("Generating insights..."):
            response = session.get(url, timeout=120)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error: {response.text}")
                return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def agent_analyze(product, analysis_type="full"):
    """Call POST /agent/analyze"""
    url = f"{get_base_url()}/agent/analyze"
    payload = {"product": product, "analysis_type": analysis_type}
    try:
        with st.spinner(f"Analyzing {product}..."):
            response = session.post(url, json=payload, timeout=120)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error: {response.text}")
                return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def get_comparison(product: str, years: list):
    """Call GET /data/compare"""
    year_str = ",".join(map(str, years))
    url = f"{get_base_url()}/data/compare"
    params = {"product": product, "years": year_str}
    try:
        with st.spinner("Fetching comparison data..."):
            response = session.get(url, params=params, timeout=60)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error fetching comparison: {response.text}")
                return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None
