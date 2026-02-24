import streamlit as st
import uuid
from datetime import datetime
from utils.api_client import (
    agent_query, 
    agent_insights, 
    agent_analyze, 
    get_history, 
    get_all_sessions, 
    delete_session,
    get_products
)

st.set_page_config(page_title="AI Agent - PharmaIQ", page_icon="🤖", layout="wide")

# Sidebar Branding
st.sidebar.title("PharmaIQ")
st.sidebar.caption("Sales Intelligence Platform")

st.markdown("# 🤖 AI Agent")

# ── Session Management ──────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Fetch history if session_id changes or on first load
def load_history():
    history = get_history(st.session_state["session_id"])
    if history:
        # Convert model field names if necessary (API returns {role, message, timestamp, context})
        st.session_state["chat_history"] = history
    else:
        st.session_state["chat_history"] = []

# Initial load
if not st.session_state["chat_history"]:
    load_history()

# ── Sidebar Actions ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("---")
    st.header("💬 Session Control")
    st.info(f"Current Session: `{st.session_state['session_id'][:8]}`")
    
    if st.button("➕ New Session", use_container_width=True):
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["chat_history"] = []
        st.rerun()

    # Load Previous Sessions
    st.markdown("### Load History")
    all_sessions = get_all_sessions()
    if all_sessions:
        session_options = {}
        for s in all_sessions:
            # Format label: context first, then timestamp
            ctx = s.get("context") or {}
            prod = ctx.get("product")
            cat = ctx.get("category")
            
            context_str = f"{prod} · {cat}" if prod else "General Chat"
            
            try:
                dt = datetime.fromisoformat(s["start_time"])
                time_str = dt.strftime("%d %b %Y %H:%M")
            except:
                time_str = s["start_time"][:16]

            label = f"{context_str} — {time_str}"
            session_options[label] = s['session_id']
            
        selected_label = st.selectbox("Select past session", options=list(session_options.keys()))
        if st.button("Load Session"):
            st.session_state["session_id"] = session_options[selected_label]
            load_history()
            st.rerun()
    else:
        st.caption("No past sessions found.")

    if st.button("🗑️ Delete This Session", type="secondary", use_container_width=True):
        if delete_session(st.session_state["session_id"]):
            st.success("Session deleted.")
            st.session_state["session_id"] = str(uuid.uuid4())
            st.session_state["chat_history"] = []
            st.rerun()

    st.markdown("---")
    st.header("⚙️ Optional Filters")
    st.caption("Override auto-detection (optional)")
    include_context = st.checkbox("Include Forecast Context", value=True)
    
    # Context selectors (Product/Category)
    products_data = get_products()
    product_options = ["None"]
    category_options = ["None"]
    if products_data:
        import pandas as pd
        df_prods = pd.DataFrame(products_data)
        product_options += sorted(df_prods["product_name"].unique().tolist())
        category_options += sorted(df_prods["product_category"].astype(str).unique().tolist())

    selected_product = st.selectbox("Product Filter", product_options, index=0)
    selected_category = st.selectbox("Category Filter", category_options, index=0)

# Tabs for different AI modes
tab1, tab2, tab3 = st.tabs(["💬 Chat", "🧠 Insights", "🔍 Deep Analysis"])

# --- Chat Tab ---
with tab1:
    st.markdown("## Chat with PharmaCast AI")
    
    # Display chat messages from session state
    for msg in st.session_state["chat_history"]:
        role = msg["role"]
        content = msg["message"]
        timestamp = msg.get("timestamp", "")
        context = msg.get("context")

        with st.chat_message(role):
            st.markdown(content)
            
            # Timestamp & Context Tags
            footer_cols = st.columns([1, 1])
            with footer_cols[0]:
                if timestamp:
                    dt = datetime.fromisoformat(timestamp)
                    st.caption(f"🕒 {dt.strftime('%Y-%m-%d %H:%M')}")
            
            with footer_cols[1]:
                if context:
                    tags = []
                    if context.get("product"): tags.append(f"`{context['product']}`")
                    if context.get("category"): tags.append(f"`{context['category']}`")
                    if tags:
                        st.markdown(f"{' '.join(tags)}", help="Context used for this prompt")

    # User Input
    if prompt := st.chat_input("Ask about sales trends, forecasts, or anomalies..."):
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Call AI
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                prod = selected_product if selected_product != "None" else None
                cat = selected_category if selected_category != "None" else None
                
                response_data = agent_query(
                    prompt, 
                    session_id=st.session_state["session_id"],
                    include_context=include_context,
                    product=prod,
                    category=cat
                )
                
                response_text = response_data.get("response", "Error processing request.")
                st.markdown(response_text)
                
                # Get final context used by the agent
                final_p = response_data.get("detected_product")
                final_c = response_data.get("detected_category")
                chat_ctx = {"product": final_p, "category": final_c}
                ts = datetime.now().isoformat()
                
                # Update local history
                st.session_state["chat_history"].append({
                    "role": "user", "message": prompt, "timestamp": ts, "context": chat_ctx
                })
                st.session_state["chat_history"].append({
                    "role": "assistant", "message": response_text, "timestamp": datetime.now().isoformat(), "context": chat_ctx
                })
                
                st.rerun()

# --- Insights Tab ---
with tab2:
    if "agent_insights" not in st.session_state:
        st.session_state["agent_insights"] = None

    st.markdown("## Automated Market Insights")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Generate Insights", type="primary"):
            res = agent_insights()
            if res:
                st.session_state["agent_insights"] = res

    if st.session_state["agent_insights"]:
        insights_data = st.session_state["agent_insights"]
        st.markdown(insights_data.get("insights", ""))
    else:
        st.info("Click 'Generate Insights' to get an overview of market performance.")

# --- Deep Analysis Tab ---
with tab3:
    if "agent_analysis" not in st.session_state:
        st.session_state["agent_analysis"] = None

    st.markdown("## Deep Product Analysis")
    
    col_a, col_b, col_c = st.columns([3, 1, 1])
    with col_a:
        target_product = st.text_input("Product Name or Category to Analyze", placeholder="e.g. OMEPRAZOLE")
    with col_b:
        analysis_type = st.selectbox("Analysis Type", ["full", "trend", "anomaly", "comparison"])
    with col_c:
        st.write("") # Spacer
        st.write("")
        if st.button("Analyze", type="primary"):
            if not target_product:
                st.error("Please enter a subject.")
            else:
                res = agent_analyze(target_product, analysis_type=analysis_type)
                if res:
                    st.session_state["agent_analysis"] = res

    if st.session_state["agent_analysis"]:
        analysis_result = st.session_state["agent_analysis"]
        st.markdown(f"### Analysis for: {analysis_result.get('subject', 'Unknown')}")
        st.markdown(analysis_result.get("insights", ""))
        
        recs = analysis_result.get("recommendations", [])
        if recs:
            st.markdown("#### Key Recommendations")
            for rec in recs:
                st.info(f"💡 {rec}")
