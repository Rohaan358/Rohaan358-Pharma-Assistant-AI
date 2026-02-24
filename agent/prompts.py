from typing import Optional, Dict, Any, List


CATEGORY_KEYWORDS = {
    "CEFIXIME": ["cefixime", "caricef", "cefspan", "cebosh"],
    "OMEPRAZOLE": ["omeprazole", "omez", "risek"],
    "DICLOFENAC SODIUM": ["diclofenac", "voltaren", "dicloran"],
    "ESCITALOPRAM": ["escitalopram", "lexapro", "esciram"],
    "EMPAGLIFLOZIN": ["empagliflozin", "jardiance", "empa"],
    "DAPAGLIFLOZIN": ["dapagliflozin", "forxiga", "dapa"],
    "SITAGLIPTIN": ["sitagliptin", "januvia", "sita"],
}


def detect_context(query: str, products: List[str]) -> tuple:
    """Detect product or category from query text."""
    query_lower = query.lower()
    detected_product = None
    detected_category = None

    # Detect Product
    for p in products:
        if p.lower() in query_lower:
            detected_product = p
            break
            
    # Detect Category
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw.lower() in query_lower for kw in keywords):
            detected_category = cat
            break
            
    return detected_product, detected_category


SYSTEM_BASE = """You are PharmaCast AI — an expert pharmaceutical sales analyst and forecasting agent powered by Meta Llama 3.3 70B.

Your capabilities:
- Analyze pharmaceutical sales trends and patterns
- Interpret ML forecasting results (Prophet, XGBoost, SARIMAX, Hybrid)
- Detect anomalies and unusual sales behavior
- Compare product performance across categories
- Provide actionable business recommendations
- Explain forecast accuracy metrics (MAE, RMSE, MAPE) in plain language

Guidelines:
- Be precise, data-driven, and concise
- Always cite specific numbers from the context when available
- Flag data quality issues or insufficient data clearly
- Distinguish between correlation and causation
- Use pharmaceutical domain knowledge (seasonality, disease cycles, prescription patterns)
"""


def build_system_prompt(
    data_summary: Optional[Dict] = None,
    forecast_results: Optional[List[Dict]] = None,
    product: Optional[str] = None,
    category: Optional[str] = None,
) -> str:
    """Build a dynamic system prompt with injected context."""
    prompt = SYSTEM_BASE

    if data_summary:
        prompt += f"""
## Current Database Context
- Total Records: {data_summary.get('total_records', 'N/A')}
- Date Range: {data_summary.get('date_range', {}).get('min', 'N/A')} to {data_summary.get('date_range', {}).get('max', 'N/A')}
- Product Categories: {', '.join(data_summary.get('categories', []))}
- Products in DB: {', '.join(data_summary.get('products', [])[:20])}
"""

    if forecast_results:
        prompt += "\n## Latest Forecast Results\n"
        for r in forecast_results[:5]:  # limit context size
            metrics = r.get("metrics", {})
            prompt += (
                f"- **{r.get('product')}** ({r.get('category')}) "
                f"| Model: {r.get('model_used')} "
                f"| MAE: {metrics.get('MAE', 'N/A')} "
                f"| RMSE: {metrics.get('RMSE', 'N/A')} "
                f"| MAPE: {metrics.get('MAPE', 'N/A')}\n"
            )

    if product:
        prompt += f"\n## Current Focus\nProduct: {product}"
        if category:
            prompt += f" | Category: {category}"
        prompt += "\n"

    return prompt


def build_insights_prompt(data_summary: Dict, forecast_results: List[Dict]) -> str:
    """Build prompt for auto-generating insights."""
    system = build_system_prompt(data_summary, forecast_results)
    user_msg = """Analyze the current pharmaceutical sales data and forecast results. Provide:
1. Top 3 key observations about sales trends
2. Best and worst performing products by forecast accuracy
3. Category-level insights (which categories show strongest/weakest demand)
4. 2–3 actionable recommendations for the sales team
5. Any anomalies or data quality concerns

Format your response with clear sections and bullet points."""
    return system, user_msg


def build_analysis_prompt(
    subject: str,
    analysis_type: str,
    context_data: Dict,
) -> tuple:
    """Build prompt for deep product/category analysis."""
    system = build_system_prompt(
        data_summary=context_data.get("summary"),
        forecast_results=context_data.get("forecasts"),
        product=context_data.get("product"),
        category=context_data.get("category"),
    )

    type_instructions = {
        "trend": "Focus on long-term sales trends, growth rates, and seasonal patterns.",
        "anomaly": "Identify unusual spikes, drops, or irregular patterns. Explain potential causes.",
        "comparison": "Compare performance across time periods.",
        "full": "Provide a comprehensive analysis covering trends, anomalies, forecast accuracy, and recommendations.",
    }

    instruction = type_instructions.get(analysis_type, type_instructions["full"])
    user_msg = f"""Perform a {analysis_type} analysis for: **{subject}**

{instruction}

Include:
- Key metrics and what they indicate
- Forecast model performance assessment
- Business impact assessment
- Specific recommendations

Use the provided context data to support your analysis."""

    return system, user_msg
