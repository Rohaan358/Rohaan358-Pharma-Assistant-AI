"""
forecasting/model_selector.py — Auto model selection based on product category
"""

from typing import List

# Mapping actual categories (case-insensitive) to logical types
CATEGORY_TYPE_MAP = {
    "cefixime":          "antibiotic",
    "omeprazole":        "gastro",
    "diclofenac sodium": "acute",
    "escitalopram":      "chronic",
    "empagliflozin":     "chronic",
    "dapagliflozin":     "chronic",
    "sitagliptin":       "chronic",
}

# Mapping logical types to models
TYPE_MODEL_MAP = {
    "antibiotic": "sarimax",
    "gastro":     "prophet",
    "acute":      "xgboost",
    "chronic":    "prophet",
    "other":      "xgboost",
}

# Expanded fallback lists
TYPE_FALLBACK_MAP = {
    "antibiotic": ["prophet", "xgboost"],
    "gastro":     ["hybrid", "xgboost"],
    "acute":      ["hybrid", "prophet"],
    "chronic":    ["sarimax", "xgboost"],
    "other":      ["prophet", "sarimax"],
}


def get_category_type(category: str) -> str:
    """Resolve actual category string to logical type (antibiotic, chronic, etc.)."""
    if not category:
        return "other"
    cat_clean = category.lower().strip()
    return CATEGORY_TYPE_MAP.get(cat_clean, "other")


def select_model(category: str, override: str = "auto") -> str:
    """
    Return the recommended model name for a given category.

    Args:
        category: product category string (e.g. "CEFIXIME")
        override: 'auto' uses category mapping; otherwise uses the provided model name

    Returns:
        model name string: 'prophet' | 'xgboost' | 'sarimax' | 'hybrid'
    """
    if override and override != "auto":
        valid = {"prophet", "xgboost", "sarimax", "hybrid"}
        if override not in valid:
            raise ValueError(f"Invalid model override '{override}'. Choose from {valid}")
        return override

    cat_type = get_category_type(category)
    return TYPE_MODEL_MAP.get(cat_type, "xgboost")


def get_fallback_list(category: str) -> List[str]:
    """Return a list of fallback models to try if the primary fails."""
    cat_type = get_category_type(category)
    defaults = ["prophet", "xgboost"]
    return TYPE_FALLBACK_MAP.get(cat_type, defaults)
