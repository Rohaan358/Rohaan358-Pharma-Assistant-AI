# 💊 PharmaIQ - Intelligent Sales Forecasting Agent

PharmaIQ is an end-to-end pharmaceutical sales analytics and forecasting platform. It combines traditional machine learning with modern Generative AI to provide predictive insights and natural language analysis of sales data.

## 🚀 Key Features

-   **Intelligent Forecasting**: Automatically selects the best model (**Prophet**, **XGBoost**, **SARIMAX**, or **Hybrid**) based on product category.
-   **AI Analytics Agent**: Natural language chat interface powered by **Meta Llama 3.3 70B** for deep insights and trend analysis.
-   **Interactive Dashboard**: Real-time sales summaries, historical comparisons, and KPI tracking.
-   **One-Click Upload**: Easy ingestion of sales data via CSV or Excel.
-   **Automated Backtesting**: Compares 2025 forecasts against actual sales data to measure accuracy (MAE, RMSE, MAPE).

---

## 🏗️ Architecture

```text
pharma-agent/
├── main.py                     # FastAPI Backend Entry
├── streamlit_app/              # Streamlit Frontend
│   └── app.py                  # Home Page
├── forecasting/                # Business Logic & Model Selection
├── models/                     # ML Model Implementations
├── agent/                      # Llama AI Integration
├── database/                   # MongoDB (Motor) Async Helpers
├── routes/                     # FastAPI Endpoint Definitions
└── schemas/                    # Pydantic Data Models
```

---

## ⚙️ Setup & Installation

### 1. Prerequisite
Ensure you have **Python 3.9+** installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: If `prophet` installation fails on Windows, install via `conda install -c conda-forge prophet`.*

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
# Database
MONGODB_URL=mongodb+srv://<user>:<password>@cluster.mongodb.net/pharma_db

# AI Agent (OpenRouter, Together, or Groq)
LLAMA_API_KEY=your_api_key_here
LLAMA_API_BASE=https://openrouter.ai/api
LLAMA_MODEL=meta-llama/Llama-3.3-70B-Instruct
```

### 4. Run the Backend (FastAPI)
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
The API documentation will be available at `http://localhost:8000/docs`.

### 5. Run the Frontend (Streamlit)
```bash
cd streamlit_app
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 📈 Model Mapping Logic

The system automatically routes products to specific models for optimal accuracy:

| Category | Primary Model | Why? |
| :--- | :--- | :--- |
| **Chronic** | Prophet | Best for long-term trends and seasonality. |
| **Antibiotic** | SARIMAX | Handles acute peaks and external features well. |
| **Gastro** | Prophet | Smooths out high variance. |
| **Acute** | XGBoost | Captures complex, non-linear relationships. |
| **Other** | Hybrid | Blends Prophet and XGBoost for stability. |

---

## 🤖 AI Agent Usage

The built-in AI Agent can answer questions like:
- *"Which products in the Antibiotic category had the highest growth in 2024?"*
- *"Analyze the trend for OMEPRAZOLE and tell me if the 2025 forecast is realistic."*
- *"Give me a summary of the top 5 products by sales volume."*

---

## ⚠️ Security Notice
This project follows strictly secure practices:
- **Zero-Touch Credentials**: The frontend never handles the MongoDB URL; it is managed exclusively by the backend `.env`.
- **Data Privacy**: No 2025 data is ever used for training; it is strictly reserved for evaluation (Zero Leakage).

---

## 📄 License
Custom Enterprise License - Created for Pharmaceutical Sales Intelligence.
