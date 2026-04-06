# XAU/USD ML Signal Pipeline

A production-grade machine learning pipeline that predicts the next-day direction of Gold (XAU/USD) using macro indicators, technical features, and order flow data. Built as a portfolio project targeting ML/AI Engineer roles at investment banks.

---

## What this does

Takes in daily gold price data, US macro data (real yields, DXY, VIX), and CFTC order flow positioning — engineers 34 features across 4 categories — trains a LightGBM classifier with walk-forward validation — and serves predictions via a REST API.

**Output:** `LONG`, `SHORT`, or `NEUTRAL` signal with probability and confidence score.

---

## Architecture

```
xauusd_pipeline/
├── data/
│   └── ingestion.py        # Fetches gold prices (Yahoo), macro (FRED), COT (CFTC)
├── features/
│   └── engineering.py      # 34 features: technical + macro + order flow + HMM regime
├── model/
│   ├── train.py            # LightGBM + walk-forward CV + MLflow tracking
│   └── artifacts/          # Trained model saved here (model.lgb)
├── api/
│   ├── main.py             # FastAPI app with /predict, /health, /model/info
│   ├── predictor.py        # Singleton model loader + inference engine
│   └── schemas.py          # Pydantic request/response validation
├── tests/
│   └── test_api.py         # API endpoint tests (pytest)
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # API + MLflow tracking server
└── requirements.txt
```

---

## Quickstart

### Run locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (logs to MLflow automatically)
python model/train.py

# 3. Start the API
uvicorn api.main:app --reload

# 4. Open API docs in browser
http://localhost:8000/docs
```

### Run with Docker

```bash
docker-compose up --build
```

### Run tests

```bash
pytest tests/ -v
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — model loaded status |
| `GET` | `/model/info` | Model version, features, training metrics |
| `POST` | `/predict` | Next-day gold direction signal |
| `POST` | `/predict/batch` | Signals over a date range |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.5}'
```

### Example response

```json
{
  "signal": "LONG",
  "probability": 0.73,
  "confidence": "HIGH",
  "regime": "BULLISH",
  "real_yield_signal": "NEGATIVE (-0.45%) → Supportive for gold",
  "prediction_date": "2024-06-15",
  "model_version": "lgbm-v20240615",
  "features_used": 34
}
```

---

## Features

### Macro (top predictors)
- US 10Y real yield (TIPS) — strongest single predictor of gold
- DXY trade-weighted USD index + momentum
- VIX level, z-score, and spike indicator
- Fed funds rate, CPI YoY, 10Y breakeven inflation

### Technical
- RSI (14, 28), ROC, Stochastic
- MACD, EMA crossovers (10/50/200)
- Bollinger Band width and position
- ATR, realized volatility (10d, 30d)
- OBV divergence (CVD proxy)

### Order Flow (COT-based)
- Managed Money net positioning z-score
- Commercial net positioning z-score
- COT momentum (4-week change)
- Retail sentiment proxy (contrarian)
- CVD/price divergence signal

### Regime
- HMM-based 3-state regime detection (bull / neutral / bear)
- Regime change flag

---

## Model & Validation

| Component | Detail |
|---|---|
| Model | LightGBM (gradient boosted trees) |
| Validation | Walk-forward cross-validation, 5 folds, 5-day gap |
| Primary metric | AUC + annualized Sharpe ratio |
| Experiment tracking | MLflow (all runs logged automatically) |
| Target | Next-day binary direction (UP=1, DOWN=0) |

Walk-forward validation is used instead of random splits to prevent lookahead bias — a critical requirement for financial time series.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | LightGBM, scikit-learn, hmmlearn |
| API | FastAPI, Pydantic v2, Uvicorn |
| Tracking | MLflow |
| Data | yfinance, FRED API, CFTC public data |
| Deployment | Docker, docker-compose |
| Testing | pytest, httpx |

---

## Disclaimer

This project is for research and portfolio purposes only. It is not financial advice and should not be used to make real trading decisions.