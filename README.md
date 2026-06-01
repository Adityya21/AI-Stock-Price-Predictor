<div align="center">

<h1>📈 AI Stock Price Predictor</h1>

<p><b>Ensemble ML model that forecasts stock prices using LSTM, Random Forest & XGBoost —<br/>with 30+ technical indicators and real-time Yahoo Finance data.</b></p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-189AB4?style=flat-square"/>
  <img src="https://img.shields.io/badge/Yahoo_Finance-6001D2?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=flat-square"/>
</p>

<p>
  Works on any ticker &nbsp;·&nbsp; US stocks &nbsp;·&nbsp; NSE India &nbsp;·&nbsp; Crypto &nbsp;·&nbsp; Indices
</p>

</div>

---

## What This Project Does

Most stock prediction projects train a single model on raw closing prices and call it a day. This one is different.

It builds an **ensemble of three models** (Bidirectional LSTM-GRU, Random Forest, XGBoost), engineers **30+ technical indicators** from raw OHLCV data, selects the best model by R² score, and outputs a **multi-day rolling forecast** — all from a single function call on any ticker worldwide.

```python
predictor, results = predict_stock(ticker='TSLA', period='3y', epochs=100, future_days=30)
```

---

## 🏗️ Model Architecture

### Ensemble Strategy

| Model | Weight | Role |
|-------|--------|------|
| Bidirectional LSTM-GRU | **40%** | Captures long-range temporal dependencies in price sequences |
| Random Forest | **30%** | Robust to noise, handles non-linear feature interactions |
| XGBoost | **30%** | Gradient boosting on lagged engineered features |

Final prediction = weighted average. Best individual model auto-selected by R².

### LSTM Architecture

```
Input (lookback=60 days)
│
├── BiLSTM(128)   → Dropout(0.3)
├── BiGRU(64)     → Dropout(0.3)
├── BiLSTM(32)    → Dropout(0.2)
├── Dense(64, relu)
├── Dense(32, relu)
└── Dense(forecast_days)     ← multi-step output

Loss: Huber  |  Optimizer: Adam(lr=0.001)
```

---

## 📊 Feature Engineering — 30+ Technical Indicators

| Category | Indicators |
|---|---|
| **Moving Averages** | MA (5/10/20/50/100/200), EMA (12/26/50) |
| **Momentum** | MACD, RSI(14), ROC(10/20), Stochastic %K/%D |
| **Volatility** | Bollinger Bands, ATR(14), Volatility(10/30) |
| **Volume** | Volume MA(20), Volume Ratio, Volume Change |
| **Price Action** | Momentum(5/10), Support/Resistance(20), Log Returns |
| **Time Encoding** | Cyclical day-of-week & month (sin/cos) |

All features scaled with `MinMaxScaler` per ticker and persisted to disk.

---

## 📉 Evaluation Metrics

```
RMSE   →  Root Mean Squared Error        (penalises large errors)
MAE    →  Mean Absolute Error             (average magnitude)
R²     →  Coefficient of Determination   (variance explained)
MAPE   →  Mean Absolute Percentage Error (scale-independent)
```

Best single model is selected automatically by highest R² on the test split.

---

## 🚀 Usage

### Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow yfinance joblib
```

### Option 1 — Quick single prediction

```python
from AI_Project import predict_stock

predictor, results = predict_stock(
    ticker='AAPL',
    period='3y',       # 1y | 2y | 3y | 5y
    epochs=100,
    future_days=30
)
```

### Option 2 — Interactive mode (prompts for inputs)

```python
interactive_stock_predictor()
```

### Option 3 — Batch prediction

```python
tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']

for ticker in tickers:
    predict_stock(ticker, period='2y', epochs=50, future_days=30)
```

---

## 🌍 Supported Tickers

| Market | Examples |
|--------|---------|
| 🇺🇸 US Tech | `AAPL` `GOOGL` `MSFT` `TSLA` `NVDA` `META` |
| 🏦 Finance | `JPM` `GS` `V` `MA` `BRK-B` |
| 📊 Indices | `^GSPC` (S&P 500) `^DJI` `^IXIC` (Nasdaq) |
| 🪙 Crypto | `BTC-USD` `ETH-USD` |
| 🇮🇳 NSE India | `RELIANCE.NS` `TCS.NS` `INFY.NS` `HDFCBANK.NS` |

Any valid Yahoo Finance ticker works out of the box.

---

## 📁 Output Files

After each run, the following are saved automatically:

```
{TICKER}_analysis_dashboard.png       # Multi-panel prediction dashboard
{TICKER}_future_prediction.png        # Forecast chart (next N days)
{TICKER}_feature_importance.png       # RF/XGB feature importance plot

saved_models/
├── lstm_{TICKER}.h5                  # Trained LSTM model
├── rf_{TICKER}.pkl                   # Trained Random Forest
├── xgb_{TICKER}.pkl                  # Trained XGBoost
└── scaler_{TICKER}.pkl               # Fitted MinMaxScaler
```

---

## ⚙️ Configuration

| Parameter | Default | Description |
|---|---|---|
| `lookback_days` | `60` | LSTM input sequence length (days of history per sample) |
| `period` | `3y` | Historical data window from Yahoo Finance |
| `epochs` | `100` | LSTM training epochs |
| `forecast_days` | `1` | Multi-step prediction horizon |
| `batch_size` | `32` | LSTM mini-batch size |

---

## 🗂️ Project Structure

```
AI-Stock-Price-Predictor/
├── AI_Project.ipynb          # Main notebook (models, training, inference)
├── saved_models/             # Persisted trained models per ticker
├── *.png                     # Auto-generated charts
└── README.md
```

---

## ⚠️ Disclaimer

This project is built for **educational and research purposes only**. Stock price predictions made by ML models — including this one — are not financial advice and should not be used as the basis for any real investment decisions. Markets are inherently unpredictable.

---

## 🚀 Possible Extensions

- Add Transformer / Temporal Fusion Transformer (TFT) as a fourth ensemble member
- Build a Streamlit or Gradio dashboard for interactive forecasting
- Add portfolio-level multi-ticker correlation analysis
- Deploy as a REST API (FastAPI + Docker)
- Integrate news sentiment via NewsAPI or FinBERT

---

## 👤 Author

**Aditya Yashovardhan** **Daksh Gurjar** **Dhruv Mahalle**

---

## 📄 Licence

MIT — open-source and free to use with attribution.

---

<div align="center">
<sub>Data: Yahoo Finance &nbsp;·&nbsp; Models: TensorFlow · Scikit-learn · XGBoost &nbsp;·&nbsp; Built with Python</sub>
</div>
