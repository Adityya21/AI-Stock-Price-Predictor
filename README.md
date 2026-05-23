# 🤖 AI-Stock-Price-Predictor
AI stock price predictor using ensemble of LSTM, Random Forest &amp; XGBoost models. Fetches real-time data via Yahoo Finance, engineers 30+ technical indicators, and forecasts future prices for any stock, index, or crypto ticker. Built with Python, TensorFlow &amp; Scikit-learn.
## 📌 Features

- **Real-time data** via Yahoo Finance (`yfinance`)
- **3 ML models**: Bidirectional LSTM-GRU hybrid, Random Forest, XGBoost
- **Ensemble prediction** (weighted: LSTM 40%, RF 30%, XGB 30%)
- **30+ technical indicators**: RSI, MACD, Bollinger Bands, ATR, Stochastic, EMA, etc.
- **Multi-day forecasting** (configurable horizon)
- **Auto visualizations**: dashboard, future prediction plot, feature importance chart
- **Model persistence**: saves trained models to disk
- **Works for any ticker**: stocks, indices, crypto, Indian NSE stocks


## 🛠️ Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow yfinance joblib
```


## 🚀 Usage

### Option 1 — Interactive Mode
```python
python AI_Project.ipynb
# or run the last cell in Jupyter
interactive_stock_predictor()
```
Prompts for ticker, period, epochs, forecast days.

### Option 2 — Direct Call
```python
from AI_Project import predict_stock

predictor, results = predict_stock(
    ticker='AAPL',
    period='3y',      # 1y | 2y | 3y | 5y
    epochs=100,
    future_days=30
)
```

### Option 3 — Batch Prediction
```python
stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
for stock in stocks:
    predict_stock(stock, period='2y', epochs=50, future_days=30)
```

## 🏗️ Architecture

```
AdvancedStockPredictor
├── fetch_realtime_data()       → yfinance pull
├── create_advanced_features()  → 30+ indicators
├── prepare_lstm_data()         → sequence builder (lookback window)
├── prepare_ml_data()           → lagged features for RF/XGB
├── build_advanced_lstm()       → BiLSTM → BiGRU → BiLSTM → Dense
├── train_all_models()          → trains all 3 + ensemble
├── predict_future()            → rolling forecast
├── plot_comprehensive_analysis()
├── plot_future_prediction()
├── plot_feature_importance()
├── generate_report()
└── save_models()
```

**LSTM architecture:**
```
BiLSTM(128) → Dropout(0.3) → BiGRU(64) → Dropout(0.3)
→ BiLSTM(32) → Dropout(0.2) → Dense(64) → Dense(32) → Dense(forecast_days)
Loss: Huber | Optimizer: Adam(lr=0.001)
```


## 📊 Technical Indicators Used

| Category | Indicators |
|---|---|
| Moving Averages | MA (5/10/20/50/100/200), EMA (12/26/50) |
| Momentum | MACD, RSI(14), ROC(10/20), Stochastic %K/%D |
| Volatility | Bollinger Bands, ATR(14), Volatility(10/30) |
| Volume | Volume MA(20), Volume Ratio, Volume Change |
| Price Action | Momentum(5/10), Support/Resistance(20), Log Returns |
| Time | Cyclical day-of-week & month encoding |


## 📁 Output Files

| File | Description |
|---|---|
| `{TICKER}_analysis_dashboard.png` | Multi-panel prediction dashboard |
| `{TICKER}_future_prediction.png` | Future price forecast chart |
| `{TICKER}_feature_importance.png` | RF/XGB feature importance |
| `saved_models/lstm_{TICKER}.h5` | Saved LSTM model |
| `saved_models/rf_{TICKER}.pkl` | Saved Random Forest |
| `saved_models/xgb_{TICKER}.pkl` | Saved XGBoost |
| `saved_models/scaler_{TICKER}.pkl` | MinMaxScaler |


## 📈 Supported Tickers

| Category | Examples |
|---|---|
| US Tech | `AAPL`, `GOOGL`, `MSFT`, `TSLA`, `NVDA`, `META` |
| Finance | `JPM`, `GS`, `V`, `MA`, `BRK-B` |
| Indices | `^GSPC` (S&P 500), `^DJI`, `^IXIC` |
| Crypto | `BTC-USD`, `ETH-USD` |
| India (NSE) | `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS` |


## ⚙️ Configuration

| Parameter | Default | Description |
|---|---|---|
| `lookback_days` | 60 | LSTM input sequence length |
| `period` | `3y` | Historical data period |
| `epochs` | 100 | LSTM training epochs |
| `forecast_days` | 1 | Multi-step prediction horizon |
| `batch_size` | 32 | LSTM batch size |


## 📉 Evaluation Metrics

- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **R²** — Coefficient of Determination
- **MAPE** — Mean Absolute Percentage Error

Best model auto-selected by highest R² score.



