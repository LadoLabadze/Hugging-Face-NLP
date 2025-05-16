 Stock Sentiment Analysis with Alpaca + Hugging Face in 10 Minutes
 
🧾 What You’ll Do:
Perform real-time stock sentiment analysis using headlines from Alpaca News API

Classify sentiment using Hugging Face’s FinBERT, a model trained on financial texts

Aggregate daily sentiment scores for any stock (e.g., AAPL, AMZN, SPY)

✅ Use sentiment scores as input features for machine learning trading models

# 📊 Stock Sentiment Analysis with Alpaca + Hugging Face (FinBERT)

This project demonstrates a complete pipeline for performing **daily stock sentiment analysis** using **Alpaca News API** and **Hugging Face Transformers (FinBERT)**. It also evaluates how sentiment correlates with SPY (S&P 500 ETF) price changes — both **historically and predictively**.

---

## 🚀 What You’ll Do

- ✅ Fetch stock-related news using **Alpaca's Market News API**
- ✅ Score each headline with **FinBERT**, a BERT model fine-tuned on financial text
- ✅ Aggregate sentiment **by day**
- ✅ Retrieve historical SPY prices using `yfinance`
- ✅ Calculate correlations between:
  - Past sentiment and today’s return (backward)
  - Today’s sentiment and future return (forward)
  - Lag = 0 (same-day sentiment vs return)
- ✅ Visualize sentiment, price, and correlation across **−10 to +10 day lags**
- ✅ Use sentiment scores as features for **machine learning trading models**

---

## 🛠️ Quickstart

### 1. Install Dependencies

```bash
pip install alpaca-trade-api transformers yfinance pandas matplotlib seaborn
from alpaca_trade_api import REST
from transformers import pipeline
from datetime import datetime, timedelta

api = REST("YOUR_API_KEY", "YOUR_SECRET_KEY", base_url="https://paper-api.alpaca.markets")
classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")

end = datetime.today()
start = end - timedelta(days=1)

news = api.get_news("AAPL", start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))

for item in news:
    text = item._raw.get("headline")
    if text:
        result = classifier(text)[0]
        print(f"{text}\n→ {result['label']} ({result['score']:.2f})\n")
Sentiment + SPY Price Correlation
The notebook calculates correlation between daily sentiment scores and SPY returns:

sql
Copy
Edit
Lag Day | Past Sentiment → Today Return | Today Sentiment → Future Return
--------|-------------------------------|-------------------------------
      0 |                        +0.0912 |                        +0.0912
      1 |                        +0.1231 |                        +0.1312
      2 |                        +0.0754 |                        +0.1185
...
It also visualizes sentiment vs SPY price and plots correlation curves from −10 to +10 days to analyze signal lag.

📊 Visualization Examples
SPY Price + Sentiment (Dual-Axis Line Chart)

Correlation vs Lag Plot (−10 to +10 days)

python
Copy
Edit
# Plot sentiment vs SPY
plt.plot(combined_df.index, combined_df["SPY_Close"], label="SPY")
plt.plot(combined_df.index, combined_df["sentiment_score"], label="Sentiment")
python
Copy
Edit
# Plot lag correlations
plt.plot(range(-10, 11), lag_correlation_values)
🧠 ML-Ready Features
Sentiment data can be used as input for machine learning models:

Predict future returns (regression)

Generate buy/sell signals (classification)

Combine with technical indicators for smarter trading bots

🔧 Tools & Models Used
Tool	Purpose
📰 Alpaca API	News headlines by symbol
🤗 Hugging Face	NLP transformer pipelines
📘 FinBERT (ProsusAI)	Financial sentiment classification
📈 yfinance	SPY historical prices
🧠 pandas, matplotlib	Data analysis & visualization

📘 License
This project is under the MIT License — free to use, modify, and distribute.

🙋 Want More?
Backtest with trading strategy

Train ML models using lagged sentiment

Deploy in production with Streamlit or FastAPI

Pull requests and contributions are welcome! ⭐
