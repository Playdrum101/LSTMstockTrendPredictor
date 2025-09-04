# Stock Price Predictor (LSTM + Streamlit)

A Streamlit web app that predicts the next day's stock closing price using a pre-trained Keras LSTM model, with data sourced from Yahoo Finance via `yfinance`. The app visualizes historical prices and moving averages, and compares predicted vs. actual test prices.

## Features
- Interactive UI built with Streamlit
- Data download via `yfinance` (Yahoo Finance)
- Visualizations: closing price, 100-day and 200-day moving averages
- LSTM-based next-day price prediction using a 100-day look-back window
- Side-by-side plot of predicted vs. actual test prices

## Project Structure
```
StockPricePredictor/
  ├─ app.py                # Streamlit app
  ├─ keras_model.h5        # Pre-trained Keras LSTM model
  ├─ LSTM_Model.ipynb      # Notebook used to train/save model
  └─ README.md             # This file
```

## Prerequisites
- Python 3.10+ recommended
- Windows PowerShell or a terminal

## Installation
1. Clone or download this repository.
2. Create and activate a virtual environment (recommended):
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
3. Install dependencies:
   ```powershell
   pip install --upgrade pip
   pip install streamlit numpy pandas matplotlib scikit-learn yfinance pandas_datareader keras tensorflow
   ```

## Running the App
From the project root:
```powershell
streamlit run app.py
```
Then open the local URL shown in the terminal (typically http://localhost:8501). Enter a stock ticker (e.g., AAPL) and the app will load data for 2010-01-01 to 2019-12-31, visualize charts, and generate predictions.

## How It Works
- Data is downloaded with `yfinance.download(ticker, start, end)`.
- The app visualizes:
  - Closing price over time
  - 100-day and 200-day simple moving averages
- Data is split by time: first 70% for training, last 30% for testing.
- Inputs are scaled to [0, 1] using `MinMaxScaler`.
- Sequences are created using a 100-day look-back window to predict the next day's closing price (regression).
- A pre-trained Keras LSTM model (`keras_model.h5`) generates predictions on the test set.

## Model Details
- Task: Predict next-day closing price (regression)
- Look-back window: 100 trading days
- Input features: closing price only (1 feature)
- Network (from `LSTM_Model.ipynb`):
  - LSTM(50, relu, return_sequences=True) + Dropout(0.2)
  - LSTM(60, relu, return_sequences=True) + Dropout(0.3)
  - LSTM(80, relu, return_sequences=True) + Dropout(0.3)
  - LSTM(120, relu) + Dropout(0.5)
  - Dense(1)
- Loss/Optimizer: MSE with Adam
- Training: 50 epochs (see notebook)

## Evaluation
- Training loss: Mean Squared Error (MSE)
- Visual evaluation: Plot of actual vs. predicted prices on the test set
- Suggested additions (not yet in code):
  ```python
  from sklearn.metrics import mean_absolute_error, mean_squared_error
  import numpy as np

  mae = mean_absolute_error(y_test, y_predicted)
  rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
  print("MAE:", mae)
  print("RMSE:", rmse)
  ```
- Baseline comparison (recommended): naive forecast where tomorrow's price = today's price

## Assumptions & Limitations
- Uses only closing price as input (no volume or technical indicators)
- Sequences cover trading days only (weekends/holidays omitted by data source)
- Model was trained on 2010–2019 data for AAPL; performance on other tickers may vary
- Predictions are not financial advice; use at your own risk

## Troubleshooting
- TensorFlow/Keras install issues:
  - Try upgrading pip: `python -m pip install --upgrade pip`
  - Install CPU-only TensorFlow if GPU is not required
- yfinance rate limits or network errors:
  - Retry after a short wait
  - Check ticker validity and date range
- Streamlit not opening in browser:
  - Copy/paste the URL from the terminal into your browser

## Roadmap / Next Steps
- Add evaluation metrics (MAE, RMSE, R²) in the app
- Add a naive baseline and show side-by-side metrics
- Incorporate additional features: RSI, MACD, Bollinger Bands, Volume
- Try alternative architectures (GRU, CNN-LSTM, Transformers)
- Extend date ranges and add a date picker in the UI
- Save models in the newer Keras format (`.keras`)

## License
This project is provided as-is for educational purposes. No warranty is expressed or implied. 
