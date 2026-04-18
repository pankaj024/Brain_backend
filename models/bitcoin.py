import os
import numpy as np
import pandas as pd
import tflite_runtime.interpreter as tflite
import datetime
import yfinance as yf

# Configuration
SEQ_LEN = 60
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bitcoin.tflite')

# Caching for performance
_cached_interpreter = None
_cached_data = None
_last_fetch_time = None

def load_and_clean_data(period="1d"):
    """
    Loads and cleans the Bitcoin dataset from yfinance with caching.
    """
    global _cached_data, _last_fetch_time
    
    current_time = datetime.datetime.now()
    if period == "1d" and _cached_data is not None and _last_fetch_time is not None:
        # Cache for 1 minute
        if (current_time - _last_fetch_time).total_seconds() < 60:
            return _cached_data

    try:
        print(f"Fetching real BTC-USD data for period={period}...")
        # Use a more reliable way to fetch data if yfinance is flaky
        df = yf.download("BTC-USD", period=period, interval="1m", progress=False)
        
        if df.empty:
            print("Warning: yfinance returned empty dataframe. Using cache if available.")
            return _cached_data
            
        df = df.reset_index()
        # Handle potential multi-index columns from yf.download
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]

        if 'Datetime' in df.columns: df['Datetime'] = pd.to_datetime(df['Datetime'])
        elif 'Date' in df.columns: df['Datetime'] = pd.to_datetime(df['Date'])
        
        if period == "1d":
            _cached_data = df
            _last_fetch_time = current_time
        
        return df
    except Exception as e:
        print(f"Error fetching data from yfinance: {e}")
        return _cached_data

def get_prediction_data():
    """
    Returns historical data and a future prediction using REAL real-time data and TFLite model.
    """
    global _cached_interpreter
    
    if _cached_interpreter is None:
        if not os.path.exists(MODEL_PATH):
            return {"error": "Bitcoin model file (.tflite) missing."}
        try:
            _cached_interpreter = tflite.Interpreter(model_path=MODEL_PATH)
            _cached_interpreter.allocate_tensors()
        except Exception as e:
            return {"error": f"Inference engine failed: {str(e)}"}

    try:
        df = load_and_clean_data()
        
        # Fallback to simulation ONLY if network/yfinance completely fails after retries
        if df is None or df.empty:
            print("CRITICAL: Network/Yahoo Finance failed. Falling back to simulation for UI stability.")
            now = datetime.datetime.now()
            history = []
            base_price = 76000.0
            for i in range(60):
                time_point = now - datetime.timedelta(minutes=(60-i))
                price = base_price + np.random.normal(0, 50)
                history.append({"time": time_point.strftime("%H:%M"), "price": float(price)})
            
            # Simulated prediction
            last_prices = [h['price'] for h in history]
            min_p, max_p = min(last_prices), max(last_prices)
            X_input = (np.array([last_prices], dtype=np.float32) - min_p) / (max_p - min_p + 1e-9)
            
            input_details = _cached_interpreter.get_input_details()
            output_details = _cached_interpreter.get_output_details()
            _cached_interpreter.set_tensor(input_details[0]['index'], X_input)
            _cached_interpreter.invoke()
            pred_scaled = _cached_interpreter.get_tensor(output_details[0]['index'])
            prediction = float(pred_scaled[0][0] * (max_p - min_p + 1e-9) + min_p)
            
            return {
                "history": history,
                "prediction": prediction,
                "message": "Simulated data (Yahoo Finance unavailable)"
            }
        
        # Ensure we have enough data points
        if len(df) < SEQ_LEN:
            df = load_and_clean_data(period="5d")
            if df is None or len(df) < SEQ_LEN:
                return {"error": f"Insufficient data (need {SEQ_LEN} mins, got {len(df) if df is not None else 0})"}

        # Use the last 60 minutes for prediction
        last_prices = df['Close'].values[-SEQ_LEN:].reshape(1, -1)
        
        # Manual scaling based on the window
        min_p = np.min(df['Close'].values)
        max_p = np.max(df['Close'].values)
        X_input = (last_prices - min_p) / (max_p - min_p + 1e-9)
        X_input = X_input.astype(np.float32)
        
        input_details = _cached_interpreter.get_input_details()
        output_details = _cached_interpreter.get_output_details()
        _cached_interpreter.set_tensor(input_details[0]['index'], X_input)
        _cached_interpreter.invoke()
        pred_scaled = _cached_interpreter.get_tensor(output_details[0]['index'])
        
        prediction = float(pred_scaled[0][0] * (max_p - min_p + 1e-9) + min_p)
        
        # Prepare history for frontend chart
        history_df = df.tail(SEQ_LEN)
        history = []
        for _, row in history_df.iterrows():
            history.append({
                "time": row['Datetime'].strftime("%H:%M"),
                "price": float(row['Close'])
            })
        
        return {
            "history": history,
            "prediction": prediction,
            "message": "Real-time data from Yahoo Finance (TFLite Optimized)"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Bitcoin TFLite Module (Hybrid Data)")
