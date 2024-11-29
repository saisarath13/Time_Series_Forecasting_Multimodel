import os
import time
import tensorflow as tf
from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime
import boto3

# Disable GPU usage for TensorFlow if not available
if not tf.config.list_physical_devices('GPU'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU if no GPU is found

app = Flask(__name__)

# Initialize S3 client
s3 = boto3.client('s3')

# Define the S3 bucket and model filenames
bucket_name = 'my-keras-models-stock'  # Replace with your actual S3 bucket name
rnn_model_key = 'models/fine_tuned_rnn_model.keras'  # The S3 key for the RNN model
lstm_model_key = 'models/fine_tuned_lstm_model.keras'  # The S3 key for the LSTM model

# Retry mechanism for S3 downloads
def download_model_from_s3(model_key, local_path, retries=3, delay=5):
    attempt = 0
    while attempt < retries:
        try:
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            print(f"Attempting to download {model_key} to {local_path}...")
            s3.download_file(bucket_name, model_key, local_path)
            print(f"Model {model_key} downloaded successfully.")
            return
        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                raise

# Function to load the models (download if not already downloaded)
def load_models():
    try:
        model_dir = 'models/'
        os.makedirs(model_dir, exist_ok=True)

        # Download models if they do not exist
        rnn_model_path = os.path.join(model_dir, 'fine_tuned_rnn_model.keras')
        lstm_model_path = os.path.join(model_dir, 'fine_tuned_lstm_model.keras')

        if not os.path.exists(rnn_model_path):
            download_model_from_s3(rnn_model_key, rnn_model_path)
        
        if not os.path.exists(lstm_model_path):
            download_model_from_s3(lstm_model_key, lstm_model_path)

        # Load the fine-tuned models
        rnn_model = load_model(rnn_model_path)
        lstm_model = load_model(lstm_model_path)

        return rnn_model, lstm_model
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

# Load scaler for normalization
scaler = MinMaxScaler(feature_range=(0, 1))

# Function to fetch stock data
def fetch_stock_data(ticker, start_date='2015-01-01'):
    try:
        stock_data = yf.download(ticker, start=start_date)
        if stock_data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        return stock_data[['Close']]  # Use only the closing prices
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        raise

# Function to create sequences for prediction
def create_sequences(data, sequence_length=60):
    X = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
    return np.array(X)

# Function to make predictions with the selected model
def make_prediction(model, ticker, future_date):
    # Validate future date
    today = datetime.date.today()
    try:
        future_date_obj = datetime.datetime.strptime(future_date, "%d-%m-%Y").date()
    except ValueError:
        raise ValueError("Invalid date format. Please use dd-mm-yyyy.")
    
    if future_date_obj <= today:
        raise ValueError("Future date must be later than today.")

    stock_data = fetch_stock_data(ticker)
    scaled_data = scaler.fit_transform(stock_data)
    
    # Prepare the latest data for prediction
    sequence_length = 60
    X = create_sequences(scaled_data, sequence_length=sequence_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshaping for RNN/LSTM input

    predictions = []
    current_input = X[-1]

    # Calculate the number of days until the future date
    days_ahead = (future_date_obj - today).days

    # Loop to predict stock prices for each day until the future date
    for _ in range(days_ahead):
        prediction = model.predict(current_input.reshape(1, X.shape[1], 1))
        prediction = scaler.inverse_transform(prediction)
        predictions.append(prediction[0][0])

        # Update input for next day's prediction
        current_input = np.roll(current_input, -1, axis=0)
        current_input[-1] = prediction

    # Predict the price for the final future date
    final_prediction = model.predict(current_input.reshape(1, X.shape[1], 1))
    final_prediction = scaler.inverse_transform(final_prediction)
    predictions.append(final_prediction[0][0])

    return predictions, future_date_obj

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    error = None
    ticker = None
    future_date = None
    model_name = None

    if request.method == 'POST':
        ticker = request.form['ticker'].upper()  # Get the stock ticker from the form
        model_choice = request.form['model']  # Get the model choice (RNN or LSTM)
        future_date = request.form['future_date']  # Get the future date input

        try:
            # Validate the date format
            datetime.datetime.strptime(future_date, "%d-%m-%Y")
        except ValueError:
            error = "Invalid date format. Please use dd-mm-yyyy."
            return render_template('index.html', error=error, predictions=predictions)

        try:
            print("Loading models from S3...")
            rnn_model, lstm_model = load_models()
        except Exception as e:
            error = f"Error loading models from S3: {e}"
            return render_template('index.html', error=error, predictions=predictions)

        try:
            if model_choice == 'rnn':
                predictions, future_date_obj = make_prediction(rnn_model, ticker, future_date)
                model_name = "RNN"
            else:
                predictions, future_date_obj = make_prediction(lstm_model, ticker, future_date)
                model_name = "LSTM"
        except Exception as e:
            error = f"Error during prediction: {e}"
            return render_template('index.html', error=error, predictions=predictions)

        return render_template('index.html', predictions=predictions, model=model_name, ticker=ticker, future_date=future_date_obj.strftime('%d-%m-%Y'))

    return render_template('index.html', predictions=predictions, error=error)

if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true', host='0.0.0.0', port=5000)
