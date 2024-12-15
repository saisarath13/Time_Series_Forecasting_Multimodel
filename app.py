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
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Ensure GPU is disabled
print("Running TensorFlow on CPU mode.")

app = Flask(__name__)

# Initialize S3 client using AWS credentials from environment variables
s3 = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    region_name=os.environ.get('AWS_REGION')
)

# Define the S3 bucket and model filenames
bucket_name = 'my-keras-models-stock'  # Replace with your actual S3 bucket name
rnn_model_key = 'models/fine_tuned_rnn_model.keras'  # The S3 key for the RNN model
lstm_model_key = 'models/fine_tuned_lstm_model.keras'  # The S3 key for the LSTM model

# Retry mechanism for S3 downloads
def download_model_from_s3(model_key, local_path, retries=3, delay=5):
    attempt = 0
    while attempt < retries:
        try:
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
                raise Exception(f"Failed to download {model_key} after {retries} attempts.")

# Function to load the models (download if not already downloaded)
def load_models():
    model_dir = 'models/'
    os.makedirs(model_dir, exist_ok=True)

    # Paths to local models
    rnn_model_path = os.path.join(model_dir, 'fine_tuned_rnn_model.keras')
    lstm_model_path = os.path.join(model_dir, 'fine_tuned_lstm_model.keras')

    # Download models if they do not exist locally
    if not os.path.exists(rnn_model_path):
        download_model_from_s3(rnn_model_key, rnn_model_path)
    if not os.path.exists(lstm_model_path):
        download_model_from_s3(lstm_model_key, lstm_model_path)

    # Load the fine-tuned models
    rnn_model = load_model(rnn_model_path)
    lstm_model = load_model(lstm_model_path)

    return rnn_model, lstm_model

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
    today = datetime.date.today()

    try:
        future_date_obj = datetime.datetime.strptime(future_date, "%d-%m-%Y").date()
    except ValueError:
        raise ValueError("Invalid date format. Please use dd-mm-yyyy.")
    
    if future_date_obj <= today:
        raise ValueError("Future date must be later than today.")

    stock_data = fetch_stock_data(ticker)
    scaled_data = scaler.fit_transform(stock_data)

    sequence_length = 60
    X = create_sequences(scaled_data, sequence_length=sequence_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    predictions = []
    current_input = X[-1]
    days_ahead = (future_date_obj - today).days

    for _ in range(days_ahead):
        prediction = model.predict(current_input.reshape(1, sequence_length, 1))
        prediction = scaler.inverse_transform(prediction)
        predictions.append(prediction[0][0])

        current_input = np.roll(current_input, -1, axis=0)
        current_input[-1] = scaler.transform([[prediction[0][0]]])[0]

    final_prediction = model.predict(current_input.reshape(1, sequence_length, 1))
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
        ticker = request.form['ticker'].upper()
        model_choice = request.form['model']
        future_date = request.form['future_date']

        try:
            datetime.datetime.strptime(future_date, "%d-%m-%Y")
        except ValueError:
            error = "Invalid date format. Please use dd-mm-yyyy."
            return render_template('index.html', error=error, predictions=predictions)

        try:
            print("Loading models from S3...")
            rnn_model, lstm_model = load_models()
        except Exception as e:
            error = f"Error loading models: {e}"
            return render_template('index.html', error=error, predictions=predictions)

        try:
            if model_choice == 'rnn':
                predictions, future_date_obj = make_prediction(rnn_model, ticker, future_date)
                model_name = "RNN"
            else:
                predictions, future_date_obj = make_prediction(lstm_model, ticker, future_date)
                model_name = "LSTM"
        except Exception as e:
            error = f"Prediction error: {e}"
            return render_template('index.html', error=error, predictions=predictions)

        return render_template('index.html', predictions=predictions, model=model_name, ticker=ticker, future_date=future_date_obj.strftime('%d-%m-%Y'))

    return render_template('index.html', predictions=predictions, error=error)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
