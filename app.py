import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage for TensorFlow if not available

from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime

app = Flask(__name__)

# Load the fine-tuned models
rnn_model = load_model('models/fine_tuned_rnn_model.keras')
lstm_model = load_model('models/fine_tuned_lstm_model.keras')

# Load scaler for normalization
scaler = MinMaxScaler(feature_range=(0, 1))

# Function to fetch stock data
def fetch_stock_data(ticker, start_date='2015-01-01'):
    stock_data = yf.download(ticker, start=start_date)  # No end_date means fetch till today
    return stock_data[['Close']]  # Use only the closing prices

# Function to create sequences for prediction
def create_sequences(data, sequence_length=60):
    X = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
    return np.array(X)

# Function to make predictions with the selected model
def make_prediction(model, ticker, future_date):
    stock_data = fetch_stock_data(ticker)
    scaled_data = scaler.fit_transform(stock_data)
    
    # Prepare the latest data for prediction
    X = create_sequences(scaled_data)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshaping for RNN/LSTM input

    predictions = []
    current_input = X[-1]  # Start with the last available data point

    # Calculate the number of days until the future date
    today = datetime.date.today()
    future_date_obj = datetime.datetime.strptime(future_date, "%d-%m-%Y").date()
    days_ahead = (future_date_obj - today).days

    # Loop to predict the stock price for each day from today until the future date
    for _ in range(days_ahead):
        prediction = model.predict(current_input.reshape(1, X.shape[1], 1))
        prediction = scaler.inverse_transform(prediction)  # Convert back to original scale
        predictions.append(prediction[0][0])

        # Update input for next day's prediction (rolling the window)
        current_input = np.roll(current_input, -1, axis=0)
        current_input[-1] = prediction  # Replace last value with predicted value

    # Finally predict the price for the future date
    final_prediction = model.predict(current_input.reshape(1, X.shape[1], 1))
    final_prediction = scaler.inverse_transform(final_prediction)
    predictions.append(final_prediction[0][0])

    return predictions, future_date_obj

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker'].upper()  # Get the stock ticker from the form
        model_choice = request.form['model']  # Get the model choice (RNN or LSTM)
        future_date = request.form['future_date']  # Get the future date input

        # Check if the date format is correct
        try:
            datetime.datetime.strptime(future_date, "%d-%m-%Y")
        except ValueError:
            return render_template('index.html', error="Invalid date format. Please use dd-mm-yyyy.")

        # Choose the model
        if model_choice == 'rnn':
            predictions, future_date_obj = make_prediction(rnn_model, ticker, future_date)
            model_name = "RNN"
        else:
            predictions, future_date_obj = make_prediction(lstm_model, ticker, future_date)
            model_name = "LSTM"

        return render_template('index.html', predictions=predictions, model=model_name, ticker=ticker, future_date=future_date_obj.strftime('%d-%m-%Y'))

    return render_template('index.html', predictions=None)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Listen on all IPs, necessary for Docker
