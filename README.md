# Time Series Prediction and Demand Forecasting

## Overview
**Time Series Prediction and Demand Forecasting** is a Flask-based web application designed to forecast future stock prices and demand using advanced deep learning models. It leverages **LSTM** and **RNN** models for time series analysis, and is deployed as an **API endpoint on AWS** for scalable cloud-based execution.

---

## Features
- **Multiple Models**: Fine-tuned LSTM and RNN models for accurate time series predictions.
- **Interactive Web Interface**: Predict future stock prices based on user inputs.
- **Scalable Deployment**: Hosted on AWS API Gateway and Lambda for high availability.
- **Historical Data Analysis**: Fetches and visualizes stock price trends from Yahoo Finance.

---

## Installation

### Prerequisites
- Python 3.9+
- Docker (optional, for containerized deployment)
- AWS account (for deployment)

### Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/time-series-prediction.git
   cd time-series-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   python app.py
   ```

4. Access the application at `http://127.0.0.1:8080`.

---

## Running with Docker

1. Build the Docker image:
   ```bash
   docker build -t time-series-prediction .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8080:8080 time-series-prediction
   ```

3. Access the application at `http://localhost:8080`.

---

## Deployment on AWS

1. Package the application as a Docker image:
   ```bash
   docker build -t time-series-forecasting-api .
   ```

2. Tag and push the image to Amazon Elastic Container Registry (ECR):
   ```bash
   docker tag time-series-forecasting-api:latest <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/time-series-forecasting-api
   docker push <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/time-series-forecasting-api
   ```

3. Deploy the container to AWS Elastic Container Service (ECS) or Lambda.
4. Expose the API endpoint using AWS API Gateway.

---

## Project Structure
```
time-series-prediction/
├── app.py                # Main Flask application
├── models/               # Trained models (LSTM, RNN)
│   ├── fine_tuned_lstm_model.keras
│   ├── fine_tuned_rnn_model.keras
├── templates/
│   └── index.html        # Web interface template
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── README.md             # Project documentation
└── utils.py              # Utility functions (data fetching, preprocessing)
```

---

## Usage
### Web Interface
1. Navigate to the home page.
2. Input the stock ticker, select a model (RNN or LSTM), and enter a future date.
3. Submit to view the predicted stock prices.

---

## Technologies Used
- **Flask**: Web framework for the application.
- **LSTM & RNN**: Deep learning models for time series forecasting.
- **Yahoo Finance API**: Fetch historical stock price data.
- **Docker**: Containerization for portability and scalability.
- **AWS (ECR, ECS, API Gateway)**: Cloud hosting and API management.

---

## Future Enhancements
- Add ARIMA and Prophet models for comparative analysis.
- Implement real-time data streaming and prediction updates.
- Integrate a dashboard for better visualization of predictions.
- Extend support to forecast demand for other sectors (e.g., energy, retail).

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Contact
For inquiries or contributions, please reach out to:
- **Email**: sarathk1307@gmail.com
- **GitHub**: [Your GitHub Profile](https://github.com/your-profile)

