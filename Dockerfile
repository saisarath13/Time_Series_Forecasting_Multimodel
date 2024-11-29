# Use a specific Python version as the base image
FROM python:3.9.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first (to take advantage of Docker cache)
COPY requirements.txt /app/

# Install the necessary dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application code
COPY . /app/

# Remove unnecessary files to keep the image clean
RUN rm -rf venv __pycache__ .pytest_cache .vscode .idea *.log *.env

# Expose the port your app will run on
EXPOSE 5000

# Ensure the app listens on all interfaces (important for Docker)
ENV FLASK_RUN_HOST=0.0.0.0

# Run app.py when the container launches
ENTRYPOINT ["python"]
CMD ["app.py"]
