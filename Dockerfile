# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the necessary dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable for CUDA (optional if you're using a CPU-only version)
ENV CUDA_VISIBLE_DEVICES=-1

# Run app.py when the container launches
CMD ["python", "app.py"]
