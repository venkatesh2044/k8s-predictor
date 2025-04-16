FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Create directories
RUN mkdir -p /var/lib/k8s-predictor/models /var/lib/k8s-predictor/visualizations /var/log/k8s-predictor

# Set entrypoint
ENTRYPOINT ["python", "driver.py"]