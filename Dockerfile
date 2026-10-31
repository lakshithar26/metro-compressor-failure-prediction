FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy source code and models
COPY src/ ./src/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Run the API server
CMD ["python", "-m", "src.api.scoring_service"]
