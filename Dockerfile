# Dockerfile (Updated with Increased Gunicorn Timeout)

# 1. Choose base image
FROM python:3.12-slim

# 2. Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 3. Install system dependencies required by libraries (like lightgbm)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 4. Set working directory
WORKDIR /app

# 5. Copy requirements first for caching
COPY requirements.txt .

# 6. Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Copy project code and templates
COPY . .

# 8. Expose port
EXPOSE 8000

# 9. Define container start command (WITH INCREASED TIMEOUT)
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000", "--timeout", "5000"]