# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY examples/ ./examples/
COPY telegram_bot_main.py .
COPY env.example .env

# Create non-root user
RUN useradd -m -u 1000 tradingbot && chown -R tradingbot:tradingbot /app
USER tradingbot

# Expose port (if needed for web interface)
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src.trading_bot; print('OK')" || exit 1

# Run the Telegram bot
CMD ["python", "telegram_bot_main.py"] 