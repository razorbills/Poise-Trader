#!/bin/bash
# Render.com startup script

echo "🚀 Starting Poise Trader on Render..."
echo "📦 Python version: $(python --version)"
echo "💰 Starting trading bot with web service..."

# Run with gunicorn (production server)
gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 0 web_service_launcher:app
