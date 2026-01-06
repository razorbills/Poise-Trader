#!/bin/bash
# Render.com startup script

echo "🚀 Starting Poise Trader on Render..."
echo "📦 Python version: $(python --version)"
echo "💰 Starting trading bot with web service..."

# Run with gunicorn (production server)
python render_launcher.py
