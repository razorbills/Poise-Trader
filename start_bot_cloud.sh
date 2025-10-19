#!/bin/bash
# Poise Trader - Cloud Startup Script
# This script runs the bot 24/7 with auto-restart

echo "========================================"
echo "🚀 POISE TRADER - CLOUD STARTUP"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MAX_RESTARTS=10
RESTART_DELAY=10
restart_count=0

# Function to run bot
run_bot() {
    echo -e "${GREEN}✅ Starting Poise Trader...${NC}"
    python3 cloud_launcher.py
    exit_code=$?
    return $exit_code
}

# Main loop with auto-restart
while true; do
    # Run the bot
    run_bot
    exit_code=$?
    
    # If exit code is 0, it was a clean shutdown
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ Bot stopped cleanly. Exiting...${NC}"
        exit 0
    fi
    
    # If exit code is 130 (Ctrl+C), exit
    if [ $exit_code -eq 130 ]; then
        echo -e "${YELLOW}⚠️ Received interrupt signal. Exiting...${NC}"
        exit 0
    fi
    
    # Otherwise, it crashed - restart
    restart_count=$((restart_count + 1))
    
    # Check if we've exceeded max restarts
    if [ $restart_count -ge $MAX_RESTARTS ]; then
        echo -e "${RED}❌ Maximum restart attempts ($MAX_RESTARTS) reached. Exiting...${NC}"
        exit 1
    fi
    
    echo -e "${RED}❌ Bot crashed (Exit code: $exit_code)${NC}"
    echo -e "${YELLOW}🔄 Restart attempt $restart_count/$MAX_RESTARTS in ${RESTART_DELAY}s...${NC}"
    sleep $RESTART_DELAY
done
