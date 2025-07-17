#!/bin/bash

# Fix kahinali service issues
echo "=== Fixing kahinali service ==="

# Stop the service first
systemctl stop kahinali.service

# Check if the Python virtual environment exists
if [ ! -d "/var/www/html/kahin/venv" ]; then
    echo "Creating virtual environment..."
    cd /var/www/html/kahin
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "Virtual environment exists, updating dependencies..."
    cd /var/www/html/kahin
    source venv/bin/activate
    pip install -r requirements.txt
fi

# Check if main.py exists and is executable
if [ ! -f "/var/www/html/kahin/main.py" ]; then
    echo "ERROR: main.py not found!"
    exit 1
fi

# Test the Python script manually
echo "Testing main.py..."
cd /var/www/html/kahin
source venv/bin/activate
python main.py --test

# Check the service file
echo "=== Current service configuration ==="
cat /etc/systemd/system/kahinali.service

# Reload systemd and restart service
systemctl daemon-reload
systemctl enable kahinali.service
systemctl start kahinali.service

# Check service status
echo "=== Service status ==="
systemctl status kahinali.service

# Show recent logs
echo "=== Recent logs ==="
journalctl -u kahinali.service -n 20 --no-pager 