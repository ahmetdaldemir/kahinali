# Server Fix Guide

## Current Issues
1. **Git ownership issue**: `fatal: detected dubious ownership in repository`
2. **Missing requirements.txt**: `requirements.txt bulunamadı!`
3. **Service failing**: Exit code 2

## Manual Fix Steps

### Step 1: Connect to Server
```bash
ssh root@185.27.134.10
```

### Step 2: Fix Git Ownership
```bash
cd /var/www/html/kahin
git config --global --add safe.directory /var/www/html/kahin
```

### Step 3: Check Repository Status
```bash
cd /var/www/html/kahin
git status
git log --oneline -5
```

### Step 4: Ensure requirements.txt Exists
```bash
cd /var/www/html/kahin
ls -la requirements.txt
```

If requirements.txt doesn't exist, create it:
```bash
cat > requirements.txt << 'EOF'
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
requests>=2.25.0
python-binance>=1.0.0
ccxt>=2.0.0

# Technical Analysis
ta-lib>=0.4.0
scikit-learn>=1.0.0
scipy>=1.7.0

# Machine Learning
tensorflow>=2.8.0
keras>=2.8.0
xgboost>=1.5.0
lightgbm>=3.3.0

# Web Framework
flask>=2.0.0
flask-cors>=3.0.0

# Database
psycopg2-binary>=2.9.0
sqlalchemy>=1.4.0

# Scheduling
apscheduler>=3.9.0

# Telegram Bot
python-telegram-bot>=13.0.0

# Additional utilities
tqdm>=4.62.0
colorama>=0.4.4
EOF
```

### Step 5: Update Virtual Environment
```bash
cd /var/www/html/kahin
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6: Test the Application
```bash
cd /var/www/html/kahin
source venv/bin/activate
python test_deployment.py
```

### Step 7: Fix Service Configuration
```bash
# Stop the service
systemctl stop kahinali.service

# Update service file
cat > /etc/systemd/system/kahinali.service << 'EOF'
[Unit]
Description=Kahinali Trading Bot for kahinapp.com
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/var/www/html/kahin
ExecStart=/var/www/html/kahin/venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment=PYTHONPATH=/var/www/html/kahin

[Install]
WantedBy=multi-user.target
EOF

# Reload and start service
systemctl daemon-reload
systemctl enable kahinali.service
systemctl start kahinali.service
```

### Step 8: Check Service Status
```bash
systemctl status kahinali.service
journalctl -u kahinali.service -f
```

## Alternative: Complete Reset

If the above doesn't work, try a complete reset:

```bash
# Stop service
systemctl stop kahinali.service
systemctl disable kahinali.service

# Remove old directory
rm -rf /var/www/html/kahin

# Create fresh directory
mkdir -p /var/www/html/kahin
cd /var/www/html/kahin

# Clone fresh repository
git clone https://github.com/ahmetkahin/kahinali.git .

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create service file
cat > /etc/systemd/system/kahinali.service << 'EOF'
[Unit]
Description=Kahinali Trading Bot for kahinapp.com
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/var/www/html/kahin
ExecStart=/var/www/html/kahin/venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment=PYTHONPATH=/var/www/html/kahin

[Install]
WantedBy=multi-user.target
EOF

# Start service
systemctl daemon-reload
systemctl enable kahinali.service
systemctl start kahinali.service
```

## Testing Commands

### Test 1: Manual Run
```bash
cd /var/www/html/kahin
source venv/bin/activate
python main.py
```

### Test 2: Import Test
```bash
cd /var/www/html/kahin
source venv/bin/activate
python test_deployment.py
```

### Test 3: Service Test
```bash
systemctl status kahinali.service
journalctl -u kahinali.service -n 20
```

## Monitoring

### Check Service Status
```bash
systemctl status kahinali.service
```

### Monitor Logs
```bash
journalctl -u kahinali.service -f
```

### Check Application Logs
```bash
tail -f /var/www/html/kahin/logs/kahin_ultima.log
```

### Check System Resources
```bash
ps aux | grep python
top -p $(pgrep -f main.py)
```

## Expected Results

After successful fix:
1. **Service Status**: `Active: active (running)`
2. **Git**: No ownership errors
3. **Requirements**: All dependencies installed
4. **Application**: Running without import errors

## Troubleshooting

### If service still fails:
1. Check logs: `journalctl -u kahinali.service -n 50`
2. Test manually: `cd /var/www/html/kahin && python main.py`
3. Check dependencies: `pip list`
4. Verify paths: `ls -la /var/www/html/kahin/`

### If git issues persist:
```bash
cd /var/www/html/kahin
git config --global --add safe.directory /var/www/html/kahin
git config --global user.name "deploy"
git config --global user.email "deploy@example.com"
```

### If import errors:
```bash
cd /var/www/html/kahin
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
``` 