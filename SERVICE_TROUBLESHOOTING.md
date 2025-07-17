# Kahinali Service Troubleshooting Guide

## Current Issue
The kahinali.service is in "activating (auto-restart)" state with exit code 2, indicating the service keeps failing to start.

## Manual Fix Steps

### 1. Connect to Server
```bash
ssh root@185.27.134.10
```

### 2. Check Service Status
```bash
systemctl status kahinali.service
```

### 3. Check Service Logs
```bash
journalctl -u kahinali.service -n 50 --no-pager
```

### 4. Stop the Service
```bash
systemctl stop kahinali.service
```

### 5. Check Directory Structure
```bash
ls -la /var/www/html/kahin/
```

### 6. Verify main.py Exists
```bash
ls -la /var/www/html/kahin/main.py
```

### 7. Check Virtual Environment
```bash
ls -la /var/www/html/kahin/venv/
```

### 8. Test Python Script Manually
```bash
cd /var/www/html/kahin
source venv/bin/activate
python main.py --help
```

### 9. Check Service Configuration
```bash
cat /etc/systemd/system/kahinali.service
```

### 10. Common Issues and Solutions

#### Issue: Virtual Environment Not Found
```bash
cd /var/www/html/kahin
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Issue: Missing Dependencies
```bash
cd /var/www/html/kahin
source venv/bin/activate
pip install -r requirements.txt
```

#### Issue: Wrong Python Path
Edit the service file:
```bash
nano /etc/systemd/system/kahinali.service
```

Ensure the ExecStart line points to the correct Python path:
```
ExecStart=/var/www/html/kahin/venv/bin/python main.py
```

#### Issue: Working Directory
Make sure the service file has the correct working directory:
```
[Service]
Type=simple
User=root
WorkingDirectory=/var/www/html/kahin
ExecStart=/var/www/html/kahin/venv/bin/python main.py
Restart=always
RestartSec=10
```

### 11. Reload and Restart
```bash
systemctl daemon-reload
systemctl enable kahinali.service
systemctl start kahinali.service
```

### 12. Verify Fix
```bash
systemctl status kahinali.service
journalctl -u kahinali.service -f
```

## Alternative: Run the Script
If you have access to the server, you can run the fix_service.sh script:
```bash
chmod +x fix_service.sh
./fix_service.sh
```

## Expected Service Configuration
The service file should look like this:
```ini
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

[Install]
WantedBy=multi-user.target
``` 