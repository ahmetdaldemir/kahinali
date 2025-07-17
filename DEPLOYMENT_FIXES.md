# Kahinali Service Fix Guide

## Current Issue
The kahinali.service is failing to start with exit code 2, indicating import or configuration issues.

## Root Cause Analysis
1. **Import Errors**: The main.py file has missing or incorrect imports
2. **Multiple `if __name__ == "__main__"` blocks**: This can cause issues
3. **Missing Dependencies**: Some modules might not be properly installed
4. **Path Issues**: The service might not be running from the correct directory

## Manual Fix Steps

### Step 1: Connect to Server
```bash
ssh root@185.27.134.10
```

### Step 2: Stop the Service
```bash
systemctl stop kahinali.service
```

### Step 3: Check Current Directory Structure
```bash
ls -la /var/www/html/kahin/
```

### Step 4: Test the Fixed Main File
```bash
cd /var/www/html/kahin
python3 main_fixed.py --help
```

### Step 5: Test Imports
```bash
cd /var/www/html/kahin
python3 test_imports.py
```

### Step 6: Update the Service File
```bash
nano /etc/systemd/system/kahinali.service
```

Replace the content with:
```ini
[Unit]
Description=Kahinali Trading Bot for kahinapp.com
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/var/www/html/kahin
ExecStart=/var/www/html/kahin/venv/bin/python main_fixed.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment=PYTHONPATH=/var/www/html/kahin

[Install]
WantedBy=multi-user.target
```

### Step 7: Reload and Start Service
```bash
systemctl daemon-reload
systemctl enable kahinali.service
systemctl start kahinali.service
```

### Step 8: Check Service Status
```bash
systemctl status kahinali.service
```

### Step 9: Check Logs
```bash
journalctl -u kahinali.service -f
```

## Alternative: Use the Fixed Main File

If the original main.py has issues, you can use the fixed version:

1. **Backup original**:
```bash
cd /var/www/html/kahin
cp main.py main_original.py
```

2. **Use fixed version**:
```bash
cd /var/www/html/kahin
cp main_fixed.py main.py
```

3. **Test the new main.py**:
```bash
cd /var/www/html/kahin
source venv/bin/activate
python main.py --test
```

## Common Issues and Solutions

### Issue 1: Import Errors
**Symptoms**: `ModuleNotFoundError` or `ImportError`
**Solution**:
```bash
cd /var/www/html/kahin
source venv/bin/activate
pip install -r requirements.txt
```

### Issue 2: Missing Dependencies
**Symptoms**: `AttributeError` or missing classes
**Solution**:
```bash
cd /var/www/html/kahin
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue 3: Path Issues
**Symptoms**: File not found errors
**Solution**:
```bash
cd /var/www/html/kahin
ls -la main.py
ls -la venv/bin/python
```

### Issue 4: Permission Issues
**Symptoms**: Permission denied errors
**Solution**:
```bash
chmod +x /var/www/html/kahin/main.py
chown -R root:root /var/www/html/kahin/
```

## Testing the Fix

### Test 1: Manual Run
```bash
cd /var/www/html/kahin
source venv/bin/activate
python main_fixed.py
```

### Test 2: Import Test
```bash
cd /var/www/html/kahin
source venv/bin/activate
python test_imports.py
```

### Test 3: Service Test
```bash
systemctl status kahinali.service
journalctl -u kahinali.service -n 20
```

## Expected Behavior After Fix

1. **Service Status**: `Active: active (running)`
2. **Logs**: No import errors, successful module loading
3. **Functionality**: Signal generation working properly

## Monitoring Commands

### Check Service Status
```bash
systemctl status kahinali.service
```

### Monitor Logs in Real-time
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

## Rollback Plan

If the fix doesn't work, you can rollback:

```bash
cd /var/www/html/kahin
cp main_original.py main.py
systemctl restart kahinali.service
```

## Emergency Stop

If the service causes issues:

```bash
systemctl stop kahinali.service
systemctl disable kahinali.service
```

## Contact Information

If you need help with the deployment:
- Check the logs: `journalctl -u kahinali.service -n 50`
- Check application logs: `tail -f /var/www/html/kahin/logs/kahin_ultima.log`
- Test manually: `cd /var/www/html/kahin && python test_imports.py` 