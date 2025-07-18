#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import requests
import time
import subprocess
import signal
import psutil

# Logging ayarla
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test_web_dashboard.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def test_flask_import():
    """Flask uygulamasını import etmeyi test et"""
    try:
        # Proje dizinine git
        os.chdir('/var/www/html/kahin')
        
        # Flask uygulamasını import et
        from app.web import app
        logger.info("✅ Flask uygulaması başarıyla import edildi")
        return True
    except Exception as e:
        logger.error(f"❌ Flask uygulaması import hatası: {e}")
        return False

def test_flask_run():
    """Flask uygulamasını çalıştırmayı test et"""
    try:
        # Proje dizinine git
        os.chdir('/var/www/html/kahin')
        
        # Flask uygulamasını başlat
        import subprocess
        import time
        
        # Flask uygulamasını arka planda başlat
        process = subprocess.Popen([
            'python3', '-m', 'flask', '--app', 'app.web', 'run', 
            '--host=0.0.0.0', '--port=5000'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Başlamasını bekle
        time.sleep(10)
        
        # Process'in çalışıp çalışmadığını kontrol et
        if process.poll() is None:
            logger.info("✅ Flask uygulaması başarıyla başlatıldı")
            
            # Process'i durdur
            process.terminate()
            process.wait()
            
            return True
        else:
            stdout, stderr = process.communicate()
            logger.error(f"❌ Flask uygulaması başlatılamadı")
            logger.error(f"STDOUT: {stdout.decode()}")
            logger.error(f"STDERR: {stderr.decode()}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Flask çalıştırma hatası: {e}")
        return False

def test_web_endpoints():
    """Web endpoint'lerini test et"""
    try:
        # Localhost test
        logger.info("🌐 Web endpoint'leri test ediliyor...")
        
        # Ana sayfa testi
        response = requests.get('http://localhost:5000/', timeout=10)
        if response.status_code == 200:
            logger.info("✅ Ana sayfa erişilebilir")
        else:
            logger.error(f"❌ Ana sayfa hatası: {response.status_code}")
        
        # API testi
        response = requests.get('http://localhost:5000/api/signals', timeout=10)
        if response.status_code == 200:
            logger.info("✅ API endpoint erişilebilir")
            data = response.json()
            logger.info(f"📊 API yanıtı: {len(data)} sinyal")
        else:
            logger.error(f"❌ API endpoint hatası: {response.status_code}")
        
        # Performance API testi
        response = requests.get('http://localhost:5000/api/performance', timeout=10)
        if response.status_code == 200:
            logger.info("✅ Performance API erişilebilir")
        else:
            logger.error(f"❌ Performance API hatası: {response.status_code}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        logger.error("❌ Web dashboard'a bağlanılamıyor")
        return False
    except Exception as e:
        logger.error(f"❌ Web endpoint test hatası: {e}")
        return False

def check_port_5000():
    """Port 5000'i kontrol et"""
    try:
        import socket
        
        # Port 5000'i kontrol et
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 5000))
        sock.close()
        
        if result == 0:
            logger.info("✅ Port 5000 açık")
            return True
        else:
            logger.error("❌ Port 5000 kapalı")
            return False
            
    except Exception as e:
        logger.error(f"❌ Port kontrolü hatası: {e}")
        return False

def check_flask_process():
    """Flask process'ini kontrol et"""
    try:
        # Flask process'lerini bul
        flask_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any('flask' in str(arg).lower() for arg in cmdline):
                    flask_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if flask_processes:
            logger.info(f"✅ {len(flask_processes)} Flask process bulundu")
            for proc in flask_processes:
                logger.info(f"   - PID: {proc['pid']}, CMD: {' '.join(proc['cmdline'])}")
            return True
        else:
            logger.warning("⚠️ Flask process bulunamadı")
            return False
            
    except Exception as e:
        logger.error(f"❌ Process kontrolü hatası: {e}")
        return False

def test_systemd_service():
    """Systemd servisini test et"""
    try:
        # Systemd servis durumunu kontrol et
        result = subprocess.run(['systemctl', 'status', 'kahinali-web.service'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Systemd servisi aktif")
            return True
        else:
            logger.error(f"❌ Systemd servisi hatası: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Systemd servis kontrolü hatası: {e}")
        return False

def start_web_dashboard():
    """Web dashboard'u başlat"""
    try:
        logger.info("🚀 Web dashboard başlatılıyor...")
        
        # Proje dizinine git
        os.chdir('/var/www/html/kahin')
        
        # Virtual environment'ı aktifleştir
        activate_script = '/var/www/html/kahin/venv/bin/activate_this.py'
        exec(open(activate_script).read(), {'__file__': activate_script})
        
        # Flask uygulamasını başlat
        from app.web import app
        
        logger.info("✅ Flask uygulaması başlatıldı")
        logger.info("🌐 Web dashboard erişim:")
        logger.info("   - Local: http://localhost:5000")
        logger.info("   - External: http://185.209.228.189:5000")
        logger.info("   - API: http://localhost:5000/api/signals")
        
        # Flask uygulamasını çalıştır
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except Exception as e:
        logger.error(f"❌ Web dashboard başlatma hatası: {e}")
        import traceback
        logger.error(f"Hata detayı: {traceback.format_exc()}")

def main():
    """Ana fonksiyon"""
    logger.info("🚀 Web Dashboard Test Başlatılıyor...")
    
    # 1. Flask import testi
    if not test_flask_import():
        logger.error("Flask import başarısız!")
        sys.exit(1)
    
    # 2. Flask çalıştırma testi
    if not test_flask_run():
        logger.error("Flask çalıştırma başarısız!")
        sys.exit(1)
    
    # 3. Port kontrolü
    if not check_port_5000():
        logger.error("Port 5000 kapalı!")
        sys.exit(1)
    
    # 4. Flask process kontrolü
    check_flask_process()
    
    # 5. Systemd servis kontrolü
    test_systemd_service()
    
    # 6. Web endpoint testi
    if not test_web_endpoints():
        logger.error("Web endpoint testleri başarısız!")
        sys.exit(1)
    
    logger.info("🎉 Web Dashboard Test Tamamlandı!")
    logger.info("✅ Tüm testler başarılı!")

if __name__ == "__main__":
    main() 