# Kahinali Deployment Guide

## GitHub Secrets Ayarları

GitHub repository'nizde aşağıdaki secrets'ları ayarlayın:

1. GitHub repository'nize gidin
2. Settings > Secrets and variables > Actions
3. "New repository secret" butonuna tıklayın
4. Aşağıdaki secrets'ları ekleyin:

### Gerekli Secrets:

- `HOST`: Sunucunuzun IP adresi (örn: 192.168.1.100)
- `USERNAME`: Sunucu kullanıcı adınız (örn: ubuntu)
- `PASSWORD`: Sunucu şifreniz
- `PORT`: SSH port (genellikle 22)

## Sunucu Hazırlığı

### 1. Sunucuda Python 3.11 Kurulumu

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-pip
```

### 2. Git Kurulumu

```bash
sudo apt install git
```

### 3. Proje Dizini Oluşturma

```bash
cd /home/$USER
git clone https://github.com/ahmetdaldemir/kahinali.git
cd kahinali
```

### 4. İlk Kurulum

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Deployment

### Otomatik Deployment

Her `main` branch'e push yaptığınızda otomatik olarak deploy edilir.

### Manuel Deployment

GitHub Actions sekmesinden "Deploy to Server" workflow'unu manuel olarak çalıştırabilirsiniz.

## Servis Yönetimi

### Servis Durumu Kontrolü

```bash
sudo systemctl status kahinali.service
```

### Servis Logları

```bash
sudo journalctl -u kahinali.service -f
```

### Servisi Yeniden Başlatma

```bash
sudo systemctl restart kahinali.service
```

### Servisi Durdurma

```bash
sudo systemctl stop kahinali.service
```

## Sorun Giderme

### 1. Servis Başlamıyorsa

```bash
sudo journalctl -u kahinali.service -n 50
```

### 2. Python Bağımlılık Sorunları

```bash
cd /home/$USER/kahinali
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Permission Sorunları

```bash
sudo chown -R $USER:$USER /home/$USER/kahinali
chmod +x /home/$USER/kahinali/main.py
```

## Monitoring

### Log Dosyaları

- Application logs: `/home/$USER/kahinali/app.log`
- System logs: `sudo journalctl -u kahinali.service`

### Disk Kullanımı

```bash
df -h
du -sh /home/$USER/kahinali
```

### Memory Kullanımı

```bash
ps aux | grep python
free -h
```

## Güvenlik

### Firewall Ayarları

```bash
sudo ufw allow 22
sudo ufw enable
```

### SSH Güvenliği

```bash
sudo nano /etc/ssh/sshd_config
# PasswordAuthentication no
# PubkeyAuthentication yes
sudo systemctl restart ssh
```

## Backup

### Otomatik Backup Script

```bash
#!/bin/bash
BACKUP_DIR="/home/$USER/backups"
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf "$BACKUP_DIR/kahinali_$DATE.tar.gz" /home/$USER/kahinali
```

## Notlar

- Servis otomatik olarak yeniden başlatılır
- Loglar systemd journal'da saklanır
- Virtual environment kullanılır
- Git pull ile otomatik güncelleme yapılır 