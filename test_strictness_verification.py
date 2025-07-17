#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistem sıkılığını doğrulama testi
"""

import os
import sys
import logging
from datetime import datetime

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_config_loading():
    """Konfigürasyon dosyasının doğru yüklenip yüklenmediğini test et"""
    print("=" * 60)
    print("🔍 KONFİGÜRASYON YÜKLEME TESTİ")
    print("=" * 60)
    
    try:
        from config import Config
        print("✅ config.py başarıyla yüklendi")
        
        # Config dosyasının konumunu kontrol et
        config_file = os.path.abspath('config.py')
        print(f"📁 Config dosyası: {config_file}")
        
        # Aktif config dosyasının içeriğini kontrol et
        with open('config.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'MIN_SIGNAL_CONFIDENCE = 0.55' in content:
                print("✅ Normal config.py aktif (0.55)")
            elif 'MIN_SIGNAL_CONFIDENCE = 0.65' in content:
                print("✅ Sıkı config.py aktif (0.65)")
            else:
                print("⚠️ Bilinmeyen config değeri")
        
        return Config
        
    except Exception as e:
        print(f"❌ Config yükleme hatası: {e}")
        return None

def test_strict_config():
    """Sıkı config dosyasının varlığını ve değerlerini kontrol et"""
    print("\n" + "=" * 60)
    print("🔒 SIKI KONFİGÜRASYON KONTROLÜ")
    print("=" * 60)
    
    if os.path.exists('config_strict.py'):
        print("✅ config_strict.py dosyası mevcut")
        
        try:
            # Sıkı config'i import et
            sys.path.insert(0, os.getcwd())
            import config_strict
            strict_config = config_strict.Config
            
            print(f"📊 Sıkı config değerleri:")
            print(f"  MIN_SIGNAL_CONFIDENCE: {strict_config.MIN_SIGNAL_CONFIDENCE}")
            print(f"  MIN_AI_SCORE: {strict_config.MIN_AI_SCORE}")
            print(f"  MIN_TA_STRENGTH: {strict_config.MIN_TA_STRENGTH}")
            print(f"  MIN_WHALE_SCORE: {strict_config.MIN_WHALE_SCORE}")
            print(f"  MAX_COINS_TO_TRACK: {strict_config.MAX_COINS_TO_TRACK}")
            print(f"  MAX_SIGNALS_PER_BATCH: {strict_config.MAX_SIGNALS_PER_BATCH}")
            
            return strict_config
            
        except Exception as e:
            print(f"❌ Sıkı config yükleme hatası: {e}")
            return None
    else:
        print("❌ config_strict.py dosyası bulunamadı")
        return None

def compare_configs(normal_config, strict_config):
    """Normal ve sıkı config'leri karşılaştır"""
    print("\n" + "=" * 60)
    print("📊 KONFİGÜRASYON KARŞILAŞTIRMASI")
    print("=" * 60)
    
    if not normal_config or not strict_config:
        print("❌ Karşılaştırma yapılamıyor - config eksik")
        return
    
    print("🔍 Normal Config vs Sıkı Config:")
    print(f"{'Parametre':<25} {'Normal':<10} {'Sıkı':<10} {'Fark':<10}")
    print("-" * 55)
    
    params = [
        ('MIN_SIGNAL_CONFIDENCE', normal_config.MIN_SIGNAL_CONFIDENCE, strict_config.MIN_SIGNAL_CONFIDENCE),
        ('MIN_AI_SCORE', normal_config.MIN_AI_SCORE, strict_config.MIN_AI_SCORE),
        ('MIN_TA_STRENGTH', normal_config.MIN_TA_STRENGTH, strict_config.MIN_TA_STRENGTH),
        ('MIN_WHALE_SCORE', normal_config.MIN_WHALE_SCORE, strict_config.MIN_WHALE_SCORE),
        ('MAX_COINS_TO_TRACK', normal_config.MAX_COINS_TO_TRACK, strict_config.MAX_COINS_TO_TRACK),
        ('MAX_SIGNALS_PER_BATCH', normal_config.MAX_SIGNALS_PER_BATCH, strict_config.MAX_SIGNALS_PER_BATCH),
    ]
    
    for param, normal_val, strict_val in params:
        diff = strict_val - normal_val if isinstance(normal_val, (int, float)) else "N/A"
        print(f"{param:<25} {normal_val:<10} {strict_val:<10} {diff:<10}")

def test_system_strictness():
    """Sistemin gerçek sıkılığını test et"""
    print("\n" + "=" * 60)
    print("🎯 SİSTEM SIKILIK ANALİZİ")
    print("=" * 60)
    
    try:
        from config import Config
        
        # Sıkılık hesaplama
        strictness_score = 0
        max_score = 100
        
        # AI Score sıkılığı (25 puan)
        if Config.MIN_AI_SCORE >= 0.85:
            strictness_score += 25
        elif Config.MIN_AI_SCORE >= 0.65:
            strictness_score += 20
        elif Config.MIN_AI_SCORE >= 0.45:
            strictness_score += 15
        else:
            strictness_score += 10
        
        # TA Strength sıkılığı (25 puan)
        if Config.MIN_TA_STRENGTH >= 0.90:
            strictness_score += 25
        elif Config.MIN_TA_STRENGTH >= 0.70:
            strictness_score += 20
        elif Config.MIN_TA_STRENGTH >= 0.50:
            strictness_score += 15
        else:
            strictness_score += 10
        
        # Signal Confidence sıkılığı (25 puan)
        if Config.MIN_SIGNAL_CONFIDENCE >= 0.65:
            strictness_score += 25
        elif Config.MIN_SIGNAL_CONFIDENCE >= 0.55:
            strictness_score += 20
        elif Config.MIN_SIGNAL_CONFIDENCE >= 0.45:
            strictness_score += 15
        else:
            strictness_score += 10
        
        # Coin sayısı sıkılığı (15 puan)
        if Config.MAX_COINS_TO_TRACK <= 30:
            strictness_score += 15
        elif Config.MAX_COINS_TO_TRACK <= 100:
            strictness_score += 10
        elif Config.MAX_COINS_TO_TRACK <= 200:
            strictness_score += 5
        else:
            strictness_score += 0
        
        # Signal batch sıkılığı (10 puan)
        if Config.MAX_SIGNALS_PER_BATCH <= 1:
            strictness_score += 10
        elif Config.MAX_SIGNALS_PER_BATCH <= 3:
            strictness_score += 7
        elif Config.MAX_SIGNALS_PER_BATCH <= 5:
            strictness_score += 5
        else:
            strictness_score += 0
        
        # Sonuçları göster
        strictness_percentage = (strictness_score / max_score) * 100
        strictness_level = strictness_score / 10
        
        print(f"📊 Sıkılık Puanı: {strictness_score}/{max_score}")
        print(f"📈 Sıkılık Yüzdesi: {strictness_percentage:.1f}%")
        print(f"🔒 Sıkılık Seviyesi: {strictness_level:.1f}/10")
        
        # Seviye belirleme
        if strictness_percentage >= 80:
            level = "🟢 ÇOK SIKI"
        elif strictness_percentage >= 60:
            level = "🟡 ORTA"
        elif strictness_percentage >= 40:
            level = "🟠 GEVŞEK"
        else:
            level = "🔴 ÇOK GEVŞEK"
        
        print(f"🏆 Seviye: {level}")
        
        # Detaylı analiz
        print(f"\n📋 DETAYLI ANALİZ:")
        print(f"  AI Score ({Config.MIN_AI_SCORE}): {'Sıkı' if Config.MIN_AI_SCORE >= 0.65 else 'Orta' if Config.MIN_AI_SCORE >= 0.45 else 'Gevşek'}")
        print(f"  TA Strength ({Config.MIN_TA_STRENGTH}): {'Sıkı' if Config.MIN_TA_STRENGTH >= 0.70 else 'Orta' if Config.MIN_TA_STRENGTH >= 0.50 else 'Gevşek'}")
        print(f"  Signal Confidence ({Config.MIN_SIGNAL_CONFIDENCE}): {'Sıkı' if Config.MIN_SIGNAL_CONFIDENCE >= 0.55 else 'Orta' if Config.MIN_SIGNAL_CONFIDENCE >= 0.45 else 'Gevşek'}")
        print(f"  Coin Sayısı ({Config.MAX_COINS_TO_TRACK}): {'Sıkı' if Config.MAX_COINS_TO_TRACK <= 100 else 'Orta' if Config.MAX_COINS_TO_TRACK <= 200 else 'Gevşek'}")
        print(f"  Signal Batch ({Config.MAX_SIGNALS_PER_BATCH}): {'Sıkı' if Config.MAX_SIGNALS_PER_BATCH <= 3 else 'Orta' if Config.MAX_SIGNALS_PER_BATCH <= 5 else 'Gevşek'}")
        
        return strictness_score, strictness_percentage, strictness_level
        
    except Exception as e:
        print(f"❌ Sıkılık analizi hatası: {e}")
        return None, None, None

def main():
    """Ana test fonksiyonu"""
    print("🚀 KAHIN ULTIMA SIKILIK DOĞRULAMA TESTİ")
    print(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Config yükleme testi
    normal_config = test_config_loading()
    
    # 2. Sıkı config kontrolü
    strict_config = test_strict_config()
    
    # 3. Karşılaştırma
    compare_configs(normal_config, strict_config)
    
    # 4. Sistem sıkılık analizi
    score, percentage, level = test_system_strictness()
    
    # 5. Sonuç özeti
    print("\n" + "=" * 60)
    print("📋 SONUÇ ÖZETİ")
    print("=" * 60)
    
    if score is not None:
        print(f"✅ Sistem sıkılığı doğru şekilde hesaplandı")
        print(f"🎯 Mevcut sıkılık: {level}/10 ({percentage:.1f}%)")
        
        if percentage >= 60:
            print("✅ Sistem yeterince sıkı çalışıyor")
        else:
            print("⚠️ Sistem gevşek çalışıyor, sıkılaştırma önerilir")
    else:
        print("❌ Sıkılık hesaplanamadı")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 