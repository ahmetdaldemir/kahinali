from modules.signal_manager import SignalManager
from datetime import datetime
import numpy as np

print("=== GERÇEK SİSTEM DEĞERLERİYLE TEST ===")

try:
    sm = SignalManager()
    print("✓ SignalManager yüklendi")
    
    # Loglardan alınan gerçek değerler
    ai_score = 0.2687  # Ensemble tahmin
    confidence = 0.5204  # Güven skoru
    
    # TA strength hesaplama (main.py'daki fonksiyon)
    def calculate_ta_strength(ta_data, historical_data=None):
        try:
            # RSI strength
            rsi_strength = 0.0
            if 'rsi_14' in ta_data.columns:
                rsi = ta_data['rsi_14'].iloc[-1]
                if rsi < 30 or rsi > 70:
                    rsi_strength = 0.8
                elif rsi < 40 or rsi > 60:
                    rsi_strength = 0.6
                else:
                    rsi_strength = 0.3
            
            # MACD strength
            macd_strength = 0.0
            if 'macd' in ta_data.columns and 'macd_signal' in ta_data.columns:
                macd = ta_data['macd'].iloc[-1]
                macd_signal = ta_data['macd_signal'].iloc[-1]
                if abs(macd - macd_signal) > 0.001:
                    macd_strength = 0.7
                else:
                    macd_strength = 0.3
            
            # Moving average strength
            ma_strength = 0.0
            if 'sma_20' in ta_data.columns and 'sma_50' in ta_data.columns and historical_data is not None:
                sma_20 = ta_data['sma_20'].iloc[-1]
                sma_50 = ta_data['sma_50'].iloc[-1]
                close = historical_data['close'].iloc[-1] if 'close' in historical_data.columns else 0
                
                if close > sma_20 > sma_50:
                    ma_strength = 0.8
                elif close < sma_20 < sma_50:
                    ma_strength = 0.8
                else:
                    ma_strength = 0.4
            
            # Overall TA strength
            ta_strength = (rsi_strength + macd_strength + ma_strength) / 3
            return min(1.0, max(0.0, ta_strength))
            
        except Exception as e:
            print(f"TA strength calculation hatasi: {e}")
            return 0.5
    
    # Test verisi oluştur (gerçek sistemdeki gibi)
    import pandas as pd
    ta_data = pd.DataFrame({
        'rsi_14': [45],  # Normal RSI
        'macd': [0.001],  # MACD
        'macd_signal': [0.0005],  # MACD signal
        'sma_20': [100],  # SMA20
        'sma_50': [98]   # SMA50
    })
    
    historical_data = pd.DataFrame({
        'close': [101]  # Fiyat SMA20'nin üstünde
    })
    
    # TA strength hesapla
    ta_strength = calculate_ta_strength(ta_data, historical_data)
    print(f"TA Strength: {ta_strength}")
    
    # Quality score hesapla (main.py'daki gibi)
    whale_score = 0.5
    social_score = 0.5
    news_score = 0.5
    
    quality_score = (
        ai_score * 0.4 +
        ta_strength * 0.25 +
        whale_score * 0.15 +
        social_score * 0.1 +
        news_score * 0.1
    )
    print(f"Quality Score: {quality_score}")
    
    # Test sinyali oluştur
    test_signal = {
        'symbol': 'TEST/USDT',
        'timeframe': '1h',
        'direction': 'LONG',
        'ai_score': ai_score,
        'ta_strength': ta_strength,
        'whale_score': whale_score,
        'social_score': social_score,
        'news_score': news_score,
        'current_price': 1.0,
        'entry_price': 1.0,
        'timestamp': str(datetime.now()),
        'predicted_gain': 5.0,
        'predicted_duration': '4-8 saat',
        'quality_score': quality_score,
        'quality_checks_passed': True,
        'confidence': confidence
    }
    
    print(f"\nTest sinyali: {test_signal}")
    
    # Config değerlerini kontrol et
    from config import Config
    print(f"\n--- CONFIG DEĞERLERİ ---")
    print(f"MIN_AI_SCORE: {Config.MIN_AI_SCORE}")
    print(f"MIN_TA_STRENGTH: {Config.MIN_TA_STRENGTH}")
    print(f"SIGNAL_QUALITY_THRESHOLD: {Config.SIGNAL_QUALITY_THRESHOLD}")
    print(f"MIN_CONFIDENCE_THRESHOLD: {Config.MIN_CONFIDENCE_THRESHOLD}")
    
    # Manuel filtreleme testi (main.py'daki gibi)
    print(f"\n--- MANUEL FİLTRELEME TESTİ ---")
    
    if test_signal['ai_score'] < Config.MIN_AI_SCORE:
        print(f"❌ AI skoru düşük: {test_signal['ai_score']} < {Config.MIN_AI_SCORE}")
    else:
        print(f"✅ AI skoru geçti: {test_signal['ai_score']} >= {Config.MIN_AI_SCORE}")
    
    if test_signal['ta_strength'] < Config.MIN_TA_STRENGTH:
        print(f"❌ TA skoru düşük: {test_signal['ta_strength']} < {Config.MIN_TA_STRENGTH}")
    else:
        print(f"✅ TA skoru geçti: {test_signal['ta_strength']} >= {Config.MIN_TA_STRENGTH}")
    
    if test_signal['quality_score'] < Config.SIGNAL_QUALITY_THRESHOLD:
        print(f"❌ Quality skoru düşük: {test_signal['quality_score']} < {Config.SIGNAL_QUALITY_THRESHOLD}")
    else:
        print(f"✅ Quality skoru geçti: {test_signal['quality_score']} >= {Config.SIGNAL_QUALITY_THRESHOLD}")
    
    if test_signal['confidence'] < Config.MIN_CONFIDENCE_THRESHOLD:
        print(f"❌ Confidence düşük: {test_signal['confidence']} < {Config.MIN_CONFIDENCE_THRESHOLD}")
    else:
        print(f"✅ Confidence geçti: {test_signal['confidence']} >= {Config.MIN_CONFIDENCE_THRESHOLD}")
    
    # Tüm filtreleri geçip geçmediğini kontrol et
    all_passed = (
        test_signal['ai_score'] >= Config.MIN_AI_SCORE and
        test_signal['ta_strength'] >= Config.MIN_TA_STRENGTH and
        test_signal['quality_score'] >= Config.SIGNAL_QUALITY_THRESHOLD and
        test_signal['confidence'] >= Config.MIN_CONFIDENCE_THRESHOLD
    )
    
    if all_passed:
        print(f"\n🎉 TÜM FİLTRELER GEÇİLDİ! Sinyal kaydedilecek.")
    else:
        print(f"\n❌ BAZI FİLTRELER GEÇİLEMEDİ! Sinyal kaydedilmeyecek.")
    
except Exception as e:
    print(f"❌ Hata: {e}")
    import traceback
    print(f"Hata detayı: {traceback.format_exc()}") 

def safe_format(val):
    try:
        return f"{float(val):.2f}"
    except Exception:
        return "N/A"

print(f"\nTest sinyali detayları:")
print(f"   AI Score: {safe_format(test_signal['ai_score'])}")
print(f"   TA Strength: {safe_format(test_signal['ta_strength'])}")
print(f"   Quality Score: {safe_format(test_signal['quality_score'])}")
print(f"   Confidence: {safe_format(test_signal['confidence'])}") 