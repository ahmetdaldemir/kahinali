#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("🔮 KAHİN Ultima Sistem Testi")
print("=" * 40)

try:
    print("1. Config yükleniyor...")
    from config import Config
    print("✅ Config başarıyla yüklendi")
    
    print("2. Data Collector test ediliyor...")
    from modules.data_collector import DataCollector
    dc = DataCollector()
    print("✅ Data Collector başarıyla yüklendi")
    
    print("3. Technical Analysis test ediliyor...")
    from modules.technical_analysis import TechnicalAnalysis
    ta = TechnicalAnalysis()
    print("✅ Technical Analysis başarıyla yüklendi")
    
    print("4. AI Model test ediliyor...")
    from modules.ai_model import AIModel
    ai = AIModel()
    print("✅ AI Model başarıyla yüklendi")
    
    print("5. Signal Manager test ediliyor...")
    from modules.signal_manager import SignalManager
    sm = SignalManager()
    print("✅ Signal Manager başarıyla yüklendi")
    
    print("6. Performance Analyzer test ediliyor...")
    from modules.performance import PerformanceAnalyzer
    pa = PerformanceAnalyzer()
    print("✅ Performance Analyzer başarıyla yüklendi")
    
    print("\n🎉 Tüm modüller başarıyla yüklendi!")
    print("Sistem kullanıma hazır.")
    
except Exception as e:
    print(f"❌ Hata: {e}")
    print("Lütfen gerekli bağımlılıkları yüklediğinizden emin olun.") 