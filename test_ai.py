from modules.ai_model import AIModel
from modules.data_collector import DataCollector
from modules.technical_analysis import TechnicalAnalysis
from modules.technical_analysis import FIXED_FEATURE_LIST

print("AI Model test ediliyor...")

try:
    # AI Model yükle
    ai = AIModel()
    print("✓ AI Model yüklendi")
    
    # Data Collector yükle
    dc = DataCollector()
    print("✓ Data Collector yüklendi")
    
    # Technical Analysis yükle
    ta = TechnicalAnalysis()
    print("✓ Technical Analysis yüklendi")
    
    # BTC verisi al
    data = dc.get_historical_data('BTC/USDT', '1h', 100)
    print(f"✓ BTC/USDT verisi alındı: {len(data)} satır")
    
    # Technical analysis yap
    ta_data = ta.calculate_all_indicators(data)
    print(f"✓ Technical analysis tamamlandı: {ta_data.shape}")

    # Kolonları FIXED_FEATURE_LIST ile sırala ve eksik olanları sıfırla
    for col in FIXED_FEATURE_LIST:
        if col not in ta_data.columns:
            ta_data[col] = 0
    ta_data = ta_data[FIXED_FEATURE_LIST]

    # AI tahmin yap
    result = ai.predict(ta_data)
    print(f"✓ AI Tahmin: {result}")
    
except Exception as e:
    print(f"❌ Hata: {e}")
    import traceback
    traceback.print_exc() 