import joblib
from modules.technical_analysis import FIXED_FEATURE_LIST
import pandas as pd

# Teknik analiz çıktısını örnekle
from modules.technical_analysis import TechnicalAnalysis
from modules.data_collector import DataCollector

symbol = "BTCUSDT"
ta = TechnicalAnalysis()
dc = DataCollector()
df = dc.get_historical_data(symbol, "1h", 100)
df_ta = ta.calculate_all_indicators(df)

print("Teknik analiz çıktısı feature sayısı:", len(df_ta.columns))
print("Teknik analiz feature isimleri:", list(df_ta.columns))
print("FIXED_FEATURE_LIST feature sayısı:", len(FIXED_FEATURE_LIST))

# Model feature listesi
try:
    feature_cols = joblib.load("models/feature_cols.pkl")
    print("Model feature_cols.pkl feature sayısı:", len(feature_cols))
except Exception as e:
    print("feature_cols.pkl okunamadı:", e)

# Farkları göster
print("Eksik olanlar:", set(FIXED_FEATURE_LIST) - set(df_ta.columns))
print("Fazla olanlar:", set(df_ta.columns) - set(FIXED_FEATURE_LIST))