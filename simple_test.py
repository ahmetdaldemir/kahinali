import pandas as pd
import numpy as np
from modules.technical_analysis import TechnicalAnalysis

# Test verisi oluştur
data = pd.DataFrame({
    'open': [100, 101, 102, 103, 104],
    'high': [103, 104, 105, 106, 107],
    'low': [99, 100, 101, 102, 103],
    'close': [101, 102, 103, 104, 105],
    'volume': [1000, 1100, 1200, 1300, 1400]
})

# Teknik analiz
ta = TechnicalAnalysis()
result = ta.analyze_technical_signals(data)

print("Test başarılı!")
print(f"Sonuç anahtarları: {list(result.keys())}")
print(f"EMA değerleri: {result.get('ema', {})}")
print(f"RSI değerleri: {result.get('rsi', {})}") 