import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Feature columns listesi
feature_cols = [
    'open', 'high', 'low', 'close', 'volume', 'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200', 
    'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200', 'macd', 'macd_signal', 'macd_histogram', 
    'rsi_7', 'rsi_14', 'rsi_21', 'stoch_k', 'stoch_d', 'bb_middle', 'bb_upper', 'bb_lower', 
    'bb_width', 'bb_percent', 'atr', 'obv', 'vwap', 'adx_pos', 'adx_neg', 'adx', 'cci', 'mfi', 
    'williams_r', 'psar', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span', 
    'keltner_ema', 'keltner_upper', 'keltner_lower', 'volume_roc', 'volume_ma', 'volume_ratio', 
    'roc', 'momentum', 'price_roc', 'historical_volatility', 'true_range', 'volatility_ratio', 
    'price_change', 'price_change_5', 'price_change_10', 'return_5', 'return_10', 'return_20', 
    'cumulative_return', 'momentum_5', 'momentum_10', 'volatility', 'volatility_5', 'volatility_10', 
    'volatility_20', 'volume_ma_5', 'volume_ma_10', 'dynamic_threshold', 'label_5', 'label_10', 
    'label_20', 'label_dynamic', 'day_of_week', 'hour', 'doji', 'hammer', 'shooting_star', 
    'support_level', 'resistance_level', 'price_vs_support', 'price_vs_resistance', 'volume_sma', 
    'high_volume', 'body_size', 'upper_shadow', 'lower_shadow', 'momentum_ma', 'momentum_trend', 
    'breakout_up', 'breakout_down', 'price_range', 'consolidation', 'volatility_ma', 'trend_strength', 
    'trend_ma', 'range', 'range_ma', 'market_regime', 'price_change_20', 'volatility_50', 'momentum_20', 
    'volume_ma_20', 'volume_trend', 'rsi_trend', 'rsi_momentum', 'macd_strength', 'macd_trend', 
    'bb_position', 'bb_squeeze', 'future_close_5', 'future_close_10', 'future_close_20', 'future_close_30', 
    'return_30', 'volatility_30', 'dynamic_threshold_5', 'dynamic_threshold_10', 'dynamic_threshold_20', 
    'dynamic_threshold_30', 'label_30', 'trend_5', 'trend_20', 'trend_label', 'momentum_label'
][:125]  # Tam olarak 125 feature

# Feature columns dosyasını kaydet
with open('models/feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
print(f"Feature columns kaydedildi: {len(feature_cols)} feature")

# Scaler oluştur ve kaydet
scaler = StandardScaler()
# Dummy data ile fit et (gerçek veri ile değiştirilecek)
dummy_data = np.random.randn(100, 125)
scaler.fit(dummy_data)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler kaydedildi")

print("Eksik model dosyaları oluşturuldu!") 