#!/usr/bin/env python3
"""
128 feature'lık güncel feature listesini oluşturur
"""

import os
import joblib
from config import Config

# 128 feature'lık güncel liste
feature_cols = [
    'open', 'high', 'low', 'close', 'volume', 'future_close_5', 'return_5', 'volatility_5', 'dynamic_threshold_5', 'label_5',
    'future_close_10', 'return_10', 'volatility_10', 'dynamic_threshold_10', 'label_10', 'future_close_20', 'return_20', 'volatility_20', 'dynamic_threshold_20', 'label_20',
    'future_close_30', 'return_30', 'volatility_30', 'dynamic_threshold_30', 'label_30', 'return', 'volatility', 'dynamic_threshold', 'label_dynamic',
    'trend_5', 'trend_20', 'trend_label', 'momentum_5', 'momentum_10', 'momentum_label', 'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
    'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200', 'macd', 'macd_signal', 'macd_histogram', 'rsi_7', 'rsi_14', 'rsi_21', 'stoch_k', 'stoch_d',
    'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_percent', 'atr', 'obv', 'vwap', 'adx_pos', 'adx_neg', 'adx', 'cci', 'mfi', 'williams_r',
    'psar', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span', 'keltner_ema', 'keltner_upper', 'keltner_lower',
    'volume_roc', 'volume_ma', 'volume_ratio', 'roc', 'momentum', 'price_roc', 'historical_volatility', 'true_range', 'volatility_ratio',
    'price_change', 'price_change_5', 'price_change_10', 'cumulative_return', 'volume_ma_5', 'volume_ma_10', 'day_of_week', 'hour',
    'doji', 'hammer', 'shooting_star', 'support_level', 'resistance_level', 'price_vs_support', 'price_vs_resistance', 'volume_sma',
    'high_volume', 'body_size', 'upper_shadow', 'lower_shadow', 'momentum_ma', 'momentum_trend', 'breakout_up', 'breakout_down', 'price_range',
    'consolidation', 'volatility_ma', 'trend_strength', 'trend_ma', 'range', 'range_ma', 'market_regime', 'price_change_20', 'volatility_50',
    'momentum_20', 'volume_ma_20', 'volume_trend', 'rsi_trend', 'rsi_momentum', 'macd_strength', 'macd_trend', 'bb_position', 'bb_squeeze'
]

# Models dizinini oluştur
os.makedirs(Config.MODELS_DIR, exist_ok=True)

# Sadece gerçek feature'ları seç (etiket, future, dynamic_threshold, label, trend_label, momentum_label, consolidation, breakout, price_range, vs. hariç)
exclude_keywords = [
    'label', 'future', 'dynamic_threshold', 'trend_label', 'momentum_label', 'consolidation', 'breakout', 'price_range', 'support_level', 'resistance_level', 'price_vs_support', 'price_vs_resistance', 'market_regime', 'day_of_week', 'hour'
]
filtered_feature_cols = [col for col in feature_cols if not any(ex in col for ex in exclude_keywords)]

# Feature columns dosyasını kaydet
feature_cols_path = os.path.join(Config.MODELS_DIR, 'feature_cols.pkl')
joblib.dump(filtered_feature_cols, feature_cols_path)

print(f"✅ Feature columns dosyası oluşturuldu: {feature_cols_path}")
print(f"📊 Toplam feature sayısı: {len(filtered_feature_cols)}")
print(f"🔍 İlk 10 feature: {filtered_feature_cols[:10]}")
print(f"🔍 Son 10 feature: {filtered_feature_cols[-10:]}") 