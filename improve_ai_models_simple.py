import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import pickle
from modules.whale_tracker import get_whale_score, detect_whale_trades
from modules.data_collector import DataCollector
from modules.technical_analysis import TechnicalAnalysis
import os
import glob
import json

# Teknik analiz/feature engineering fonksiyonu (örnek, basit versiyon)
def engineer_features(df, symbol):
    df = df.copy()
    # --- Teknik analiz pattern ve indikatörlerini ekle ---
    ta = TechnicalAnalysis()
    df = ta.analyze_patterns(df)
    # --- Diğer mevcut feature engineering kodları ---
    if 'close' in df.columns:
        df['price_change_5'] = df['close'].pct_change(5)
        df['volatility_5'] = df['close'].pct_change().rolling(5).std()
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    if 'volume' in df.columns:
        df['volume_ma_5'] = df['volume'].rolling(5).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma_5'] + 1e-6)
        # Hacim spike
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_spike'] = df['volume'] / (df['volume_ma_20'] + 1e-6)
    # Price-volume divergence
    if 'close' in df.columns and 'volume' in df.columns:
        df['price_change_1'] = df['close'].pct_change(1)
        df['volume_change_1'] = df['volume'].pct_change(1)
        df['price_volume_divergence'] = df['price_change_1'] - df['volume_change_1']
    # RSI divergence
    if 'rsi_14' in df.columns:
        df['rsi_change_1'] = df['rsi_14'].diff(1)
        df['rsi_divergence'] = df['price_change_1'] - df['rsi_change_1']
    # MACD divergence
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        df['macd_change_1'] = df['macd'].diff(1)
        df['macd_divergence'] = df['price_change_1'] - df['macd_change_1']
    # Whale score ekle
    whale_score = get_whale_score(symbol)
    df['whale_score'] = whale_score
    # Order book feature'ları ekle
    data_collector = DataCollector()
    ob = data_collector.get_order_book(symbol)
    # --- Order Book Mikro Feature'ları ---
    if ob:
        df['bid_volume'] = ob.get('bid_volume', 0)
        df['ask_volume'] = ob.get('ask_volume', 0)
        df['spread'] = ob.get('spread', 0)
        # Yeni mikro feature'lar
        total_volume = ob.get('bid_volume', 0) + ob.get('ask_volume', 0)
        df['order_flow_imbalance'] = (ob.get('bid_volume', 0) - ob.get('ask_volume', 0)) / (total_volume + 1e-6)
        df['market_depth'] = total_volume
        df['liquidity_ratio'] = ob.get('bid_volume', 0) / (ob.get('ask_volume', 0) + 1e-6)
    else:
        df['order_flow_imbalance'] = 0
        df['market_depth'] = 0
        df['liquidity_ratio'] = 0
    # --- Multi-Timeframe Trend Alignment ---
    try:
        # 1h, 4h, 1d verilerini yükle (örnek, dosya isimleri değişebilir)
        df_1h = pd.read_csv(f'data/{symbol.replace("/", "_")}_1h_6months.csv', index_col=0)
        df_4h = pd.read_csv(f'data/{symbol.replace("/", "_")}_4h_6months.csv', index_col=0)
        df_1d = pd.read_csv(f'data/{symbol.replace("/", "_")}_1d_6months.csv', index_col=0)
        ta = TechnicalAnalysis()
        analysis = ta.multi_timeframe_analysis(df_1h, df_4h, df_1d)
        alignment_score = ta.calculate_trend_alignment(analysis)
        df['multi_tf_alignment'] = alignment_score
    except Exception as e:
        df['multi_tf_alignment'] = 0
    # --- Combo Pattern Score ---
    bullish_patterns = [
        'hammer', 'engulfing_bullish', 'morning_star', 'double_bottom', 'triangle_ascending', 'doji'
    ]
    bearish_patterns = [
        'shooting_star', 'engulfing_bearish', 'evening_star', 'double_top', 'triangle_descending'
    ]
    df['bullish_combo'] = df[bullish_patterns].sum(axis=1)
    df['bearish_combo'] = df[bearish_patterns].sum(axis=1)
    # Combo pattern score: bullish - bearish
    df['combo_pattern_score'] = df['bullish_combo'] - df['bearish_combo']
    # --- Pre-Breakout Volatility/Volume Spike ---
    if 'breakout_up' in df.columns:
        df['pre_breakout_volatility_spike'] = (df['breakout_up'].shift(-1) == 1) & (df['volatility_5'] > df['volatility_5'].rolling(100, min_periods=20).quantile(0.85))
        df['pre_breakout_volume_spike'] = (df['breakout_up'].shift(-1) == 1) & (df['volume_spike'] > df['volume_spike'].rolling(100, min_periods=20).quantile(0.85))
        df['pre_breakout_volatility_spike'] = df['pre_breakout_volatility_spike'].astype(int)
        df['pre_breakout_volume_spike'] = df['pre_breakout_volume_spike'].astype(int)
    else:
        df['pre_breakout_volatility_spike'] = 0
        df['pre_breakout_volume_spike'] = 0
    return df

# Veri yükle (örnek: 6 aylık ana coinlerden biri)
symbol = 'ADA/USDT'
data = pd.read_csv('data/ADA_USDT_1h_6months.csv', index_col=0)

data = engineer_features(data, symbol)

# Label oluştur (örnek: 10 bar sonrası %2 getiri)
data['future_close_10'] = data['close'].shift(-10)
data['return_10'] = (data['future_close_10'] - data['close']) / data['close']
data['label'] = (data['return_10'] > 0.02).astype(int)

# NaN temizle
features = data.dropna(subset=['label'])

# Feature ve label ayır
X = features.drop(['future_close_10', 'return_10', 'label'], axis=1, errors='ignore')
y = features['label']

# NaN değerleri sıfırla
X = X.fillna(0)

# En anlamlı feature'ları seç (maksimum 20, feature sayısı 20'den azsa k='all')
k = 20 if X.shape[1] >= 20 else 'all'
selector = SelectKBest(score_func=f_classif, k=k)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# Eğitim/test böl
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

# Modeli eğit
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Tahmin ve rapor
y_pred = rf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1:', f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --- 2) Feature Importance Analizi ---
importances = rf.feature_importances_
feature_importance_report = sorted(zip(selected_features, importances), key=lambda x: x[1], reverse=True)
print('\nÖnemli Feature Sıralaması:')
for feat, score in feature_importance_report:
    print(f'{feat}: {score:.4f}')

# --- 1) Dinamik Threshold/Strictness Sistemi ---
# Son 200 tahminin olasılıklarına göre, en iyi F1 skorunu veren threshold'u bul
from sklearn.metrics import f1_score, precision_recall_curve
probs = rf.predict_proba(X_test)[:,1]
prec, rec, thresholds = precision_recall_curve(y_test, probs)
f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
best_idx = np.argmax(f1s)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
print(f'\nDinamik (Otomatik) En İyi Threshold: {best_threshold:.3f} (F1: {f1s[best_idx]:.3f})')

# --- Gereksiz Feature Temizliği ---
# Feature importance < 0.001 olanları çıkar
min_importance = 0.001
important_features = [feat for feat, score in feature_importance_report if score >= min_importance]
if len(important_features) < len(selected_features):
    print(f'\nGereksiz (önemsiz) featurelar çıkarılıyor: {set(selected_features) - set(important_features)}')
    X = X[important_features]
    # Yeniden seçici ve model eğitimi
    k = len(important_features)
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print('Temizlenmiş Feature ile Accuracy:', accuracy_score(y_test, y_pred))
    print('Temizlenmiş Feature ile Precision:', precision_score(y_test, y_pred))
    print('Temizlenmiş Feature ile Recall:', recall_score(y_test, y_pred))
    print('Temizlenmiş Feature ile F1:', f1_score(y_test, y_pred))
    importances = rf.feature_importances_
    feature_importance_report = sorted(zip(selected_features, importances), key=lambda x: x[1], reverse=True)
    print('\nTemizlenmiş Feature Sıralaması:')
    for feat, score in feature_importance_report:
        print(f'{feat}: {score:.4f}')
    # Dinamik threshold'u tekrar hesapla
    probs = rf.predict_proba(X_test)[:,1]
    prec, rec, thresholds = precision_recall_curve(y_test, probs)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f'\nTemizlenmiş Feature ile Dinamik Threshold: {best_threshold:.3f} (F1: {f1s[best_idx]:.3f})')
    # Model ve feature'ları tekrar kaydet
    joblib.dump(rf, 'models/simple_rf_model.pkl')
    with open('models/simple_rf_features.pkl', 'wb') as f:
        pickle.dump(list(selected_features), f)
    with open('models/simple_rf_threshold.txt', 'w') as f:
        f.write(str(best_threshold))

# --- Threshold'u canlıda otomatik okuyan fonksiyon ---
def get_dynamic_threshold(path='models/simple_rf_threshold.txt'):
    try:
        with open(path, 'r') as f:
            return float(f.read().strip())
    except Exception as e:
        print(f'Threshold okunamadı, varsayılan 0.5 kullanılıyor. Hata: {e}')
        return 0.5

# Model ve feature'ları kaydet
joblib.dump(rf, 'models/simple_rf_model.pkl')
with open('models/simple_rf_features.pkl', 'wb') as f:
    pickle.dump(list(selected_features), f)
with open('models/simple_rf_threshold.txt', 'w') as f:
    f.write(str(best_threshold))

print('Model, feature listesi ve threshold kaydedildi.')

def auto_update_threshold(
    signals_dir='signals/',
    threshold_file='models/simple_rf_threshold.txt',
    window=200,
    min_threshold=0.10,
    max_threshold=0.80,
    step=0.02,
    target_success=0.5
):
    """
    Son window kadar sinyalin başarı oranına göre threshold'u otomatik günceller.
    Başarı oranı düşükse threshold'u düşürür, yüksekse artırır.
    """
    # Son sinyal dosyalarını bul
    files = sorted(glob.glob(os.path.join(signals_dir, 'signal_*.json')))[-window:]
    if not files:
        print('Otomatik threshold güncelleme: Yeterli sinyal yok.')
        return
    success_count = 0
    total = 0
    for f in files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
            # Başarıyı tespit et (ör: 'result' veya 'success' anahtarı varsa)
            if 'success' in data:
                total += 1
                if data['success']:
                    success_count += 1
            elif 'result' in data:
                total += 1
                if data['result'] == 'success':
                    success_count += 1
        except Exception as e:
            continue
    if total == 0:
        print('Otomatik threshold güncelleme: Geçerli sinyal yok.')
        return
    success_rate = success_count / total
    # Mevcut threshold'u oku
    try:
        with open(threshold_file, 'r') as fp:
            threshold = float(fp.read().strip())
    except Exception:
        threshold = 0.35
    # Güncelleme mantığı
    if success_rate < target_success:
        threshold = max(min_threshold, threshold - step)
    elif success_rate > target_success:
        threshold = min(max_threshold, threshold + step)
    # Dosyaya yaz
    with open(threshold_file, 'w') as fp:
        fp.write(f'{threshold:.4f}')
    print(f'Otomatik threshold güncellendi: Başarı oranı={success_rate:.2%}, Yeni threshold={threshold:.4f}')

# Örnek periyodik kullanım (manuel veya cron-job ile çağrılabilir)
if __name__ == '__main__':
    auto_update_threshold() 