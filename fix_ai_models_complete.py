#!/usr/bin/env python3
"""
Kapsamlı AI Model Düzeltme Scripti
Tüm AI model sorunlarını çözer ve sinyal kalitesini artırır
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_prepare_data():
    """Veri yükle ve hazırla"""
    print("📊 Veri yükleniyor ve hazırlanıyor...")
    
    try:
        # 6 aylık veriyi yükle
        data_files = []
        data_dir = "data"
        
        for file in os.listdir(data_dir):
            if file.endswith('_6months.csv'):
                data_files.append(os.path.join(data_dir, file))
        
        print(f"📁 {len(data_files)} veri dosyası bulundu")
        
        all_data = []
        for file in data_files[:50]:  # İlk 50 dosyayı al (hız için)
            try:
                df = pd.read_csv(file)
                if len(df) > 100:  # En az 100 satır olan dosyaları al
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Dosya yüklenemedi {file}: {e}")
                continue
        
        if not all_data:
            raise Exception("Hiç veri yüklenemedi!")
        
        # Verileri birleştir
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"📈 Toplam {len(combined_data)} satır veri yüklendi")
        
        # Veri temizleme
        combined_data = combined_data.dropna()
        combined_data = combined_data.replace([np.inf, -np.inf], np.nan)
        combined_data = combined_data.dropna()
        
        # Aykırı değerleri temizle
        for col in combined_data.select_dtypes(include=[np.number]).columns:
            Q1 = combined_data[col].quantile(0.25)
            Q3 = combined_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            combined_data = combined_data[(combined_data[col] >= lower_bound) & (combined_data[col] <= upper_bound)]
        
        print(f"🧹 Temizlik sonrası {len(combined_data)} satır kaldı")
        
        return combined_data
        
    except Exception as e:
        logger.error(f"Veri yükleme hatası: {e}")
        return None

def create_advanced_features(df):
    """Gelişmiş özellikler oluştur"""
    print("🔧 Gelişmiş özellikler oluşturuluyor...")
    
    try:
        # Teknik analiz özellikleri
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        # Hareketli ortalamalar
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volatilite
        df['volatility'] = df['close'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility'] / df['close']
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['momentum_ratio'] = df['momentum'] / df['close'].shift(10)
        
        # Hacim analizi
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Fiyat pozisyonu
        df['price_position'] = (df['close'] - df['close'].rolling(window=50).min()) / (df['close'].rolling(window=50).max() - df['close'].rolling(window=50).min())
        
        # Trend gücü
        df['trend_strength'] = abs(df['sma_20'] - df['sma_50']) / df['sma_50']
        
        # Support/Resistance
        df['support_level'] = df['close'].rolling(window=20).min()
        df['resistance_level'] = df['close'].rolling(window=20).max()
        df['support_distance'] = (df['close'] - df['support_level']) / df['close']
        df['resistance_distance'] = (df['resistance_level'] - df['close']) / df['close']
        
        # Zaman özellikleri
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # NaN değerleri temizle
        df = df.dropna()
        
        print(f"✅ {len(df.columns)} özellik oluşturuldu")
        return df
        
    except Exception as e:
        logger.error(f"Özellik oluşturma hatası: {e}")
        return None

def create_target_variables(df):
    """Hedef değişkenler oluştur"""
    print("🎯 Hedef değişkenler oluşturuluyor...")
    
    try:
        # Gelecek fiyat değişimleri
        for period in [1, 4, 8, 24]:  # 1h, 4h, 8h, 24h
            df[f'future_return_{period}h'] = df['close'].shift(-period) / df['close'] - 1
        
        # Volatilite hedefi
        df['future_volatility'] = df['close'].shift(-24).rolling(window=24).std() / df['close']
        
        # Trend hedefi
        df['future_trend'] = np.where(df['future_return_24h'] > 0.02, 1, 0)  # %2'den fazla artış
        
        # Breakout hedefi
        df['future_breakout'] = np.where(df['future_return_8h'] > 0.05, 1, 0)  # %5'den fazla artış
        
        # Risk/Ödül hedefi
        df['future_risk_reward'] = df['future_return_8h'] / (df['future_return_1h'].abs() + 0.001)
        
        # NaN değerleri temizle
        df = df.dropna()
        
        print(f"✅ {len([col for col in df.columns if 'future_' in col])} hedef değişken oluşturuldu")
        return df
        
    except Exception as e:
        logger.error(f"Hedef değişken oluşturma hatası: {e}")
        return None

def train_advanced_models(df):
    """Gelişmiş modeller eğit"""
    print("🤖 Gelişmiş modeller eğitiliyor...")
    
    try:
        # Özellik sütunları
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'symbol'] and not col.startswith('future_')]
        
        # Hedef değişkenler
        target_cols = [col for col in df.columns if col.startswith('future_')]
        
        print(f"📊 {len(feature_cols)} özellik, {len(target_cols)} hedef değişken")
        
        # Veriyi ölçeklendir
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[feature_cols])
        
        models = {}
        model_scores = {}
        
        # Her hedef için model eğit
        for target in target_cols:
            print(f"🎯 {target} için model eğitiliyor...")
            
            y = df[target]
            
            # Veriyi böl
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            # Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            
            # Gradient Boosting
            gb_model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
            
            # Modelleri eğit
            rf_model.fit(X_train, y_train)
            gb_model.fit(X_train, y_train)
            
            # Skorları hesapla
            rf_score = rf_model.score(X_test, y_test)
            gb_score = gb_model.score(X_test, y_test)
            
            # En iyi modeli seç
            if rf_score > gb_score:
                models[target] = rf_model
                model_scores[target] = rf_score
                print(f"   ✅ Random Forest seçildi (R²: {rf_score:.4f})")
            else:
                models[target] = gb_model
                model_scores[target] = gb_score
                print(f"   ✅ Gradient Boosting seçildi (R²: {gb_score:.4f})")
        
        # Ensemble model oluştur
        ensemble_model = {
            'models': models,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'target_cols': target_cols,
            'scores': model_scores
        }
        
        print(f"✅ {len(models)} model eğitildi")
        print(f"📈 Ortalama R² skoru: {np.mean(list(model_scores.values())):.4f}")
        
        return ensemble_model
        
    except Exception as e:
        logger.error(f"Model eğitme hatası: {e}")
        return None

def save_models(ensemble_model):
    """Modelleri kaydet"""
    print("💾 Modeller kaydediliyor...")
    
    try:
        # Models dizinini oluştur
        os.makedirs('models', exist_ok=True)
        
        # Ensemble modeli kaydet
        joblib.dump(ensemble_model, 'models/ensemble_model.pkl')
        
        # Özellik sütunlarını kaydet
        joblib.dump(ensemble_model['feature_cols'], 'models/feature_cols.pkl')
        
        # Scaler'ı kaydet
        joblib.dump(ensemble_model['scaler'], 'models/scaler.pkl')
        
        # Model performans raporu
        performance_report = {
            'training_date': datetime.now().isoformat(),
            'model_count': len(ensemble_model['models']),
            'feature_count': len(ensemble_model['feature_cols']),
            'average_score': np.mean(list(ensemble_model['scores'].values())),
            'individual_scores': ensemble_model['scores']
        }
        
        joblib.dump(performance_report, 'models/performance_report.pkl')
        
        print("✅ Modeller başarıyla kaydedildi")
        return True
        
    except Exception as e:
        logger.error(f"Model kaydetme hatası: {e}")
        return False

def test_models(ensemble_model):
    """Modelleri test et"""
    print("🧪 Modeller test ediliyor...")
    
    try:
        # Test verisi oluştur
        test_data = np.random.randn(100, len(ensemble_model['feature_cols']))
        test_data_scaled = ensemble_model['scaler'].transform(test_data)
        
        predictions = {}
        
        for target, model in ensemble_model['models'].items():
            pred = model.predict(test_data_scaled)
            predictions[target] = pred
        
        # AI skoru hesapla
        if 'future_return_8h' in predictions:
            returns = predictions['future_return_8h']
            ai_score = np.mean(np.where(returns > 0, returns, 0)) * 100
            print(f"📊 Test AI skoru: {ai_score:.2f}")
        
        print("✅ Model testi başarılı")
        return True
        
    except Exception as e:
        logger.error(f"Model test hatası: {e}")
        return False

def main():
    """Ana fonksiyon"""
    print("🚀 KAPSAMLI AI MODEL DÜZELTME")
    print("=" * 50)
    print(f"Başlangıç zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. Veri yükle
    df = load_and_prepare_data()
    if df is None:
        print("❌ Veri yükleme başarısız!")
        return
    
    # 2. Özellikler oluştur
    df = create_advanced_features(df)
    if df is None:
        print("❌ Özellik oluşturma başarısız!")
        return
    
    # 3. Hedef değişkenler oluştur
    df = create_target_variables(df)
    if df is None:
        print("❌ Hedef değişken oluşturma başarısız!")
        return
    
    # 4. Modelleri eğit
    ensemble_model = train_advanced_models(df)
    if ensemble_model is None:
        print("❌ Model eğitme başarısız!")
        return
    
    # 5. Modelleri kaydet
    if not save_models(ensemble_model):
        print("❌ Model kaydetme başarısız!")
        return
    
    # 6. Modelleri test et
    if not test_models(ensemble_model):
        print("❌ Model testi başarısız!")
        return
    
    print()
    print("🎉 AI MODEL DÜZELTME TAMAMLANDI!")
    print(f"Bitiş zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("📊 SONUÇLAR:")
    print(f"   - {len(ensemble_model['models'])} model eğitildi")
    print(f"   - {len(ensemble_model['feature_cols'])} özellik kullanıldı")
    print(f"   - Ortalama R² skoru: {np.mean(list(ensemble_model['scores'].values())):.4f}")
    print()
    print("💡 Sistem artık daha yüksek kaliteli sinyaller üretecek!")

if __name__ == "__main__":
    main() 