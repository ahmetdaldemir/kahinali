import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from modules.data_collector import DataCollector
from modules.technical_analysis import TechnicalAnalysis
from modules.ai_model import AIModel
from modules.market_analysis import MarketAnalysis
from config import Config
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from keras.models import Sequential, load_model
from keras.layers import LSTM, BatchNormalization, Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import pickle
import joblib
from modules.technical_analysis import FIXED_FEATURE_LIST

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def collect_extended_training_data():
    """Genişletilmiş eğitim verisi toplama"""
    logger.info("Genişletilmiş eğitim verisi toplanıyor...")
    
    data_collector = DataCollector()
    
    # Binance formatında USDT çiftleri
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'UNIUSDT',
        'ATOMUSDT', 'LTCUSDT', 'BCHUSDT', 'XLMUSDT', 'VETUSDT', 'TRXUSDT', 'FILUSDT', 'THETAUSDT', 'XMRUSDT', 'NEOUSDT',
        'ALGOUSDT', 'ICPUSDT', 'FTTUSDT', 'XTZUSDT', 'AAVEUSDT', 'SUSHIUSDT', 'COMPUSDT', 'MKRUSDT', 'SNXUSDT', 'YFIUSDT',
        'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT', 'BONKUSDT', 'WIFUSDT', 'JUPUSDT', 'PYTHUSDT', 'JTOUSDT', 'BOMEUSDT',
        'BOOKUSDT', 'GMEUSDT', 'TRUMPUSDT', 'BIDENUSDT', 'PALANTIRUSDT', 'NVIDIAUSDT', 'TSLAUSDT', 'APPLEUSDT', 'METAUSDT'
    ]
    
    # Çoklu timeframe
    timeframes = ['1h', '4h', '1d']
    
    all_data = []
    total_combinations = len(symbols) * len(timeframes)
    current = 0
    
    for symbol in symbols:
        for timeframe in timeframes:
            current += 1
            logger.info(f"Veri toplama {current}/{total_combinations}: {symbol} {timeframe}")
            
            try:
                # Daha uzun geçmiş veri (365 gün)
                data = data_collector.get_historical_data(symbol, timeframe, 365)
                if data is not None and not data.empty:
                    data['symbol'] = symbol
                    data['timeframe'] = timeframe
                    all_data.append(data)
                    logger.info(f"{symbol} {timeframe}: {len(data)} satır")
                else:
                    logger.warning(f"{symbol} {timeframe} için veri alınamadı")
            except Exception as e:
                logger.error(f"{symbol} {timeframe} veri toplama hatası: {e}")
                continue
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Toplam {len(combined_df)} satır veri toplandı")
        
        # Veriyi kaydet
        combined_df.to_csv('data/extended_training_data.csv', index=False)
        logger.info("Genişletilmiş eğitim verisi kaydedildi: data/extended_training_data.csv")
        
        return combined_df
    else:
        logger.error("Hiç veri toplanamadı!")
        return None

def prepare_advanced_features(df):
    """Gelişmiş feature hazırlama"""
    logger.info("Gelişmiş feature'lar hazırlanıyor...")
    
    if df is None or df.empty:
        logger.error("Input DataFrame is empty or None")
        return None
    
    logger.info(f"Input data shape: {df.shape}")
    
    ai_model = AIModel()
    technical_analysis = TechnicalAnalysis()
    
    try:
        # 1. Etiketleri oluştur (çoklu horizon)
        logger.info("Çoklu horizon etiketleri oluşturuluyor...")
        df = ai_model.create_labels(df, threshold=0.02, horizon=10)
        logger.info(f"Etiketler oluşturuldu: {df.shape}")
        
        if df is None or df.empty:
            logger.error("DataFrame etiketleme sonrası boş")
            return None
        
        # 2. Teknik analiz (tüm indikatörler)
        logger.info("Teknik analiz uygulanıyor...")
        df = technical_analysis.calculate_all_indicators(df)
        logger.info(f"Teknik analiz tamamlandı: {df.shape}")
        
        # 3. Gelişmiş pattern recognition
        logger.info("Pattern recognition uygulanıyor...")
        df = technical_analysis.calculate_advanced_patterns(df)
        logger.info(f"Pattern recognition tamamlandı: {df.shape}")
        
        # 4. Market analizi
        logger.info("Market analizi uygulanıyor...")
        market_analysis = MarketAnalysis()
        df = market_analysis.analyze_market_regime(df)
        logger.info(f"Market analizi tamamlandı: {df.shape}")
        
        # 5. Feature engineering
        logger.info("Feature engineering uygulanıyor...")
        df = ai_model.engineer_features(df)
        logger.info(f"Feature engineering tamamlandı: {df.shape}")

        # --- LABEL SÜTUNU KONTROLÜ ---
        if 'label' not in df.columns:
            logger.warning("label sütunu kaybolmuş, tekrar ekleniyor...")
            df = ai_model.create_labels(df, threshold=0.02, horizon=10)
            logger.info(f"label sütunu tekrar eklendi: {df.shape}")
        
        # 6. NaN değerleri temizle
        logger.info("NaN değerler temizleniyor...")
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        logger.info(f"NaN temizleme: {initial_rows} -> {final_rows} satır")
        
        if df.empty:
            logger.error("DataFrame NaN temizleme sonrası boş")
            return None
        
        # 7. Feature seçimi
        logger.info("Feature seçimi yapılıyor...")
        # Sadece feature_cols listesinden çıkar, DataFrame'den silme
        feature_cols = [col for col in df.columns if col not in ['symbol', 'timeframe', 'timestamp', 'label']]
        logger.info(f"Toplam {len(feature_cols)} feature seçildi")
        
        # 8. Veri kalitesi kontrolü
        logger.info("Veri kalitesi kontrol ediliyor...")
        for col in feature_cols:
            if df[col].isnull().sum() > 0:
                logger.warning(f"{col} sütununda {df[col].isnull().sum()} NaN değer var")
        
        # market_regime sütununu sayısal olarak kodla
        if 'market_regime' in df.columns:
            regime_map = {'NEUTRAL': 0, 'RANGING': 1, 'LOW_VOL': 2, 'TRENDING': 3, 'VOLATILE': 4}
            df['market_regime'] = df['market_regime'].map(regime_map).fillna(0).astype(int)

        # symbol ve timeframe sütunlarını model eğitiminden çıkar
        drop_cols = [col for col in ['symbol', 'timeframe'] if col in df.columns]  # 'label' asla silinmemeli
        if drop_cols:
            df = df.drop(columns=drop_cols)
        
        # 9. Veri kaydet
        df.to_csv('data/processed_training_data.csv', index=False)
        logger.info("İşlenmiş eğitim verisi kaydedildi: data/processed_training_data.csv")
        
        return df
        
    except Exception as e:
        logger.error(f"Feature hazırlama hatası: {e}")
        return None

def train_advanced_models(df):
    """Gelişmiş model eğitimi"""
    logger.info("Gelişmiş model eğitimi başlıyor...")
    
    if df is None or df.empty:
        logger.error("Eğitim verisi boş")
        return False
    
    try:
        # Feature'ları hazırla
        # Eğitimde kullanılacak feature listesi
        feature_cols = FIXED_FEATURE_LIST
        # Model ve scaler eğitiminden sonra feature_cols.pkl dosyasını güncelle
        if os.path.exists(os.path.join(Config.MODELS_DIR, 'feature_cols.pkl')):
            os.remove(os.path.join(Config.MODELS_DIR, 'feature_cols.pkl'))
        joblib.dump(feature_cols, os.path.join(Config.MODELS_DIR, 'feature_cols.pkl'))
        
        # Veriyi böl
        X = df[feature_cols]
        y = df['label']
        
        # Stratified split (label dağılımını koru)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Eğitim seti: {X_train.shape}, Test seti: {X_test.shape}")
        logger.info(f"Label dağılımı - Eğitim: {y_train.value_counts().to_dict()}")
        logger.info(f"Label dağılımı - Test: {y_test.value_counts().to_dict()}")
        
        # Eğitim ve test setlerinde NaN ve sonsuz değer temizliği
        X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
        X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna()

        # Eğer y_train ve y_test ile indeks uyumsuzluğu olursa, onları da X_train.index ve X_test.index ile eşitle:
        y_train = y_train.loc[X_train.index]
        y_test = y_test.loc[X_test.index]
        
        # 1. LSTM Modeli (Gelişmiş)
        logger.info("Gelişmiş LSTM modeli eğitiliyor...")
        train_lstm_advanced(X_train, y_train, X_test, y_test, feature_cols)
        
        # 2. Random Forest (basit)
        logger.info("Random Forest (basit) eğitiliyor...")
        train_rf_simple(X_train, y_train, X_test, y_test, feature_cols)
        
        # 3. Gradient Boosting (basit)
        logger.info("Gradient Boosting (basit) eğitiliyor...")
        train_gb_simple(X_train, y_train, X_test, y_test, feature_cols)
        
        # 4. Ensemble Model
        logger.info("Ensemble model eğitiliyor...")
        train_ensemble_model(X_train, y_train, X_test, y_test, feature_cols)
        
        logger.info("Tüm modeller başarıyla eğitildi!")
        return True
        
    except Exception as e:
        logger.error(f"Model eğitimi hatası: {e}")
        return False

def train_lstm_advanced(X_train, y_train, X_test, y_test, feature_cols):
    """Gelişmiş LSTM eğitimi"""
    try:
        # Veri tiplerini düzelt ve NumPy array'e çevir
        X_train = X_train.values.astype(np.float32)
        X_test = X_test.values.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
        
        # LSTM için 3D reshape
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        # Gelişmiş model mimarisi
        model = Sequential([
            LSTM(128, input_shape=(1, X_train.shape[1]), return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        # Optimizer ve callbacks
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint('models/lstm_model.h5', save_best_only=True, monitor='val_accuracy')
        ]
        
        # Eğitim
        history = model.fit(
            X_train_lstm, y_train,
            validation_data=(X_test_lstm, y_test),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Değerlendirme
        test_loss, test_accuracy = model.evaluate(X_test_lstm, y_test, verbose=0)
        logger.info(f"LSTM Test Accuracy: {test_accuracy:.4f}")
        
        # Model kaydet
        model.save('models/lstm_model.h5')
        logger.info("LSTM modeli kaydedildi")
        
        # Feature listesini kaydet
        joblib.dump(feature_cols, os.path.join(Config.MODELS_DIR, 'feature_cols.pkl'))
        
    except Exception as e:
        logger.error(f"LSTM advanced eğitim hatası: {e}")
        import traceback
        logger.error(traceback.format_exc())

def train_rf_simple(X_train, y_train, X_test, y_test, feature_cols):
    """Basit Random Forest eğitimi"""
    try:
        print("Random Forest (basit) eğitiliyor...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # Metrikler
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Random Forest Test Accuracy: {accuracy:.4f}")
        print(f"Random Forest Precision: {precision:.4f}")
        print(f"Random Forest Recall: {recall:.4f}")
        print(f"Random Forest F1-Score: {f1:.4f}")
        
        # Modeli kaydet
        joblib.dump(rf, os.path.join(Config.MODELS_DIR, 'rf_model.pkl'))
        print("Random Forest modeli kaydedildi")
        
        return rf
        
    except Exception as e:
        print(f"Random Forest eğitim hatası: {e}")
        return None

def train_gb_simple(X_train, y_train, X_test, y_test, feature_cols):
    """Basit Gradient Boosting eğitimi"""
    try:
        print("Gradient Boosting (basit) eğitiliyor...")
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        gb.fit(X_train, y_train)
        y_pred = gb.predict(X_test)
        
        # Metrikler
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Gradient Boosting Test Accuracy: {accuracy:.4f}")
        print(f"Gradient Boosting Precision: {precision:.4f}")
        print(f"Gradient Boosting Recall: {recall:.4f}")
        print(f"Gradient Boosting F1-Score: {f1:.4f}")
        
        # Modeli kaydet
        joblib.dump(gb, os.path.join(Config.MODELS_DIR, 'gb_model.pkl'))
        print("Gradient Boosting modeli kaydedildi")
        
        return gb
        
    except Exception as e:
        print(f"Gradient Boosting eğitim hatası: {e}")
        return None

def train_ensemble_model(X_train, y_train, X_test, y_test, feature_cols):
    """Ensemble model eğitimi"""
    try:
        # Base modelleri yükle
        with open('models/rf_model.pkl', 'rb') as f:
            rf_model = joblib.load(f)
        
        with open('models/gb_model.pkl', 'rb') as f:
            gb_model = joblib.load(f)
        
        # LSTM modelini yükle
        lstm_model = load_model('models/lstm_model.h5')
        
        # LSTM için veriyi hazırla
        lookback = 60
        X_lstm_test, _ = prepare_lstm_data(X_test, y_test, lookback)
        # LSTM inputunu (batch, 1, feature) shape'e zorla
        if len(X_lstm_test.shape) == 3 and X_lstm_test.shape[1] != 1:
            X_lstm_test = X_lstm_test[:, -1:, :]
        # Tahminler
        rf_pred = rf_model.predict_proba(X_test)[:, 1]
        gb_pred = gb_model.predict_proba(X_test)[:, 1]
        lstm_pred = lstm_model.predict(X_lstm_test).flatten()
        # Broadcast hatasını önle: tüm dizileri minimum uzunluğa eşitle
        min_len = min(len(y_test), len(rf_pred), len(gb_pred), len(lstm_pred))
        y_test = y_test[:min_len]
        rf_pred = rf_pred[:min_len]
        gb_pred = gb_pred[:min_len]
        lstm_pred = lstm_pred[:min_len]
        
        # Ensemble tahmin (ağırlıklı ortalama)
        ensemble_pred = (0.4 * rf_pred + 0.4 * gb_pred + 0.2 * lstm_pred)
        ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
        
        # Performans değerlendirme
        accuracy = accuracy_score(y_test, ensemble_pred_binary)
        precision = precision_score(y_test, ensemble_pred_binary)
        recall = recall_score(y_test, ensemble_pred_binary)
        f1 = f1_score(y_test, ensemble_pred_binary)
        
        logger.info(f"Ensemble Model - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        # Ensemble modeli kaydet
        ensemble_model = {
            'rf_model': rf_model,
            'gb_model': gb_model,
            'lstm_model': lstm_model,
            'weights': [0.4, 0.4, 0.2]
        }
        
        with open('models/ensemble_model.pkl', 'wb') as f:
            pickle.dump(ensemble_model, f)
        
    except Exception as e:
        logger.error(f"Ensemble model eğitim hatası: {e}")

def prepare_lstm_data(X, y, lookback=60):
    """LSTM için veriyi hazırla"""
    try:
        X_lstm, y_lstm = [], []
        
        for i in range(lookback, len(X)):
            X_lstm.append(X.iloc[i-lookback:i].values)
            y_lstm.append(y.iloc[i])
        
        return np.array(X_lstm), np.array(y_lstm)
        
    except Exception as e:
        logger.error(f"LSTM veri hazırlama hatası: {e}")
        return np.array([]), np.array([])

def test_models_on_live_data():
    """Canlı veri üzerinde model testi"""
    logger.info("Canlı veri üzerinde model testi başlıyor...")
    
    # Test edilecek coinler
    test_symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']
    
    ai_model = AIModel()
    data_collector = DataCollector()
    
    for symbol in test_symbols:
        try:
            logger.info(f"{symbol} test ediliyor...")
            
            # Son 100 veriyi al
            data = data_collector.get_historical_data(symbol, '1h', 100)
            if data is None or data.empty:
                logger.warning(f"{symbol} için veri alınamadı")
                continue
            
            # Teknik analiz uygula
            technical_analysis = TechnicalAnalysis()
            data = technical_analysis.calculate_all_indicators(data)
            
            # Tahmin yap
            score = ai_model.predict(data)
            
            logger.info(f"{symbol} skoru: {score:.3f}")
            
            if score > 0.7:
                logger.info(f"{symbol} - Yüksek skor! Potansiyel sinyal")
            elif score > 0.5:
                logger.info(f"{symbol} - Orta skor")
            else:
                logger.info(f"{symbol} - Düşük skor")
                
        except Exception as e:
            logger.error(f"{symbol} test hatası: {e}")

def main():
    """Ana eğitim süreci"""
    logger.info("Gelişmiş model eğitimi başlıyor...")
    logger.info("=" * 80)
    
    # Config directories oluştur
    Config.create_directories()
    
    # 1. Genişletilmiş veri topla
    df = collect_extended_training_data()
    if df is None:
        logger.error("Veri toplanamadı, eğitim durduruluyor")
        return
    
    # 2. Gelişmiş feature'ları hazırla
    df = prepare_advanced_features(df)
    if df is None or df.empty:
        logger.error("Feature hazırlama başarısız, eğitim durduruluyor")
        return
    
    # 3. Gelişmiş modelleri eğit
    success = train_advanced_models(df)
    if not success:
        logger.error("Model eğitimi başarısız")
        return
    
    # 4. Canlı veri üzerinde test et
    test_models_on_live_data()
    
    logger.info("Gelişmiş model eğitimi tamamlandı!")
    logger.info("=" * 80)

if __name__ == "__main__":
    main() 