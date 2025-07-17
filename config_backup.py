import os
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

class Config:
    # API Keys
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
    
    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # Twitter API
    TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', '')
    TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET', '')
    TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN', '')
    TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET', '')
    
    # Reddit API
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', '')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '')
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'KAHIN_Ultima_Bot/1.0')
    
    # News API
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
    
    # PostgreSQL Database Configuration
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', 5432))
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'kahin_ultima')
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', '3010726904')
    
    # Database URL - PostgreSQL 15 kullan, password'ü URL encode et
    DATABASE_URL = os.getenv('DATABASE_URL', 
                            f'postgresql://{POSTGRES_USER}:{quote_plus(POSTGRES_PASSWORD)}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}')
    
    # Flask
    FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'kahin-ultima-secret-key-2024')
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
    
    # Trading Parameters - Optimize edilmiş kriterler
    MIN_SIGNAL_CONFIDENCE = float(os.getenv('MIN_SIGNAL_CONFIDENCE', 0.25))  # 0.35'ten 0.25'e düşürüldü
    MAX_COINS_TO_TRACK = int(os.getenv('MAX_COINS_TO_TRACK', 100))  # 150'den 100'e düşürüldü (daha kaliteli)
    SIGNAL_TIMEFRAMES = ['1h', '4h', '1d']  # Çoklu timeframe
    
    # AI Model Parameters - Gelişmiş eğitim
    LSTM_LOOKBACK_DAYS = int(os.getenv('LSTM_LOOKBACK_DAYS', 180))  # 120'den 180'e çıkarıldı
    MODEL_RETRAIN_INTERVAL_HOURS = int(os.getenv('MODEL_RETRAIN_INTERVAL_HOURS', 4))  # 6'dan 4'e düşürüldü
    
    # Whale Tracking - Daha hassas
    WHALE_THRESHOLD_USDT = float(os.getenv('WHALE_THRESHOLD_USDT', 50000))  # 25k'dan 50k'ya çıkarıldı
    
    # Social Media - Optimize edilmiş
    SOCIAL_MEDIA_UPDATE_INTERVAL_MINUTES = int(os.getenv('SOCIAL_MEDIA_UPDATE_INTERVAL_MINUTES', 10))  # 5'ten 10'a çıkarıldı
    
    # News Analysis - Optimize edilmiş
    NEWS_UPDATE_INTERVAL_MINUTES = int(os.getenv('NEWS_UPDATE_INTERVAL_MINUTES', 5))  # 3'ten 5'e çıkarıldı
    
    # Signal Quality - Gelişmiş filtreler
    MIN_AI_SCORE = float(os.getenv('MIN_AI_SCORE', 0.55))  # 0.45'ten 0.55'e yükseltildi
    MIN_TA_STRENGTH = float(os.getenv('MIN_TA_STRENGTH', 0.65))  # 0.55'ten 0.65'e yükseltildi
    MIN_WHALE_SCORE = float(os.getenv('MIN_WHALE_SCORE', 0.35))  # 0.25'ten 0.35'e yükseltildi
    
    # Breakout Prediction - Gelişmiş
    MIN_BREAKOUT_PROBABILITY = float(os.getenv('MIN_BREAKOUT_PROBABILITY', 0.45))  # 0.35'ten 0.45'e yükseltildi
    MAX_BREAKOUT_THRESHOLD = float(os.getenv('MAX_BREAKOUT_THRESHOLD', 0.12))  # 0.15'ten 0.12'ye düşürüldü
    
    # Performance Tracking - Gelişmiş
    SIGNAL_EXPIRY_HOURS = int(os.getenv('SIGNAL_EXPIRY_HOURS', 6))  # 8'den 6'ya düşürüldü
    MIN_PROFIT_PERCENTAGE = float(os.getenv('MIN_PROFIT_PERCENTAGE', 5.0))  # 4.0'dan 5.0'a yükseltildi
    
    # Yeni parametreler - Gelişmiş kriterler
    MAX_SIGNALS_PER_BATCH = int(os.getenv('MAX_SIGNALS_PER_BATCH', 3))  # 5'ten 3'e düşürüldü
    MAX_NEUTRAL_SIGNALS = int(os.getenv('MAX_NEUTRAL_SIGNALS', 0))  # 1'den 0'a düşürüldü
    
    # Gelişmiş AI Model Parametreleri
    USE_ENSEMBLE_MODEL = os.getenv('USE_ENSEMBLE_MODEL', 'True').lower() == 'true'
    FEATURE_SELECTION_ENABLED = os.getenv('FEATURE_SELECTION_ENABLED', 'True').lower() == 'true'
    HYPERPARAMETER_OPTIMIZATION = os.getenv('HYPERPARAMETER_OPTIMIZATION', 'True').lower() == 'true'
    
    # Market Analysis Parametreleri
    MARKET_REGIME_DETECTION = os.getenv('MARKET_REGIME_DETECTION', 'True').lower() == 'true'
    SENTIMENT_ANALYSIS_ENABLED = os.getenv('SENTIMENT_ANALYSIS_ENABLED', 'True').lower() == 'true'
    FEAR_GREED_API_ENABLED = os.getenv('FEAR_GREED_API_ENABLED', 'False').lower() == 'true'
    
    # Advanced Signal Filtering
    ADVANCED_FILTERING_ENABLED = os.getenv('ADVANCED_FILTERING_ENABLED', 'True').lower() == 'true'
    TIME_BASED_FILTERING = os.getenv('TIME_BASED_FILTERING', 'True').lower() == 'true'
    VOLUME_BASED_FILTERING = os.getenv('VOLUME_BASED_FILTERING', 'True').lower() == 'true'
    
    # Performance Monitoring
    REAL_TIME_MONITORING = os.getenv('REAL_TIME_MONITORING', 'True').lower() == 'true'
    AUTO_TUNING_ENABLED = os.getenv('AUTO_TUNING_ENABLED', 'True').lower() == 'true'
    PERFORMANCE_ALERTS = os.getenv('PERFORMANCE_ALERTS', 'True').lower() == 'true'
    
    # Risk Management - Gelişmiş
    MAX_DRAWDOWN_THRESHOLD = float(os.getenv('MAX_DRAWDOWN_THRESHOLD', 0.10))  # 15%'den 10%'a düşürüldü
    MIN_PROFIT_FACTOR = float(os.getenv('MIN_PROFIT_FACTOR', 1.5))  # 1.2'den 1.5'e yükseltildi
    POSITION_SIZE_LIMIT = float(os.getenv('POSITION_SIZE_LIMIT', 0.015))  # 2%'den 1.5%'e düşürüldü
    
    # Technical Analysis Geliştirmeleri
    ADVANCED_INDICATORS_ENABLED = os.getenv('ADVANCED_INDICATORS_ENABLED', 'True').lower() == 'true'
    PATTERN_RECOGNITION_ENABLED = os.getenv('PATTERN_RECOGNITION_ENABLED', 'True').lower() == 'true'
    FIBONACCI_LEVELS_ENABLED = os.getenv('FIBONACCI_LEVELS_ENABLED', 'True').lower() == 'true'
    
    # Model Retraining - Gelişmiş
    ADAPTIVE_RETRAINING = os.getenv('ADAPTIVE_RETRAINING', 'True').lower() == 'true'
    PERFORMANCE_BASED_RETRAINING = os.getenv('PERFORMANCE_BASED_RETRAINING', 'True').lower() == 'true'
    MIN_PERFORMANCE_FOR_RETRAINING = float(os.getenv('MIN_PERFORMANCE_FOR_RETRAINING', 0.6))  # 0.5'ten 0.6'ya yükseltildi
    
    # Data Quality - Gelişmiş
    MIN_DATA_QUALITY_SCORE = float(os.getenv('MIN_DATA_QUALITY_SCORE', 0.85))  # 0.8'den 0.85'e yükseltildi
    OUTLIER_DETECTION_ENABLED = os.getenv('OUTLIER_DETECTION_ENABLED', 'True').lower() == 'true'
    DATA_VALIDATION_ENABLED = os.getenv('DATA_VALIDATION_ENABLED', 'True').lower() == 'true'
    
    # Yeni Gelişmiş Parametreler
    SIGNAL_QUALITY_THRESHOLD = float(os.getenv('SIGNAL_QUALITY_THRESHOLD', 0.7))  # Minimum sinyal kalite skoru
    MARKET_CONDITION_FILTER = os.getenv('MARKET_CONDITION_FILTER', 'True').lower() == 'true'
    VOLATILITY_FILTER = os.getenv('VOLATILITY_FILTER', 'True').lower() == 'true'
    TREND_ALIGNMENT_FILTER = os.getenv('TREND_ALIGNMENT_FILTER', 'True').lower() == 'true'
    
    # AI Model Ağırlıkları
    LSTM_WEIGHT = float(os.getenv('LSTM_WEIGHT', 0.4))  # LSTM model ağırlığı
    RF_WEIGHT = float(os.getenv('RF_WEIGHT', 0.3))      # Random Forest ağırlığı
    GB_WEIGHT = float(os.getenv('GB_WEIGHT', 0.3))      # Gradient Boosting ağırlığı
    
    # Skor Normalizasyon Parametreleri
    SCORE_AMPLIFICATION_FACTOR = float(os.getenv('SCORE_AMPLIFICATION_FACTOR', 2.0))  # Skor güçlendirme faktörü
    MIN_CONFIDENCE_THRESHOLD = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', 0.35))  # Minimum güven eşiği
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/kahin_ultima.log')
    
    # File Paths
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    SIGNALS_DIR = 'signals'
    LOGS_DIR = 'logs'
    
    # Create directories if they don't exist
    @staticmethod
    def create_directories():
        directories = [
            Config.DATA_DIR,
            Config.MODELS_DIR,
            Config.SIGNALS_DIR,
            Config.LOGS_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True) 