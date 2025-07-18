import os
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

class Config:
    # API Keys
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', None)
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', None)
    
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
    DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    DB_PORT = int(os.getenv('POSTGRES_PORT', 5432))
    DB_NAME = os.getenv('POSTGRES_DB', 'kahin_ultima')
    DB_USER = os.getenv('POSTGRES_USER', 'laravel')
    DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'secret')
    
    # Database URL - PostgreSQL 15 kullan, password'ü URL encode et
    DATABASE_URL = os.getenv('DATABASE_URL', 
                            f'postgresql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    
    # Flask
    FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'kahin-ultima-secret-key-2024')
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
    
    # Trading Parameters - SIKI KRİTERLER
    MIN_SIGNAL_CONFIDENCE = 0.30  # Daha gevşek
    MAX_COINS_TO_TRACK = int(os.getenv('MAX_COINS_TO_TRACK', 400))  # 400 coin aktif
    SIGNAL_TIMEFRAMES = ['1h', '4h', '1d']  # Çoklu timeframe
    
    # AI Model Parameters - Gelişmiş eğitim
    LSTM_LOOKBACK_DAYS = int(os.getenv('LSTM_LOOKBACK_DAYS', 180))  # 120'den 180'e çıkarıldı
    MODEL_RETRAIN_INTERVAL_HOURS = int(os.getenv('MODEL_RETRAIN_INTERVAL_HOURS', 4))  # 6'dan 4'e düşürüldü
    
    # Whale Tracking - Daha sıkı
    WHALE_THRESHOLD_USDT = float(os.getenv('WHALE_THRESHOLD_USDT', 200000))  # 100k'dan 200k'ya yükseltildi
    
    # Social Media - Daha az sıklık
    SOCIAL_MEDIA_UPDATE_INTERVAL_MINUTES = int(os.getenv('SOCIAL_MEDIA_UPDATE_INTERVAL_MINUTES', 30))  # 10'dan 30'a çıkarıldı
    
    # News Analysis - Daha az sıklık
    NEWS_UPDATE_INTERVAL_MINUTES = int(os.getenv('NEWS_UPDATE_INTERVAL_MINUTES', 20))  # 5'ten 20'ye çıkarıldı
    
    # Signal Quality - SIKI FİLTRELER
    MIN_AI_SCORE = 0.30  # Daha gevşek
    MIN_TA_STRENGTH = 0.20  # Daha gevşek
    MIN_WHALE_SCORE = 0.20  # Daha gevşek
    
    # Breakout Prediction - SIKI
    MIN_BREAKOUT_PROBABILITY = 0.30  # Daha gevşek
    MAX_BREAKOUT_THRESHOLD = float(os.getenv('MAX_BREAKOUT_THRESHOLD', 0.99))  # 0.99'da kaldı
    
    # Performance Tracking - SIKI
    SIGNAL_EXPIRY_HOURS = int(os.getenv('SIGNAL_EXPIRY_HOURS', 12))  # 24'ten 12'ye düşürüldü
    MIN_PROFIT_PERCENTAGE = float(os.getenv('MIN_PROFIT_PERCENTAGE', 0.0))  # 0.0'da kaldı
    
    # Yeni parametreler - SIKI
    MAX_SIGNALS_PER_BATCH = 20  # Daha fazla sinyal
    MAX_NEUTRAL_SIGNALS = int(os.getenv('MAX_NEUTRAL_SIGNALS', 2))  # 8'den 2'ye düşürüldü
    
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
    
    # Risk Management - GEVŞEK
    MAX_DRAWDOWN_THRESHOLD = float(os.getenv('MAX_DRAWDOWN_THRESHOLD', 0.50))  # 0.40'dan 0.50'ye yükseltildi
    MIN_PROFIT_FACTOR = float(os.getenv('MIN_PROFIT_FACTOR', 1.0))  # 1.2'den 1.0'a düşürüldü
    POSITION_SIZE_LIMIT = float(os.getenv('POSITION_SIZE_LIMIT', 0.10))  # 0.08'den 0.10'a yükseltildi
    
    # Technical Analysis Geliştirmeleri
    ADVANCED_INDICATORS_ENABLED = os.getenv('ADVANCED_INDICATORS_ENABLED', 'True').lower() == 'true'
    PATTERN_RECOGNITION_ENABLED = os.getenv('PATTERN_RECOGNITION_ENABLED', 'True').lower() == 'true'
    FIBONACCI_LEVELS_ENABLED = os.getenv('FIBONACCI_LEVELS_ENABLED', 'True').lower() == 'true'
    
    # Model Retraining - GEVŞEK
    ADAPTIVE_RETRAINING = os.getenv('ADAPTIVE_RETRAINING', 'True').lower() == 'true'
    PERFORMANCE_BASED_RETRAINING = os.getenv('PERFORMANCE_BASED_RETRAINING', 'True').lower() == 'true'
    MIN_PERFORMANCE_FOR_RETRAINING = float(os.getenv('MIN_PERFORMANCE_FOR_RETRAINING', 0.30))  # 0.40'dan 0.30'a düşürüldü
    
    # Data Quality - GEVŞEK
    MIN_DATA_QUALITY_SCORE = float(os.getenv('MIN_DATA_QUALITY_SCORE', 0.50))  # 0.60'dan 0.50'ye düşürüldü
    OUTLIER_DETECTION_ENABLED = os.getenv('OUTLIER_DETECTION_ENABLED', 'True').lower() == 'true'
    DATA_VALIDATION_ENABLED = os.getenv('DATA_VALIDATION_ENABLED', 'True').lower() == 'true'
    
    # Yeni Gelişmiş Parametreler - SIKI
    SIGNAL_QUALITY_THRESHOLD = 0.35  # Daha gevşek
    MIN_CONFIDENCE_THRESHOLD = 0.35  # Daha gevşek
    
    # AI Model Ağırlıkları
    LSTM_WEIGHT = float(os.getenv('LSTM_WEIGHT', 0.4))  # LSTM model ağırlığı
    RF_WEIGHT = float(os.getenv('RF_WEIGHT', 0.3))      # Random Forest ağırlığı
    GB_WEIGHT = float(os.getenv('GB_WEIGHT', 0.3))      # Gradient Boosting ağırlığı
    
    # Skor Normalizasyon Parametreleri - SIKI
    SCORE_AMPLIFICATION_FACTOR = float(os.getenv('SCORE_AMPLIFICATION_FACTOR', 1.0))  # 1.0'da kaldı
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')  # DEBUG'dan INFO'ya düşürüldü
    LOG_FILE = os.getenv('LOG_FILE', 'logs/kahin_ultima.log')
    
    # File Paths
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    SIGNALS_DIR = 'signals'
    LOGS_DIR = 'logs'
    
    # Kalite kontrol kriterleri - SIKI
    MIN_TOTAL_SCORE = 0.35  # Daha gevşek
    MIN_VOLUME = 2000000  # Daha gevşek
    MIN_MARKET_CAP = 20000000  # Daha gevşek
    
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