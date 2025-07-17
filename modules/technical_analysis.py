import pandas as pd
import numpy as np
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from ta.others import CumulativeReturnIndicator
import logging

# SABİT FEATURE LİSTESİ (150+ GÜÇLÜ FEATURE)
FIXED_FEATURE_LIST = [
    # Temel OHLCV
    'open', 'high', 'low', 'close', 'volume',
    
    # Moving Averages (10)
    'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
    'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200',
    
    # MACD (3)
    'macd', 'macd_signal', 'macd_histogram',
    
    # RSI (3)
    'rsi_7', 'rsi_14', 'rsi_21',
    
    # Stochastic (2)
    'stoch_k', 'stoch_d',
    
    # Bollinger Bands (5)
    'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_percent',
    
    # Volatility (3)
    'atr', 'historical_volatility', 'volatility_ratio',
    
    # Volume (4)
    'obv', 'vwap', 'volume_roc', 'volume_ma',
    
    # Trend (4)
    'adx_pos', 'adx_neg', 'adx', 'volume_ratio',
    
    # Momentum (6)
    'cci', 'mfi', 'williams_r', 'psar', 'roc', 'price_roc',
    
    # Ichimoku (5)
    'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span',
    
    # Keltner Channels (3)
    'keltner_ema', 'keltner_upper', 'keltner_lower',
    
    # Rolling Statistics (12)
    'rolling_mean_5', 'rolling_mean_10', 'rolling_mean_20',
    'rolling_std_5', 'rolling_std_10', 'rolling_std_20',
    'rolling_min_5', 'rolling_max_5', 'rolling_min_10', 'rolling_max_10',
    'rolling_median_5', 'rolling_median_10',
    
    # Price Changes (8)
    'price_change_1', 'price_change_3', 'price_change_5', 'price_change_10',
    'price_change_15', 'price_change_20', 'price_change_30', 'price_change_50',
    
    # Volume Changes (6)
    'volume_change_1', 'volume_change_3', 'volume_change_5',
    'volume_change_10', 'volume_change_15', 'volume_change_20',
    
    # Price Positions (4)
    'price_position_5', 'price_position_10', 'price_position_20', 'price_position_50',
    
    # Momentum Indicators (8)
    'momentum_1', 'momentum_3', 'momentum_5', 'momentum_10',
    'momentum_15', 'momentum_20', 'momentum_30', 'momentum_50',
    
    # Volatility Indicators (8)
    'volatility_1', 'volatility_3', 'volatility_5', 'volatility_10',
    'volatility_15', 'volatility_20', 'volatility_30', 'volatility_50',
    
    # Volume Moving Averages (6)
    'volume_ma_1', 'volume_ma_3', 'volume_ma_5', 'volume_ma_10',
    'volume_ma_15', 'volume_ma_20',
    
    # Price Ratios (6)
    'price_ratio_5', 'price_ratio_10', 'price_ratio_20',
    'price_ratio_30', 'price_ratio_50', 'price_ratio_200',
    
    # Spread Indicators (4)
    'high_low_ratio', 'open_close_ratio', 'close_open_ratio', 'high_close_ratio',
    
    # Z-Score Indicators (6)
    'price_zscore_5', 'price_zscore_10', 'price_zscore_20',
    'price_zscore_30', 'price_zscore_50', 'price_zscore_200',
    
    # Advanced Momentum (8)
    'rate_of_change_5', 'rate_of_change_10', 'rate_of_change_20',
    'momentum_roc_5', 'momentum_roc_10', 'momentum_roc_20',
    'price_momentum_5', 'price_momentum_10',
    
    # Advanced Volatility (6)
    'true_range', 'average_true_range_5', 'average_true_range_10',
    'volatility_std_5', 'volatility_std_10', 'volatility_std_20',
    
    # Volume Analysis (8)
    'volume_sma_5', 'volume_sma_10', 'volume_sma_20',
    'volume_ema_5', 'volume_ema_10', 'volume_ema_20',
    'volume_ratio_5', 'volume_ratio_10',
    
    # Price Patterns (6)
    'price_gap_up', 'price_gap_down', 'price_breakout_up',
    'price_breakout_down', 'price_consolidation', 'price_trend_strength',
    
    # Support/Resistance (4)
    'near_support', 'near_resistance', 'support_strength', 'resistance_strength',
    
    # Time Features (4)
    'day_of_week', 'hour', 'month', 'quarter',
    
    # Statistical Features (8)
    'skewness_5', 'skewness_10', 'skewness_20',
    'kurtosis_5', 'kurtosis_10', 'kurtosis_20',
    'correlation_5', 'correlation_10',
    
    # Advanced Technical (10)
    'parabolic_sar', 'commodity_channel_index', 'money_flow_index',
    'ultimate_oscillator', 'stochastic_rsi', 'williams_alligator',
    'fractal_chaos_bands', 'ease_of_movement', 'mass_index', 'detrended_price_oscillator',
    
    # Market Microstructure (6)
    'bid_ask_spread', 'order_flow_imbalance', 'market_depth',
    'liquidity_ratio', 'volume_profile', 'price_impact',
    
    # Cross-Timeframe (8)
    'higher_tf_trend', 'lower_tf_momentum', 'timeframe_alignment',
    'multi_tf_support', 'multi_tf_resistance', 'timeframe_divergence',
    'higher_tf_volatility', 'lower_tf_volume',
    
    # Pattern Recognition (10)
    'doji_pattern', 'hammer_pattern', 'shooting_star_pattern',
    'engulfing_bullish', 'engulfing_bearish', 'morning_star',
    'evening_star', 'double_bottom', 'double_top', 'head_shoulders',
    # AI modelinin beklediği pattern feature'lar
    'triangle_descending', 'breakout_down', 'doji',
    
    # Market Regime (4)
    'trending_market', 'ranging_market', 'volatile_market', 'low_volatility_market',
    
    # Risk Metrics (6)
    'var_95_5', 'var_95_10', 'var_95_20',
    'expected_shortfall_5', 'expected_shortfall_10', 'expected_shortfall_20',
    
    # Sentiment Indicators (4)
    'fear_greed_index', 'market_sentiment', 'social_sentiment', 'news_sentiment',
    
    # Divergence Indicators (6)
    'price_rsi_divergence', 'price_macd_divergence', 'price_volume_divergence',
    'momentum_divergence', 'volatility_divergence', 'trend_divergence',
    
    # Volume Patterns
    'volume_spike',
]

def calculate_fibonacci_levels(df, window=50):
    """Son window bar için Fibonacci retracement seviyelerini hesapla ve feature olarak ekle."""
    if len(df) < window:
        # Yeterli veri yoksa feature'ları 0 ile doldur
        for level in ['fibo_0', 'fibo_236', 'fibo_382', 'fibo_500', 'fibo_618', 'fibo_786', 'fibo_100', 'fibo_proximity', 'fibo_break']:
            df[level] = 0
        return df
    
    high = df['high'].iloc[-window:].max()
    low = df['low'].iloc[-window:].min()
    close = df['close'].iloc[-1]
    # Fibo seviyeleri
    fibo_levels = {
        'fibo_0': low,
        'fibo_236': high - 0.236 * (high - low),
        'fibo_382': high - 0.382 * (high - low),
        'fibo_500': high - 0.5 * (high - low),
        'fibo_618': high - 0.618 * (high - low),
        'fibo_786': high - 0.786 * (high - low),
        'fibo_100': high
    }
    for k, v in fibo_levels.items():
        df[k] = v
    # Fiyat en yakın hangi fibo seviyesinde?
    fibo_vals = np.array(list(fibo_levels.values()))
    proximity = np.abs(fibo_vals - close).min() / (high - low + 1e-9)
    df['fibo_proximity'] = proximity
    # Fiyat, bir fibo seviyesini kırdı mı? (son kapanış bir seviyeyi geçtiyse 1, değilse 0)
    fibo_break = int(any(np.isclose(close, fibo_vals, atol=0.002 * (high - low))))
    df['fibo_break'] = fibo_break
    return df

class TechnicalAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_all_indicators(self, df):
        """Tüm teknik indikatörleri hesapla - 119 FEATURE SABİT + Fibo"""
        try:
            df_copy = df.copy()
            df_copy = self._add_volume_patterns(df_copy)
            
            # Temel indikatörler
            df_copy = self._add_moving_averages(df_copy)
            df_copy = self._add_ema(df_copy)
            df_copy = self._add_macd(df_copy)
            df_copy = self._add_rsi(df_copy)
            df_copy = self._add_stochastic(df_copy)
            df_copy = self._add_bollinger_bands(df_copy)
            df_copy = self._add_atr(df_copy)
            df_copy = self._add_obv(df_copy)
            df_copy = self._add_vwap(df_copy)
           
            # --- EK FEATURE'LAR ---
            # Uzun periyotlu hareketli ortalamalar
            for period in [100]:
                df_copy[f'sma_{period}'] = df_copy['close'].rolling(period).mean()
                df_copy[f'ema_{period}'] = df_copy['close'].ewm(span=period, adjust=False).mean()
            # RSI
            for period in [5, 10, 14, 20, 50, 100]:
                try:
                    df_copy[f'rsi_{period}'] = ta.momentum.RSIIndicator(df_copy['close'], window=period).rsi()
                except Exception:
                    df_copy[f'rsi_{period}'] = 50.0
            # MACD
            for period in [5, 10, 20, 50, 100]:
                try:
                    macd = ta.trend.MACD(df_copy['close'], window_slow=period, window_fast=max(2, period//2), window_sign=9)
                    df_copy[f'macd_{period}'] = macd.macd()
                except Exception:
                    df_copy[f'macd_{period}'] = 0.0
            # Bollinger Bands
            for period in [5, 10, 20, 50, 100]:
                try:
                    bb = ta.volatility.BollingerBands(df_copy['close'], window=period)
                    df_copy[f'bb_upper_{period}'] = bb.bollinger_hband()
                    df_copy[f'bb_lower_{period}'] = bb.bollinger_lband()
                    df_copy[f'bb_middle_{period}'] = bb.bollinger_mavg()
                except Exception:
                    df_copy[f'bb_upper_{period}'] = 0.0
                    df_copy[f'bb_lower_{period}'] = 0.0
                    df_copy[f'bb_middle_{period}'] = 0.0
            # Normalizasyon ve yüzde değişim
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_copy[f'{col}_normalized'] = (df_copy[col] - df_copy[col].min()) / (df_copy[col].max() - df_copy[col].min() + 1e-9)
                df_copy[f'{col}_pct_change'] = df_copy[col].pct_change().fillna(0)
            # Diğer teknikler (örnek)
            df_copy['momentum_14'] = df_copy['close'] / df_copy['close'].shift(14) - 1
            df_copy['roc_5'] = df_copy['close'].pct_change(5)
            df_copy['roc_14'] = df_copy['close'].pct_change(14)
            # ... ve benzeri şekilde diğer eksik feature'ları da ekleyebilirsiniz.
            
            
            # Gelişmiş indikatörler
            df_copy = self._add_adx(df_copy)
            df_copy = self._add_cci(df_copy)
            df_copy = self._add_mfi(df_copy)
            df_copy = self._add_williams_r(df_copy)
            df_copy = self._add_psar(df_copy)
            df_copy = self._add_ichimoku(df_copy)
            df_copy = self._add_keltner_channels(df_copy)
            df_copy = self._add_extra_indicators(df_copy)
            
            # Fibo seviyeleri ekle
            df_copy = calculate_fibonacci_levels(df_copy, window=50)
            
            # Pattern fonksiyonlarından ÖNCE momentum sütununu garantiye al
            if 'momentum' not in df_copy.columns:
                df_copy['momentum'] = df_copy['close'].diff().fillna(0)
            # Ekstra pattern ve volume feature'ları ekle
            df_copy = self.add_pattern_recognition(df_copy)
            df_copy = self.calculate_advanced_patterns(df_copy)
            df_copy = self.analyze_patterns(df_copy)
            # Volume spike feature'ı ekle
            df_copy['volume_spike'] = (df_copy['volume'] > df_copy['volume'].rolling(20).mean() * 2).astype(int)
            # Eksik pattern feature'larını garantiye al
            if 'triangle_descending' not in df_copy.columns:
                if 'triangle_descending_pattern' in df_copy.columns:
                    df_copy['triangle_descending'] = df_copy['triangle_descending_pattern']
                else:
                    df_copy['triangle_descending'] = 0
            if 'breakout_down' not in df_copy.columns:
                if 'price_breakout_down' in df_copy.columns:
                    df_copy['breakout_down'] = df_copy['price_breakout_down']
                else:
                    df_copy['breakout_down'] = 0
            if 'doji' not in df_copy.columns:
                if 'doji_pattern' in df_copy.columns:
                    df_copy['doji'] = df_copy['doji_pattern']
                else:
                    df_copy['doji'] = 0
            
            # SADECE 119 FEATURE KULLAN - FIXED_FEATURE_LIST'ten
            available_features = [col for col in FIXED_FEATURE_LIST if col in df_copy.columns]
            
            # Eksik feature'ları 0 ile doldur
            for feature in FIXED_FEATURE_LIST:
                if feature not in df_copy.columns:
                    self.logger.error(f"{feature} eksik, teknik analiz başarısız. Sinyal üretimi iptal edildi.")
                    return None
            
            # --- KRİTİK: pattern_score ve momentum kontrolü ---
            kritik_features = ['pattern_score', 'momentum']
            for feature in kritik_features:
                if feature not in df_copy.columns:
                    self.logger.error(f"{feature} eksik, teknik analiz zinciri başarısız. Sinyal üretimi iptal edildi.")
                    return None
                # Anlamsız (sürekli 0 veya sabit) ise de logla
                if df_copy[feature].nunique() <= 1:
                    self.logger.error(f"{feature} sütunu sabit veya anlamsız (tekil değer: {df_copy[feature].iloc[-1]}). Teknik analiz başarısız. Sinyal üretimi iptal edildi.")
                    return None

            # Sadece FIXED_FEATURE_LIST'teki sütunları al
            df_final = df_copy[FIXED_FEATURE_LIST].copy()
            df_final = df_final.fillna(0)
            return df_final
            
        except Exception as e:
            self.logger.error(f"Teknik analiz hatası: {e}")
            return df
    
    def _add_trend_indicators(self, df):
        """Trend indikatörlerini ekle"""
        try:
            # Minimum veri kontrolü
            if len(df) < 200:
                self.logger.warning(f'Trend indikatörleri için yetersiz veri: {len(df)} satır')
                return df
            
            # SMA (Simple Moving Average)
            for period in [5, 10, 20, 50, 200]:
                if len(df) >= period:
                    df[f'sma_{period}'] = SMAIndicator(close=df['close'], window=period).sma_indicator()
                else:
                    df[f'sma_{period}'] = df['close']
            
            # EMA (Exponential Moving Average)
            for period in [5, 10, 20, 50, 200]:
                if len(df) >= period:
                    df[f'ema_{period}'] = EMAIndicator(close=df['close'], window=period).ema_indicator()
                else:
                    df[f'ema_{period}'] = df['close']
            
            # MACD
            if len(df) >= 26:
                macd = MACD(close=df['close'])
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()
                df['macd_histogram'] = macd.macd_diff()
            else:
                df['macd'] = 0
                df['macd_signal'] = 0
                df['macd_histogram'] = 0
            
            # ADX (Average Directional Index)
            if len(df) >= 14:
                adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'])
                df['adx'] = adx.adx()
                df['adx_pos'] = adx.adx_pos()
                df['adx_neg'] = adx.adx_neg()
            else:
                df['adx'] = 25
                df['adx_pos'] = 25
                df['adx_neg'] = 25
            
            return df
            
        except Exception as e:
            self.logger.error(f"Trend indikatörleri eklenirken hata: {e}")
            # Hata durumunda varsayılan değerler
            for period in [5, 10, 20, 50, 200]:
                df[f'sma_{period}'] = df['close']
                df[f'ema_{period}'] = df['close']
            df['macd'] = 0
            df['macd_signal'] = 0
            df['macd_histogram'] = 0
            df['adx'] = 25
            df['adx_pos'] = 25
            df['adx_neg'] = 25
            return df
    
    def _add_momentum_indicators(self, df):
        """Momentum indikatörlerini ekle"""
        try:
            # RSI
            for period in [7, 14, 21]:
                df[f'rsi_{period}'] = RSIIndicator(close=df['close'], window=period).rsi()
            
            # Stochastic Oscillator
            stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Williams %R
            df['williams_r'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()
            
            # CCI (Commodity Channel Index)
            df['cci'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'], close=df['close']).cci()
            
            # MFI (Money Flow Index) - düzeltilmiş import
            df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).money_flow_index()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Momentum indikatörleri eklenirken hata: {e}")
            return df
    
    def _add_volatility_indicators(self, df):
        """Volatilite indikatörlerini ekle"""
        try:
            # Minimum veri kontrolü
            if len(df) < 20:
                self.logger.warning(f'Volatilite indikatörleri için yetersiz veri: {len(df)} satır')
                return df
            
            # Bollinger Bands
            if len(df) >= 20:
                bb = BollingerBands(close=df['close'])
                df['bb_upper'] = bb.bollinger_hband()
                df['bb_middle'] = bb.bollinger_mavg()
                df['bb_lower'] = bb.bollinger_lband()
                df['bb_width'] = bb.bollinger_wband()
                df['bb_percent'] = bb.bollinger_pband()
            else:
                df['bb_upper'] = df['close'] * 1.02
                df['bb_middle'] = df['close']
                df['bb_lower'] = df['close'] * 0.98
                df['bb_width'] = 0.04
                df['bb_percent'] = 0.5
            
            # ATR (Average True Range)
            if len(df) >= 14:
                df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
            else:
                df['atr'] = (df['high'] - df['low']).rolling(5).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Volatilite indikatörleri eklenirken hata: {e}")
            # Hata durumunda varsayılan değerler
            df['bb_upper'] = df['close'] * 1.02
            df['bb_middle'] = df['close']
            df['bb_lower'] = df['close'] * 0.98
            df['bb_width'] = 0.04
            df['bb_percent'] = 0.5
            df['atr'] = (df['high'] - df['low']).rolling(5).mean()
            return df
    
    def _add_volume_indicators(self, df):
        """Hacim indikatörlerini ekle"""
        try:
            # OBV (On Balance Volume)
            df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
            
            # VWAP (Volume Weighted Average Price)
            df['vwap'] = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).volume_weighted_average_price()
            
            # Money Flow Index (MFI) - düzeltilmiş
            df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).money_flow_index()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Hacim indikatörleri eklenirken hata: {e}")
            return df
    
    def _add_other_indicators(self, df):
        """Diğer indikatörleri ekle"""
        try:
            # Cumulative Return
            df['cumulative_return'] = CumulativeReturnIndicator(close=df['close']).cumulative_return()
            
            # Price Rate of Change
            df['roc'] = ta.momentum.ROCIndicator(close=df['close']).roc()
            
            # Parabolic SAR
            df['psar'] = ta.trend.PSARIndicator(high=df['high'], low=df['low'], close=df['close']).psar()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Diğer indikatörler eklenirken hata: {e}")
            return df
    
    def generate_signals(self, df):
        """Teknik analiz sinyallerini üret"""
        try:
            signals = {}
            
            # RSI Sinyalleri
            if 'rsi_14' in df.columns:
                signals['rsi_oversold'] = df['rsi_14'] < 30
                signals['rsi_overbought'] = df['rsi_14'] > 70
            else:
                signals['rsi_oversold'] = pd.Series([False] * len(df))
                signals['rsi_overbought'] = pd.Series([False] * len(df))
            
            # MACD Sinyalleri
            if 'macd' in df.columns and 'macd_signal' in df.columns and 'macd_histogram' in df.columns:
                signals['macd_bullish'] = (df['macd'] > df['macd_signal']) & (df['macd_histogram'] > 0)
                signals['macd_bearish'] = (df['macd'] < df['macd_signal']) & (df['macd_histogram'] < 0)
            else:
                signals['macd_bullish'] = pd.Series([False] * len(df))
                signals['macd_bearish'] = pd.Series([False] * len(df))
            
            # Bollinger Bands Sinyalleri
            if 'bb_lower' in df.columns and 'bb_upper' in df.columns:
                signals['bb_oversold'] = df['close'] < df['bb_lower']
                signals['bb_overbought'] = df['close'] > df['bb_upper']
            else:
                signals['bb_oversold'] = pd.Series([False] * len(df))
                signals['bb_overbought'] = pd.Series([False] * len(df))
            
            # Moving Average Sinyalleri
            if 'ema_20' in df.columns:
                signals['ma_bullish'] = df['close'] > df['ema_20']
                signals['ma_bearish'] = df['close'] < df['ema_20']
            else:
                signals['ma_bullish'] = pd.Series([False] * len(df))
                signals['ma_bearish'] = pd.Series([False] * len(df))
            
            # Stochastic Sinyalleri
            if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
                signals['stoch_oversold'] = (df['stoch_k'] < 20) & (df['stoch_d'] < 20)
                signals['stoch_overbought'] = (df['stoch_k'] > 80) & (df['stoch_d'] > 80)
            else:
                signals['stoch_oversold'] = pd.Series([False] * len(df))
                signals['stoch_overbought'] = pd.Series([False] * len(df))
            
            # ADX Trend Gücü
            if 'adx' in df.columns:
                signals['strong_trend'] = df['adx'] > 25
            else:
                signals['strong_trend'] = pd.Series([False] * len(df))
            
            # Volume Sinyalleri
            if 'volume' in df.columns:
                signals['high_volume'] = df['volume'] > df['volume'].rolling(20).mean() * 1.5
            else:
                signals['high_volume'] = pd.Series([False] * len(df))
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Sinyaller üretilirken hata: {e}")
            # Hata durumunda boş sinyaller döndür
            empty_signals = {}
            for signal_name in ['rsi_oversold', 'rsi_overbought', 'macd_bullish', 'macd_bearish', 
                               'bb_oversold', 'bb_overbought', 'ma_bullish', 'ma_bearish',
                               'stoch_oversold', 'stoch_overbought', 'strong_trend', 'high_volume']:
                empty_signals[signal_name] = pd.Series([False] * len(df) if not df.empty else [])
            return empty_signals
    
    def calculate_signal_strength(self, df):
        print("[DEBUG] calculate_signal_strength input df.tail():", df.tail())
        try:
            if df is None or (hasattr(df, 'empty') and df.empty) or len(df) < 20:
                print("[DEBUG] calculate_signal_strength: df boş veya yetersiz uzunlukta")
                self.logger.error("Teknik analiz verisi eksik veya yetersiz, sinyal gücü hesaplanamadı. Sinyal üretimi iptal edildi.")
                return None
            if isinstance(df, dict):
                self.logger.warning('DataFrame dict olarak geldi, 0.0 döndürülüyor')
                return None
            current = df.iloc[-1]
            prev = df.iloc[-2]
            strength = 0.5
            if 'rsi_14' in df.columns and not pd.isna(current['rsi_14']):
                rsi = current['rsi_14']
                if rsi < 20:
                    strength += 0.15
                elif rsi > 80:
                    strength -= 0.15
                elif 30 < rsi < 70:
                    strength += 0.05
            elif 'rsi' in df.columns and not pd.isna(current['rsi']):
                rsi = current['rsi']
                if rsi < 20:
                    strength += 0.15
                elif rsi > 80:
                    strength -= 0.15
                elif 30 < rsi < 70:
                    strength += 0.05
            else:
                self.logger.error("RSI verisi eksik, sinyal gücü hesaplanamadı. Sinyal üretimi iptal edildi.")
                return None
            if ('macd' in df.columns and 'macd_signal' in df.columns and 
                not pd.isna(current['macd']) and not pd.isna(current['macd_signal'])):
                macd = current['macd']
                macd_signal = current['macd_signal']
                if macd > macd_signal:
                    strength += 0.1
                elif macd < macd_signal:
                    strength -= 0.1
            else:
                self.logger.error("MACD verisi eksik, sinyal gücü hesaplanamadı. Sinyal üretimi iptal edildi.")
                return None
            return strength
        except Exception as e:
            self.logger.error(f"Sinyal gücü hesaplama hatası: {e}")
            return None

    def get_trend_direction(self, df):
        """Trend yönünü belirle"""
        try:
            if df.empty:
                return 'neutral'
            
            # EMA trend analizi
            ema_20 = df['ema_20'].iloc[-1]
            ema_50 = df['ema_50'].iloc[-1]
            ema_200 = df['ema_200'].iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Güçlü yukarı trend
            if current_price > ema_20 > ema_50 > ema_200:
                return 'strong_up'
            # Yukarı trend
            elif current_price > ema_20 and ema_20 > ema_50:
                return 'up'
            # Güçlü aşağı trend
            elif current_price < ema_20 < ema_50 < ema_200:
                return 'strong_down'
            # Aşağı trend
            elif current_price < ema_20 and ema_20 < ema_50:
                return 'down'
            else:
                return 'neutral'
                
        except Exception as e:
            self.logger.error(f"Trend yönü belirleme hatası: {e}")
            return 'neutral'
    
    def calculate_trend_strength(self, df):
        """Trend gücünü hesapla"""
        try:
            if df.empty:
                return 0.0
            
            # ADX ile trend gücü
            adx = df['adx'].iloc[-1] if 'adx' in df.columns else 25
            
            # RSI trend gücü
            rsi = df['rsi_14'].iloc[-1] if 'rsi_14' in df.columns else 50
            rsi_strength = abs(rsi - 50) / 50  # 0-1 arası
            
            # MACD trend gücü
            macd = df['macd'].iloc[-1] if 'macd' in df.columns else 0
            macd_signal = df['macd_signal'].iloc[-1] if 'macd_signal' in df.columns else 0
            macd_strength = abs(macd - macd_signal) / (abs(macd) + abs(macd_signal) + 0.001)
            
            # Toplam trend gücü
            adx_weight = 0.4
            rsi_weight = 0.3
            macd_weight = 0.3
            
            trend_strength = (
                (adx / 100) * adx_weight +
                rsi_strength * rsi_weight +
                macd_strength * macd_weight
            )
            
            return min(trend_strength, 1.0)
            
        except Exception as e:
            self.logger.error(f"Trend gücü hesaplama hatası: {e}")
            return 0.0
    
    def find_dynamic_support(self, df):
        """Dinamik destek seviyelerini bul"""
        try:
            if df.empty:
                return []
            
            supports = []
            current_price = df['close'].iloc[-1]
            
            # Bollinger Bands alt bandı
            if 'bb_lower' in df.columns:
                bb_lower = df['bb_lower'].iloc[-1]
                if bb_lower < current_price:
                    supports.append(bb_lower)
            
            # Keltner Channel alt bandı
            if 'keltner_lower' in df.columns:
                keltner_lower = df['keltner_lower'].iloc[-1]
                if keltner_lower < current_price:
                    supports.append(keltner_lower)
            
            # Son 20 mumun en düşük noktaları
            recent_lows = df['low'].tail(20).nsmallest(3).tolist()
            for low in recent_lows:
                if low < current_price:
                    supports.append(low)
            
            # EMA seviyeleri
            for ema_col in ['ema_20', 'ema_50', 'ema_200']:
                if ema_col in df.columns:
                    ema_value = df[ema_col].iloc[-1]
                    if ema_value < current_price:
                        supports.append(ema_value)
            
            # Tekrarlanan değerleri kaldır ve sırala
            supports = sorted(list(set(supports)), reverse=True)
            
            return supports[:5]  # En yakın 5 destek
            
        except Exception as e:
            self.logger.error(f"Destek seviyesi bulma hatası: {e}")
            return []
    
    def find_dynamic_resistance(self, df):
        """Dinamik direnç seviyelerini bul"""
        try:
            if df.empty:
                return []
            
            resistances = []
            current_price = df['close'].iloc[-1]
            
            # Bollinger Bands üst bandı
            if 'bb_upper' in df.columns:
                bb_upper = df['bb_upper'].iloc[-1]
                if bb_upper > current_price:
                    resistances.append(bb_upper)
            
            # Keltner Channel üst bandı
            if 'keltner_upper' in df.columns:
                keltner_upper = df['keltner_upper'].iloc[-1]
                if keltner_upper > current_price:
                    resistances.append(keltner_upper)
            
            # Son 20 mumun en yüksek noktaları
            recent_highs = df['high'].tail(20).nlargest(3).tolist()
            for high in recent_highs:
                if high > current_price:
                    resistances.append(high)
            
            # EMA seviyeleri
            for ema_col in ['ema_20', 'ema_50', 'ema_200']:
                if ema_col in df.columns:
                    ema_value = df[ema_col].iloc[-1]
                    if ema_value > current_price:
                        resistances.append(ema_value)
            
            # Tekrarlanan değerleri kaldır ve sırala
            resistances = sorted(list(set(resistances)))
            
            return resistances[:5]  # En yakın 5 direnç
            
        except Exception as e:
            self.logger.error(f"Direnç seviyesi bulma hatası: {e}")
            return []
    
    def calculate_momentum_score(self, df):
        print("[DEBUG] calculate_momentum_score input df.tail():", df.tail())
        try:
            if df.empty:
                print("[DEBUG] calculate_momentum_score: df boş")
                return 'Veri Yok'
            momentum_score = 0.0
            factors = 0
            # RSI momentum
            if 'rsi_14' in df.columns:
                rsi = df['rsi_14'].iloc[-1]
                if rsi > 70:
                    momentum_score += 0.8  # Güçlü yukarı momentum
                elif rsi > 60:
                    momentum_score += 0.6  # Orta yukarı momentum
                elif rsi < 30:
                    momentum_score += 0.2  # Zayıf momentum
                elif rsi < 40:
                    momentum_score += 0.4  # Orta aşağı momentum
                else:
                    momentum_score += 0.5  # Nötr
                factors += 1
            # MACD momentum
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                macd = df['macd'].iloc[-1]
                macd_signal = df['macd_signal'].iloc[-1]
                macd_hist = df['macd_histogram'].iloc[-1]
                if macd > macd_signal and macd_hist > 0:
                    momentum_score += 0.8
                elif macd > macd_signal:
                    momentum_score += 0.6
                elif macd < macd_signal and macd_hist < 0:
                    momentum_score += 0.2
                elif macd < macd_signal:
                    momentum_score += 0.4
                else:
                    momentum_score += 0.5
                factors += 1
            # Stochastic momentum
            if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
                stoch_k = df['stoch_k'].iloc[-1]
                stoch_d = df['stoch_d'].iloc[-1]
                if stoch_k > 80 and stoch_d > 80:
                    momentum_score += 0.8
                elif stoch_k > stoch_d and stoch_k > 60:
                    momentum_score += 0.6
                elif stoch_k < 20 and stoch_d < 20:
                    momentum_score += 0.2
                elif stoch_k < stoch_d and stoch_k < 40:
                    momentum_score += 0.4
                else:
                    momentum_score += 0.5
                factors += 1
            # Ortalama momentum skoru
            if factors > 0:
                result = momentum_score / factors
                print("[DEBUG] calculate_momentum_score output:", result)
                return result
            else:
                print("[DEBUG] calculate_momentum_score: factors=0, Veri Yok")
                return 'Veri Yok'
        except Exception as e:
            print(f"[DEBUG] calculate_momentum_score exception: {e}")
            self.logger.error(f"Momentum skoru hesaplama hatası: {e}")
            return 'Veri Yok'

    def calculate_volume_score(self, df):
        print("[DEBUG] calculate_volume_score input df.tail():", df.tail())
        try:
            if df.empty or 'volume' not in df.columns:
                print("[DEBUG] calculate_volume_score: df boş veya 'volume' yok")
                self.logger.error("Hacim verisi eksik, hacim skoru hesaplanamadı. Sinyal üretimi iptal edildi.")
                return None
            volume = df['volume'].iloc[-1]
            volume_ma = df['volume'].rolling(20).mean().iloc[-1] if 'volume' in df.columns else None
            if volume_ma is None or np.isnan(volume_ma) or volume_ma == 0:
                print("[DEBUG] calculate_volume_score: volume_ma yok veya sıfır")
                self.logger.error("Hacim ortalaması eksik, hacim skoru hesaplanamadı. Sinyal üretimi iptal edildi.")
                return None
            ratio = volume / volume_ma
            if ratio > 2.0:
                print("[DEBUG] calculate_volume_score output: 1.0")
                return 1.0
            elif ratio > 1.5:
                print("[DEBUG] calculate_volume_score output: 0.8")
                return 0.8
            elif ratio > 1.2:
                print("[DEBUG] calculate_volume_score output: 0.6")
                return 0.6
            elif ratio > 0.8:
                print("[DEBUG] calculate_volume_score output: 0.4")
                return 0.4
            else:
                print("[DEBUG] calculate_volume_score output: 0.2")
                return 0.2
        except Exception as e:
            print(f"[DEBUG] calculate_volume_score exception: {e}")
            self.logger.error(f"Hacim skoru hesaplama hatası: {e}")
            return None

    def calculate_trend_alignment(self, analysis):
        """Trend uyumu skorunu hesapla"""
        try:
            if not analysis:
                return 0.0
            
            timeframes = ['1h', '4h', '1d']
            trends = []
            strengths = []
            
            for tf in timeframes:
                if tf in analysis:
                    trend = analysis[tf]['trend']
                    strength = analysis[tf]['strength']
                    
                    # Trend yönünü sayısal değere çevir
                    if 'up' in trend:
                        trends.append(1)
                    elif 'down' in trend:
                        trends.append(-1)
                    else:
                        trends.append(0)
                    
                    strengths.append(strength)
            
            if not trends:
                return 0.0
            
            # Trend uyumu hesapla
            if len(set(trends)) == 1:  # Tüm zaman dilimleri aynı yönde
                alignment = 1.0
            elif len(set(trends)) == 2:  # İki farklı yön
                alignment = 0.5
            else:  # Üç farklı yön
                alignment = 0.0
            
            # Güç ağırlıklı ortalama
            avg_strength = sum(strengths) / len(strengths) if strengths else 0.0
            
            return alignment * avg_strength
            
        except Exception as e:
            self.logger.error(f"Trend uyumu hesaplama hatası: {e}")
            return 0.0

    def get_market_regime(self, df):
        """Piyasa rejimini belirle (Trending/Ranging/Volatile)"""
        try:
            if df.empty or len(df) < 20:
                return 'unknown'
            
            recent_df = df.tail(20)
            
            # Volatilite hesapla
            volatility = recent_df['close'].pct_change().std()
            
            # ADX gücü
            adx = recent_df['adx'].iloc[-1] if 'adx' in recent_df.columns else 0
            
            # Bollinger Band genişliği
            bb_width = recent_df['bb_width'].iloc[-1] if 'bb_width' in recent_df.columns else 0
            
            # Rejim belirleme
            if adx > 25 and volatility < 0.05:
                return 'trending'
            elif adx < 20 and bb_width < 0.1:
                return 'ranging'
            elif volatility > 0.08:
                return 'volatile'
            else:
                return 'mixed'
                
        except Exception as e:
            self.logger.error(f"Piyasa rejimi analizi hatası: {e}")
            return 'unknown'
    
    def get_support_resistance(self, df, window=20):
        """Destek ve direnç seviyelerini hesapla"""
        try:
            highs = df['high'].rolling(window=window, center=True).max()
            lows = df['low'].rolling(window=window, center=True).min()
            
            # Son değerler
            current_high = highs.iloc[-1]
            current_low = lows.iloc[-1]
            current_price = df['close'].iloc[-1]
            
            return {
                'resistance': current_high,
                'support': current_low,
                'distance_to_resistance': ((current_high - current_price) / current_price) * 100,
                'distance_to_support': ((current_price - current_low) / current_price) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Destek/direnç hesaplanırken hata: {e}")
            return {}

    def _add_ichimoku(self, df):
        """Ichimoku Cloud indikatörü ekle"""
        try:
            # Tenkan-sen (Conversion Line)
            df['tenkan_sen'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
            
            # Kijun-sen (Base Line)
            df['kijun_sen'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
            
            # Senkou Span A (Leading Span A)
            df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
            
            # Senkou Span B (Leading Span B)
            df['senkou_span_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
            
            # Chikou Span (Lagging Span)
            df['chikou_span'] = df['close'].shift(-26)
            
            return df
        except Exception as e:
            self.logger.warning(f"Ichimoku hesaplama hatası: {e}")
            return df

    def _add_keltner_channels(self, df):
        """Keltner Channels ekle"""
        try:
            if len(df) < 20:
                return df
            
            # ATR hesapla
            atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
            
            # EMA hesapla
            ema = df['close'].ewm(span=20).mean()
            
            # Keltner Channels
            df['keltner_ema'] = ema
            df['keltner_upper'] = ema + (2 * atr)
            df['keltner_lower'] = ema - (2 * atr)
            
            return df
        except Exception as e:
            self.logger.warning(f"Keltner Channels hesaplama hatası: {e}")
            return df

    def _add_volume_profile(self, df):
        """Volume Profile indikatörü ekle"""
        try:
            # Volume Weighted Average Price
            df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            
            # Volume Rate of Change
            df['volume_roc'] = df['volume'].pct_change(periods=10)
            
            # Volume Moving Average
            df['volume_ma'] = df['volume'].rolling(20).mean()
            
            # Volume Ratio
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            return df
        except Exception as e:
            self.logger.warning(f"Volume Profile hesaplama hatası: {e}")
            return df

    def _add_momentum_indicators(self, df):
        """Momentum indikatörleri ekle"""
        try:
            # Rate of Change
            df['roc'] = df['close'].pct_change(periods=10) * 100
            
            # Momentum
            df['momentum'] = df['close'] - df['close'].shift(10)
            
            # Price Rate of Change
            df['price_roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
            
            # Williams %R
            df['williams_r'] = ((df['high'].rolling(14).max() - df['close']) / 
                               (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * -100
            
            return df
        except Exception as e:
            self.logger.warning(f"Momentum indikatörleri hesaplama hatası: {e}")
            return df

    def _add_volatility_indicators(self, df):
        """Volatilite indikatörleri ekle"""
        try:
            # Historical Volatility
            df['historical_volatility'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
            
            # True Range
            df['true_range'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            
            # Average True Range
            df['atr'] = df['true_range'].rolling(14).mean()
            
            # Volatility Ratio
            df['volatility_ratio'] = df['atr'] / df['close']
            
            return df
        except Exception as e:
            self.logger.warning(f"Volatilite indikatörleri hesaplama hatası: {e}")
            return df

    def _add_moving_averages(self, df):
        """Moving Average indikatörleri ekle"""
        try:
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['sma_200'] = df['close'].rolling(200).mean()
            return df
        except Exception as e:
            self.logger.warning(f"Moving Average hesaplama hatası: {e}")
            return df

    def _add_ema(self, df):
        """Exponential Moving Average indikatörleri ekle"""
        try:
            df['ema_5'] = df['close'].ewm(span=5).mean()
            df['ema_10'] = df['close'].ewm(span=10).mean()
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['ema_200'] = df['close'].ewm(span=200).mean()
            return df
        except Exception as e:
            self.logger.warning(f"EMA hesaplama hatası: {e}")
            return df

    def _add_macd(self, df):
        """MACD indikatörü ekle"""
        try:
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            return df
        except Exception as e:
            self.logger.warning(f"MACD hesaplama hatası: {e}")
            return df

    def _add_rsi(self, df):
        """RSI indikatörü ekle"""
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_7'] = 100 - (100 / (1 + rs))
            
            # Farklı periyotlar için RSI
            gain_14 = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss_14 = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs_14 = gain_14 / loss_14
            df['rsi_14'] = 100 - (100 / (1 + rs_14))
            
            gain_21 = (delta.where(delta > 0, 0)).rolling(window=21).mean()
            loss_21 = (-delta.where(delta < 0, 0)).rolling(window=21).mean()
            rs_21 = gain_21 / loss_21
            df['rsi_21'] = 100 - (100 / (1 + rs_21))
            
            return df
        except Exception as e:
            self.logger.warning(f"RSI hesaplama hatası: {e}")
            return df

    def _add_stochastic(self, df):
        """Stochastic Oscillator ekle"""
        try:
            df['stoch_k'] = ((df['close'] - df['low'].rolling(14).min()) / 
                            (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * 100
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            return df
        except Exception as e:
            self.logger.warning(f"Stochastic hesaplama hatası: {e}")
            return df

    def _add_bollinger_bands(self, df):
        """Bollinger Bands ekle"""
        try:
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            return df
        except Exception as e:
            self.logger.warning(f"Bollinger Bands hesaplama hatası: {e}")
            return df

    def _add_atr(self, df):
        """Average True Range ekle"""
        try:
            df['atr'] = self._calculate_atr(df, period=14)
            return df
        except Exception as e:
            self.logger.warning(f"ATR hesaplama hatası: {e}")
            return df

    def _add_obv(self, df):
        """On Balance Volume ekle"""
        try:
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            return df
        except Exception as e:
            self.logger.warning(f"OBV hesaplama hatası: {e}")
            return df

    def _add_vwap(self, df):
        """Volume Weighted Average Price ekle"""
        try:
            df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            return df
        except Exception as e:
            self.logger.warning(f"VWAP hesaplama hatası: {e}")
            return df

    def _add_adx(self, df):
        """Average Directional Index ekle"""
        try:
            # True Range
            tr = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            
            # Directional Movement
            dm_plus = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                              np.maximum(df['high'] - df['high'].shift(1), 0), 0)
            dm_minus = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                               np.maximum(df['low'].shift(1) - df['low'], 0), 0)
            
            # Smoothed values
            tr_smooth = tr.rolling(14).mean()
            dm_plus_smooth = pd.Series(dm_plus).rolling(14).mean()
            dm_minus_smooth = pd.Series(dm_minus).rolling(14).mean()
            
            # DI values
            df['adx_pos'] = (dm_plus_smooth / tr_smooth) * 100
            df['adx_neg'] = (dm_minus_smooth / tr_smooth) * 100
            
            # ADX
            dx = abs(df['adx_pos'] - df['adx_neg']) / (df['adx_pos'] + df['adx_neg']) * 100
            df['adx'] = dx.rolling(14).mean()
            
            return df
        except Exception as e:
            self.logger.warning(f"ADX hesaplama hatası: {e}")
            return df

    def _add_cci(self, df):
        """Commodity Channel Index ekle"""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = typical_price.rolling(20).mean()
            mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
            df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
            return df
        except Exception as e:
            self.logger.warning(f"CCI hesaplama hatası: {e}")
            return df

    def _add_mfi(self, df):
        """Money Flow Index ekle"""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            
            mfi_ratio = positive_flow / negative_flow
            df['mfi'] = 100 - (100 / (1 + mfi_ratio))
            return df
        except Exception as e:
            self.logger.warning(f"MFI hesaplama hatası: {e}")
            return df

    def _add_williams_r(self, df):
        """Williams %R ekle"""
        try:
            df['williams_r'] = ((df['high'].rolling(14).max() - df['close']) / 
                               (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * -100
            return df
        except Exception as e:
            self.logger.warning(f"Williams %R hesaplama hatası: {e}")
            return df

    def _add_psar(self, df):
        """Parabolic SAR ekle"""
        try:
            # Basit PSAR implementasyonu
            df['psar'] = df['close'].rolling(5).mean()  # Basitleştirilmiş
            return df
        except Exception as e:
            self.logger.warning(f"PSAR hesaplama hatası: {e}")
            return df

    def add_advanced_indicators(self, df):
        """Gelişmiş teknik analiz göstergeleri ekle"""
        try:
            # Fibonacci Retracement
            high = df['high'].max()
            low = df['low'].min()
            diff = high - low
            df['fib_23.6'] = high - 0.236 * diff
            df['fib_38.2'] = high - 0.382 * diff
            df['fib_50.0'] = high - 0.500 * diff
            df['fib_61.8'] = high - 0.618 * diff
            
            # Pivot Points
            df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
            df['r1'] = 2 * df['pivot'] - df['low']
            df['s1'] = 2 * df['pivot'] - df['high']
            df['r2'] = df['pivot'] + (df['high'] - df['low'])
            df['s2'] = df['pivot'] - (df['high'] - df['low'])
            
            # Volume Weighted Average Price (VWAP)
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            
            # Money Flow Index (MFI)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            
            mfi_ratio = positive_flow / negative_flow
            df['mfi'] = 100 - (100 / (1 + mfi_ratio))
            
            # Average True Range (ATR) - zaten var ama geliştir
            df['atr_14'] = df['atr'].rolling(14).mean()
            df['atr_21'] = df['atr'].rolling(21).mean()
            
            # Stochastic RSI
            df['stoch_rsi'] = (df['rsi_14'] - df['rsi_14'].rolling(14).min()) / \
                             (df['rsi_14'].rolling(14).max() - df['rsi_14'].rolling(14).min())
            
            # Williams %R
            df['williams_r'] = ((df['high'].rolling(14).max() - df['close']) / \
                               (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * -100
            
            # Commodity Channel Index (CCI)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = typical_price.rolling(20).mean()
            mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
            df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
            
            # Detrended Price Oscillator (DPO)
            df['dpo'] = df['close'] - df['close'].rolling(20).mean().shift(10)
            
            # Price Rate of Change (ROC)
            df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
            
            # Ultimate Oscillator
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            bp = df['close'] - df['low']
            bp_7 = bp.rolling(7).sum()
            bp_14 = bp.rolling(14).sum()
            bp_28 = bp.rolling(28).sum()
            
            tr_7 = tr.rolling(7).sum()
            tr_14 = tr.rolling(14).sum()
            tr_28 = tr.rolling(28).sum()
            
            df['ultimate_osc'] = 100 * ((4 * bp_7 / tr_7) + (2 * bp_14 / tr_14) + (bp_28 / tr_28)) / 7
            
            # Pattern Recognition
            df = self.add_pattern_recognition(df)
            
            self.logger.info(f"Gelişmiş göstergeler eklendi. Toplam sütun: {len(df.columns)}")
            return df
            
        except Exception as e:
            self.logger.error(f"Gelişmiş göstergeler eklenirken hata: {e}")
            return df
    
    def add_pattern_recognition(self, df):
        if 'momentum' not in df.columns:
            df['momentum'] = df['close'].diff().fillna(0)
        try:
            # Doji pattern
            body_size = abs(df['close'] - df['open'])
            total_range = df['high'] - df['low']
            df['doji'] = (body_size <= 0.1 * total_range).astype(int)
            
            # Hammer pattern
            body_size = abs(df['close'] - df['open'])
            lower_shadow = df['low'] - df[['open', 'close']].min(axis=1)
            upper_shadow = df[['open', 'close']].max(axis=1) - df['high']
            df['hammer'] = ((lower_shadow > 2 * body_size) & (upper_shadow < body_size)).astype(int)
            
            # Shooting Star pattern
            df['shooting_star'] = ((upper_shadow > 2 * body_size) & (lower_shadow < body_size)).astype(int)
            
            # Engulfing patterns
            df['bullish_engulfing'] = ((df['open'] < df['close'].shift(1)) & 
                                      (df['close'] > df['open'].shift(1)) &
                                      (df['open'] < df['close'].shift(1)) &
                                      (df['close'] > df['open'].shift(1))).astype(int)
            
            df['bearish_engulfing'] = ((df['open'] > df['close'].shift(1)) & 
                                      (df['close'] < df['open'].shift(1)) &
                                      (df['open'] > df['close'].shift(1)) &
                                      (df['close'] < df['open'].shift(1))).astype(int)
            
            # Support/Resistance levels
            df['support_level'] = df['low'].rolling(20).min()
            df['resistance_level'] = df['high'].rolling(20).max()
            
            # Price position relative to support/resistance
            df['price_vs_support'] = (df['close'] - df['support_level']) / df['support_level']
            df['price_vs_resistance'] = (df['resistance_level'] - df['close']) / df['resistance_level']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Pattern recognition eklenirken hata: {e}")
            return df

    def _calculate_atr(self, df, period=14):
        """Average True Range hesaplama"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr
        except Exception as e:
            self.logger.warning(f"ATR hesaplama hatası: {e}")
            return pd.Series([0] * len(df), index=df.index)

    def _calculate_keltner_channels(self, df, period=20, multiplier=2):
        """Keltner Channels hesaplama"""
        try:
            atr = self._calculate_atr(df, period)
            ema = df['close'].ewm(span=period).mean()
            
            upper = ema + (multiplier * atr)
            lower = ema - (multiplier * atr)
            
            return upper, ema, lower
        except Exception as e:
            self.logger.warning(f"Keltner Channels hesaplama hatası: {e}")
            return pd.Series([0] * len(df), index=df.index), pd.Series([0] * len(df), index=df.index), pd.Series([0] * len(df), index=df.index)

    def _add_advanced_features(self, df):
        """Gelişmiş özellikler ekle"""
        try:
            # Fiyat değişimi
            df['price_change'] = df['close'].pct_change()
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_10'] = df['close'].pct_change(10)
            
            # Getiri hesaplamaları
            df['return_5'] = df['close'].pct_change(5)
            df['return_10'] = df['close'].pct_change(10)
            df['return_20'] = df['close'].pct_change(20)
            
            # Kümülatif getiri
            df['cumulative_return'] = (1 + df['price_change']).cumprod() - 1
            
            # Momentum
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            
            # Volatilite
            df['volatility'] = df['price_change'].rolling(window=20).std()
            df['volatility_5'] = df['price_change'].rolling(window=5).std()
            df['volatility_10'] = df['price_change'].rolling(window=10).std()
            df['volatility_20'] = df['price_change'].rolling(window=20).std()
            
            # Volume MA
            df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
            df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
            
            # Dinamik eşik
            df['dynamic_threshold'] = df['volatility'] * 2
            
            # Label'lar (gelecek getiriye göre)
            df['label_5'] = (df['return_5'].shift(-5) > 0.02).astype(int)
            df['label_10'] = (df['return_10'].shift(-10) > 0.03).astype(int)
            df['label_20'] = (df['return_20'].shift(-20) > 0.05).astype(int)
            df['label_dynamic'] = (df['return_5'].shift(-5) > df['dynamic_threshold']).astype(int)
            
            # Zaman özellikleri
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['hour'] = df['timestamp'].dt.hour
            else:
                # Timestamp yoksa index'i kullan
                df['day_of_week'] = 0  # Varsayılan değer
                df['hour'] = 12  # Varsayılan değer
            
            return df
            
        except Exception as e:
            self.logger.error(f"Gelişmiş özellikler eklenirken hata: {e}")
            return df

    def calculate_advanced_patterns(self, df):
        if 'momentum' not in df.columns:
            df['momentum'] = df['close'].diff().fillna(0)
        try:
            df = df.copy()
            
            # Candlestick Patterns
            df['doji'] = ((df['high'] - df['low']) <= (df['close'] - df['open']) * 0.1).astype(int)
            df['hammer'] = ((df['close'] - df['low']) > 2 * (df['high'] - df['close'])) & \
                          ((df['high'] - df['low']) > 3 * (df['close'] - df['open']))
            df['shooting_star'] = ((df['high'] - df['close']) > 2 * (df['close'] - df['low'])) & \
                                 ((df['high'] - df['low']) > 3 * (df['close'] - df['open']))
            
            # Support/Resistance Levels
            df['support_level'] = df['low'].rolling(20).min()
            df['resistance_level'] = df['high'].rolling(20).max()
            df['price_vs_support'] = (df['close'] - df['support_level']) / df['support_level']
            df['price_vs_resistance'] = (df['resistance_level'] - df['close']) / df['resistance_level']
            
            # Volume Analysis
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['high_volume'] = (df['volume_ratio'] > 2.0).astype(int)
            
            # Price Action
            df['body_size'] = abs(df['close'] - df['open']) / df['close']
            df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
            df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
            
            # Momentum Patterns
            df['momentum_ma'] = df['momentum'].rolling(10).mean()
            df['momentum_trend'] = (df['momentum'] > df['momentum_ma']).astype(int)
            
            # Breakout Detection
            df['breakout_up'] = (df['close'] > df['resistance_level'].shift(1)).astype(int)
            df['breakout_down'] = (df['close'] < df['support_level'].shift(1)).astype(int)
            
            # Consolidation Detection
            df['price_range'] = (df['high'] - df['low']) / df['close']
            df['consolidation'] = (df['price_range'] < df['price_range'].rolling(20).mean() * 0.5).astype(int)
            
            # --- Bollinger Sıkışması (Squeeze) Özelliği ---
            if 'bb_width' in df.columns:
                bb_squeeze_threshold = df['bb_width'].rolling(100, min_periods=20).quantile(0.15)
                df['bb_squeeze'] = (df['bb_width'] < bb_squeeze_threshold).astype(int)
            else:
                df['bb_squeeze'] = 0
            
            # --- Pattern Score Özelliği ---
            bullish_patterns = [
                'hammer', 'engulfing_bullish', 'morning_star', 'double_bottom', 'inverse_head_shoulders', 'triangle_ascending', 'flag_bullish'
            ]
            bearish_patterns = [
                'shooting_star', 'engulfing_bearish', 'evening_star', 'double_top', 'head_shoulders', 'triangle_descending', 'flag_bearish'
            ]
            
            # Ağırlıklar: Hammer/Engulfing/MorningStar/DoubleBottom/InverseHS/AscTriangle/FlagBullish = 2, diğerleri = 1
            bullish_score = sum([2*df[p] if p in ['hammer','engulfing_bullish','morning_star','double_bottom','inverse_head_shoulders','triangle_ascending','flag_bullish'] else df[p] for p in bullish_patterns if p in df.columns])
            bearish_score = sum([2*df[p] if p in ['shooting_star','engulfing_bearish','evening_star','double_top','head_shoulders','triangle_descending','flag_bearish'] else df[p] for p in bearish_patterns if p in df.columns])
            df['pattern_score'] = bullish_score - bearish_score
            
            return df
            
        except Exception as e:
            self.logger.error(f"Pattern recognition hatası: {e}")
            return df

    def analyze_patterns(self, df):
        # Eksik sütunları ekle
        required_cols = ['momentum', 'triangle_ascending']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
        if df is None or df.empty:
            return df
        df = df.copy()
        
        # Candlestick Patterns
        df['doji'] = self._is_doji(df)
        df['hammer'] = self._is_hammer(df)
        df['shooting_star'] = self._is_shooting_star(df)
        df['engulfing_bullish'] = self._is_engulfing_bullish(df)
        df['engulfing_bearish'] = self._is_engulfing_bearish(df)
        df['morning_star'] = self._is_morning_star(df)
        df['evening_star'] = self._is_evening_star(df)
        
        # Chart Patterns
        df['double_bottom'] = self._is_double_bottom(df)
        df['double_top'] = self._is_double_top(df)
        df['head_shoulders'] = self._is_head_shoulders(df)
        df['inverse_head_shoulders'] = self._is_inverse_head_shoulders(df)
        df['triangle_ascending'] = self._is_triangle_ascending(df)
        df['triangle_descending'] = self._is_triangle_descending(df)
        df['flag_bullish'] = self._is_flag_bullish(df)
        df['flag_bearish'] = self._is_flag_bearish(df)
        
        # Support/Resistance Levels
        df['support_level'] = self._find_support_level(df)
        df['resistance_level'] = self._find_resistance_level(df)
        df['near_support'] = (df['close'] <= df['support_level'] * 1.02).astype(int)
        df['near_resistance'] = (df['close'] >= df['resistance_level'] * 0.98).astype(int)
        
        # Breakout Detection
        df['breakout_up'] = self._detect_breakout_up(df)
        df['breakout_down'] = self._detect_breakout_down(df)
        
        # --- Bollinger Sıkışması (Squeeze) Özelliği ---
        if 'bb_width' in df.columns:
            bb_squeeze_threshold = df['bb_width'].rolling(100, min_periods=20).quantile(0.15)
            df['bb_squeeze'] = (df['bb_width'] < bb_squeeze_threshold).astype(int)
        else:
            df['bb_squeeze'] = 0
        
        # --- Pattern Score Özelliği ---
        bullish_patterns = [
            'hammer', 'engulfing_bullish', 'morning_star', 'double_bottom', 'inverse_head_shoulders', 'triangle_ascending', 'flag_bullish'
        ]
        bearish_patterns = [
            'shooting_star', 'engulfing_bearish', 'evening_star', 'double_top', 'head_shoulders', 'triangle_descending', 'flag_bearish'
        ]
        
        # Ağırlıklar: Hammer/Engulfing/MorningStar/DoubleBottom/InverseHS/AscTriangle/FlagBullish = 2, diğerleri = 1
        bullish_score = sum([2*df[p] if p in ['hammer','engulfing_bullish','morning_star','double_bottom','inverse_head_shoulders','triangle_ascending','flag_bullish'] else df[p] for p in bullish_patterns if p in df.columns])
        bearish_score = sum([2*df[p] if p in ['shooting_star','engulfing_bearish','evening_star','double_top','head_shoulders','triangle_descending','flag_bearish'] else df[p] for p in bearish_patterns if p in df.columns])
        df['pattern_score'] = bullish_score - bearish_score
        
        return df
    
    def _is_doji(self, df, tolerance=0.1):
        """Doji pattern detection"""
        body_size = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        result = (body_size <= total_range * tolerance).astype(int)
        return pd.Series(result, index=df.index)
    
    def _is_hammer(self, df):
        """Hammer pattern detection"""
        body_size = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        return ((lower_shadow > 2 * body_size) & (upper_shadow < body_size)).astype(int)
    
    def _is_shooting_star(self, df):
        """Shooting star pattern detection"""
        body_size = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        return ((upper_shadow >= 2 * body_size) & (lower_shadow <= 0.1 * body_size)).astype(int)
    
    def _is_engulfing_bullish(self, df):
        """Bullish engulfing pattern detection"""
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        return ((df['open'] < prev_close) & (df['close'] > prev_open) & 
                (prev_close < prev_open)).astype(int)
    
    def _is_engulfing_bearish(self, df):
        """Bearish engulfing pattern detection"""
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        return ((df['open'] > prev_close) & (df['close'] < prev_open) & 
                (prev_close > prev_open)).astype(int)
    
    def _is_morning_star(self, df):
        """Morning star pattern detection"""
        # Basitleştirilmiş morning star detection
        prev_close = df['close'].shift(1)
        prev2_close = df['close'].shift(2)
        return ((df['close'] > prev_close) & (prev_close < prev2_close)).astype(int)
    
    def _is_evening_star(self, df):
        """Evening star pattern detection"""
        # Basitleştirilmiş evening star detection
        prev_close = df['close'].shift(1)
        prev2_close = df['close'].shift(2)
        return ((df['close'] < prev_close) & (prev_close > prev2_close)).astype(int)
    
    def _is_double_bottom(self, df, window=20):
        """Double bottom pattern detection"""
        lows = df['low'].rolling(window=window, center=True).min()
        return (df['low'] == lows).astype(int)
    
    def _is_double_top(self, df, window=20):
        """Double top pattern detection"""
        highs = df['high'].rolling(window=window, center=True).max()
        return (df['high'] == highs).astype(int)
    
    def _is_head_shoulders(self, df):
        """Head and shoulders pattern detection (basitleştirilmiş)"""
        # Bu pattern detection daha karmaşık, şimdilik basit bir yaklaşım
        return pd.Series(0, index=df.index)
    
    def _is_inverse_head_shoulders(self, df):
        """Inverse head and shoulders pattern detection (basitleştirilmiş)"""
        return pd.Series(0, index=df.index)
    
    def _is_triangle_ascending(self, df, window=20):
        """Ascending triangle pattern detection"""
        highs = df['high'].rolling(window=window).max()
        lows = df['low'].rolling(window=window).min()
        return ((highs.diff() < 0.001) & (lows.diff() > 0)).astype(int)
    
    def _is_triangle_descending(self, df, window=20):
        """Descending triangle pattern detection"""
        highs = df['high'].rolling(window=window).max()
        lows = df['low'].rolling(window=window).min()
        result = ((highs.diff() < 0) & (lows.diff() < 0.001)).astype(int)
        return pd.Series(result, index=df.index)
    
    def _is_flag_bullish(self, df):
        """Bullish flag pattern detection (basitleştirilmiş)"""
        return pd.Series(0, index=df.index)
    
    def _is_flag_bearish(self, df):
        """Bearish flag pattern detection (basitleştirilmiş)"""
        return pd.Series(0, index=df.index)
    
    def _find_support_level(self, df, window=50):
        """Support level detection"""
        return df['low'].rolling(window=window).min()
    
    def _find_resistance_level(self, df, window=50):
        """Resistance level detection"""
        return df['high'].rolling(window=window).max()
    
    def _detect_breakout_up(self, df, window=20):
        """Upward breakout detection"""
        resistance = df['high'].rolling(window=window).max().shift(1)
        return (df['close'] > resistance).astype(int)
    
    def _detect_breakout_down(self, df, window=20):
        """Downward breakout detection"""
        support = df['low'].rolling(window=window).min().shift(1)
        result = (df['close'] < support).astype(int)
        return pd.Series(result, index=df.index)

    def analyze_technical_signals(self, df):
        """
        Tüm ana teknik indikatörleri, formasyonları ve önemli sinyal noktalarını JSON/dict olarak döndürür.
        """
        result = {}
        try:
            df = df.copy()
            df = self.calculate_all_indicators(df)
            df = self.add_advanced_indicators(df)
            df = self.analyze_patterns(df)

            last = df.iloc[-1]
            # 1) Trend ve ortalamalar
            result['ema'] = {p: float(last.get(f'ema_{p}', np.nan)) for p in [5,10,20,50,200]}
            result['sma'] = {p: float(last.get(f'sma_{p}', np.nan)) for p in [50,200]}
            result['ema_cross'] = {
                'golden_cross': last.get('ema_50', 0) > last.get('ema_200', 0),
                'death_cross': last.get('ema_50', 0) < last.get('ema_200', 0)
            }
            result['sma_cross'] = {
                'golden_cross': last.get('sma_50', 0) > last.get('sma_200', 0),
                'death_cross': last.get('sma_50', 0) < last.get('sma_200', 0)
            }
            result['ichimoku'] = {
                'kumo_break': float(last['close']) > float(last.get('senkou_span_a', 0)) and float(last['close']) > float(last.get('senkou_span_b', 0)),
                'tenkan_kijun_cross': last.get('tenkan_sen', 0) > last.get('kijun_sen', 0)
            }

            # 2) Momentum
            result['rsi'] = {p: float(last.get(f'rsi_{p}', np.nan)) for p in [7,14]}
            result['rsi_signal'] = {
                'overbought': last.get('rsi_14', 0) > 70,
                'oversold': last.get('rsi_14', 0) < 30
            }
            result['stoch_rsi'] = float(last.get('stoch_rsi', np.nan))
            result['stoch_signal'] = {
                'k_above_d': last.get('stoch_k', 0) > last.get('stoch_d', 0),
                'k_below_d': last.get('stoch_k', 0) < last.get('stoch_d', 0)
            }
            result['cci'] = float(last.get('cci', np.nan))
            result['cci_signal'] = {
                'overbought': last.get('cci', 0) > 100,
                'oversold': last.get('cci', 0) < -100
            }
            result['williams_r'] = float(last.get('williams_r', np.nan))
            result['williams_r_signal'] = {
                'overbought': last.get('williams_r', 0) > -20,
                'oversold': last.get('williams_r', 0) < -80
            }

            # 3) Volatilite
            result['bollinger'] = {
                'upper': float(last.get('bb_upper', np.nan)),
                'lower': float(last.get('bb_lower', np.nan)),
                'middle': float(last.get('bb_middle', np.nan)),
                'width': float(last.get('bb_width', np.nan)),
                'percent': float(last.get('bb_percent', np.nan)),
                'touch_upper': last.get('close', 0) >= last.get('bb_upper', 0),
                'touch_lower': last.get('close', 0) <= last.get('bb_lower', 0)
            }
            result['atr'] = float(last.get('atr', np.nan))
            result['atr_stop_loss'] = float(last.get('close', 0)) - float(last.get('atr', 0))

            # 4) Trend gücü
            result['adx'] = float(last.get('adx', np.nan))
            result['adx_signal'] = {'strong_trend': last.get('adx', 0) > 25}
            result['macd'] = {
                'macd': float(last.get('macd', np.nan)),
                'signal': float(last.get('macd_signal', np.nan)),
                'histogram': float(last.get('macd_histogram', np.nan)),
                'bullish_cross': last.get('macd', 0) > last.get('macd_signal', 0),
                'bearish_cross': last.get('macd', 0) < last.get('macd_signal', 0)
            }

            # 5) Hacim
            result['obv'] = float(last.get('obv', np.nan))
            # OBV uyumsuzluk için basit bir kontrol (fiyat artarken obv düşüyor mu?)
            result['obv_divergence'] = (last.get('close', 0) > df['close'].iloc[-5]) and (last.get('obv', 0) < df['obv'].iloc[-5])

            # 6) Formasyonlar
            result['patterns'] = {
                'double_top': bool(last.get('double_top', 0)),
                'double_bottom': bool(last.get('double_bottom', 0)),
                'head_shoulders': bool(last.get('head_shoulders', 0)),
                'inverse_head_shoulders': bool(last.get('inverse_head_shoulders', 0)),
                'flag_bullish': bool(last.get('flag_bullish', 0)),
                'flag_bearish': bool(last.get('flag_bearish', 0)),
                'triangle_ascending': bool(last.get('triangle_ascending', 0)),
                'triangle_descending': bool(last.get('triangle_descending', 0))
            }

            # 7) Destek/direnç ve fibonacci
            result['pivot'] = float(last.get('pivot', np.nan))
            result['support'] = float(last.get('support_level', np.nan))
            result['resistance'] = float(last.get('resistance_level', np.nan))
            result['fibonacci'] = {
                'fib_23_6': float(last.get('fib_23.6', np.nan)),
                'fib_38_2': float(last.get('fib_38.2', np.nan)),
                'fib_50_0': float(last.get('fib_50.0', np.nan)),
                'fib_61_8': float(last.get('fib_61.8', np.nan))
            }

            return result
        except Exception as e:
            self.logger.error(f"analyze_technical_signals hata: {e}")
            return {'error': str(e)}

    def multi_timeframe_analysis(self, df_1h, df_4h, df_1d):
        """Çoklu zaman dilimi analizi - 1h, 4h, 1d birleştirme"""
        try:
            analysis = {}
            # 1 saatlik analiz
            if df_1h is not None and not df_1h.empty:
                indicators_1h = self.calculate_all_indicators(df_1h)
                # Pattern analizi uygula
                patterns_1h = self.analyze_patterns(indicators_1h)
                # Ek skorlar
                volume_score = self.calculate_volume_score(patterns_1h) if hasattr(self, 'calculate_volume_score') else None
                momentum_score = self.calculate_momentum_score(patterns_1h) if hasattr(self, 'calculate_momentum_score') else None
                # Pattern skoru son satırdan al
                pattern_score = None
                if 'pattern_score' in patterns_1h.columns:
                    pattern_score = patterns_1h['pattern_score'].iloc[-1]
                # LOG EKLE
                self.logger.info(f"[DEBUG][TA] 1h volume_score={volume_score}, momentum_score={momentum_score}, pattern_score={pattern_score}")
                self.logger.info(f"[DEBUG][TA] 1h last_row={patterns_1h.iloc[-1].to_dict()}")
                analysis['1h'] = {
                    'trend': self.get_trend_direction(patterns_1h),
                    'strength': self.calculate_trend_strength(patterns_1h),
                    'volume_score': volume_score if volume_score is not None else 0.0,
                    'momentum': momentum_score if momentum_score is not None else 0.0,
                    'pattern_score': pattern_score if pattern_score is not None else 0.0
                }
            # 4 saatlik analiz
            if df_4h is not None and not df_4h.empty:
                indicators_4h = self.calculate_all_indicators(df_4h)
                analysis['4h'] = {
                    'trend': self.get_trend_direction(indicators_4h),
                    'strength': self.calculate_trend_strength(indicators_4h)
                }
            # 1 günlük analiz
            if df_1d is not None and not df_1d.empty:
                indicators_1d = self.calculate_all_indicators(df_1d)
                analysis['1d'] = {
                    'trend': self.get_trend_direction(indicators_1d),
                    'strength': self.calculate_trend_strength(indicators_1d)
                }
            return analysis
        except Exception as e:
            self.logger.error(f"[DEBUG][TA] Çoklu zaman dilimi analizi hatası: {e}")
            return {}

    def _add_extra_indicators(self, df):
        """Ekstra indikatörler ekle - 150+ GÜÇLÜ FEATURE"""
        try:
            # ROC (Rate of Change)
            df['roc'] = ta.momentum.ROCIndicator(close=df['close']).roc()
            df['price_roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
            df['historical_volatility'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
            df['volatility_ratio'] = df['atr'] / df['close']
            
            # Rolling Statistics
            df['rolling_mean_5'] = df['close'].rolling(5).mean()
            df['rolling_mean_10'] = df['close'].rolling(10).mean()
            df['rolling_mean_20'] = df['close'].rolling(20).mean()
            df['rolling_std_5'] = df['close'].rolling(5).std()
            df['rolling_std_10'] = df['close'].rolling(10).std()
            df['rolling_std_20'] = df['close'].rolling(20).std()
            df['rolling_min_5'] = df['close'].rolling(5).min()
            df['rolling_max_5'] = df['close'].rolling(5).max()
            df['rolling_min_10'] = df['close'].rolling(10).min()
            df['rolling_max_10'] = df['close'].rolling(10).max()
            df['rolling_median_5'] = df['close'].rolling(5).median()
            df['rolling_median_10'] = df['close'].rolling(10).median()
            
            # Price Changes
            df['price_change_1'] = df['close'].pct_change(1)
            df['price_change_3'] = df['close'].pct_change(3)
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_10'] = df['close'].pct_change(10)
            df['price_change_15'] = df['close'].pct_change(15)
            df['price_change_20'] = df['close'].pct_change(20)
            df['price_change_30'] = df['close'].pct_change(30)
            df['price_change_50'] = df['close'].pct_change(50)
            
            # Volume Changes
            df['volume_change_1'] = df['volume'].pct_change(1)
            df['volume_change_3'] = df['volume'].pct_change(3)
            df['volume_change_5'] = df['volume'].pct_change(5)
            df['volume_change_10'] = df['volume'].pct_change(10)
            df['volume_change_15'] = df['volume'].pct_change(15)
            df['volume_change_20'] = df['volume'].pct_change(20)
            
            # Price Positions
            df['price_position_5'] = (df['close'] - df['rolling_min_5']) / (df['rolling_max_5'] - df['rolling_min_5'])
            df['price_position_10'] = (df['close'] - df['rolling_min_10']) / (df['rolling_max_10'] - df['rolling_min_10'])
            df['price_position_20'] = (df['close'] - df['close'].rolling(20).min()) / (df['close'].rolling(20).max() - df['close'].rolling(20).min())
            df['price_position_50'] = (df['close'] - df['close'].rolling(50).min()) / (df['close'].rolling(50).max() - df['close'].rolling(50).min())
            
            # Momentum Indicators
            df['momentum_1'] = df['close'] - df['close'].shift(1)
            df['momentum_3'] = df['close'] - df['close'].shift(3)
            df['momentum_5'] = df['close'] - df['close'].shift(5)
            df['momentum_10'] = df['close'] - df['close'].shift(10)
            df['momentum_15'] = df['close'] - df['close'].shift(15)
            df['momentum_20'] = df['close'] - df['close'].shift(20)
            df['momentum_30'] = df['close'] - df['close'].shift(30)
            df['momentum_50'] = df['close'] - df['close'].shift(50)
            
            # Volatility Indicators
            df['volatility_1'] = df['close'].pct_change().rolling(1).std()
            df['volatility_3'] = df['close'].pct_change().rolling(3).std()
            df['volatility_5'] = df['close'].pct_change().rolling(5).std()
            df['volatility_10'] = df['close'].pct_change().rolling(10).std()
            df['volatility_15'] = df['close'].pct_change().rolling(15).std()
            df['volatility_20'] = df['close'].pct_change().rolling(20).std()
            df['volatility_30'] = df['close'].pct_change().rolling(30).std()
            df['volatility_50'] = df['close'].pct_change().rolling(50).std()
            
            # Volume Moving Averages
            df['volume_ma_1'] = df['volume'].rolling(1).mean()
            df['volume_ma_3'] = df['volume'].rolling(3).mean()
            df['volume_ma_5'] = df['volume'].rolling(5).mean()
            df['volume_ma_10'] = df['volume'].rolling(10).mean()
            df['volume_ma_15'] = df['volume'].rolling(15).mean()
            df['volume_ma_20'] = df['volume'].rolling(20).mean()
            
            # Price Ratios
            df['price_ratio_5'] = df['close'] / df['rolling_mean_5']
            df['price_ratio_10'] = df['close'] / df['rolling_mean_10']
            df['price_ratio_20'] = df['close'] / df['rolling_mean_20']
            df['price_ratio_30'] = df['close'] / df['close'].rolling(30).mean()
            df['price_ratio_50'] = df['close'] / df['close'].rolling(50).mean()
            df['price_ratio_200'] = df['close'] / df['close'].rolling(200).mean()
            
            # Spread Indicators
            df['high_low_ratio'] = df['high'] / df['low']
            df['open_close_ratio'] = df['open'] / df['close']
            df['close_open_ratio'] = df['close'] / df['open']
            df['high_close_ratio'] = df['high'] / df['close']
            
            # Z-Score Indicators
            df['price_zscore_5'] = (df['close'] - df['rolling_mean_5']) / df['rolling_std_5']
            df['price_zscore_10'] = (df['close'] - df['rolling_mean_10']) / df['rolling_std_10']
            df['price_zscore_20'] = (df['close'] - df['rolling_mean_20']) / df['rolling_std_20']
            df['price_zscore_30'] = (df['close'] - df['close'].rolling(30).mean()) / df['close'].rolling(30).std()
            df['price_zscore_50'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()
            df['price_zscore_200'] = (df['close'] - df['close'].rolling(200).mean()) / df['close'].rolling(200).std()
            
            # Advanced Momentum
            df['rate_of_change_5'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100
            df['rate_of_change_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
            df['rate_of_change_20'] = ((df['close'] - df['close'].shift(20)) / df['close'].shift(20)) * 100
            df['momentum_roc_5'] = df['momentum_5'] / df['close'].shift(5) * 100
            df['momentum_roc_10'] = df['momentum_10'] / df['close'].shift(10) * 100
            df['momentum_roc_20'] = df['momentum_20'] / df['close'].shift(20) * 100
            df['price_momentum_5'] = df['close'] / df['close'].shift(5)
            df['price_momentum_10'] = df['close'] / df['close'].shift(10)
            
            # Advanced Volatility
            df['true_range'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
            df['average_true_range_5'] = df['true_range'].rolling(5).mean()
            df['average_true_range_10'] = df['true_range'].rolling(10).mean()
            df['volatility_std_5'] = df['close'].pct_change().rolling(5).std()
            df['volatility_std_10'] = df['close'].pct_change().rolling(10).std()
            df['volatility_std_20'] = df['close'].pct_change().rolling(20).std()
            
            # Volume Analysis
            df['volume_sma_5'] = df['volume'].rolling(5).mean()
            df['volume_sma_10'] = df['volume'].rolling(10).mean()
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ema_5'] = df['volume'].ewm(span=5).mean()
            df['volume_ema_10'] = df['volume'].ewm(span=10).mean()
            df['volume_ema_20'] = df['volume'].ewm(span=20).mean()
            df['volume_ratio_5'] = df['volume'] / df['volume_sma_5']
            df['volume_ratio_10'] = df['volume'] / df['volume_sma_10']
            
            # Time Features
            df['day_of_week'] = pd.to_datetime(df.index).dayofweek
            df['hour'] = pd.to_datetime(df.index).hour
            df['month'] = pd.to_datetime(df.index).month
            df['quarter'] = pd.to_datetime(df.index).quarter
            
            # Statistical Features
            df['skewness_5'] = df['close'].pct_change().rolling(5).skew()
            df['skewness_10'] = df['close'].pct_change().rolling(10).skew()
            df['skewness_20'] = df['close'].pct_change().rolling(20).skew()
            df['kurtosis_5'] = df['close'].pct_change().rolling(5).kurt()
            df['kurtosis_10'] = df['close'].pct_change().rolling(10).kurt()
            df['kurtosis_20'] = df['close'].pct_change().rolling(20).kurt()
            df['correlation_5'] = df['close'].rolling(5).corr(df['volume'])
            df['correlation_10'] = df['close'].rolling(10).corr(df['volume'])
            
            # Placeholder features (will be filled with 0 if not available)
            placeholder_features = [
                'price_gap_up', 'price_gap_down', 'price_breakout_up', 'price_breakout_down', 'price_consolidation', 'price_trend_strength',
                'near_support', 'near_resistance', 'support_strength', 'resistance_strength',
                'parabolic_sar', 'commodity_channel_index', 'money_flow_index', 'ultimate_oscillator', 'stochastic_rsi', 'williams_alligator',
                'fractal_chaos_bands', 'ease_of_movement', 'mass_index', 'detrended_price_oscillator',
                'bid_ask_spread', 'order_flow_imbalance', 'market_depth', 'liquidity_ratio', 'volume_profile', 'price_impact',
                'higher_tf_trend', 'lower_tf_momentum', 'timeframe_alignment', 'multi_tf_support', 'multi_tf_resistance', 'timeframe_divergence',
                'higher_tf_volatility', 'lower_tf_volume',
                'doji_pattern', 'hammer_pattern', 'shooting_star_pattern', 'engulfing_bullish', 'engulfing_bearish', 'morning_star',
                'evening_star', 'double_bottom', 'double_top', 'head_shoulders',
                'trending_market', 'ranging_market', 'volatile_market', 'low_volatility_market',
                'var_95_5', 'var_95_10', 'var_95_20', 'expected_shortfall_5', 'expected_shortfall_10', 'expected_shortfall_20',
                'fear_greed_index', 'market_sentiment', 'social_sentiment', 'news_sentiment',
                'price_rsi_divergence', 'price_macd_divergence', 'price_volume_divergence', 'momentum_divergence', 'volatility_divergence', 'trend_divergence'
            ]
            
            for feature in placeholder_features:
                df[feature] = 0
            
            return df
        except Exception as e:
            self.logger.error(f"Ekstra indikatörler eklenirken hata: {e}")
            return df
    
    def _add_price_patterns(self, df):
        """Fiyat pattern'leri ekle"""
        try:
            df['price_change'] = df['close'].pct_change()
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_10'] = df['close'].pct_change(10)
            df['return_5'] = df['close'].pct_change(5)
            df['return_10'] = df['close'].pct_change(10)
            df['return_20'] = df['close'].pct_change(20)
            df['cumulative_return'] = (1 + df['price_change']).cumprod()
            df['momentum_5'] = df['close'] - df['close'].shift(5)
            df['momentum_10'] = df['close'] - df['close'].shift(10)
            return df
        except Exception as e:
            self.logger.error(f"Fiyat pattern'leri eklenirken hata: {e}")
            return df
    
    def _add_volume_patterns(self, df):
        """Hacim pattern'leri ekle"""
        try:
            df['volume_roc'] = df['volume'].pct_change()
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ma_5'] = df['volume'].rolling(5).mean()
            df['volume_ma_10'] = df['volume'].rolling(10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            return df
        except Exception as e:
            self.logger.error(f"Hacim pattern'leri eklenirken hata: {e}")
            return df
    
    def _add_trend_patterns(self, df):
        """Trend pattern'leri ekle"""
        try:
            df['volatility'] = df['close'].pct_change().rolling(20).std()
            df['volatility_5'] = df['close'].pct_change().rolling(5).std()
            df['volatility_10'] = df['close'].pct_change().rolling(10).std()
            df['volatility_20'] = df['close'].pct_change().rolling(20).std()
            return df
        except Exception as e:
            self.logger.error(f"Trend pattern'leri eklenirken hata: {e}")
            return df
    
    def _add_momentum_patterns(self, df):
        """Momentum pattern'leri ekle"""
        try:
            df['dynamic_threshold'] = df['close'].rolling(20).std() * 2
            return df
        except Exception as e:
            self.logger.error(f"Momentum pattern'leri eklenirken hata: {e}")
            return df
    
    def _add_volatility_patterns(self, df):
        """Volatilite pattern'leri ekle"""
        try:
            df['label_5'] = (df['close'].shift(-5) > df['close']).astype(int)
            df['label_10'] = (df['close'].shift(-10) > df['close']).astype(int)
            df['label_20'] = (df['close'].shift(-20) > df['close']).astype(int)
            df['label_dynamic'] = (df['close'].shift(-5) > df['close'] + df['dynamic_threshold']).astype(int)
            return df
        except Exception as e:
            self.logger.error(f"Volatilite pattern'leri eklenirken hata: {e}")
            return df
    
    def _add_support_resistance_features(self, df):
        """Support/Resistance özellikleri ekle"""
        try:
            df['support_level'] = df['low'].rolling(20).min()
            df['resistance_level'] = df['high'].rolling(20).max()
            df['near_support'] = (df['close'] <= df['support_level'] * 1.02).astype(int)
            df['near_resistance'] = (df['close'] >= df['resistance_level'] * 0.98).astype(int)
            return df
        except Exception as e:
            self.logger.error(f"Support/Resistance özellikleri eklenirken hata: {e}")
            return df
    
    def _add_time_features(self, df):
        """Zaman özellikleri ekle"""
        try:
            df['day_of_week'] = pd.to_datetime(df.index).dayofweek
            df['hour'] = pd.to_datetime(df.index).hour
            return df
        except Exception as e:
            self.logger.error(f"Zaman özellikleri eklenirken hata: {e}")
            return df
    
    def _add_statistical_features(self, df):
        """İstatistiksel özellikler ekle"""
        try:
            df['z_score'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['rsi_divergence'] = df['rsi_14'] - df['rsi_14'].shift(1)
            df['macd_divergence'] = df['macd'] - df['macd'].shift(1)
            return df
        except Exception as e:
            self.logger.error(f"İstatistiksel özellikler eklenirken hata: {e}")
            return df

    def calculate_advanced_ta_strength(self, df, multi_tf_analysis):
        """Gelişmiş teknik analiz gücü hesaplama"""
        try:
            if df.empty:
                return 0.0
            # Temel TA gücü
            base_ta_strength = self.calculate_ta_strength(df)
            # Çoklu zaman dilimi uyumu
            trend_alignment = multi_tf_analysis.get('trend_alignment', 0.0)
            # Momentum skoru
            momentum_score = multi_tf_analysis.get('1h', {}).get('momentum', 0.0)
            # Trend gücü
            trend_strength = multi_tf_analysis.get('1h', {}).get('strength', 0.0)
            # Ağırlıklı ortalama
            advanced_ta_strength = (
                base_ta_strength * 0.3 +
                trend_alignment * 0.3 +
                momentum_score * 0.2 +
                trend_strength * 0.2
            )
            return min(advanced_ta_strength, 1.0)
        except Exception as e:
            self.logger.error(f"Gelişmiş TA gücü hesaplama hatası: {e}")
            return 0.0

def convert_string_features_to_numeric(df):
    """String feature'ları sayısal değere çevirir."""
    # Trend
    if 'trend' in df.columns:
        df['trend'] = df['trend'].map({'neutral': 0, 'up': 1, 'down': -1}).fillna(0)
    # Market regime
    if 'market_regime' in df.columns:
        df['market_regime'] = df['market_regime'].map({'trending': 1, 'ranging': 0, 'volatile': 2}).fillna(0)
    # Sentiment
    for col in ['market_sentiment', 'social_sentiment', 'news_sentiment']:
        if col in df.columns:
            df[col] = df[col].map({'bullish': 1, 'bearish': -1, 'neutral': 0}).fillna(0)
    # Pattern (ör: doji_pattern, hammer_pattern, vb. varsa)
    pattern_cols = [c for c in df.columns if 'pattern' in c]
    for col in pattern_cols:
        if df[col].dtype == object:
            df[col] = df[col].map({'bullish': 1, 'bearish': -1, 'neutral': 0, 'none': 0}).fillna(0)
    return df