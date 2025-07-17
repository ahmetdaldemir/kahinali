import pandas as pd
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import requests
import json
from datetime import datetime, timedelta
from config import Config

class MarketAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.market_regime = 'unknown'
        self.volatility_regime = 'normal'
        self.sentiment_score = 0.0
        
    def detect_market_regime(self, btc_data):
        """Market regime detection - trend, sideways, volatile"""
        try:
            # Volatility calculation
            returns = btc_data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std()
            
            # Trend strength
            sma_20 = btc_data['close'].rolling(20).mean()
            sma_50 = btc_data['close'].rolling(50).mean()
            trend_strength = (sma_20 - sma_50) / sma_50
            
            # Price momentum
            momentum = btc_data['close'].pct_change(20)
            
            # RSI hesapla eğer yoksa
            if 'rsi_14' not in btc_data.columns:
                # Basit RSI hesaplama
                delta = btc_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                btc_data['rsi_14'] = rsi
            
            # Volume ratio hesapla
            volume_ratio = btc_data['volume'] / btc_data['volume'].rolling(20).mean()
            
            # Market regime features
            features = pd.DataFrame({
                'volatility': volatility,
                'trend_strength': trend_strength,
                'momentum': momentum,
                'rsi': btc_data['rsi_14'],
                'volume_ratio': volume_ratio
            }).dropna()
            
            # K-means clustering for regime detection
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(features_scaled)
            
            # Regime classification
            cluster_centers = kmeans.cluster_centers_
            
            # Identify regimes based on cluster characteristics
            regimes = []
            for i, center in enumerate(cluster_centers):
                if center[1] > 0.5:  # High trend strength
                    regimes.append('trending')
                elif center[0] > 0.5:  # High volatility
                    regimes.append('volatile')
                else:
                    regimes.append('sideways')
            
            # Current regime
            current_cluster = clusters[-1]
            self.market_regime = regimes[current_cluster]
            
            # Volatility regime
            current_vol = volatility.iloc[-1]
            if current_vol > volatility.quantile(0.8):
                self.volatility_regime = 'high'
            elif current_vol < volatility.quantile(0.2):
                self.volatility_regime = 'low'
            else:
                self.volatility_regime = 'normal'
                
            self.logger.info(f"Market regime: {self.market_regime}, Volatility: {self.volatility_regime}")
            return self.market_regime, self.volatility_regime
            
        except Exception as e:
            self.logger.error(f"Market regime detection hatası: {e}")
            return 'unknown', 'normal'
    
    def analyze_market_sentiment(self):
        """Market sentiment analysis - fear & greed, social media"""
        try:
            # Fear & Greed Index (simulated)
            # Gerçek uygulamada API'den alınabilir
            fear_greed = self.get_fear_greed_index()
            
            # Social media sentiment
            social_sentiment = self.get_social_sentiment()
            
            # News sentiment
            news_sentiment = self.get_news_sentiment()
            
            # Combined sentiment score
            self.sentiment_score = (fear_greed * 0.4 + social_sentiment * 0.3 + news_sentiment * 0.3)
            
            self.logger.info(f"Market sentiment score: {self.sentiment_score:.2f}")
            return self.sentiment_score
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis hatası: {e}")
            return 0.5
    
    def get_fear_greed_index(self):
        """Fear & Greed Index simulation"""
        # Gerçek uygulamada https://api.alternative.me/fng/ kullanılabilir
        try:
            # Simulated fear & greed index (0-100)
            # 0-25: Extreme Fear, 26-45: Fear, 46-55: Neutral, 56-75: Greed, 76-100: Extreme Greed
            base_score = 50
            # Market conditions'den etkilenen score
            if self.market_regime == 'trending':
                base_score += 10
            elif self.market_regime == 'volatile':
                base_score -= 15
            elif self.market_regime == 'sideways':
                base_score += 5
                
            # Normalize to 0-1
            return max(0, min(1, base_score / 100))
            
        except Exception as e:
            self.logger.error(f"Fear & Greed Index hatası: {e}")
            return 0.5
    
    def get_social_sentiment(self):
        """Social media sentiment analysis"""
        try:
            # Twitter, Reddit sentiment analysis
            # Gerçek uygulamada Twitter API ve Reddit API kullanılabilir
            
            # Simulated social sentiment
            base_sentiment = 0.5
            
            # Market regime'den etkilenen sentiment
            if self.market_regime == 'trending':
                base_sentiment += 0.1
            elif self.market_regime == 'volatile':
                base_sentiment -= 0.2
                
            return max(0, min(1, base_sentiment))
            
        except Exception as e:
            self.logger.error(f"Social sentiment hatası: {e}")
            return 0.5
    
    def get_news_sentiment(self):
        """News sentiment analysis"""
        try:
            # News API sentiment analysis
            # Gerçek uygulamada News API kullanılabilir
            
            # Simulated news sentiment
            base_sentiment = 0.5
            
            # Market conditions'den etkilenen sentiment
            if self.volatility_regime == 'high':
                base_sentiment -= 0.15
            elif self.volatility_regime == 'low':
                base_sentiment += 0.1
                
            return max(0, min(1, base_sentiment))
            
        except Exception as e:
            self.logger.error(f"News sentiment hatası: {e}")
            return 0.5
    
    def get_market_conditions(self):
        """Genel market conditions summary"""
        return {
            'regime': self.market_regime,
            'volatility': self.volatility_regime,
            'sentiment': self.sentiment_score,
            'timestamp': str(datetime.now())
        }

    def analyze_market_regime(self, df):
        """Market regime analizi - trend, range, volatility"""
        try:
            df = df.copy()
            
            # Volatilite hesapla
            df['volatility'] = df['close'].pct_change().rolling(20).std()
            df['volatility_ma'] = df['volatility'].rolling(50).mean()
            
            # Trend gücü hesapla
            df['trend_strength'] = abs(df['close'] - df['close'].shift(20)) / df['close'].shift(20)
            df['trend_ma'] = df['trend_strength'].rolling(50).mean()
            
            # Range hesapla
            df['range'] = (df['high'] - df['low']) / df['close']
            df['range_ma'] = df['range'].rolling(20).mean()
            
            # Market regime belirleme
            df['market_regime'] = 'NEUTRAL'
            
            # Trending market
            trending_condition = (
                (df['trend_strength'] > df['trend_ma'] * 1.2) &
                (df['volatility'] > df['volatility_ma'] * 0.8)
            )
            df.loc[trending_condition, 'market_regime'] = 'TRENDING'
            
            # Ranging market
            ranging_condition = (
                (df['trend_strength'] < df['trend_ma'] * 0.8) &
                (df['range'] > df['range_ma'] * 1.2)
            )
            df.loc[ranging_condition, 'market_regime'] = 'RANGING'
            
            # Volatile market
            volatile_condition = (
                (df['volatility'] > df['volatility_ma'] * 1.5) &
                (df['range'] > df['range_ma'] * 1.5)
            )
            df.loc[volatile_condition, 'market_regime'] = 'VOLATILE'
            
            # Low volatility market
            low_vol_condition = (
                (df['volatility'] < df['volatility_ma'] * 0.5) &
                (df['range'] < df['range_ma'] * 0.5)
            )
            df.loc[low_vol_condition, 'market_regime'] = 'LOW_VOL'
            
            return df
            
        except Exception as e:
            self.logger.error(f"Market regime analizi hatası: {e}")
            return df

    def calculate_market_sentiment(self, df):
        """Market sentiment skoru hesapla"""
        try:
            df = df.copy()
            
            # RSI sentiment
            rsi = df['rsi_14']
            rsi_sentiment = np.where(rsi > 70, -1, np.where(rsi < 30, 1, 0))
            
            # MACD sentiment
            macd = df['macd']
            macd_signal = df['macd_signal']
            macd_sentiment = np.where(macd > macd_signal, 1, -1)
            
            # Bollinger Band sentiment
            bb_position = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            bb_sentiment = np.where(bb_position > 0.8, -1, np.where(bb_position < 0.2, 1, 0))
            
            # Volume sentiment
            volume_ma = df['volume'].rolling(20).mean()
            volume_sentiment = np.where(df['volume'] > volume_ma * 1.5, 1, np.where(df['volume'] < volume_ma * 0.5, -1, 0))
            
            # Price momentum sentiment
            price_momentum = df['close'].pct_change(5)
            momentum_sentiment = np.where(price_momentum > 0.05, 1, np.where(price_momentum < -0.05, -1, 0))
            
            # Ağırlıklı sentiment skoru
            sentiment_score = (
                rsi_sentiment * 0.2 +
                macd_sentiment * 0.25 +
                bb_sentiment * 0.15 +
                volume_sentiment * 0.2 +
                momentum_sentiment * 0.2
            )
            
            # Normalize et (0-1 arası)
            df['sentiment_score'] = (sentiment_score + 1) / 2
            
            return df
            
        except Exception as e:
            self.logger.error(f"Market sentiment hesaplama hatası: {e}")
            df['sentiment_score'] = 0.5
            return df

    def detect_market_anomalies(self, df):
        """Market anomalilerini tespit et"""
        try:
            df = df.copy()
            
            # Volume spike
            volume_ma = df['volume'].rolling(20).mean()
            volume_std = df['volume'].rolling(20).std()
            df['volume_spike'] = (df['volume'] > volume_ma + 2 * volume_std).astype(int)
            
            # Price gap
            df['price_gap'] = abs(df['close'] - df['close'].shift(1)) / df['close'].shift(1)
            df['price_gap_anomaly'] = (df['price_gap'] > df['price_gap'].rolling(20).quantile(0.95)).astype(int)
            
            # Volatility spike
            volatility_ma = df['volatility'].rolling(20).mean()
            volatility_std = df['volatility'].rolling(20).std()
            df['volatility_spike'] = (df['volatility'] > volatility_ma + 2 * volatility_std).astype(int)
            
            # Support/Resistance break
            df['support_break'] = (df['close'] < df['low'].rolling(20).min()).astype(int)
            df['resistance_break'] = (df['close'] > df['high'].rolling(20).max()).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Market anomaly tespit hatası: {e}")
            return df

    def calculate_market_efficiency(self, df):
        """Market verimliliğini hesapla"""
        try:
            df = df.copy()
            
            # Hurst exponent (trend strength)
            def hurst_exponent(series):
                lags = range(2, 100)
                tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
                reg = np.polyfit(np.log(lags), np.log(tau), 1)
                return reg[0]
            
            # Rolling Hurst exponent
            df['hurst_exponent'] = df['close'].rolling(100).apply(hurst_exponent, raw=True)
            
            # Market efficiency ratio
            df['efficiency_ratio'] = abs(df['close'] - df['close'].shift(20)) / df['close'].rolling(20).apply(lambda x: sum(abs(x.diff().dropna())), raw=True)
            
            # Mean reversion vs momentum
            df['mean_reversion'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Market efficiency hesaplama hatası: {e}")
            return df

    def analyze_correlation_structure(self, df):
        """Korelasyon yapısını analiz et"""
        try:
            df = df.copy()
            
            # Price-volume correlation
            df['price_volume_corr'] = df['close'].rolling(20).corr(df['volume'])
            
            # Price-momentum correlation
            df['price_momentum_corr'] = df['close'].rolling(20).corr(df['close'].pct_change())
            
            # Volatility clustering
            df['volatility_clustering'] = df['volatility'].rolling(5).mean() / df['volatility'].rolling(20).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Korelasyon analizi hatası: {e}")
            return df 