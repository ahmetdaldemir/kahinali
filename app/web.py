import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
import time
from datetime import datetime, date
import pandas as pd
import numpy as np
import collections.abc
from functools import wraps
from sqlalchemy import create_engine, text
from config import Config
import math

from modules.signal_manager import SignalManager
from modules.performance import PerformanceAnalyzer
from modules.data_collector import DataCollector
from modules.technical_analysis import TechnicalAnalysis
from modules.signal_tracker import SignalTracker
from modules.dynamic_strictness import DynamicStrictness
from modules.whale_tracker import get_order_book_heatmap, get_trade_tape_analysis, get_stop_hunt_analysis, get_spread_volatility_analysis, get_orderbook_anomaly_analysis, check_alarm_conditions

# FAILED_COINS global değişkenini main.py'den import et
try:
    from main import FAILED_COINS
except ImportError:
    FAILED_COINS = []

app = Flask(__name__)
CORS(app)
app.secret_key = Config.FLASK_SECRET_KEY

signal_manager = SignalManager()
performance_analyzer = PerformanceAnalyzer()

# Basit cache sistemi
cache = {}
CACHE_DURATION = 60  # 60 saniye

engine = create_engine(Config.DATABASE_URL + "?client_encoding=utf8")

def cache_result(duration=CACHE_DURATION):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            cache_key = f.__name__ + str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if current_time - timestamp < duration:
                    return result
            
            result = f(*args, **kwargs)
            cache[cache_key] = (result, current_time)
            return result
        return decorated_function
    return decorator

def to_serializable(obj):
    """Numpy, pandas ve diğer JSON'a uygun olmayan tipleri dönüştür"""
    if obj is None:
        return None
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, pd.Series):
        # Pandas Series'i güvenli şekilde dict'e çevir
        try:
            result = {}
            for key in obj.index:
                value = obj[key]
                result[key] = to_serializable(value)
            return result
        except:
            return {}
    elif isinstance(obj, pd.DataFrame):
        try:
            return obj.to_dict('records')
        except:
            return []
    elif hasattr(obj, 'dtype'):  # Numpy array veya scalar
        try:
            if hasattr(obj, 'size') and obj.size == 1:  # Scalar
                try:
                    item_val = obj.item()
                    return to_serializable(item_val)
                except:
                    return None
            else:  # Array
                return obj.tolist()
        except:
            return None
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    else:
        return obj

def nan_to_zero(val):
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return 0
    return val

def deep_nan_to_zero(obj):
    if isinstance(obj, dict):
        return {k: deep_nan_to_zero(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_nan_to_zero(v) for v in obj]
    else:
        return nan_to_zero(obj)

def safe_format(val):
    try:
        return f"{float(val):.2f}"
    except Exception:
        return "N/A"

# Sinyal detaylarını render ederken:
def render_score(value):
    if value == 'Eksik Veri':
        return '<span style="color:orange;font-weight:bold;" title="Bu skor eksik veriyle hesaplandı.">Eksik Veri</span>'
    try:
        return f'{float(value)*100:.1f}%' if isinstance(value, (float, int)) else str(value)
    except Exception:
        return str(value)

@app.route('/')
def index():
    # Hatalı coinleri de index.html'e gönder
    failed_coins = FAILED_COINS if FAILED_COINS else []
    return render_template('index.html', failed_coins=failed_coins)

@app.route('/api/signals')
def get_signals():
    """Sinyalleri al - Gelişmiş sayfalama ile"""
    try:
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 50, type=int)
        offset = (page - 1) * limit
        
        # Engine tanımı
        from sqlalchemy import create_engine, text
        from config import Config
        engine = create_engine(Config.DATABASE_URL)
        
        # Önce toplam sinyal sayısını al
        count_query = "SELECT COUNT(*) as total FROM signals"
        
        with engine.connect() as conn:
            # Toplam sayıyı al
            count_result = conn.execute(text(count_query))
            total_count = count_result.fetchone()[0]
            
            # Sinyalleri al - WHERE koşulunu kaldırdık, tüm sinyalleri getir
            query = """
                SELECT id, symbol, timeframe, direction, entry_price, current_price, timestamp, 
                       ai_score, ta_strength, whale_score, social_score, news_score,
                       predicted_gain, predicted_duration, quality_score, market_regime, 
                       volatility_regime, volume_score, momentum_score, pattern_score,
                       breakout_probability, risk_reward_ratio, confidence_level, 
                       signal_strength, market_sentiment, result, realized_gain, 
                       exit_price, exit_time, duration_hours, exit_reason, current_pnl,
                       take_profit, stop_loss, support_level, resistance_level, target_time_hours,
                       predicted_breakout_threshold, predicted_breakout_time_hours,
                       order_book_imbalance, top_bid_walls, top_ask_walls, whale_direction_score
                FROM signals 
                ORDER BY timestamp DESC
                LIMIT :limit OFFSET :offset
            """
            
            result = conn.execute(text(query), {'limit': limit, 'offset': offset})
            signals = []
            
            for row in result.fetchall():
                signal = dict(row._mapping)
                
                # Result durumunu belirle
                if signal['result'] is None or signal['result'] == 'None' or signal['result'] == '':
                    signal['status'] = 'OPEN'
                    signal['result'] = None
                else:
                    signal['status'] = 'CLOSED'
                
                # Market regime, regime_note ve acil fırsat bilgisi ekle
                signal['market_regime'] = signal.get('market_regime', None)
                signal['regime_note'] = signal.get('regime_note', None)
                signal['acil_firsat'] = signal.get('acil_firsat', None)
                
                # Timestamp'i string'e çevir
                if signal['timestamp']:
                    signal['timestamp'] = str(signal['timestamp'])
                if signal.get('exit_time'):
                    signal['exit_time'] = str(signal['exit_time'])
                
                # Kar/Zarar hedefini yüzde olarak ekle
                entry_price = float(signal.get('entry_price') or 0)
                take_profit = float(signal.get('take_profit') or 0)
                stop_loss = float(signal.get('stop_loss') or 0)
                if entry_price > 0 and take_profit > 0:
                    signal['kar_hedefi_yuzde'] = safe_format((take_profit / entry_price - 1) * 100)
                else:
                    signal['kar_hedefi_yuzde'] = None
                if entry_price > 0 and stop_loss > 0:
                    signal['zarar_hedefi_yuzde'] = safe_format((1 - stop_loss / entry_price) * 100)
                else:
                    signal['zarar_hedefi_yuzde'] = None
                # AI skorunda veya önemli metriklerde None veya dict varsa hata mesajı ekle
                if signal.get('ai_score') is None or isinstance(signal.get('ai_score'), dict):
                    signal['error'] = 'AI skoru hesaplanamadı, model veya veri eksik.'
                signals.append(signal)
        
        # Sayfalama bilgilerini hesapla
        total_pages = (total_count + limit - 1) // limit  # Yuvarlama ile sayfa sayısı
        
        return jsonify({
            'signals': signals,
            'pagination': {
                'page': page,
                'limit': limit,
                'total': total_count,
                'pages': total_pages,
                'has_next': page < total_pages,
                'has_prev': page > 1
            }
        })
        
    except Exception as e:
        app.logger.error(f"Sinyaller alma hatası: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/signals/open')
@cache_result(duration=15)  # 15 saniye cache
def get_open_signals():
    """Açık sinyalleri getir"""
    try:
        df = signal_manager.get_open_signals()
        
        # Sinyal durumlarını güncelle
        signals_with_status = []
        for _, signal in df.iterrows():
            signal_dict = signal.to_dict()
            status_info = signal_manager.check_signal_status(signal_dict)
            if status_info:
                signal_dict['status'] = status_info.get('status', None)
                signal_dict['duration_hours'] = status_info.get('duration_hours', None)
            # Kar/Zarar hedefini yüzde olarak ekle
            entry_price = float(signal_dict.get('entry_price') or 0)
            take_profit = float(signal_dict.get('take_profit') or 0)
            stop_loss = float(signal_dict.get('stop_loss') or 0)
            if entry_price > 0 and take_profit > 0:
                signal_dict['kar_hedefi_yuzde'] = safe_format((take_profit / entry_price - 1) * 100)
            else:
                signal_dict['kar_hedefi_yuzde'] = None
            if entry_price > 0 and stop_loss > 0:
                signal_dict['zarar_hedefi_yuzde'] = safe_format((1 - stop_loss / entry_price) * 100)
            else:
                signal_dict['zarar_hedefi_yuzde'] = None
            # AI skorunda veya önemli metriklerde None varsa hata mesajı ekle
            if signal_dict.get('ai_score') is None:
                signal_dict['error'] = 'AI skoru hesaplanamadı, model veya veri eksik.'
            signals_with_status.append(signal_dict)
        
        return jsonify({
            'signals': signals_with_status,
            'count': len(signals_with_status)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/signals/closed')
@cache_result(duration=60)  # 1 dakika cache
def get_closed_signals():
    """Kapalı sinyalleri getir"""
    try:
        days = request.args.get('days', 30, type=int)
        df = signal_manager.get_closed_signals(days=days)
        
        signals_list = df.to_dict('records')
        return jsonify({
            'signals': signals_list,
            'count': len(signals_list)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/signal/<int:signal_id>/status')
def get_signal_status(signal_id):
    """Belirli bir sinyalin durumunu kontrol et"""
    try:
        status_info = signal_manager.check_signal_status(signal_id)
        if status_info:
            return jsonify(status_info)
        else:
            return jsonify({'error': 'Sinyal bulunamadı'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance')
@cache_result(duration=60)
def get_performance():
    """Performans verilerini al"""
    try:
        # Basit performans verisi döndür
        performance = {
            'labels': ['1G', '1H', '4H', '1D', '1H'],
            'values': [2.5, 5.2, 8.1, 12.3, 15.7],
            'total_profit': 15.7,
            'success_rate': 78.5,
            'total_trades': 45
        }
        return jsonify(performance)
    except Exception as e:
        app.logger.error(f"Performance data error: {e}")
        return jsonify({'error': 'Performance data not available'}), 500

@app.route('/api/coin_performance/<coin>')
def get_coin_performance(coin):
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    performance = performance_analyzer.coin_performance(coin, start_date, end_date)
    return jsonify(performance)

@app.route('/api/daily_performance')
def get_daily_performance():
    days = request.args.get('days', 30, type=int)
    performance = performance_analyzer.get_daily_performance(days=days)
    return jsonify(performance)

@app.route('/api/top_coins')
def get_top_coins():
    """En iyi performans gösteren coinler"""
    try:
        days = request.args.get('days', 30, type=int)
        limit = request.args.get('limit', 10, type=int)
        
        analyzer = PerformanceAnalyzer()
        top_coins = analyzer.get_top_coins(limit)
        
        return jsonify(top_coins)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance_summary')
def get_performance_summary():
    summary = performance_analyzer.get_performance_summary()
    return jsonify(to_serializable(summary))

@app.route('/api/stats')
@cache_result(duration=30)  # 30 saniye cache
def get_stats():
    """Sistem istatistiklerini al"""
    try:
        # Signal tracker'dan performans özeti al
        tracker = SignalTracker()
        performance = tracker.get_performance_summary()
        
        if performance:
            total_signals = performance.get('total_signals', 0)
            open_signals = performance.get('open_signals', 0)
            success_rate = performance.get('success_rate', 0.0)
            avg_profit = performance.get('avg_profit', 0.0)
            
            stats_data = {
                'total_signals': total_signals,
                'open_signals': open_signals,
                'success_rate': success_rate,
                'avg_profit': avg_profit,
                'system_status': 'running'
            }
        else:
            # Fallback - eski yöntem
            query = """
                SELECT 
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN result IS NULL OR result = 'None' THEN 1 END) as open_signals
                FROM signals 
                WHERE timestamp::timestamp >= NOW() - INTERVAL '7 days'
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(query))
                row = result.fetchone()
                
                if row:
                    stats_data = {
                        'total_signals': row.total_signals,
                        'open_signals': row.open_signals,
                        'success_rate': 0.0,
                        'avg_profit': 0.0,
                        'system_status': 'running'
                    }
                else:
                    stats_data = {
                        'total_signals': 0,
                        'open_signals': 0,
                        'success_rate': 0.0,
                        'avg_profit': 0.0,
                        'system_status': 'running'
                    }
        
        return jsonify(stats_data)
        
    except Exception as e:
        app.logger.error(f"Stats alma hatası: {e}")
        return jsonify({
            'total_signals': 0,
            'open_signals': 0,
            'success_rate': 0.0,
            'avg_profit': 0.0,
            'system_status': 'error'
        })

@app.route('/api/signal_summary')
def get_signal_summary():
    """Sinyal özeti API"""
    try:
        days = request.args.get('days', 7, type=int)
        summary = signal_manager.get_signal_summary(days)
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/signals/filtered')
def get_filtered_signals():
    """Filtrelenmiş sinyaller API"""
    try:
        min_confidence = request.args.get('min_confidence', 0.7, type=float)
        max_signals = request.args.get('max_signals', 10, type=int)
        timeframe = request.args.get('timeframe', None)
        symbol = request.args.get('symbol', None)
        
        # Tüm sinyalleri al
        signals = signal_manager.load_signals()
        
        if signals.empty:
            return jsonify([])
        
        # DataFrame'i dict'e çevir
        signals_list = signals.to_dict('records')
        
        # Filtreleme
        if timeframe:
            signals_list = [s for s in signals_list if s.get('timeframe') == timeframe]
        if symbol:
            signals_list = [s for s in signals_list if s.get('symbol') == symbol]
        
        # Skorlama ve filtreleme
        filtered_signals = signal_manager.filter_signals(signals_list, min_confidence, max_signals)
        
        return jsonify(filtered_signals)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/detailed/<symbol>')
def get_detailed_analysis(symbol):
    """Detaylı coin analizi"""
    try:
        timeframe = request.args.get('timeframe', '1h')
        
        # Veri topla
        collector = DataCollector()
        data = collector.get_historical_data(symbol, timeframe, 500)
        
        if data.empty:
            return jsonify({'error': 'Veri bulunamadı'}), 404
        
        # Teknik analiz
        ta = TechnicalAnalysis()
        ta_result = ta.analyze_technical_signals(data)
        
        if not ta_result or 'error' in ta_result:
            return jsonify({'error': 'Teknik analiz hatası'}), 500
        
        # Son değerler
        latest = data.iloc[-1]
        
        analysis = {
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': float(latest['close']),
            'price_change_24h': float(data['close'].pct_change().iloc[-1] * 100),
            'volume_24h': float(latest['volume']) if 'volume' in latest else 0,
            'technical_indicators': ta_result,
            'support_resistance': {
                'support': ta_result.get('support', 0),
                'resistance': ta_result.get('resistance', 0),
                'pivot': ta_result.get('pivot', 0)
            }
        }
        
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/signal/<int:signal_id>/analysis')
def get_signal_analysis(signal_id):
    """Sinyal analizi ve tahminleri getir"""
    try:
        # Sinyal bilgilerini al
        df = signal_manager.load_signals()
        signal = df[df['id'] == signal_id]
        
        if signal.empty:
            return jsonify({'error': 'Sinyal bulunamadı'}), 404
        
        signal_data = signal.iloc[0].to_dict()
        
        # Dinamik analiz yap
        prediction = signal_manager.predict_breakout_probability(signal_data)
        
        analysis = {
            'signal_id': signal_id,
            'symbol': signal_data.get('symbol'),
            'current_status': signal_data.get('status', 'OPEN'),
            'prediction': prediction,
            'dynamic_targets': {
                'profit_target': f"{prediction['profit_target']*100:.1f}%",
                'loss_target': f"{prediction['profit_target']*0.6*100:.1f}%",
                'breakout_threshold': f"{prediction['breakout_threshold']*100:.1f}%"
            },
            'risk_assessment': {
                'probability': f"{prediction['probability']*100:.1f}%",
                'category': prediction['category'],
                'confidence': prediction['confidence']
            },
            'recommendations': generate_recommendations(prediction, signal_data)
        }
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_recommendations(prediction, signal_data):
    """Sinyal için öneriler oluştur"""
    recommendations = []
    
    probability = prediction['probability']
    category = prediction['category']
    
    if probability >= 0.8:
        recommendations.append("🚀 Çok güçlü sinyal - Yüksek kar potansiyeli")
        recommendations.append("💰 Dinamik kar hedefi: %{:.1f}".format(prediction['profit_target']*100))
        recommendations.append("⚡ Ani yükseliş olasılığı yüksek")
    elif probability >= 0.6:
        recommendations.append("✅ Güçlü sinyal - İyi kar potansiyeli")
        recommendations.append("📊 Orta risk seviyesi")
        recommendations.append("⏰ 24-48 saat takip önerilir")
    elif probability >= 0.4:
        recommendations.append("⚠️ Orta güçte sinyal - Dikkatli takip")
        recommendations.append("🔍 Daha fazla doğrulama gerekli")
        recommendations.append("📈 Küçük pozisyon önerilir")
    else:
        recommendations.append("❌ Zayıf sinyal - Yüksek risk")
        recommendations.append("🚫 Pozisyon açılması önerilmez")
        recommendations.append("🔄 Daha iyi fırsat bekleyin")
    
    # Coin tipine özel öneriler
    symbol = signal_data.get('symbol', '')
    if any(coin in symbol for coin in ['DOGE', 'SHIB', 'PEPE', 'BONK']):
        recommendations.append("🎭 Meme coin - Yüksek volatilite")
        recommendations.append("⚡ Hızlı giriş/çıkış stratejisi")
    
    return recommendations

@app.route('/api/market/breakout-predictions')
def get_breakout_predictions():
    """En yüksek breakout olasılığına sahip sinyalleri getir"""
    try:
        # Son 24 saatteki sinyalleri al
        signals = signal_manager.get_recent_signals(1000)  # Daha fazla sinyal al
        
        # DataFrame kontrolü - güvenli şekilde
        if signals is None or (hasattr(signals, 'empty') and signals.empty):
            return jsonify({'predictions': []})
        
        # Eğer signals bir DataFrame ise, dict listesine çevir
        if hasattr(signals, 'to_dict'):
            signals = signals.to_dict('records')
        
        # Breakout predictions'ları filtrele ve işle
        breakout_predictions = []
        coin_predictions = {}  # Her coin için en iyi prediction'ı tut
        
        for signal in signals:
            try:
                # Sinyal verilerini güvenli şekilde dönüştür
                safe_signal = {}
                for key, value in signal.items():
                    safe_signal[key] = to_serializable(value)
                
                # Sadece açık sinyalleri göster
                if not (safe_signal.get('result') is None or safe_signal.get('result') == '' or safe_signal.get('result') == 'None'):
                    continue
                # Breakout probability kontrolü
                breakout_prob = safe_signal.get('breakout_probability', 0)
                if breakout_prob and breakout_prob > 0.25:  # Minimum threshold
                    symbol = safe_signal.get('symbol', '')
                    coin = symbol.split('/')[0] if '/' in symbol else symbol
                    
                    # Her coin için sadece en yüksek probability olan sinyali tut
                    if coin not in coin_predictions or breakout_prob > coin_predictions[coin]['breakout_probability']:
                        # Kategori belirleme
                        if breakout_prob >= 0.8:
                            category = "YÜKSEK"
                            confidence = "Çok güçlü sinyal"
                        elif breakout_prob >= 0.6:
                            category = "ORTA"
                            confidence = "Güçlü sinyal"
                        elif breakout_prob >= 0.4:
                            category = "DÜŞÜK"
                            confidence = "Zayıf sinyal"
                        else:
                            category = "ÇOK DÜŞÜK"
                            confidence = "Riski yüksek"
                        
                        # Kar ve zarar hedeflerini hesapla
                        entry_price = safe_signal.get('entry_price', 0)
                        take_profit = safe_signal.get('take_profit', 0)
                        stop_loss = safe_signal.get('stop_loss', 0)
                        
                        profit_target = 0
                        loss_target = 0
                        
                        if entry_price > 0 and take_profit > 0:
                            profit_target = (take_profit / entry_price - 1)
                        
                        if entry_price > 0 and stop_loss > 0:
                            loss_target = (stop_loss / entry_price - 1)
                        
                        coin_predictions[coin] = {
                            'symbol': symbol,
                            'breakout_probability': breakout_prob,
                            'breakout_threshold': safe_signal.get('predicted_breakout_threshold', 0),
                            'ai_score': safe_signal.get('ai_score', 0),
                            'ta_strength': safe_signal.get('ta_strength', 0),
                            'confidence': confidence,
                            'category': category,
                            'entry_price': entry_price,
                            'profit_target': profit_target,
                            'loss_target': loss_target,
                            'take_profit': take_profit,
                            'stop_loss': stop_loss,
                            'timeframe': safe_signal.get('timeframe', '1h'),
                            'timestamp': safe_signal.get('timestamp', ''),
                            'action': safe_signal.get('action', 'BUY'),
                            'signal_id': safe_signal.get('id', 0)
                        }
                        
            except Exception as e:
                print(f"Breakout prediction işleme hatası: {e}")
                continue
        
        # En yüksek probability'ye sahip 10 coin'i seç
        sorted_predictions = sorted(
            coin_predictions.values(), 
            key=lambda x: x['breakout_probability'], 
            reverse=True
        )[:10]
        
        return jsonify({'predictions': sorted_predictions})
        
    except Exception as e:
        print(f"Breakout predictions hatası: {e}")
        return jsonify({'predictions': []})

@app.route('/api/signal/<int:signal_id>/advanced-analysis')
def get_advanced_signal_analysis(signal_id):
    """Sinyal için gelişmiş analiz ve başarı kriterleri"""
    try:
        # Sinyal bilgilerini al
        df = signal_manager.load_signals()
        signal = df[df['id'] == signal_id]
        
        if signal.empty:
            return jsonify({'error': 'Sinyal bulunamadı'}), 404
        
        signal_data = signal.iloc[0].to_dict()
        
        # Gelişmiş metrikleri hesapla
        success_metrics = signal_manager.calculate_success_metrics(signal_data)
        
        # Breakout tahmini
        breakout_prediction = signal_manager.predict_breakout_probability(signal_data)
        
        # Yükseliş süresi tahmini
        predicted_time = signal_manager.predict_breakout_time(signal_data)
        
        # Risk/Ödül oranı
        risk_reward = signal_manager.calculate_risk_reward_ratio(signal_data)
        
        # Volatilite ve trend analizi
        volatility_score = signal_manager.calculate_volatility_score(signal_data)
        trend_strength = signal_manager.calculate_trend_strength(signal_data)
        market_regime = signal_manager.determine_market_regime(signal_data)
        
        analysis = {
            'signal_id': signal_id,
            'symbol': signal_data.get('symbol'),
            'timeframe': signal_data.get('timeframe'),
            'current_status': signal_data.get('status', 'OPEN'),
            
            'breakout_analysis': {
                'probability': f"{breakout_prediction['probability']*100:.1f}%",
                'category': breakout_prediction['category'],
                'confidence': breakout_prediction['confidence'],
                'threshold': f"{breakout_prediction['breakout_threshold']*100:.1f}%",
                'achieved': signal_data.get('breakout_achieved', False)
            },
            
            'time_analysis': {
                'predicted_breakout_time': f"{predicted_time:.1f} saat",
                'predicted_breakout_time_hours': predicted_time,
                'actual_breakout_time': signal_data.get('breakout_time_hours'),
                'time_accuracy': _calculate_time_accuracy(predicted_time, signal_data.get('breakout_time_hours'))
            },
            
            'risk_analysis': {
                'risk_reward_ratio': risk_reward,
                'predicted_profit_target': f"{success_metrics.get('predicted_profit_target', 0.05)*100:.1f}%",
                'predicted_loss_target': f"{success_metrics.get('predicted_profit_target', 0.05)*0.6*100:.1f}%",
                'actual_max_gain': f"{signal_data.get('actual_max_gain', 0)*100:.1f}%" if signal_data.get('actual_max_gain') else "N/A",
                'actual_max_loss': f"{signal_data.get('actual_max_loss', 0)*100:.1f}%" if signal_data.get('actual_max_loss') else "N/A"
            },
            
            'market_analysis': {
                'volatility_score': f"{volatility_score*100:.1f}%",
                'trend_strength': f"{trend_strength*100:.1f}%",
                'market_regime': market_regime,
                'signal_quality': f"{success_metrics.get('signal_quality', 0.5)*100:.1f}%"
            },
            
            'success_metrics': {
                'breakout_success_rate': _calculate_breakout_success_rate(signal_data),
                'time_prediction_accuracy': _calculate_time_accuracy(predicted_time, signal_data.get('breakout_time_hours')),
                'risk_reward_accuracy': _calculate_rr_accuracy(risk_reward, signal_data.get('actual_risk_reward_ratio')),
                'overall_success_score': _calculate_overall_success_score(signal_data, success_metrics)
            },
            
            'recommendations': _generate_advanced_recommendations(signal_data, success_metrics)
        }
        
        return jsonify(to_serializable(analysis))
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _calculate_time_accuracy(predicted_time, actual_time):
    """Zaman tahmin doğruluğunu hesapla"""
    if not actual_time or actual_time <= 0:
        return "Hesaplanamadı"
    
    accuracy = 100 - abs(predicted_time - actual_time) / actual_time * 100
    return f"{max(0, min(100, accuracy)):.1f}%"

def _calculate_breakout_success_rate(signal_data):
    """Breakout başarı oranını hesapla"""
    if signal_data.get('breakout_achieved'):
        return "100%"
    elif signal_data.get('actual_max_gain'):
        max_gain = signal_data.get('actual_max_gain', 0)
        threshold = signal_data.get('predicted_breakout_threshold', 0.025)
        success_rate = (max_gain / threshold) * 100 if threshold > 0 else 0
        return f"{min(100, success_rate):.1f}%"
    else:
        return "Hesaplanamadı"

def _calculate_rr_accuracy(predicted_rr, actual_rr):
    """Risk/Ödül oranı doğruluğunu hesapla"""
    if not actual_rr or actual_rr <= 0:
        return "Hesaplanamadı"
    
    accuracy = 100 - abs(predicted_rr - actual_rr) / actual_rr * 100
    return f"{max(0, min(100, accuracy)):.1f}%"

def _calculate_overall_success_score(signal_data, success_metrics):
    """Genel başarı skorunu hesapla"""
    try:
        # Farklı faktörlerin ağırlıklı ortalaması
        factors = {
            'breakout_probability': success_metrics.get('breakout_probability', 0.5) * 100,
            'signal_quality': success_metrics.get('signal_quality', 0.5) * 100,
            'trend_strength': success_metrics.get('trend_strength', 0.5) * 100,
            'risk_reward_ratio': min(100, success_metrics.get('risk_reward_ratio', 1.67) * 20)
        }
        
        weights = {
            'breakout_probability': 0.4,
            'signal_quality': 0.3,
            'trend_strength': 0.2,
            'risk_reward_ratio': 0.1
        }
        
        total_score = 0
        total_weight = 0
        
        for factor, weight in weights.items():
            total_score += factors[factor] * weight
            total_weight += weight
        
        final_score = total_score / total_weight if total_weight > 0 else 0
        return f"{final_score:.1f}%"
        
    except Exception as e:
        return "Hesaplanamadı"

def _generate_advanced_recommendations(signal_data, success_metrics):
    """Gelişmiş öneriler oluştur"""
    recommendations = []
    
    # Breakout olasılığına göre
    probability = success_metrics.get('breakout_probability', 0.5)
    if probability >= 0.8:
        recommendations.append("🚀 Çok yüksek breakout olasılığı - Güçlü pozisyon önerilir")
    elif probability >= 0.6:
        recommendations.append("✅ Yüksek breakout olasılığı - Orta pozisyon önerilir")
    elif probability >= 0.4:
        recommendations.append("⚠️ Orta breakout olasılığı - Küçük pozisyon önerilir")
    else:
        recommendations.append("❌ Düşük breakout olasılığı - Pozisyon önerilmez")
    
    # Zaman tahminine göre
    predicted_time = success_metrics.get('predicted_breakout_time', 24.0)
    if predicted_time <= 6:
        recommendations.append("⚡ Hızlı breakout bekleniyor - Kısa vadeli strateji")
    elif predicted_time <= 24:
        recommendations.append("📈 Orta vadeli breakout - Günlük takip")
    else:
        recommendations.append("🕐 Uzun vadeli breakout - Sabırlı olun")
    
    # Risk/Ödül oranına göre
    rr_ratio = success_metrics.get('risk_reward_ratio', 1.67)
    if rr_ratio >= 3:
        recommendations.append("💰 Mükemmel risk/ödül oranı - Yüksek pozisyon")
    elif rr_ratio >= 2:
        recommendations.append("📊 İyi risk/ödül oranı - Orta pozisyon")
    else:
        recommendations.append("⚠️ Düşük risk/ödül oranı - Dikkatli pozisyon")
    
    # Volatiliteye göre
    volatility = success_metrics.get('volatility_score', 0.5)
    if volatility >= 0.8:
        recommendations.append("🎭 Yüksek volatilite - Hızlı giriş/çıkış")
    elif volatility >= 0.6:
        recommendations.append("📊 Orta volatilite - Dengeli strateji")
    else:
        recommendations.append("📈 Düşük volatilite - Uzun vadeli pozisyon")
    
    # Market rejimine göre
    market_regime = success_metrics.get('market_regime', 'BİLİNMEYEN')
    if 'GÜÇLÜ YÜKSELIŞ' in market_regime:
        recommendations.append("🔥 Güçlü yükseliş trendi - Agresif strateji")
    elif 'YÜKSELIŞ' in market_regime:
        recommendations.append("📈 Yükseliş trendi - Pozitif strateji")
    elif 'SIDEWAYS' in market_regime:
        recommendations.append("↔️ Yatay trend - Dikkatli strateji")
    else:
        recommendations.append("📉 Düşüş trendi - Koruyucu strateji")
    
    return recommendations

@app.route('/api/performance/advanced-stats')
def get_advanced_performance_stats():
    """Gelişmiş performans istatistikleri"""
    try:
        # Kapalı sinyalleri al
        closed_signals = signal_manager.get_closed_signals(days=30)
        
        if closed_signals.empty:
            return jsonify({
                'total_signals': 0,
                'breakout_success_rate': 0,
                'time_prediction_accuracy': 0,
                'risk_reward_accuracy': 0,
                'avg_signal_quality': 0,
                'market_regime_distribution': {},
                'volatility_analysis': {},
                'success_metrics': {}
            })
        
        # Breakout başarı oranı
        breakout_success = closed_signals['breakout_achieved'].sum()
        breakout_success_rate = (breakout_success / len(closed_signals)) * 100
        
        # Zaman tahmin doğruluğu
        time_accuracy = []
        for _, signal in closed_signals.iterrows():
            predicted = signal.get('predicted_breakout_time_hours', 24)
            actual = signal.get('breakout_time_hours', 24)
            if actual and actual > 0:
                accuracy = 100 - abs(predicted - actual) / actual * 100
                time_accuracy.append(max(0, min(100, accuracy)))
        
        avg_time_accuracy = sum(time_accuracy) / len(time_accuracy) if time_accuracy else 0
        
        # Risk/Ödül doğruluğu
        rr_accuracy = []
        for _, signal in closed_signals.iterrows():
            predicted = signal.get('risk_reward_ratio', 1.67)
            actual = signal.get('actual_risk_reward_ratio', 1.67)
            if actual and actual > 0:
                accuracy = 100 - abs(predicted - actual) / actual * 100
                rr_accuracy.append(max(0, min(100, accuracy)))
        
        avg_rr_accuracy = sum(rr_accuracy) / len(rr_accuracy) if rr_accuracy else 0
        
        # Ortalama sinyal kalitesi
        avg_signal_quality = closed_signals['signal_quality_score'].mean() * 100
        
        # Market rejimi dağılımı
        market_regime_dist = closed_signals['market_regime'].value_counts().to_dict()
        
        # Volatilite analizi
        volatility_analysis = {
            'high_volatility_signals': len(closed_signals[closed_signals['volatility_score'] >= 0.7]),
            'medium_volatility_signals': len(closed_signals[(closed_signals['volatility_score'] >= 0.4) & (closed_signals['volatility_score'] < 0.7)]),
            'low_volatility_signals': len(closed_signals[closed_signals['volatility_score'] < 0.4]),
            'avg_volatility': closed_signals['volatility_score'].mean() * 100
        }
        
        stats = {
            'total_signals': len(closed_signals),
            'breakout_success_rate': round(breakout_success_rate, 2),
            'time_prediction_accuracy': round(avg_time_accuracy, 2),
            'risk_reward_accuracy': round(avg_rr_accuracy, 2),
            'avg_signal_quality': round(avg_signal_quality, 2),
            'market_regime_distribution': market_regime_dist,
            'volatility_analysis': volatility_analysis,
            'success_metrics': {
                'avg_trend_strength': round(closed_signals['trend_strength'].mean() * 100, 2),
                'avg_risk_reward_ratio': round(closed_signals['risk_reward_ratio'].mean(), 2),
                'avg_breakout_threshold': round(closed_signals['predicted_breakout_threshold'].mean() * 100, 2)
            }
        }
        
        return jsonify(deep_nan_to_zero(to_serializable(stats)))
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/signal/<int:signal_id>/detailed-analysis')
def get_detailed_signal_analysis(signal_id):
    """Sinyal için detaylı analiz ve yorumlar"""
    try:
        # Sinyal verilerini al
        signal_data = signal_manager.get_signal_by_id(signal_id)
        
        # Güvenli tip kontrolü
        if signal_data is None:
            return jsonify({'error': 'Sinyal bulunamadı'}), 404
        
        # Pandas Series kontrolü - kesin güvenli şekilde
        if hasattr(signal_data, 'empty'):
            if signal_data.empty:
                return jsonify({'error': 'Sinyal bulunamadı'}), 404
        
        # Series ise dict'e çevir
        if hasattr(signal_data, 'to_dict'):
            try:
                signal_data = signal_data.to_dict()
            except Exception as e:
                print(f"Series to dict conversion error: {e}")
                return jsonify({'error': 'Sinyal verisi dönüştürülemedi'}), 500
        
        # Dict kontrolü
        if not isinstance(signal_data, dict):
            return jsonify({'error': 'Geçersiz sinyal verisi'}), 500
        
        # Sinyal verilerini güvenli şekilde dönüştür
        safe_signal = {}
        for key, value in signal_data.items():
            # Decimal.Decimal tipini güvenli şekilde float'a çevir
            if hasattr(value, 'as_tuple'):  # Decimal.Decimal kontrolü
                try:
                    safe_signal[key] = float(value)
                except:
                    safe_signal[key] = 0.0
            # Numpy array veya Series kontrolü - kesin güvenli
            elif hasattr(value, 'dtype'):  # Numpy array veya scalar
                try:
                    if hasattr(value, 'size') and value.size == 1:  # Scalar
                        safe_signal[key] = to_serializable(value.item())
                    else:  # Array
                        safe_signal[key] = to_serializable(value.tolist())
                except Exception:
                    safe_signal[key] = to_serializable(value)
            elif hasattr(value, '__len__') and len(value) > 1:
                # Array-like object with multiple elements
                try:
                    safe_signal[key] = to_serializable(list(value))
                except Exception:
                    safe_signal[key] = to_serializable(value)
            elif hasattr(value, 'empty') and not value.empty:
                # Pandas Series with data
                try:
                    safe_signal[key] = to_serializable(value.tolist())
                except Exception:
                    safe_signal[key] = to_serializable(value)
            elif hasattr(value, 'dtype') and hasattr(value, 'size') and value.size > 1:
                # Numpy array with multiple elements
                try:
                    safe_signal[key] = to_serializable(value.tolist())
                except Exception:
                    safe_signal[key] = to_serializable(value)
            else:
                safe_signal[key] = to_serializable(value)
        
        # Detaylı analiz bilgileri
        # Kriterler için eşik ve ağırlıklar (daha sıkı)
        criteria_config = {
            'ai_score':     {'required': 0.85, 'weight': 0.4, 'description': 'Yapay zeka model güven skoru'},
            'ta_strength': {'required': 0.7,  'weight': 0.35, 'description': 'Teknik analiz güç skoru'},
            'whale_score': {'required': 0.4,  'weight': 0.25, 'description': 'Whale (büyük yatırımcı) etkisi'},
            # Sosyal ve haber skorları pasif
        }
        
        # Kriter analizi
        criteria_analysis = {}
        all_criteria_passed = True
        for key, conf in criteria_config.items():
            val = safe_signal.get(key, 0)
            try:
                val_f = float(val)
            except:
                val_f = 0
            passed = val_f >= conf['required']
            if not passed:
                all_criteria_passed = False
            # Yüzde olarak gösterilecekse
            if key in ['ai_score', 'ta_strength', 'whale_score']:
                percentage = val_f * 100
            else:
                percentage = val_f * 100 if val_f < 1.5 else val_f
            criteria_analysis[key] = {
                'passed': passed,
                'percentage': percentage,
                'value': val_f,
                'description': conf['description'],
                'required': conf['required'],
                'weight': conf['weight']
            }
        # Sosyal ve haber badge'leri pasif
        criteria_analysis['social_score'] = {
            'passed': False,
            'percentage': 0,
            'value': 0.0,
            'description': 'Sosyal medya duyarlılığı (PASİF)',
            'required': 0.0,
            'weight': 0.0
        }
        criteria_analysis['news_score'] = {
            'passed': False,
            'percentage': 0,
            'value': 0.0,
            'description': 'Haber akışı etkisi (PASİF)',
            'required': 0.0,
            'weight': 0.0
        }
        
        # Sinyal temel bilgileri
        direction_val = safe_signal.get('direction', '')
        if isinstance(direction_val, list):
            direction_val = direction_val[0] if direction_val else ''
        if not isinstance(direction_val, str):
            direction_val = str(direction_val)
        
        # Symbol değerini düzelt
        symbol_val = safe_signal.get('symbol', '')
        if isinstance(symbol_val, list):
            symbol_val = ''.join(symbol_val) if symbol_val else ''
        if not isinstance(symbol_val, str):
            symbol_val = str(symbol_val)
        
        # Eksik alanları varsayılan değerlerle doldur
        entry_price = safe_signal.get('entry_price', 0)
        current_price = safe_signal.get('current_price', entry_price)
        profit_target = safe_signal.get('profit_target', entry_price * 1.05)  # %5 kar hedefi
        stop_loss = safe_signal.get('stop_loss', entry_price * 0.95)  # %5 zarar hedefi
        
        signal_info = {
            'symbol': symbol_val,
            'timeframe': safe_signal.get('timeframe', ''),
            'direction': direction_val,
            'timestamp': safe_signal.get('timestamp', ''),
            'entry_price': entry_price,
            'profit_target': profit_target,
            'stop_loss': stop_loss,
            'confidence': safe_signal.get('confidence', 0),
        }
        
        # Üretim nedeni analizi
        # Ağırlıklı skor hesaplama
        def safe_float(value, default=0.0):
            """Güvenli float dönüşümü"""
            if value is None:
                return default
            if isinstance(value, list):
                value = value[0] if value else default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        ai_score = safe_float(safe_signal.get('ai_score', 0))
        ta_strength = safe_float(safe_signal.get('ta_strength', 0))
        whale_score = safe_float(safe_signal.get('whale_score', 0))
        social_score = safe_float(safe_signal.get('social_score', 0))
        news_score = safe_float(safe_signal.get('news_score', 0))
        breakout_prob = safe_float(safe_signal.get('breakout_probability', 0))
        
        # Ağırlıklı toplam skor hesaplama (sadece aktif skorlar)
        weighted_score = (
            ai_score * 0.4 +
            ta_strength * 0.35 +
            whale_score * 0.25
        )
        
        # Minimum güven eşiği daha yüksek
        MIN_CONFIDENCE = 0.7
        confidence_level = max(weighted_score, float(safe_signal.get('confidence', 0)))
        
        # Genel başarı durumu
        overall_passed = all_criteria_passed and (confidence_level >= MIN_CONFIDENCE)
        
        production_reason = {
            'total_percentage': confidence_level * 100,
            'signal_produced': overall_passed,
            'reasons': [
                {
                    'type': 'success' if ai_score >= 0.85 else 'warning',
                    'message': f"AI: {ai_score:.1%} (Gerekli: 85%)"
                },
                {
                    'type': 'success' if ta_strength >= 0.7 else 'warning',
                    'message': f"TA: {ta_strength:.1%} (Gerekli: 70%)"
                },
                {
                    'type': 'success' if whale_score >= 0.4 else 'warning',
                    'message': f"Whale: {whale_score:.1%} (Gerekli: 40%)"
                },
                {
                    'type': 'secondary',
                    'message': "Sosyal medya: PASİF"
                },
                {
                    'type': 'secondary',
                    'message': "Haber etkisi: PASİF"
                }
            ]
        }
        
        # Yatırım önerileri
        confidence_text = "YÜKSEK" if confidence_level >= 0.8 else "ORTA" if confidence_level >= 0.6 else "DÜŞÜK"
            
        investment_recommendations = [
            {
                'title': 'Giriş Stratejisi',
                'description': f"Fiyat {entry_price:.6f} seviyesinden giriş yapın. Stop loss: {stop_loss:.6f}",
                'confidence': confidence_text
            },
            {
                'title': 'Hedef Seviyeler',
                'description': f"İlk hedef: {profit_target:.6f}, Risk/Ödül oranı: {safe_signal.get('risk_reward_ratio', 1.67):.2f}",
                'confidence': confidence_text
            }
        ]
        
        # Risk uyarıları
        risk_warnings = []
        if safe_float(safe_signal.get('volatility_score', 0)) > 0.7:
            risk_warnings.append({
                'level': 'high',
                'message': 'Yüksek volatilite - Risk yönetimi kritik'
            })
        if safe_float(safe_signal.get('trend_strength', 0)) < 0.4:
            risk_warnings.append({
                'level': 'medium',
                'message': 'Zayıf trend gücü - Dikkatli olun'
            })
        
        # Özet bilgiler
        summary = {
            'overall_percentage': confidence_level * 100,
            'signal_strength': 'GÜÇLÜ' if overall_passed else 'ZAYIF',
            'recommended_action': direction_val.upper(),
            'confidence_level': confidence_text
        }
        
        analysis = {
            'signal_id': signal_id,
            'criteria_analysis': criteria_analysis,
            'signal_info': signal_info,
            'production_reason': production_reason,
            'investment_recommendations': investment_recommendations,
            'risk_warnings': risk_warnings,
            'summary': summary,
            # Eski anahtarlar da kalsın (geri uyumluluk için)
            'production_criteria': {
                'ai_score': safe_signal.get('ai_score', 0),
                'ta_strength': safe_signal.get('ta_strength', 0),
                'whale_score': safe_signal.get('whale_score', 0),
                'social_score': safe_signal.get('social_score', 0),
                'news_score': safe_signal.get('news_score', 0),
                'breakout_probability': safe_signal.get('breakout_probability', 0)
            },
            'weighted_scores': {
                'overall_confidence': confidence_level,
                'signal_quality': safe_signal.get('signal_quality_score', 0),
                'risk_reward_ratio': safe_signal.get('risk_reward_ratio', 0),
                'volatility_score': safe_signal.get('volatility_score', 0),
                'trend_strength': safe_signal.get('trend_strength', 0)
            },
            'investment_recommendation': {
                'action': safe_signal.get('action', 'HOLD'),
                'entry_price': entry_price,
                'profit_target': profit_target,
                'stop_loss': stop_loss,
                'confidence_level': confidence_level,
                'timeframe': safe_signal.get('timeframe', '1h'),
                'expected_duration': safe_signal.get('breakout_duration', '24h')
            }
        }
        return jsonify(analysis)
        
    except Exception as e:
        print(f"Sinyal detay analizi hatası: {str(e)}")
        import traceback
        print(f"Hata detayı: {traceback.format_exc()}")
        return jsonify({'error': 'Sinyal analizi yüklenirken hata oluştu'}), 500

@app.route('/api/strictness', methods=['GET'])
def get_strictness_data():
    """Dinamik sıkılık verilerini getir"""
    try:
        # DynamicStrictness instance'ı oluştur
        dynamic_strictness = DynamicStrictness()
        
        # Gerçek market verilerini topla
        data_collector = DataCollector()
        technical_analyzer = TechnicalAnalysis()
        
        # Popüler coinlerden market verisi topla
        popular_coins = data_collector.get_popular_usdt_pairs(max_pairs=10)
        
        market_data = {
            'price_data': [],
            'technical_indicators': {},
            'volume_data': [],
            'sentiment_data': {'overall_sentiment': 0.5},
            'ai_predictions': {'confidence': 0.5}
        }
        
        # İlk 5 coin'den veri topla
        for coin in popular_coins[:5]:
            try:
                data = data_collector.get_historical_data(coin, '1h', 50)
                if not data.empty:
                    # Fiyat verisi
                    market_data['price_data'].extend(data['close'].tolist())
                    
                    # Teknik analiz
                    ta_data = technical_analyzer.calculate_all_indicators(data)
                    if 'rsi_14' in ta_data.columns:
                        market_data['technical_indicators']['rsi'] = ta_data['rsi_14'].iloc[-1]
                    if 'macd' in ta_data.columns:
                        market_data['technical_indicators']['macd'] = ta_data['macd'].iloc[-1]
                    
                    # Hacim verisi
                    if 'volume' in data.columns:
                        market_data['volume_data'].extend(data['volume'].tolist())
                        
            except Exception as e:
                app.logger.warning(f"Market verisi toplama hatası ({coin}): {e}")
                continue
        
        # Sıkılığı güncelle
        updated_status = dynamic_strictness.update_strictness(market_data)
        
        # Dinamik eşikleri hesapla
        current_strictness = updated_status['current_strictness']
        dynamic_min_confidence = max(0.3, min(0.8, current_strictness))
        dynamic_min_ai_score = max(0.3, min(0.7, current_strictness - 0.1))
        dynamic_min_ta_strength = max(0.2, min(0.6, current_strictness - 0.15))
        
        # Ek bilgiler ekle
        updated_status['dynamic_thresholds'] = {
            'min_confidence': dynamic_min_confidence,
            'min_ai_score': dynamic_min_ai_score,
            'min_ta_strength': dynamic_min_ta_strength,
            'static_thresholds': {
                'min_confidence': Config.MIN_SIGNAL_CONFIDENCE,
                'min_ai_score': Config.MIN_AI_SCORE,
                'min_ta_strength': Config.MIN_TA_STRENGTH
            }
        }
        
        # Performans bilgilerini ekle
        performance_summary = dynamic_strictness.get_performance_summary()
        updated_status['performance_tracking'] = {
            'success_rate': performance_summary.get('success_rate', 0.5),
            'total_signals': performance_summary.get('total_signals', 0),
            'auto_adjustment_status': performance_summary.get('auto_adjustment_status', 'No data'),
            'last_adjustment': performance_summary.get('last_adjustment'),
            'auto_adjustment_enabled': updated_status.get('auto_adjustment', {}).get('enabled', True),
            'target_success_rate': {
                'min': updated_status.get('auto_adjustment', {}).get('min_target', 0.6),
                'max': updated_status.get('auto_adjustment', {}).get('max_target', 0.8)
            }
        }
        # Kaçan fırsat (örnek: missed_opportunity)
        missed_opportunity = performance_summary.get('missed_opportunity', 0)
        if 'performance' in updated_status:
            updated_status['performance']['missed_opportunity'] = missed_opportunity
        else:
            updated_status['performance'] = {'missed_opportunity': missed_opportunity}
        return jsonify(updated_status)
        
    except Exception as e:
        app.logger.error(f"Sıkılık veri alma hatası: {e}")
        return jsonify({
            'current_strictness': 0.55,
            'strictness_level': '🟡 ORTA SIKILIK',
            'recommendation': 'Sistem analiz ediliyor...',
            'error': str(e)
        }), 500

@app.route('/api/system/health', methods=['GET'])
def get_system_health():
    health = performance_analyzer.check_system_health()
    return jsonify(to_serializable(health))

@app.route('/api/system/optimize', methods=['POST'])
def optimize_system():
    """Sistem otomatik optimizasyonu"""
    try:
        analyzer = PerformanceAnalyzer()
        optimization_result = analyzer.auto_optimize_system()
        return jsonify(optimization_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """Detaylı sistem durumu"""
    try:
        analyzer = PerformanceAnalyzer()
        
        # Sistem sağlığı
        health = analyzer.system_health_check()
        
        # Performans özeti
        performance = analyzer.get_performance_summary()
        
        # Son sinyaller
        recent_signals = analyzer.signal_manager.load_signals()
        recent_count = len(recent_signals) if not recent_signals.empty else 0
        
        # Sistem durumu
        system_status = {
            'health': health,
            'performance': performance,
            'recent_signals': recent_count,
            'uptime': get_system_uptime(),
            'last_update': str(datetime.now())
        }
        
        return jsonify(system_status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_system_uptime():
    """Sistem çalışma süresini hesapla"""
    try:
        # Basit uptime hesaplama (gerçek uygulamada daha gelişmiş olabilir)
        import os
        import time
        
        # Log dosyasının oluşturulma zamanını kullan
        log_file = 'logs/kahin_ultima.log'
        if os.path.exists(log_file):
            start_time = os.path.getctime(log_file)
            uptime_seconds = time.time() - start_time
            
            # Gün, saat, dakika formatına çevir
            days = int(uptime_seconds // 86400)
            hours = int((uptime_seconds % 86400) // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            
            return f"{days}g {hours}s {minutes}d"
        else:
            return "Bilinmiyor"
    except:
        return "Bilinmiyor"

@app.route('/api/failed_coins')
def get_failed_coins():
    """Hatalı coinler ve hata mesajlarını döndür"""
    return jsonify({'failed_coins': FAILED_COINS})

@app.route('/failed_coins')
def failed_coins_page():
    """Hatalı coinler ve hata mesajlarını gösteren sayfa"""
    return render_template('failed_coins.html', failed_coins=FAILED_COINS)

@app.route('/api/logs')
def get_logs():
    """Son 50 log satırını döndür (UTF-8, emoji destekli)"""
    import os
    log_file = 'logs/kahin_ultima.log'
    if not os.path.exists(log_file):
        return jsonify({'logs': ['Log dosyası bulunamadı.']}), 200
    try:
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            last_lines = lines[-50:] if len(lines) > 50 else lines
            # Satır sonu karakterlerini temizle
            last_lines = [line.rstrip('\n') for line in last_lines]
        return jsonify({'logs': last_lines})
    except Exception as e:
        return jsonify({'logs': [f'Log okuma hatası: {str(e)}']}), 200

@app.route('/api/orderbook/heatmap/<symbol>')
def get_orderbook_heatmap(symbol):
    try:
        heatmap = get_order_book_heatmap(symbol)
        return jsonify({'success': True, 'data': heatmap})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/tradetape/analysis/<symbol>')
def get_tradetape_analysis(symbol):
    try:
        result = get_trade_tape_analysis(symbol)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stophunt/analysis/<symbol>')
def get_stophunt_analysis(symbol):
    try:
        result = get_stop_hunt_analysis(symbol)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/spreadvol/analysis/<symbol>')
def get_spreadvol_analysis(symbol):
    try:
        result = get_spread_volatility_analysis(symbol)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/orderbook/anomaly/<symbol>')
def get_orderbook_anomaly(symbol):
    try:
        result = get_orderbook_anomaly_analysis(symbol)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/alarm/check/<symbol>')
def check_alarm(symbol):
    try:
        result = check_alarm_conditions(symbol)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/docs')
def api_docs():
    """API endpoint dökümantasyonu ve açıklamaları"""
    docs = [
        {'endpoint': '/api/signals', 'desc': 'Tüm sinyalleri listeler'},
        {'endpoint': '/api/signal/<id>/detailed-analysis', 'desc': 'Sinyal detay analizini getirir'},
        {'endpoint': '/api/orderbook/heatmap/<symbol>', 'desc': 'Order book heatmap (likidite yoğunluğu) datası'},
        {'endpoint': '/api/tradetape/analysis/<symbol>', 'desc': 'Trade tape (gerçekleşen işlemler) analizi'},
        {'endpoint': '/api/stophunt/analysis/<symbol>', 'desc': 'Likidite avı (stop hunt) analizi'},
        {'endpoint': '/api/spreadvol/analysis/<symbol>', 'desc': 'Spread ve volatilite spike analizi'},
        {'endpoint': '/api/orderbook/anomaly/<symbol>', 'desc': 'Order book pattern/anomali analizi'},
        {'endpoint': '/api/alarm/check/<symbol>', 'desc': 'Alarm durumu sorgulama'},
        {'endpoint': '/api/report/performance', 'desc': 'Son 7 günün performans raporu'},
    ]
    return jsonify({'success': True, 'docs': docs})

@app.route('/api/report/performance')
def api_report_performance():
    """Son 7 günün sinyal performans özetini JSON olarak döndürür"""
    try:
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        df = signal_manager.load_signals(start_date, end_date)
        total = len(df)
        success = (df['result'] == 'profit').sum() if 'result' in df else 0
        avg_gain = df['realized_gain'].mean() if 'realized_gain' in df else 0
        alarm_count = 0
        for symbol in df['symbol'].unique():
            alarm_result = None
            try:
                from modules.whale_tracker import check_alarm_conditions
                alarm_result = check_alarm_conditions(symbol)
            except:
                pass
            if alarm_result and alarm_result.get('alarm_count', 0) > 0:
                alarm_count += alarm_result['alarm_count']
        return jsonify({'success': True, 'total_signals': total, 'success_count': int(success), 'success_rate': round((success/total)*100,2) if total else 0, 'avg_gain': round(avg_gain,4), 'alarm_count': alarm_count})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/backtest/run')
def api_backtest_run():
    """Belirli bir tarih aralığında geçmiş sinyaller ve analizler için backtest raporu döndürür"""
    try:
        from datetime import datetime
        start = request.args.get('start')
        end = request.args.get('end')
        if not start or not end:
            return jsonify({'success': False, 'error': 'start ve end parametreleri gereklidir (YYYY-MM-DD)'})
        start_date = datetime.strptime(start, '%Y-%m-%d')
        end_date = datetime.strptime(end, '%Y-%m-%d')
        df = signal_manager.load_signals(start_date, end_date)
        total = len(df)
        alarm_count = 0
        anomaly_count = 0
        whale_alert_count = 0
        obi_alert_count = 0
        success = (df['result'] == 'profit').sum() if 'result' in df else 0
        for idx, row in df.iterrows():
            symbol = row['symbol']
            try:
                from modules.whale_tracker import check_alarm_conditions, get_orderbook_anomaly_analysis, detect_whale_trades
                alarm_result = check_alarm_conditions(symbol)
                if alarm_result and alarm_result.get('alarm_count', 0) > 0:
                    alarm_count += alarm_result['alarm_count']
                anomaly = get_orderbook_anomaly_analysis(symbol)
                if anomaly.get('anomaly_score', 0) > 1.5:
                    anomaly_count += 1
                whale = detect_whale_trades(symbol)
                if abs(whale.get('whale_direction_score', 0)) > 0.7:
                    whale_alert_count += 1
                if abs(whale.get('order_book_imbalance', 0)) > 0.7:
                    obi_alert_count += 1
            except:
                continue
        return jsonify({'success': True, 'total_signals': total, 'success_count': int(success), 'success_rate': round((success/total)*100,2) if total else 0, 'alarm_count': alarm_count, 'anomaly_count': anomaly_count, 'whale_alert_count': whale_alert_count, 'obi_alert_count': obi_alert_count})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🔮 Kahin Ultima Web Panel Başlatılıyor...")
    print("🌐 Web arayüzü: http://localhost:5000")
    print("📊 API endpoint: http://localhost:5000/api/")
    print("=" * 50)
    
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
    except Exception as e:
        print(f"❌ Web sunucusu başlatılamadı: {e}")
        print("🔧 Lütfen port 5000'in kullanılabilir olduğundan emin olun.") 