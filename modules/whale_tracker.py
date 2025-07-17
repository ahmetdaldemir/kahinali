import pandas as pd
import numpy as np
import logging
from config import Config
from modules.data_collector import DataCollector
import threading

# Order book snapshot arşivi (sembol bazlı, thread-safe)
ORDER_BOOK_SNAPSHOTS = {}
ORDER_BOOK_SNAPSHOTS_LOCK = threading.Lock()

def get_dynamic_whale_threshold(symbol):
    data_collector = DataCollector()
    try:
        stats = data_collector.get_24h_stats(symbol)
        if stats and 'quote_volume_24h' in stats and stats['quote_volume_24h']:
            volume_24h = stats['quote_volume_24h']
            threshold = max(volume_24h * 0.005, 1000)
            return threshold
    except Exception as e:
        logging.getLogger(__name__).warning(f"[Whale] Dinamik threshold alınamadı: {e}")
    return 10000  # fallback

def detect_whale_trades(symbol, limit=100):
    print(f"[DEBUG] detect_whale_trades symbol={symbol}")
    logger = logging.getLogger(__name__)
    data_collector = DataCollector()
    whale_threshold = get_dynamic_whale_threshold(symbol)
    try:
        # 1) Order book ile tek büyük işlemler
        order_book = data_collector.get_order_book(symbol, limit=limit)
        print(f"[DEBUG] order_book={order_book}")
        if not order_book:
            logger.warning(f"[Whale] Order book alınamadı veya boş: {symbol}")
            print(f"[DEBUG] detect_whale_trades: order_book boş")
            return {'whale_score': 0, 'whale_trades': []}
        logger.debug(f"[Whale] {symbol} order book örnek: bids={order_book.get('bids', [])[:3]}, asks={order_book.get('asks', [])[:3]}")
        bids = order_book['bids']
        asks = order_book['asks']
        whale_trades = []
        # Büyük bid işlemleri
        large_bids = bids[bids['amount'] * bids['price'] > whale_threshold]
        for _, row in large_bids.iterrows():
            whale_trades.append({'side': 'buy', 'price': row['price'], 'amount': row['amount'], 'usdt': row['amount'] * row['price']})
        # Büyük ask işlemleri
        large_asks = asks[asks['amount'] * asks['price'] > whale_threshold]
        for _, row in large_asks.iterrows():
            whale_trades.append({'side': 'sell', 'price': row['price'], 'amount': row['amount'], 'usdt': row['amount'] * row['price']})
        whale_score = min(len(whale_trades) / 5, 1.0)
        # Whale yönü analizi
        buy_whale_volume = sum([t['usdt'] for t in whale_trades if t['side'] == 'buy'])
        sell_whale_volume = sum([t['usdt'] for t in whale_trades if t['side'] == 'sell'])
        whale_direction_score = (buy_whale_volume - sell_whale_volume) / (buy_whale_volume + sell_whale_volume + 1e-6)
        # Order book imbalance (OBI)
        bid_volume = order_book['bid_volume'] if 'bid_volume' in order_book else bids['amount'].sum()
        ask_volume = order_book['ask_volume'] if 'ask_volume' in order_book else asks['amount'].sum()
        order_book_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-6)
        # Likidite duvarları (en büyük 3 bid ve ask)
        top_bid_walls = bids.nlargest(3, 'amount')[['price', 'amount']].to_dict('records')
        top_ask_walls = asks.nlargest(3, 'amount')[['price', 'amount']].to_dict('records')
        print(f"[DEBUG] whale_trades={whale_trades}")
        print(f"[DEBUG] whale_score={whale_score}, whale_direction_score={whale_direction_score}, order_book_imbalance={order_book_imbalance}")
        print(f"[DEBUG] top_bid_walls={top_bid_walls}, top_ask_walls={top_ask_walls}")
        # --- Order book snapshot arşivleme ---
        snapshot = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'bids': order_book['bids'][['price', 'amount']].head(20).values.tolist(),
            'asks': order_book['asks'][['price', 'amount']].head(20).values.tolist()
        }
        with ORDER_BOOK_SNAPSHOTS_LOCK:
            if symbol not in ORDER_BOOK_SNAPSHOTS:
                ORDER_BOOK_SNAPSHOTS[symbol] = []
            ORDER_BOOK_SNAPSHOTS[symbol].append(snapshot)
            if len(ORDER_BOOK_SNAPSHOTS[symbol]) > 20:
                ORDER_BOOK_SNAPSHOTS[symbol] = ORDER_BOOK_SNAPSHOTS[symbol][-20:]
        return {'whale_score': whale_score, 'whale_trades': whale_trades, 'whale_direction_score': whale_direction_score,
                'order_book_imbalance': order_book_imbalance, 'top_bid_walls': top_bid_walls, 'top_ask_walls': top_ask_walls}
    except Exception as e:
        print(f"[DEBUG] detect_whale_trades exception: {e}")
        logger.error(f"Whale işlemleri tespit edilirken hata: {e}")
        return {'whale_score': 0, 'whale_trades': []}

def get_whale_score(symbol):
    """Whale etkisini sinyal modeline dahil et (doğru isimlendirme)"""
    whale_data = detect_whale_trades(symbol)
    return whale_data['whale_score']

# Eski isimle de erişim için alias bırak
get_whale_signal = get_whale_score 

def get_order_book_heatmap(symbol, depth=20, snapshots=20):
    """Son N snapshot'tan fiyat seviyelerine göre likidite yoğunluğu matrisini döndürür."""
    import numpy as np
    import pandas as pd
    with ORDER_BOOK_SNAPSHOTS_LOCK:
        snap_list = ORDER_BOOK_SNAPSHOTS.get(symbol, [])[-snapshots:]
    if not snap_list:
        return {'bids': [], 'asks': [], 'prices': []}
    # Fiyat seviyelerini topla
    all_bid_prices = []
    all_ask_prices = []
    for snap in snap_list:
        all_bid_prices += [b[0] for b in snap['bids']]
        all_ask_prices += [a[0] for a in snap['asks']]
    unique_prices = sorted(set(all_bid_prices + all_ask_prices))
    # Sadece depth kadarını al
    if len(unique_prices) > depth:
        unique_prices = unique_prices[:depth]
    # Matrisleri oluştur
    bid_matrix = np.zeros((len(snap_list), len(unique_prices)))
    ask_matrix = np.zeros((len(snap_list), len(unique_prices)))
    for i, snap in enumerate(snap_list):
        for b in snap['bids']:
            if b[0] in unique_prices:
                j = unique_prices.index(b[0])
                bid_matrix[i, j] = b[1]
        for a in snap['asks']:
            if a[0] in unique_prices:
                j = unique_prices.index(a[0])
                ask_matrix[i, j] = a[1]
    # Toplam yoğunluk (her fiyat seviyesi için toplam miktar)
    bid_heat = bid_matrix.sum(axis=0).tolist()
    ask_heat = ask_matrix.sum(axis=0).tolist()
    return {
        'prices': unique_prices,
        'bid_heat': bid_heat,
        'ask_heat': ask_heat,
        'snapshots': len(snap_list)
    } 

def get_trade_tape_analysis(symbol, minutes=2):
    """Son X dakikadaki market işlemlerini analiz eder."""
    data_collector = DataCollector()
    try:
        trades = data_collector.get_recent_trades(symbol, minutes=minutes)
        if not trades or len(trades) == 0:
            return {'success': False, 'error': 'İşlem verisi yok'}
        buy_trades = [t for t in trades if t.get('side', '').lower() == 'buy']
        sell_trades = [t for t in trades if t.get('side', '').lower() == 'sell']
        buy_volume = sum(t['amount'] * t['price'] for t in buy_trades)
        sell_volume = sum(t['amount'] * t['price'] for t in sell_trades)
        total_volume = buy_volume + sell_volume
        buy_ratio = buy_volume / total_volume if total_volume > 0 else 0.0
        sell_ratio = sell_volume / total_volume if total_volume > 0 else 0.0
        # En büyük işlemler
        largest_buys = sorted(buy_trades, key=lambda t: t['amount'] * t['price'], reverse=True)[:3]
        largest_sells = sorted(sell_trades, key=lambda t: t['amount'] * t['price'], reverse=True)[:3]
        # Sweep detection: bir işlemin birden fazla fiyat seviyesini geçmesi (ör: price değişimi > %0.2)
        sweep_trades = []
        last_price = None
        for t in trades:
            if last_price is not None and abs(t['price'] - last_price) / last_price > 0.002:
                sweep_trades.append(t)
            last_price = t['price']
        return {
            'success': True,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'buy_ratio': buy_ratio,
            'sell_ratio': sell_ratio,
            'largest_buys': largest_buys,
            'largest_sells': largest_sells,
            'sweep_count': len(sweep_trades),
            'sweep_trades': sweep_trades[:5],
            'total_trades': len(trades)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)} 

def get_stop_hunt_analysis(symbol, minutes=10):
    """Likidite avı (stop hunt) tespiti: Büyük duvar öncesi/sonrası ani fiyat hareketi var mı?"""
    import numpy as np
    data_collector = DataCollector()
    try:
        # Son X dakikadaki işlemler
        trades = data_collector.get_recent_trades(symbol, minutes=minutes)
        if not trades or len(trades) < 5:
            return {'success': False, 'error': 'Yeterli işlem verisi yok'}
        # Son snapshot'lardaki en büyük bid/ask duvarlarını bul
        with ORDER_BOOK_SNAPSHOTS_LOCK:
            snap_list = ORDER_BOOK_SNAPSHOTS.get(symbol, [])[-5:]
        if not snap_list:
            return {'success': False, 'error': 'Order book snapshot yok'}
        # Her snapshot için en büyük bid/ask fiyatını bul
        bid_walls = []
        ask_walls = []
        for snap in snap_list:
            if snap['bids']:
                max_bid = max(snap['bids'], key=lambda x: x[1])
                bid_walls.append(max_bid[0])
            if snap['asks']:
                max_ask = max(snap['asks'], key=lambda x: x[1])
                ask_walls.append(max_ask[0])
        # Son 10 işlemde fiyat, bu duvarların hemen öncesinde veya sonrasında ani hareket yaptı mı?
        recent_trades = trades[-10:]
        stop_hunt_events = []
        for t in recent_trades:
            price = t['price']
            # Bid duvarına yakınlık ve ani hareket
            for wall in bid_walls:
                if abs(price - wall) / wall < 0.002:  # %0.2'den yakın
                    stop_hunt_events.append({'type': 'bid', 'trade': t, 'wall': wall})
            # Ask duvarına yakınlık ve ani hareket
            for wall in ask_walls:
                if abs(price - wall) / wall < 0.002:
                    stop_hunt_events.append({'type': 'ask', 'trade': t, 'wall': wall})
        return {
            'success': True,
            'stop_hunt_count': len(stop_hunt_events),
            'events': stop_hunt_events[:5],
            'bid_walls': bid_walls,
            'ask_walls': ask_walls
        }
    except Exception as e:
        return {'success': False, 'error': str(e)} 

def get_spread_volatility_analysis(symbol, minutes=10):
    """Spread ve volatilite spike analizi: Son X dakikada spread ve volatilite ani değişimleri tespit edilir."""
    import numpy as np
    import pandas as pd
    try:
        # Son X dakikadaki snapshot'ları al
        with ORDER_BOOK_SNAPSHOTS_LOCK:
            snap_list = ORDER_BOOK_SNAPSHOTS.get(symbol, [])[-minutes:]
        if not snap_list or len(snap_list) < 3:
            return {'success': False, 'error': 'Yeterli snapshot yok'}
        spreads = []
        mid_prices = []
        timestamps = []
        for snap in snap_list:
            if snap['bids'] and snap['asks']:
                best_bid = snap['bids'][0][0]
                best_ask = snap['asks'][0][0]
                spread = best_ask - best_bid
                mid = (best_ask + best_bid) / 2
                spreads.append(spread)
                mid_prices.append(mid)
                timestamps.append(snap['timestamp'])
        # Spread spike tespiti
        spread_spikes = []
        for i in range(1, len(spreads)):
            if spreads[i-1] > 0 and (spreads[i] - spreads[i-1]) / spreads[i-1] > 0.5:
                spread_spikes.append({'timestamp': timestamps[i], 'spread': spreads[i], 'prev_spread': spreads[i-1]})
        # Volatilite spike tespiti (mid price değişimi)
        volatility_spikes = []
        for i in range(1, len(mid_prices)):
            if abs(mid_prices[i] - mid_prices[i-1]) / mid_prices[i-1] > 0.01:
                volatility_spikes.append({'timestamp': timestamps[i], 'mid': mid_prices[i], 'prev_mid': mid_prices[i-1]})
        return {
            'success': True,
            'spread_spike_count': len(spread_spikes),
            'volatility_spike_count': len(volatility_spikes),
            'spread_spikes': spread_spikes[:5],
            'volatility_spikes': volatility_spikes[:5],
            'last_spread': spreads[-1] if spreads else None,
            'last_mid': mid_prices[-1] if mid_prices else None
        }
    except Exception as e:
        return {'success': False, 'error': str(e)} 

def get_orderbook_anomaly_analysis(symbol, snapshots=10):
    """Order book pattern/anomali analizi: spoofing, layering, iceberg ve genel anomali skoru."""
    import numpy as np
    try:
        with ORDER_BOOK_SNAPSHOTS_LOCK:
            snap_list = ORDER_BOOK_SNAPSHOTS.get(symbol, [])[-snapshots:]
        if not snap_list or len(snap_list) < 3:
            return {'success': False, 'error': 'Yeterli snapshot yok'}
        spoofing_events = 0
        layering_events = 0
        iceberg_events = 0
        total_anomaly_score = 0
        # Fiyat seviyelerinde ani büyük emir ekle/çek (spoofing)
        for i in range(1, len(snap_list)):
            prev_bids = {b[0]: b[1] for b in snap_list[i-1]['bids']}
            curr_bids = {b[0]: b[1] for b in snap_list[i]['bids']}
            prev_asks = {a[0]: a[1] for a in snap_list[i-1]['asks']}
            curr_asks = {a[0]: a[1] for a in snap_list[i]['asks']}
            # Spoofing: bir seviyede emir miktarı bir anda 3x artıp sonra kayboluyorsa
            for price in curr_bids:
                if price in prev_bids and prev_bids[price] > 0 and curr_bids[price] / prev_bids[price] > 3:
                    spoofing_events += 1
            for price in curr_asks:
                if price in prev_asks and prev_asks[price] > 0 and curr_asks[price] / prev_asks[price] > 3:
                    spoofing_events += 1
            # Iceberg: bir seviyede büyük emir bir anda kayboluyorsa
            for price in prev_bids:
                if price in curr_bids and prev_bids[price] > 0 and curr_bids[price] == 0:
                    iceberg_events += 1
            for price in prev_asks:
                if price in curr_asks and prev_asks[price] > 0 and curr_asks[price] == 0:
                    iceberg_events += 1
            # Layering: farklı seviyelerde aynı anda büyük emir kümeleri
            bid_layers = [b[1] for b in curr_bids.items() if b[1] > np.percentile(list(curr_bids.values()), 90)]
            ask_layers = [a[1] for a in curr_asks.items() if a[1] > np.percentile(list(curr_asks.values()), 90)]
            if len(bid_layers) >= 3:
                layering_events += 1
            if len(ask_layers) >= 3:
                layering_events += 1
        # Genel anomali skoru (basit): olayların toplamı / snapshot sayısı
        total_anomaly_score = (spoofing_events + layering_events + iceberg_events) / max(1, len(snap_list))
        return {
            'success': True,
            'spoofing_events': spoofing_events,
            'layering_events': layering_events,
            'iceberg_events': iceberg_events,
            'anomaly_score': round(total_anomaly_score, 3),
            'snapshots': len(snap_list)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)} 

def check_alarm_conditions(symbol):
    """OBI, whale_direction_score, anomaly_score, spread spike, stop hunt gibi metriklerde alarm tespiti."""
    alarms = []
    # Whale yönü
    whale_data = detect_whale_trades(symbol)
    if abs(whale_data.get('whale_direction_score', 0)) > 0.7:
        alarms.append({'type': 'Whale Yönü', 'value': whale_data['whale_direction_score'], 'desc': 'Aşırı balina baskısı'})
    # OBI
    if abs(whale_data.get('order_book_imbalance', 0)) > 0.7:
        alarms.append({'type': 'Order Book Imbalance', 'value': whale_data['order_book_imbalance'], 'desc': 'Aşırı dengesizlik'})
    # Anomali skoru
    anomaly = get_orderbook_anomaly_analysis(symbol)
    if anomaly.get('anomaly_score', 0) > 1.5:
        alarms.append({'type': 'Order Book Anomali', 'value': anomaly['anomaly_score'], 'desc': 'Yüksek anomali skoru'})
    # Spread spike
    spreadvol = get_spread_volatility_analysis(symbol)
    if spreadvol.get('spread_spike_count', 0) > 2:
        alarms.append({'type': 'Spread Spike', 'value': spreadvol['spread_spike_count'], 'desc': 'Ani spread artışı'})
    # Stop hunt
    stophunt = get_stop_hunt_analysis(symbol)
    if stophunt.get('stop_hunt_count', 0) > 0:
        alarms.append({'type': 'Stop Hunt', 'value': stophunt['stop_hunt_count'], 'desc': 'Likidite avı tespit edildi'})
    return {'success': True, 'alarm_count': len(alarms), 'alarms': alarms} 