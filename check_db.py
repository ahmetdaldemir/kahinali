import psycopg2
from datetime import datetime

try:
    conn = psycopg2.connect('postgresql://postgres:123456@localhost:5432/kahin_ultima')
    cur = conn.cursor()
    
    # Toplam sinyal sayısı
    cur.execute('SELECT COUNT(*) FROM signals')
    total_count = cur.fetchone()[0]
    print(f'Toplam sinyal sayisi: {total_count}')
    
    # Son 10 sinyal
    cur.execute('SELECT timestamp, symbol, result FROM signals ORDER BY timestamp DESC LIMIT 10')
    results = cur.fetchall()
    
    print('\nSon 10 sinyal:')
    for r in results:
        timestamp = r[0]
        symbol = r[1] if r[1] else 'None'
        result = r[2] if r[2] else 'None'
        print(f'{timestamp} - {symbol} - {result}')
    
    # Son sinyal zamanı
    if results:
        last_signal_time = results[0][0]
        now = datetime.now()
        time_diff = now - last_signal_time
        print(f'\nSon sinyal zamani: {last_signal_time}')
        print(f'Su anki zaman: {now}')
        print(f'Zaman farki: {time_diff}')
    
    conn.close()
    
except Exception as e:
    print(f'Hata: {e}') 