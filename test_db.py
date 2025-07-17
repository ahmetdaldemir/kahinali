from modules.signal_manager import SignalManager

def test_database():
    sm = SignalManager()
    
    print("=== Veritabanı Testi ===")
    
    # Süresi dolmuş sinyalleri kapat
    updated = sm.update_expired_signals()
    print(f"Kapatılan (timeout): {updated}")
    
    # Toplam sinyal sayısı
    total = sm.get_total_signal_count()
    print(f"Toplam sinyal sayısı: {total}")
    
    # Son 5 sinyal
    df = sm.get_latest_signals(limit=5)
    print(f"\nSon 5 sinyal:")
    if not df.empty:
        for _, row in df.iterrows():
            print(f"ID: {row['id']}, Symbol: {row['symbol']}, Timestamp: {row['timestamp']}, Result: {row['result']}, Direction: {row['direction']}")
    else:
        print("Hiç sinyal bulunamadı!")
    
    # Açık sinyaller
    open_df = sm.get_open_signals()
    print(f"\nAçık sinyal sayısı: {len(open_df)}")
    
    # Kapalı sinyaller
    closed_df = sm.get_closed_signals(days=30)
    print(f"Kapalı sinyal sayısı (son 30 gün): {len(closed_df)}")

    # İlk 10 sinyalin id ve timestamp'ini yazdır
    print("\nİlk 10 sinyalin id ve timestamp:")
    df10 = sm.get_latest_signals(limit=10)
    if not df10.empty:
        for _, row in df10.iterrows():
            print(f"ID: {row['id']}, Timestamp: {row['timestamp']}")
    else:
        print("Yok")

if __name__ == "__main__":
    test_database() 