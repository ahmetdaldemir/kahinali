from modules.data_collector import DataCollector

def main():
    dc = DataCollector()
    coins = dc.get_popular_usdt_pairs(400)
    failed = []
    for i, coin in enumerate(coins):
        print(f"[{i+1}/{len(coins)}] {coin} kontrol ediliyor...")
        df = dc.get_historical_data(coin, '1h', 10)
        if df.empty:
            failed.append(coin)
    print(f"\nVeri çekilemeyen coin sayısı: {len(failed)}")
    print("İlk 10 başarısız coin:", failed[:10])
    if failed:
        with open('unavailable_coins.txt', 'w', encoding='utf-8') as f:
            for coin in failed:
                f.write(coin + '\n')
        print("Tüm başarısız coinler unavailable_coins.txt dosyasına kaydedildi.")

if __name__ == "__main__":
    main() 