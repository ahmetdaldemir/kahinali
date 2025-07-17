import traceback
from modules.data_collector import DataCollector
from modules.technical_analysis import TechnicalAnalysis
from modules.ai_model import AIModel

# İlk 5 coin için pipeline testi
with open('valid_coins.txt') as f:
    coins = [line.strip() for line in f if line.strip()]

print(f'Test edilecek coinler: {coins[:5]}')

dc = DataCollector()
ta = TechnicalAnalysis()
ai = AIModel()

for c in coins[:5]:
    print(f'\n--- {c} ---')
    try:
        # Veri çekme
        df = dc.get_historical_data(c, '1h', 100)
        if df is None or df.empty:
            print('Veri yok veya çekilemedi.')
            continue
        print(f'Veri çekildi: {len(df)} satır')
        # Teknik analiz
        try:
            df_ta = ta.calculate_all_indicators(df)
            print(f'Teknik analiz OK, {len(df_ta.columns)} sütun')
        except Exception as e:
            print(f'Teknik analiz HATA: {e}')
            traceback.print_exc()
            continue
        # AI analizi
        try:
            ai_result = ai.predict_simple(df_ta)
            if ai_result is None:
                print('AI sonucu None')
                continue
            print(f"AI skor: {ai_result.get('prediction')}, Confidence: {ai_result.get('confidence')}")
        except Exception as e:
            print(f'AI analizi HATA: {e}')
            traceback.print_exc()
            continue
        # Skor ve filtreleme örneği
        ai_score = ai_result.get('prediction', 0)
        if ai_score < 0.5:
            print(f'Skor düşük, sinyal üretilmez (AI skor: {ai_score})')
        else:
            print(f'Sinyal üretilebilir (AI skor: {ai_score})')
    except Exception as e:
        print(f'Genel HATA: {e}')
        traceback.print_exc() 