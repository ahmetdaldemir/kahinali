from modules.signal_manager import SignalManager
import pandas as pd

sm = SignalManager()

print("=== SİNYAL SİSTEMİ DURUMU ===")
open_signals = sm.get_open_signals()
signals = sm.get_signals()
print(f"Açık sinyal sayısı: {len(open_signals)}")
print(f"Tüm sinyal sayısı: {len(signals)}")

print("\n[DEBUG] signals tipi:", type(signals))
if isinstance(signals, pd.DataFrame) and not signals.empty:
    print("[DEBUG] signals.iloc[0] tipi:", type(signals.iloc[0]))
    print("[DEBUG] signals.iloc[0] içeriği:\n", signals.iloc[0])

print("\n[DEBUG] open_signals tipi:", type(open_signals))
if isinstance(open_signals, pd.DataFrame) and not open_signals.empty:
    print("[DEBUG] open_signals.iloc[0] tipi:", type(open_signals.iloc[0]))
    print("[DEBUG] open_signals.iloc[0] içeriği:\n", open_signals.iloc[0])

if isinstance(signals, pd.DataFrame) and not signals.empty:
    print("\n=== SON 5 SİNYAL ===")
    for i in range(max(0, len(signals)-5), len(signals)):
        row = signals.iloc[i]
        print(f"{row['timestamp']} - {row['symbol']} - {row['direction']} - Skor: {row['ai_score']}")
else:
    print("Hiç sinyal bulunamadı!")

print("\n=== AÇIK SİNYALLER ===")
if isinstance(open_signals, pd.DataFrame) and not open_signals.empty:
    print(f"Toplam {len(open_signals)} açık sinyal var")
    for i in range(min(5, len(open_signals))):
        row = open_signals.iloc[i]
        print(f"{i+1}. {row['symbol']} - {row['direction']} - Skor: {row['ai_score']}")
else:
    print("Hiç açık sinyal yok!") 