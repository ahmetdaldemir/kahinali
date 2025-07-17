import pandas as pd

# Veriyi yükle
df = pd.read_csv('data/processed_training_data.csv')

print("Tüm sütunlar ve veri tipleri:")
for col in df.columns:
    print(f"{col}: {df[col].dtype}")

print(f"\nToplam {len(df.columns)} sütun var.")

# String sütunları bul
string_columns = [col for col in df.columns if df[col].dtype == 'object']

print(f"\nString sütunlar ({len(string_columns)} adet):")
for col in string_columns:
    unique_values = df[col].unique()[:10]  # İlk 10 benzersiz değer
    print(f"{col}: {unique_values}") 