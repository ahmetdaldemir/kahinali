import pickle
import pandas as pd
import numpy as np

def check_model_features():
    """AI modelinin beklediği feature'ları kontrol et"""
    try:
        # Model feature'larını yükle
        with open('models/feature_cols.pkl', 'rb') as f:
            model_features = pickle.load(f)
        
        print("🤖 AI Model Feature Analizi")
        print("=" * 50)
        print(f"Model beklenen feature sayısı: {len(model_features)}")
        print("\n📋 Model Feature'ları:")
        for i, feature in enumerate(model_features, 1):
            print(f"{i:2d}. {feature}")
        
        # Teknik analiz feature'larını kontrol et
        print("\n" + "=" * 50)
        print("🔧 Teknik Analiz Feature'ları:")
        
        # Örnek veri oluştur
        sample_data = pd.DataFrame({
            'open': [1.0] * 100,
            'high': [1.0] * 100,
            'low': [1.0] * 100,
            'close': [1.0] * 100,
            'volume': [1.0] * 100
        })
        
        # Teknik analiz modülünü import et
        from modules.technical_analysis import TechnicalAnalysis
        ta = TechnicalAnalysis()
        
        # Teknik analiz yap
        df_with_indicators = ta.calculate_all_indicators(sample_data)
        
        print(f"Teknik analiz sonrası feature sayısı: {len(df_with_indicators.columns)}")
        print("\nTeknik analiz feature'ları:")
        for i, col in enumerate(df_with_indicators.columns, 1):
            print(f"{i:2d}. {col}")
        
        # Uyumsuzlukları bul
        print("\n" + "=" * 50)
        print("⚠️ Uyumsuzluk Analizi:")
        
        ta_features = set(df_with_indicators.columns)
        model_features_set = set(model_features)
        
        eksik_features = model_features_set - ta_features
        fazla_features = ta_features - model_features_set
        
        if eksik_features:
            print(f"\n❌ Modelde eksik feature'lar ({len(eksik_features)} adet):")
            for feature in sorted(eksik_features):
                print(f"  - {feature}")
        
        if fazla_features:
            print(f"\n⚠️ Modelde fazla feature'lar ({len(fazla_features)} adet):")
            for feature in sorted(fazla_features):
                print(f"  - {feature}")
        
        if not eksik_features and not fazla_features:
            print("✅ Feature uyumluluğu mükemmel!")
        
        # Öneriler
        print("\n" + "=" * 50)
        print("💡 Öneriler:")
        if eksik_features or fazla_features:
            print("1. Teknik analiz modülünü model feature'larına uygun hale getirin")
            print("2. Veya modeli yeni feature'larla yeniden eğitin")
        else:
            print("1. Feature uyumluluğu tamam, diğer sorunları kontrol edin")
        
    except Exception as e:
        print(f"❌ Hata: {e}")

if __name__ == "__main__":
    check_model_features() 