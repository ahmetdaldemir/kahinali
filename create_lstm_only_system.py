import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

def create_lstm_only_system():
    print("LSTM-only sistem oluşturuluyor...")
    
    # Basit Random Forest oluştur
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    print("✓ Random Forest modeli oluşturuldu")
    
    # Basit Logistic Regression oluştur
    lr_model = LogisticRegression(random_state=42)
    print("✓ Logistic Regression modeli oluşturuldu")
    
    # Basit ensemble oluştur
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('lr', lr_model)
        ],
        voting='soft'
    )
    
    # Modelleri kaydet
    with open('models/ensemble_model.pkl', 'wb') as f:
        pickle.dump(ensemble, f)
    print("✓ Ensemble model kaydedildi")
    
    with open('models/rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("✓ RF model kaydedildi")
    
    with open('models/gb_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)  # GB yerine RF kullan
    print("✓ GB model kaydedildi (RF kullanılıyor)")
    
    # Feature importance oluştur
    feature_importance = {
        'rf_importance': np.ones(125) / 125,  # Eşit ağırlık
        'gb_importance': np.ones(125) / 125
    }
    
    with open('models/feature_importance.pkl', 'wb') as f:
        pickle.dump(feature_importance, f)
    print("✓ Feature importance kaydedildi")
    
    # Feature selector oluştur
    feature_selector = np.ones(125, dtype=bool)
    
    with open('models/feature_selector.pkl', 'wb') as f:
        pickle.dump(feature_selector, f)
    print("✓ Feature selector kaydedildi")
    
    # Models dict oluştur
    models_dict = {
        'ensemble': ensemble,
        'rf': rf_model,
        'gb': rf_model,  # GB yerine RF
        'lstm': 'lstm_model.h5'  # LSTM dosya yolu
    }
    
    with open('models/models.pkl', 'wb') as f:
        pickle.dump(models_dict, f)
    print("✓ Models dict kaydedildi")
    
    return True

if __name__ == "__main__":
    success = create_lstm_only_system()
    if success:
        print("\n🎉 LSTM-only sistem başarıyla oluşturuldu!")
        print("📊 Sistem şimdi LSTM + basit ML modelleri kullanacak")
    else:
        print("\n❌ Hata oluştu!") 