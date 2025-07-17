import pickle
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

# Basit ensemble model oluştur
def create_simple_ensemble():
    print("Basit ensemble model oluşturuluyor...")
    
    # Mevcut modelleri yükle
    try:
        with open('models/gb_model.pkl', 'rb') as f:
            gb_model = pickle.load(f)
        print("✓ Gradient Boosting modeli yüklendi")
    except:
        print("✗ Gradient Boosting modeli yüklenemedi")
        return False
    
    try:
        with open('models/rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        print("✓ Random Forest modeli yüklendi")
    except:
        print("✗ Random Forest modeli yüklenemedi")
        return False
    
    # Basit ensemble oluştur
    ensemble = VotingClassifier(
        estimators=[
            ('gb', gb_model),
            ('rf', rf_model)
        ],
        voting='soft'
    )
    
    # Kaydet
    with open('models/ensemble_model.pkl', 'wb') as f:
        pickle.dump(ensemble, f)
    
    print("✓ Ensemble model kaydedildi")
    
    # Feature importance oluştur
    feature_importance = {
        'gb_importance': gb_model.feature_importances_ if hasattr(gb_model, 'feature_importances_') else np.ones(125),
        'rf_importance': rf_model.feature_importances_ if hasattr(rf_model, 'feature_importances_') else np.ones(125)
    }
    
    with open('models/feature_importance.pkl', 'wb') as f:
        pickle.dump(feature_importance, f)
    
    print("✓ Feature importance kaydedildi")
    
    # Feature selector oluştur (basit)
    feature_selector = np.ones(125, dtype=bool)  # Tüm featureları kullan
    
    with open('models/feature_selector.pkl', 'wb') as f:
        pickle.dump(feature_selector, f)
    
    print("✓ Feature selector kaydedildi")
    
    # Models dict oluştur
    models_dict = {
        'ensemble': ensemble,
        'gb': gb_model,
        'rf': rf_model
    }
    
    with open('models/models.pkl', 'wb') as f:
        pickle.dump(models_dict, f)
    
    print("✓ Models dict kaydedildi")
    
    return True

if __name__ == "__main__":
    success = create_simple_ensemble()
    if success:
        print("\n🎉 Tüm modeller başarıyla oluşturuldu!")
    else:
        print("\n❌ Hata oluştu!") 