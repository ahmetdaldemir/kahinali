#!/usr/bin/env python3
"""
KAHİN ULTIMA - AI ve Teknik Analiz Uyumluluk Testi
AI model ve teknik analiz modüllerinin uyumluluğunu test eder
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Proje kök dizinini ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.technical_analysis import TechnicalAnalysis, FIXED_FEATURE_LIST
from modules.ai_model import AIModel
from modules.data_collector import DataCollector

def test_feature_compatibility():
    """Feature uyumluluğunu test et"""
    print("🔧 FEATURE UYUMLULUK TESTİ")
    print("=" * 60)
    
    try:
        # 1. AI model feature'larını kontrol et
        print("1. AI Model feature'ları kontrol ediliyor...")
        ai_model = AIModel()
        
        # Model feature'larını yükle
        try:
            with open('models/feature_cols.pkl', 'rb') as f:
                model_features = pd.read_pickle(f)
            print(f"✅ AI model feature'ları yüklendi: {len(model_features)} adet")
        except Exception as e:
            print(f"⚠ AI model feature'ları yüklenemedi: {e}")
            model_features = []
        
        # 2. Teknik analiz feature'larını kontrol et
        print("2. Teknik analiz feature'ları kontrol ediliyor...")
        ta = TechnicalAnalysis()
        
        # Test verisi oluştur
        test_data = pd.DataFrame({
            'open': [100] * 100,
            'high': [105] * 100,
            'low': [95] * 100,
            'close': [102] * 100,
            'volume': [1000] * 100
        })
        
        # Teknik analiz uygula
        ta_result = ta.calculate_all_indicators(test_data)
        print(f"✅ Teknik analiz tamamlandı: {ta_result.shape[1]} feature")
        
        # 3. Feature uyumluluğunu kontrol et
        print("3. Feature uyumluluğu analiz ediliyor...")
        
        ta_features = set(ta_result.columns)
        model_features_set = set(model_features) if model_features else set()
        
        # Eksik feature'lar
        missing_features = model_features_set - ta_features
        # Fazla feature'lar
        extra_features = ta_features - model_features_set
        # Ortak feature'lar
        common_features = ta_features & model_features_set
        
        print(f"📊 Feature Analizi:")
        print(f"   - Teknik analiz feature'ları: {len(ta_features)}")
        print(f"   - AI model feature'ları: {len(model_features_set)}")
        print(f"   - Ortak feature'lar: {len(common_features)}")
        print(f"   - Eksik feature'lar: {len(missing_features)}")
        print(f"   - Fazla feature'lar: {len(extra_features)}")
        
        # Uyumluluk oranı
        if model_features_set:
            compatibility_ratio = len(common_features) / len(model_features_set)
            print(f"   - Uyumluluk oranı: {compatibility_ratio:.2%}")
        else:
            compatibility_ratio = 0
            print(f"   - Uyumluluk oranı: Hesaplanamadı (model feature'ları yok)")
        
        return {
            'ta_features': ta_features,
            'model_features': model_features_set,
            'common_features': common_features,
            'missing_features': missing_features,
            'extra_features': extra_features,
            'compatibility_ratio': compatibility_ratio
        }
        
    except Exception as e:
        print(f"❌ Feature uyumluluk testi hatası: {e}")
        return None

def test_data_flow():
    """Veri akışını test et"""
    print("\n📊 VERİ AKIŞI TESTİ")
    print("=" * 60)
    
    try:
        # 1. Veri toplama
        print("1. Veri toplama test ediliyor...")
        collector = DataCollector()
        df = collector.get_historical_data('BTC/USDT', '1h', 100)
        
        if df is None or df.empty:
            print("❌ Veri toplanamadı")
            return None
            
        print(f"✅ Veri toplandı: {len(df)} satır")
        
        # 2. Teknik analiz
        print("2. Teknik analiz test ediliyor...")
        ta = TechnicalAnalysis()
        df_with_indicators = ta.calculate_all_indicators(df)
        
        if df_with_indicators is None or df_with_indicators.empty:
            print("❌ Teknik analiz başarısız")
            return None
            
        print(f"✅ Teknik analiz tamamlandı: {df_with_indicators.shape[1]} feature")
        
        # 3. AI model
        print("3. AI model test ediliyor...")
        ai_model = AIModel()
        
        # Feature'ları FIXED_FEATURE_LIST ile uyumlu hale getir
        for col in FIXED_FEATURE_LIST:
            if col not in df_with_indicators.columns:
                df_with_indicators[col] = 0
        
        # Sadece AI model'in beklediği feature'ları kullan
        if hasattr(ai_model, 'feature_columns') and ai_model.feature_columns:
            available_features = [col for col in ai_model.feature_columns if col in df_with_indicators.columns]
            df_for_ai = df_with_indicators[available_features]
        else:
            df_for_ai = df_with_indicators[FIXED_FEATURE_LIST]
        
        # AI tahmini yap
        ai_result = ai_model.predict(df_for_ai)
        
        if ai_result is None:
            print("❌ AI tahmini başarısız")
            return None
            
        print(f"✅ AI tahmini tamamlandı")
        print(f"   - Tahmin: {ai_result.get('prediction', 0):.4f}")
        print(f"   - Güven: {ai_result.get('confidence', 0):.4f}")
        
        return {
            'data_shape': df.shape,
            'ta_shape': df_with_indicators.shape,
            'ai_features': len(df_for_ai.columns),
            'ai_prediction': ai_result.get('prediction', 0),
            'ai_confidence': ai_result.get('confidence', 0)
        }
        
    except Exception as e:
        print(f"❌ Veri akışı testi hatası: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_integration_workflow():
    """Entegrasyon iş akışını test et"""
    print("\n🔄 ENTEGRASYON İŞ AKIŞI TESTİ")
    print("=" * 60)
    
    try:
        # Test sembolleri
        test_symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
        results = []
        
        for symbol in test_symbols:
            print(f"\n📊 {symbol} test ediliyor...")
            
            try:
                # 1. Veri topla
                collector = DataCollector()
                df = collector.get_historical_data(symbol, '1h', 200)
                
                if df is None or df.empty:
                    print(f"❌ {symbol} için veri alınamadı")
                    continue
                
                # 2. Teknik analiz
                ta = TechnicalAnalysis()
                df_with_indicators = ta.calculate_all_indicators(df)
                
                if df_with_indicators is None or df_with_indicators.empty:
                    print(f"❌ {symbol} teknik analiz başarısız")
                    continue
                
                # 3. AI model
                ai_model = AIModel()
                
                # Feature'ları hazırla
                for col in FIXED_FEATURE_LIST:
                    if col not in df_with_indicators.columns:
                        df_with_indicators[col] = 0
                
                df_for_ai = df_with_indicators[FIXED_FEATURE_LIST]
                
                # AI tahmini
                ai_result = ai_model.predict(df_for_ai)
                
                if ai_result is None:
                    print(f"❌ {symbol} AI tahmini başarısız")
                    continue
                
                # 4. Sonuçları analiz et
                ai_score = ai_result.get('prediction', 0)
                confidence = ai_result.get('confidence', 0)
                
                # Teknik analiz gücü
                ta_signals = ta.generate_signals(df_with_indicators)
                ta_strength = ta.calculate_signal_strength(ta_signals)
                
                # Trend analizi
                trend_direction = ta.get_trend_direction(df_with_indicators)
                trend_strength = ta.calculate_trend_strength(df_with_indicators)
                
                result = {
                    'symbol': symbol,
                    'ai_score': ai_score,
                    'confidence': confidence,
                    'ta_strength': ta_strength,
                    'trend_direction': trend_direction,
                    'trend_strength': trend_strength,
                    'success': True
                }
                
                results.append(result)
                
                print(f"✅ {symbol} analizi tamamlandı")
                print(f"   - AI Skor: {ai_score:.4f}")
                print(f"   - Güven: {confidence:.4f}")
                print(f"   - TA Gücü: {ta_strength:.4f}")
                print(f"   - Trend: {trend_direction}")
                
            except Exception as e:
                print(f"❌ {symbol} analiz hatası: {e}")
                results.append({
                    'symbol': symbol,
                    'success': False,
                    'error': str(e)
                })
        
        return results
        
    except Exception as e:
        print(f"❌ Entegrasyon testi hatası: {e}")
        return []

def test_performance_metrics():
    """Performans metriklerini test et"""
    print("\n⚡ PERFORMANS METRİKLERİ TESTİ")
    print("=" * 60)
    
    try:
        import time
        
        # Test verisi
        collector = DataCollector()
        df = collector.get_historical_data('BTC/USDT', '1h', 500)
        
        if df is None or df.empty:
            print("❌ Test verisi alınamadı")
            return None
        
        # Teknik analiz performansı
        print("🔧 Teknik analiz performansı ölçülüyor...")
        ta = TechnicalAnalysis()
        
        start_time = time.time()
        df_with_indicators = ta.calculate_all_indicators(df)
        ta_time = time.time() - start_time
        
        print(f"✅ Teknik analiz: {ta_time:.2f} saniye")
        
        # AI model performansı
        print("🤖 AI model performansı ölçülüyor...")
        ai_model = AIModel()
        
        # Feature'ları hazırla
        for col in FIXED_FEATURE_LIST:
            if col not in df_with_indicators.columns:
                df_with_indicators[col] = 0
        
        df_for_ai = df_with_indicators[FIXED_FEATURE_LIST]
        
        start_time = time.time()
        ai_result = ai_model.predict(df_for_ai)
        ai_time = time.time() - start_time
        
        print(f"✅ AI model: {ai_time:.2f} saniye")
        
        # Toplam performans
        total_time = ta_time + ai_time
        throughput = len(df) / total_time
        
        print(f"\n📊 PERFORMANS SONUÇLARI:")
        print(f"   - Teknik analiz süresi: {ta_time:.2f} saniye")
        print(f"   - AI model süresi: {ai_time:.2f} saniye")
        print(f"   - Toplam süre: {total_time:.2f} saniye")
        print(f"   - Veri işleme hızı: {throughput:.1f} satır/saniye")
        print(f"   - Feature sayısı: {len(df_with_indicators.columns)}")
        
        return {
            'ta_time': ta_time,
            'ai_time': ai_time,
            'total_time': total_time,
            'throughput': throughput,
            'feature_count': len(df_with_indicators.columns)
        }
        
    except Exception as e:
        print(f"❌ Performans testi hatası: {e}")
        return None

def create_compatibility_report(feature_compat, data_flow, integration_results, performance):
    """Uyumluluk raporu oluştur"""
    print("\n📋 AI VE TEKNİK ANALİZ UYUMLULUK RAPORU")
    print("=" * 80)
    print(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Feature uyumluluğu
    if feature_compat:
        print("🔧 FEATURE UYUMLULUĞU:")
        print(f"   - Teknik analiz feature'ları: {len(feature_compat['ta_features'])}")
        print(f"   - AI model feature'ları: {len(feature_compat['model_features'])}")
        print(f"   - Ortak feature'lar: {len(feature_compat['common_features'])}")
        print(f"   - Eksik feature'lar: {len(feature_compat['missing_features'])}")
        print(f"   - Uyumluluk oranı: {feature_compat['compatibility_ratio']:.2%}")
        
        if feature_compat['compatibility_ratio'] >= 0.8:
            print("   ✅ Yüksek uyumluluk")
        elif feature_compat['compatibility_ratio'] >= 0.6:
            print("   ⚠ Orta uyumluluk")
        else:
            print("   ❌ Düşük uyumluluk")
    
    # Veri akışı
    if data_flow:
        print(f"\n📊 VERİ AKIŞI:")
        print(f"   - Veri boyutu: {data_flow['data_shape']}")
        print(f"   - Teknik analiz boyutu: {data_flow['ta_shape']}")
        print(f"   - AI feature sayısı: {data_flow['ai_features']}")
        print(f"   - AI tahmin: {data_flow['ai_prediction']:.4f}")
        print(f"   - AI güven: {data_flow['ai_confidence']:.4f}")
    
    # Entegrasyon sonuçları
    if integration_results:
        print(f"\n🔄 ENTEGRASYON SONUÇLARI:")
        successful_tests = [r for r in integration_results if r['success']]
        failed_tests = [r for r in integration_results if not r['success']]
        
        print(f"   - Başarılı test: {len(successful_tests)}/{len(integration_results)}")
        print(f"   - Başarı oranı: {(len(successful_tests)/len(integration_results))*100:.1f}%")
        
        if successful_tests:
            avg_ai_score = np.mean([r['ai_score'] for r in successful_tests])
            avg_confidence = np.mean([r['confidence'] for r in successful_tests])
            avg_ta_strength = np.mean([r['ta_strength'] for r in successful_tests])
            
            print(f"   - Ortalama AI skor: {avg_ai_score:.4f}")
            print(f"   - Ortalama güven: {avg_confidence:.4f}")
            print(f"   - Ortalama TA gücü: {avg_ta_strength:.4f}")
    
    # Performans metrikleri
    if performance:
        print(f"\n⚡ PERFORMANS METRİKLERİ:")
        print(f"   - Teknik analiz süresi: {performance['ta_time']:.2f} saniye")
        print(f"   - AI model süresi: {performance['ai_time']:.2f} saniye")
        print(f"   - Toplam süre: {performance['total_time']:.2f} saniye")
        print(f"   - İşleme hızı: {performance['throughput']:.1f} satır/saniye")
        print(f"   - Feature sayısı: {performance['feature_count']}")
    
    # Genel değerlendirme
    print(f"\n🎯 GENEL DEĞERLENDİRME:")
    
    overall_score = 0
    total_checks = 0
    
    if feature_compat and feature_compat['compatibility_ratio'] >= 0.8:
        overall_score += 1
    total_checks += 1
    
    if data_flow and data_flow['ai_prediction'] > 0:
        overall_score += 1
    total_checks += 1
    
    if integration_results and len([r for r in integration_results if r['success']]) >= 2:
        overall_score += 1
    total_checks += 1
    
    if performance and performance['total_time'] < 10:
        overall_score += 1
    total_checks += 1
    
    overall_percentage = (overall_score / total_checks) * 100 if total_checks > 0 else 0
    
    print(f"   - Genel uyumluluk skoru: {overall_percentage:.1f}%")
    
    if overall_percentage >= 80:
        print("   ✅ MÜKEMMEL UYUMLULUK")
    elif overall_percentage >= 60:
        print("   ⚠ İYİ UYUMLULUK")
    elif overall_percentage >= 40:
        print("   ⚠ ORTA UYUMLULUK")
    else:
        print("   ❌ DÜŞÜK UYUMLULUK")
    
    print("=" * 80)

def main():
    """Ana test fonksiyonu"""
    print("🚀 KAHİN ULTIMA - AI VE TEKNİK ANALİZ UYUMLULUK TESTİ")
    print("=" * 80)
    
    try:
        # 1. Feature uyumluluğu testi
        feature_compat = test_feature_compatibility()
        
        # 2. Veri akışı testi
        data_flow = test_data_flow()
        
        # 3. Entegrasyon iş akışı testi
        integration_results = test_integration_workflow()
        
        # 4. Performans metrikleri testi
        performance = test_performance_metrics()
        
        # 5. Rapor oluştur
        create_compatibility_report(feature_compat, data_flow, integration_results, performance)
        
        print(f"\n✅ Uyumluluk testi tamamlandı!")
        return True
        
    except Exception as e:
        print(f"❌ Uyumluluk testi hatası: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 