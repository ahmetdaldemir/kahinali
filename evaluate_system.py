import sqlalchemy
from sqlalchemy import text
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Veritabanı bağlantısı
DATABASE_URL = 'postgresql://laravel:secret@localhost:5432/kahin_ultima'
engine = sqlalchemy.create_engine(DATABASE_URL)

def evaluate_system_performance():
    """Sistem performansını kapsamlı değerlendir"""
    try:
        print("🔍 SİSTEM PERFORMANS DEĞERLENDİRMESİ")
        print("=" * 50)
        
        with engine.connect() as conn:
            # 1. Genel İstatistikler
            print("\n📊 GENEL İSTATİSTİKLER")
            print("-" * 30)
            
            # Toplam sinyal sayısı
            total_query = "SELECT COUNT(*) as total FROM signals"
            total_result = conn.execute(text(total_query))
            total_signals = total_result.scalar()
            print(f"Toplam Sinyal: {total_signals}")
            
            # Son 30 günlük sinyaller
            recent_query = """
                SELECT COUNT(*) as recent FROM signals 
                WHERE timestamp::timestamp >= NOW() - INTERVAL '30 days'
            """
            recent_result = conn.execute(text(recent_query))
            recent_signals = recent_result.scalar()
            print(f"Son 30 Gün: {recent_signals}")
            
            # 2. Sinyal Kalitesi Analizi
            print("\n🎯 SİNYAL KALİTESİ")
            print("-" * 30)
            
            # AI skorları
            ai_score_query = """
                SELECT 
                    AVG(ai_score) as avg_ai_score,
                    MIN(ai_score) as min_ai_score,
                    MAX(ai_score) as max_ai_score,
                    STDDEV(ai_score) as std_ai_score
                FROM signals 
                WHERE ai_score IS NOT NULL
            """
            ai_score_result = conn.execute(text(ai_score_query))
            ai_stats = ai_score_result.fetchone()
            
            print(f"Ortalama AI Skoru: {ai_stats[0]:.3f}")
            print(f"En Düşük AI Skoru: {ai_stats[1]:.3f}")
            print(f"En Yüksek AI Skoru: {ai_stats[2]:.3f}")
            print(f"AI Skor Standart Sapması: {ai_stats[3]:.3f}")
            
            # Yüksek kaliteli sinyaller (AI skoru > 0.8)
            high_quality_query = """
                SELECT COUNT(*) as high_quality
                FROM signals 
                WHERE ai_score > 0.8
            """
            high_quality_result = conn.execute(text(high_quality_query))
            high_quality_count = high_quality_result.scalar()
            high_quality_percentage = (high_quality_count / total_signals * 100) if total_signals > 0 else 0
            print(f"Yüksek Kaliteli Sinyaller (>0.8): {high_quality_count} ({high_quality_percentage:.1f}%)")
            
            # 3. Başarı Oranı Analizi
            print("\n📈 BAŞARI ORANI")
            print("-" * 30)
            
            # Tamamlanmış sinyaller
            completed_query = """
                SELECT 
                    COUNT(*) as completed,
                    COUNT(CASE WHEN result = 'profit' THEN 1 END) as profitable,
                    COUNT(CASE WHEN result = 'loss' THEN 1 END) as loss,
                    COUNT(CASE WHEN result = 'TIMEOUT' THEN 1 END) as timeout
                FROM signals 
                WHERE result IS NOT NULL AND result != 'None'
            """
            completed_result = conn.execute(text(completed_query))
            completed_stats = completed_result.fetchone()
            
            completed_count = completed_stats[0]
            profitable_count = completed_stats[1]
            loss_count = completed_stats[2]
            timeout_count = completed_stats[3]
            
            if completed_count > 0:
                success_rate = (profitable_count / completed_count) * 100
                loss_rate = (loss_count / completed_count) * 100
                timeout_rate = (timeout_count / completed_count) * 100
                
                print(f"Tamamlanmış Sinyal: {completed_count}")
                print(f"Kazançlı: {profitable_count} ({success_rate:.1f}%)")
                print(f"Zararlı: {loss_count} ({loss_rate:.1f}%)")
                print(f"Zaman Aşımı: {timeout_count} ({timeout_rate:.1f}%)")
            else:
                print("Henüz tamamlanmış sinyal yok")
            
            # 4. Ortalama Kazanç/Kayıp
            print("\n💰 KAZANÇ/KAYIP ANALİZİ")
            print("-" * 30)
            
            if completed_count > 0:
                profit_query = """
                    SELECT 
                        AVG(realized_gain) as avg_gain,
                        SUM(realized_gain) as total_gain,
                        MIN(realized_gain) as min_gain,
                        MAX(realized_gain) as max_gain
                    FROM signals 
                    WHERE result IS NOT NULL AND result != 'None' AND realized_gain IS NOT NULL
                """
                profit_result = conn.execute(text(profit_query))
                profit_stats = profit_result.fetchone()
                
                print(f"Ortalama Kazanç/Kayıp: {profit_stats[0]:.2f}%")
                print(f"Toplam Kazanç/Kayıp: {profit_stats[1]:.2f}%")
                print(f"En Düşük: {profit_stats[2]:.2f}%")
                print(f"En Yüksek: {profit_stats[3]:.2f}%")
            
            # 5. Model Performansı
            print("\n🤖 MODEL PERFORMANSI")
            print("-" * 30)
            
            # Teknik analiz skorları
            ta_query = """
                SELECT 
                    AVG(ta_strength) as avg_ta,
                    AVG(whale_score) as avg_whale,
                    AVG(social_score) as avg_social,
                    AVG(news_score) as avg_news,
                    AVG(breakout_probability) as avg_breakout
                FROM signals 
                WHERE ta_strength IS NOT NULL
            """
            ta_result = conn.execute(text(ta_query))
            ta_stats = ta_result.fetchone()
            
            print(f"Ortalama Teknik Analiz: {ta_stats[0]:.3f}")
            print(f"Ortalama Whale Skoru: {ta_stats[1]:.3f}")
            print(f"Ortalama Sosyal Skor: {ta_stats[2]:.3f}")
            print(f"Ortalama Haber Skoru: {ta_stats[3]:.3f}")
            print(f"Ortalama Breakout Olasılığı: {ta_stats[4]:.3f}")
            
            # 6. Sistem Sağlığı
            print("\n🏥 SİSTEM SAĞLIĞI")
            print("-" * 30)
            
            # Son sinyal tarihi
            last_signal_query = "SELECT MAX(timestamp) as last_signal FROM signals"
            last_signal_result = conn.execute(text(last_signal_query))
            last_signal_time = last_signal_result.scalar()
            
            if last_signal_time:
                # String'i datetime'a çevir
                if isinstance(last_signal_time, str):
                    last_signal_time = datetime.fromisoformat(last_signal_time.replace('Z', '+00:00'))
                
                time_diff = datetime.now() - last_signal_time
                hours_since_last = time_diff.total_seconds() / 3600
                print(f"Son Sinyal: {last_signal_time}")
                print(f"Son Sinyalden Bu Yana: {hours_since_last:.1f} saat")
                
                if hours_since_last < 24:
                    print("✅ Sistem aktif")
                    system_health_score = 15
                elif hours_since_last < 72:
                    print("⚠️ Sistem yavaş")
                    system_health_score = 10
                else:
                    print("❌ Sistem pasif")
                    system_health_score = 5
            else:
                print("Son sinyal bilgisi bulunamadı")
                system_health_score = 0
            
            # 7. Sinyal Dağılımı
            print("\n📊 SİNYAL DAĞILIMI")
            print("-" * 30)
            
            # En çok sinyal üretilen coinler
            top_coins_query = """
                SELECT symbol, COUNT(*) as count
                FROM signals 
                GROUP BY symbol 
                ORDER BY count DESC 
                LIMIT 10
            """
            top_coins_result = conn.execute(text(top_coins_query))
            top_coins = top_coins_result.fetchall()
            
            print("En Çok Sinyal Üretilen Coinler:")
            for i, (symbol, count) in enumerate(top_coins, 1):
                print(f"{i}. {symbol}: {count} sinyal")
            
            # 8. Genel Puanlama
            print("\n⭐ GENEL SİSTEM PUANI")
            print("-" * 30)
            
            # Puanlama kriterleri
            scores = {}
            
            # 1. Sinyal Aktivitesi (0-20 puan)
            if recent_signals > 100:
                scores['aktivite'] = 20
            elif recent_signals > 50:
                scores['aktivite'] = 15
            elif recent_signals > 20:
                scores['aktivite'] = 10
            else:
                scores['aktivite'] = 5
            
            # 2. AI Skor Kalitesi (0-20 puan)
            if ai_stats[0] > 0.8:
                scores['ai_kalite'] = 20
            elif ai_stats[0] > 0.7:
                scores['ai_kalite'] = 15
            elif ai_stats[0] > 0.6:
                scores['ai_kalite'] = 10
            else:
                scores['ai_kalite'] = 5
            
            # 3. Başarı Oranı (0-30 puan)
            if completed_count > 0:
                if success_rate > 70:
                    scores['basari'] = 30
                elif success_rate > 60:
                    scores['basari'] = 25
                elif success_rate > 50:
                    scores['basari'] = 20
                elif success_rate > 40:
                    scores['basari'] = 15
                else:
                    scores['basari'] = 10
            else:
                scores['basari'] = 0
            
            # 4. Sistem Sağlığı (0-15 puan)
            scores['saglik'] = system_health_score
            
            # 5. Veri Kalitesi (0-15 puan)
            if high_quality_percentage > 50:
                scores['veri_kalite'] = 15
            elif high_quality_percentage > 30:
                scores['veri_kalite'] = 10
            elif high_quality_percentage > 15:
                scores['veri_kalite'] = 5
            else:
                scores['veri_kalite'] = 0
            
            # Toplam puan
            total_score = sum(scores.values())
            max_score = 100
            
            print("Puanlama Detayları:")
            for category, score in scores.items():
                print(f"  {category.replace('_', ' ').title()}: {score}/20")
            
            print(f"\n🎯 TOPLAM SİSTEM PUANI: {total_score}/{max_score}")
            
            # Puan değerlendirmesi
            if total_score >= 85:
                grade = "A+ (Mükemmel)"
                emoji = "🌟"
            elif total_score >= 75:
                grade = "A (Çok İyi)"
                emoji = "⭐"
            elif total_score >= 65:
                grade = "B+ (İyi)"
                emoji = "👍"
            elif total_score >= 55:
                grade = "B (Orta)"
                emoji = "😐"
            elif total_score >= 45:
                grade = "C (Geliştirilmeli)"
                emoji = "⚠️"
            else:
                grade = "D (Kritik)"
                emoji = "🚨"
            
            print(f"{emoji} SİSTEM NOTU: {grade}")
            
            # İyileştirme önerileri
            print("\n💡 İYİLEŞTİRME ÖNERİLERİ")
            print("-" * 30)
            
            if scores['aktivite'] < 15:
                print("• Sinyal üretim sıklığını artırın")
            
            if scores['ai_kalite'] < 15:
                print("• AI modellerini yeniden eğitin")
            
            if scores['basari'] < 20:
                print("• Sinyal filtreleme kriterlerini sıkılaştırın")
            
            if scores['saglik'] < 10:
                print("• Sistem bakımı yapın")
            
            if scores['veri_kalite'] < 10:
                print("• Veri kalitesini artırın")
            
            return total_score, grade
            
    except Exception as e:
        logger.error(f"Sistem değerlendirme hatası: {e}")
        return 0, "Hata"

if __name__ == "__main__":
    score, grade = evaluate_system_performance()
    print(f"\n📋 ÖZET: Sistem {score}/100 puan aldı - {grade}") 