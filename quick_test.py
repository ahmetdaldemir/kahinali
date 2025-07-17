from config import Config

print("Config değerleri (1/10 seviye - EN GEVŞEK):")
print(f"MIN_SIGNAL_CONFIDENCE: {Config.MIN_SIGNAL_CONFIDENCE}")
print(f"MIN_AI_SCORE: {Config.MIN_AI_SCORE}")
print(f"MIN_TA_STRENGTH: {Config.MIN_TA_STRENGTH}")
print(f"SIGNAL_QUALITY_THRESHOLD: {Config.SIGNAL_QUALITY_THRESHOLD}")
print(f"MIN_CONFIDENCE_THRESHOLD: {Config.MIN_CONFIDENCE_THRESHOLD}")
print(f"MIN_TOTAL_SCORE: {Config.MIN_TOTAL_SCORE}")

# Test sinyal skorları (loglardan)
ai_score = 0.2754
ta_strength = 0.2667
quality_score = 0.3518
confidence = 0.7000

print(f"\nTest sinyal skorları:")
print(f"AI Score: {ai_score}")
print(f"TA Strength: {ta_strength}")
print(f"Quality Score: {quality_score}")
print(f"Confidence: {confidence}")

# Filtreleme testi
print(f"\nFiltreleme sonuçları:")
print(f"AI Score geçti mi: {ai_score >= Config.MIN_AI_SCORE}")
print(f"TA Strength geçti mi: {ta_strength >= Config.MIN_TA_STRENGTH}")
print(f"Quality Score geçti mi: {quality_score >= Config.SIGNAL_QUALITY_THRESHOLD}")
print(f"Confidence geçti mi: {confidence >= Config.MIN_CONFIDENCE_THRESHOLD}")

# Toplam skor hesaplama (basit)
total_score = (ai_score + ta_strength + quality_score + confidence) / 4
print(f"Toplam Score: {total_score:.4f}")
print(f"Toplam Score geçti mi: {total_score >= Config.MIN_TOTAL_SCORE}")

# Signal confidence hesaplama
signal_confidence = (ai_score * 0.4 + ta_strength * 0.3 + quality_score * 0.2 + confidence * 0.1)
print(f"Signal Confidence: {signal_confidence:.4f}")
print(f"Signal Confidence geçti mi: {signal_confidence >= Config.MIN_SIGNAL_CONFIDENCE}")

print(f"\nSonuç: Test sinyali {'GEÇER' if (ai_score >= Config.MIN_AI_SCORE and ta_strength >= Config.MIN_TA_STRENGTH and quality_score >= Config.SIGNAL_QUALITY_THRESHOLD and confidence >= Config.MIN_CONFIDENCE_THRESHOLD and signal_confidence >= Config.MIN_SIGNAL_CONFIDENCE) else 'GEÇMEZ'}") 