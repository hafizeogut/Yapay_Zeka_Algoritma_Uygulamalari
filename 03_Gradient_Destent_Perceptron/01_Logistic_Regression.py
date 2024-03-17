import numpy as np

def mezun_olasiligi_hesapla(SAT, gelir):
    # Ölçeklendirme faktörleri
    SAT_scale = 0.001
    gelir_scale = 0.00001
    
    # Özelliklerin ölçeklendirilmesi
    scaled_SAT = SAT * SAT_scale
    scaled_gelir = gelir * gelir_scale
    
    # Lojistik regresyon modeli denklemi
    exponent = -6.24 + 0.439 * scaled_SAT + 0.222 * scaled_gelir
    
    # Mezun olma olasılığını hesapla
    mezun_olasiligi = 1 / (1 + np.exp(exponent))
    
    return mezun_olasiligi

# Örnek SAT puanı ve ebeveyn geliri
SAT = 1600
gelir = 80000

# Mezun olma olasılığını hesapla
olasilik = mezun_olasiligi_hesapla(SAT, gelir)
print("Mezun olma olasılığı:", olasilik)
