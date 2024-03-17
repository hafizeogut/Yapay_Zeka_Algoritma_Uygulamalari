import numpy as np
import matplotlib.pyplot as plt

# Ağırlıkları varsayılan değerlere başlat
w0 = 1
w1 = 1
w2 = 1

# Öğrenme oranı
alpha = 0.1

# Veriler
X1 = [1,2]
X2 = [3, 4]
y = [1, -1]

# İterasyon sayısı
iterations = 5

# Ağırlıkların ve hataların kaydedileceği listeleri oluştur
w0_values = [w0]
w1_values = [w1]
w2_values = [w2]
errors = []

# Tekrar döngüsünün ilk iterasyonu
for _ in range(iterations):
    error = 0
    
    # Veri noktaları üzerinde dön
    for i in range(len(X1)):
        # Ağırlıklı toplamı hesapla
        z = w0 + w1 * X1[i] + w2 * X2[i]
        
        # Tahmini hesapla
        if z > 0:
            y_pred = 1
        else:
            y_pred = -1
        
        # Hata hesapla
        error += abs(y_pred - y[i])
        
        # Ağırlıkları güncelle
        w0 = w0 - alpha * (y_pred - y[i])
        print(w0)
        w1 = w1 - alpha * (y_pred - y[i]) * X1[i]
        # print("w1",w1)
        w2 = w2 - alpha * (y_pred - y[i]) * X2[i]
        # print("w1",w1)
        
    # Ağırlıkları ve hataları kaydet
    w0_values.append(w0)
    w1_values.append(w1)
    w2_values.append(w2)
    errors.append(error) 
# Güncellenmiş ağırlıkları yazdır
print("8 iterasyon sonrasi guncellenmis agırlıklar:")
print("w0 =", w0)
print("w1 =", w1)
print("w2 =", w2)

# Ağırlıkların ve hataların iterasyonlar boyunca nasıl değiştiğini görselleştir
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(w0_values, label='w0')
plt.plot(w1_values, label='w1')
plt.plot(w2_values, label='w2')
plt.xlabel('İterasyon')
plt.ylabel('Ağırlıklar')
plt.title('Ağırlıkların Değişimi')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(errors, color='red')
plt.xlabel('İterasyon')
plt.ylabel('Toplam Hata')
plt.title('Toplam Hata Değişimi')

plt.tight_layout()
plt.show()
