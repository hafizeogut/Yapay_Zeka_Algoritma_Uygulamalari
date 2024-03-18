# Ağırlıkları varsayılan değerlere başlat.
w0 = 1
w1 = 1
w2 = 1

# Öğrenme oranı
alpha = 0.1

# Veriler
X1 = [1,2]
X2 = [3, 4]
y = [1, -1]

# Tekrar döngüsünün ilk iterasyonu
for _ in range(8):
    # Veri noktaları üzerinde dön
    for i in range(len(X1)):
        # Ağırlıklı toplamı hesapla
        z = w0 + w1 * X1[i] + w2 * X2[i]
        
        # Tahmini hesapla
        if z > 0:
            y_pred = 1
        else:
            y_pred = -1
        
        # Ağırlıkları güncelle
        w0 = w0 - alpha * (y_pred - y[i])
        w1 = w1 - alpha * (y_pred - y[i]) * X1[i]
        w2 = w2 - alpha * (y_pred - y[i]) * X2[i]

# Güncellenmiş ağırlıkları yazdır
print("Iki iterasyon sonrasi guncellenmis agırlıklar:")
print(z)
print("w0 =", w0)
print("w1 =", w1)
print("w2 =", w2)
