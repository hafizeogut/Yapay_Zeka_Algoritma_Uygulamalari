import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sigmoid fonksiyonunu tanımla
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Verileri numpy dizilerine dönüştür
X1 = np.array([1, 2])
X2 = np.array([3, 4])
y = np.array([1, -1])

# Lojistik regresyon modelini oluştur
def logistic_regression(X1, X2):
    # Ağırlıkları varsayılan değerlere başlat
    w0 = 0
    w1 = 0
    w2 = 0

    # Öğrenme oranı
    alpha = 0.1

    # İterasyon sayısı
    iterations = 8

    # Ağırlık güncelleme kayıtları
    w0_updates = [w0]
    w1_updates = [w1]
    w2_updates = [w2]

    # Tekrar döngüsünün ilk iterasyonu
    for _ in range(iterations):
        w0_grad = 0
        w1_grad = 0
        w2_grad = 0

        # Veri noktaları üzerinde dön
        for i in range(len(X1)):
            # Ağırlıklı toplamı hesapla
            z = w0 + w1 * X1[i] + w2 * X2[i]

            # Tahmini hesapla
            y_pred = sigmoid(z)

            # Gradyanları hesapla
            w0_grad += y_pred - y[i]
            w1_grad += (y_pred - y[i]) * X1[i]
            w2_grad += (y_pred - y[i]) * X2[i]

        # Ağırlıkları güncelle
        w0 -= alpha * w0_grad
        w1 -= alpha * w1_grad
        w2 -= alpha * w2_grad

        # Güncellenmiş ağırlıkları kaydet
        w0_updates.append(w0)
        w1_updates.append(w1)
        w2_updates.append(w2)

    return w0, w1, w2, w0_updates, w1_updates, w2_updates

# Lojistik regresyon modelini uygula
w0, w1, w2, w0_updates, w1_updates, w2_updates = logistic_regression(X1, X2)

# Gradyan inişini gösteren grafik
plt.figure(figsize=(10, 6))
plt.plot(range(len(w0_updates)), w0_updates, marker='o', label='w0 Güncellemeleri')
plt.plot(range(len(w1_updates)), w1_updates, marker='o', label='w1 Güncellemeleri')
plt.plot(range(len(w2_updates)), w2_updates,  marker='o',label='w2 Güncellemeleri')
plt.xlabel('İterasyon')
plt.ylabel('Ağırlıklar')
plt.title('Gradyan İnişi: Ağırlık Güncellemeleri')
plt.legend()

plt.tight_layout()
plt.show()
