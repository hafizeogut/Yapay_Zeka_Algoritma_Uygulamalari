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
    iterations = 100

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

    return w0, w1, w2

# Lojistik regresyon modelini uygula
w0, w1, w2 = logistic_regression(X1, X2)

# Doğrusal regresyon modelini oluştur
X = np.vstack([X1, X2]).T
regressor = LinearRegression()
regressor.fit(X, y)

# Doğrusal regresyon doğrusunu çiz
plt.figure(figsize=(8, 6))
plt.scatter(X1[y == 1], X2[y == 1], color='blue', label='Etiket 1')
plt.scatter(X1[y == -1], X2[y == -1], color='red', label='Etiket -1')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Girdiler ve Etiketler Arasındaki Bağlantı')
plt.legend()

# Doğrusal regresyon doğrusunu çiz
x_values = np.array([min(X1), max(X1)])
y_values = (-regressor.coef_[0] * x_values - regressor.intercept_) / regressor.coef_[1]
plt.plot(x_values, y_values, color='green', linestyle='--', label='Doğrusal Regresyon Doğrusu')

plt.legend()
plt.show()
