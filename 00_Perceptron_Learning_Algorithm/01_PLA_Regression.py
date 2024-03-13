import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.00001):
        # Ağırlıkları ve sapmayı rastgele başlat
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        # Öğrenme oranı
        self.learning_rate = learning_rate

    def activation_function(self, x):
        # Aktivasyon fonksiyonu olarak doğrusal fonksiyon kullanıyoruz.
        return x  # Doğrusal aktivasyon fonksiyonu

    def predict(self, inputs):
        # Ağırlıklı toplamı hesapla
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        # Aktivasyon fonksiyonunu uygula
        return self.activation_function(weighted_sum)

    def train(self, training_inputs, labels, epochs):
        # Belirli bir epoch sayısı kadar iterasyon yaparak eğitimi gerçekleştir
        for epoch in range(epochs):
            # Her veri örneği ve etiketi için eğitimi gerçekleştir
            for inputs, label in zip(training_inputs, labels):
                # Tahmini hesapla
                prediction = self.predict(inputs)
                # Hata hesapla
                error = label - prediction
                # Ağırlıkları güncelle
                self.weights += self.learning_rate * error * inputs
                # Sapmayı güncelle
                self.bias += self.learning_rate * error

# Örnek eğitim veri kümesi ve beklenen çıktılar (doğrusal ilişki)
training_inputs = np.array([[1], [2], [3], [4], [5]])
labels = np.array([3, 5, 7, 9, 11])  # 2x+1 doğrusal ilişkisi

# Perceptron'u oluştur ve eğit
perceptron = Perceptron(input_size=1)
perceptron.train(training_inputs, labels, epochs=1000)

# Test et
test_inputs = np.array([[6], [7], [8]])
for inputs in test_inputs:
    prediction = perceptron.predict(inputs)
    print(f"Girdi: {inputs}, Tahmin: {prediction}")
