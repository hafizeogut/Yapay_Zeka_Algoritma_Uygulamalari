import numpy as np

class Adaline:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        # Modelin ağırlıklarını ve sapmasını rastgele başlatır
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        # Öğrenme oranı ve epoch sayısı
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        # Aktivasyon fonksiyonu olarak doğrusal fonksiyon kullanılır.
        return x

    def predict(self, inputs):
        # Ağırlıklı toplamı hesapla
        weighted_sum = np.dot(inputs, self.weights) + self.bias#dot iki dizinin çarpımı
        # Aktivasyon fonksiyonunu uygula
        return self.activation_function(weighted_sum)

    def train(self, training_inputs, labels):
        # Belirli sayıda epoch için eğitimi gerçekleştirir
        for _ in range(self.epochs):
            # Her bir eğitim örneği için ağırlıkları güncelle
            for inputs, label in zip(training_inputs, labels):#zip listeleri birleştir.
                # Tahmini hesapla
                prediction = self.predict(inputs)
                
                # Hata hesapla
                error = label - prediction
                
                # Ağırlıkları güncelle (gradient inişi)
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

# Örnek eğitim veri kümesi ve beklenen çıktılar (doğrusal ilişki)
training_inputs = np.array([[1], [2], [3], [4], [5]])
labels = np.array([3, 5, 7, 9, 11])  # 2x+1 doğrusal ilişkisi

# Adaline modelini oluştur ve eğit
adaline = Adaline(input_size=1, learning_rate=0.01, epochs=1000)
adaline.train(training_inputs, labels)

# Test et
test_inputs = np.array([[6], [7], [8]])
for inputs in test_inputs:
    # Her bir test girdisi için tahmin yap
    prediction = adaline.predict(inputs)
    # Tahmin edilen çıktıyı ekrana yazdır
    print(f"Girdi: {inputs}, Tahmin: {prediction}")
