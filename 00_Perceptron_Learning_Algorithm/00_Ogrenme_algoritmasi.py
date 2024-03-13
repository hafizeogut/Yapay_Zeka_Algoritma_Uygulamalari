# Ağırlıkları ve sapmaları küçük rastgele değerlere sıfırlayın.
# Algılayıcıya bir girdi örneği sunun ve girdilerin ağırlıklı toplamını hesaplayın.
# Algılayıcının çıktısını belirlemek için ağırlıklı toplama bir aktivasyon fonksiyonu (tipik olarak bir adım fonksiyonu) uygulayın.
# Giriş örneği için tahmin edilen çıktıyı istenen çıktıyla karşılaştırın.
# Tahmin edilen çıktı ile istenen çıktı arasındaki hataya göre ağırlıkları ve sapmaları ayarlayın.
# Eğitim veri kümesindeki tüm giriş örnekleri için 2-5 arasındaki adımları tekrarlayın.
# Perceptron tatmin edici bir doğruluk veya yakınsama elde edene kadar eğitim veri kümesini birden çok kez (dönemler) yineleyin.

import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(weighted_sum)

    def train(self, training_inputs, labels, epochs):
        for epoch in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

# Örnek eğitim veri kümesi
training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 0, 0, 1])  # AND işlemi için beklenen çıktılar

# Perceptron'u oluştur ve eğit
perceptron = Perceptron(input_size=2)
perceptron.train(training_inputs, labels, epochs=100)

# Test et
test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for inputs in test_inputs:
    prediction = perceptron.predict(inputs)
    print(f"Girdiler: {inputs}, Tahmin: {prediction}")
