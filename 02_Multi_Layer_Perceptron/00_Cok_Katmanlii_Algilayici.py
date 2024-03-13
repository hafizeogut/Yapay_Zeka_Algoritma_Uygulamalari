import numpy as np

class MLP:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, epochs=100):
        # MLP modelinin temel özelliklerini tanımla
        self.input_size = input_size  # Giriş boyutu
        self.hidden_layers = hidden_layers  # Gizli katmanların düğüm sayıları
        self.output_size = output_size  # Çıkış boyutu
        self.learning_rate = learning_rate  # Öğrenme oranı
        self.epochs = epochs  # Eğitim epoch sayısı

        # Ağırlıkları ve sapmaları rastgele başlat
        self.weights = [np.random.randn(input_size, hidden_layers[0])]  # Giriş katmanı ile ilk gizli katman arasındaki ağırlıklar
        self.biases = [np.zeros(hidden_layers[0])]  # İlk gizli katmanın sapmaları
        
        # Gizli katmanlar arasındaki ağırlıklar ve sapmalar
        for i in range(1, len(hidden_layers)):
            self.weights.append(np.random.randn(hidden_layers[i-1], hidden_layers[i]))
            self.biases.append(np.zeros(hidden_layers[i]))
        # Son gizli katman ile çıkış katmanı arasındaki ağırlıklar ve sapmalar
        self.weights.append(np.random.randn(hidden_layers[-1], output_size))
        self.biases.append(np.zeros(output_size))

    def sigmoid(self, x):
        # Sigmoid aktivasyon fonksiyonu
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Sigmoid fonksiyonunun türevi
        return x * (1 - x)

    def forward_propagation(self, inputs):
        # İleri yayılım adımı
        activations = [inputs]
        weighted_sums = []
        for i in range(len(self.weights)):
            weighted_sum = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            weighted_sums.append(weighted_sum)
            activation = self.sigmoid(weighted_sum)
            activations.append(activation)
        return activations, weighted_sums

    def backpropagation(self, activations, weighted_sums, labels):
        # Geri yayılım adımı
        errors = [labels - activations[-1]]
        deltas = [errors[-1] * self.sigmoid_derivative(activations[-1])]
        for i in range(len(self.weights)-1, 0, -1):
            error = np.dot(deltas[-1], self.weights[i].T)
            errors.append(error)
            delta = error * self.sigmoid_derivative(activations[i])
            deltas.append(delta)
        deltas.reverse()
        return errors, deltas

    def update_weights(self, activations, deltas):
        # Ağırlıkları güncelle
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * np.dot(activations[i].T, deltas[i])

    def train(self, training_inputs, labels):
        # Eğitim adımı
        for epoch in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                inputs = inputs.reshape(1, -1)  # Girdi vektörünü uygun boyuta dönüştür
                activations, weighted_sums = self.forward_propagation(inputs)
                errors, deltas = self.backpropagation(activations, weighted_sums, label)
                self.update_weights(activations, deltas)

    def predict(self, inputs):
        # Tahmin adımı
        inputs = inputs.reshape(1, -1)  # Girdi vektörünü uygun boyuta dönüştür
        activations, _ = self.forward_propagation(inputs)
        return activations[-1]

# Örnek eğitim veri kümesi ve beklenen çıktılar
training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([[0], [1], [1], [0]])

# MLP modelini oluştur ve eğit

mlp = MLP(input_size=2, hidden_layers=[4, 3], output_size=1, learning_rate=0.1, epochs=10000)
mlp.train(training_inputs, labels)


#hidden_layers=[4, 3]: İlk gizli katmanda 4 düğüm bulunur.
#İkinci gizli katmanda ise 3 düğüm bulunur.

# Test et
test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for inputs in test_inputs:
    # Her bir giriş için tahmin yap
    prediction = mlp.predict(inputs)
    print(f"Girdi: {inputs}, Tahmin: {prediction}")
