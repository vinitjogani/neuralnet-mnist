import csv, numpy as np, matplotlib.pyplot as plt, pickle, random

from pca import PCA
from neural_network import NeuralNetwork

print("[+] Loading training data...")
reader = csv.reader(open("mnist_train.csv", "r"))
data = np.array(list(reader))

print("[+] Processing data...")
X = (data[:, 1:].astype(np.int) - 127.5) / 127.5
y = data[:, 0].astype(np.int)

print("[+] Running PCA...")
pca = PCA()
X = pca.fit_compress(X, 500)

print("[+] Fitting neural net...")
model = NeuralNetwork((500, 300, 100, 10), alpha=8e-2, reg=1e-3, batch_size=60, epochs=3, momentum = 0.8)
model.fit(X, y)

print("[+] Loading test data...")
reader = csv.reader(open("mnist_test.csv", "r"))
data = np.array(list(reader))

print("[+] Processing data...")
X = (data[:, 1:].astype(np.int) - 127.5) / 127.5
y = data[:, 0].astype(np.int)

print("[+] Compressing data...")
X = pca.compress(X)

print("[+] Making predictions...")
predictions = np.array(model.predict(X))

print("[+] Calculating accuracy...")
accuracy = sum(predictions == y) / len(y)
print(accuracy)
