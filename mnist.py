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
model = NeuralNetwork((500, 450, 10), alpha=0.3, reg=1e-3, batch_size=60, epochs=3)
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

print("[+] Calculating error...")
accuracy = sum(predictions == y) / len(y)
print(accuracy)
