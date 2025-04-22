# Neural Network from Scratch (NumPy-based)

This project implements a deep neural network from scratch using only NumPy. It includes custom layers, activation functions, a categorical cross-entropy loss function, and optimizers (SGD and Adam). The model is trained and evaluated on the **MNIST** dataset (handwritten digit recognition), and also supports training on a synthetic **spiral** dataset.

---

## 📦 Features

- Custom `Dense` layers
- ReLU and Softmax activation functions
- Categorical Cross-Entropy loss
- Backpropagation and gradient computation
- Optimizers: SGD and Adam
- Training loop with batching and validation
- Accuracy and loss tracking
- Supports both spiral dataset and MNIST dataset

---

## 🧠 Architecture


---

## 🧪 Datasets Used

- **Spiral Data**: Used for testing simple classification with 3 classes.
- **MNIST (784-dim)**: Used for training and evaluating digit classification.

---

## 🛠️ Dependencies

- Python 3.12+
- `numpy`
- `scikit-learn`
- `nnfs` (for spiral dataset)
- `openml` (via `sklearn.datasets.fetch_openml`)

Install required packages:

```bash
pip install numpy scikit-learn nnfs
sys.path.append('./venv/lib/python3.12.3/site-packages')

model.train(
    X_train, y_train,
    epochs=1000,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=100
)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

sample_input = np.array([[0.5, -0.2], [-1.0, 1.0]])
predictions = np.argmax(model.forward(sample_input), axis=1)
print(f"Sample predictions: {predictions}")

Epoch 1/1000 - loss: 2.3016 - acc: 0.1132 - val_loss: 2.3020 - val_acc: 0.1135
...
Epoch 1000/1000 - loss: 0.0475 - acc: 0.9861 - val_loss: 0.0920 - val_acc: 0.9710

Test accuracy: 0.9710, Test loss: 0.0920
Sample predictions: [1 0]

.
├── main.py                 # Entire neural net code with training & evaluation
├── README.md               # Project description (this file)

X, y = spiral_data(100, 3)


