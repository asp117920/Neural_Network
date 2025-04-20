import sys
sys.path.append('./venv/lib/python3.12.3/site-packages')
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from sklearn.datasets import fetch_openml

nnfs.init()

X, y = spiral_data(100, 3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2./n_inputs)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs

class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
        return self.dinputs

class Loss:
    def calculate(self, y_pred, y_true):
        sample_losses = self.forward(y_pred, y_true)
        return np.mean(sample_losses)

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        return -np.log(correct_confidences)
    
    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = y_pred.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
        return self.dinputs

class Optimizer_SGD:
    def __init__(self, learning_rate=1):
        self.learning_rate = learning_rate
    
    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iterations = 0
    
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Update momentum and cache
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * (layer.dweights**2)
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * (layer.dbiases**2)

        # Bias correction
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Parameter update
        layer.weights -= self.learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases -= self.learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

        self.iterations += 1


class NeuralNetwork:
    def __init__(self, layers, loss, optimizer):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
    
    def forward(self, X, training=True):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, dvalues):
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)
    
    def train(self, X, y, epochs, batch_size=32, validation_data=None, verbose=1):
        history = {'loss': [], 'accuracy': []}
        if validation_data:
            history['val_loss'] = []
            history['val_accuracy'] = []
        
        for epoch in range(1, epochs + 1):
            # Shuffle data
            permutation = np.random.permutation(len(X))
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            
            batch_loss = 0
            batch_acc = 0
            batch_count = 0
            
            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                output = self.forward(X_batch)
                
                # Loss and accuracy
                loss = self.loss.calculate(output, y_batch)
                preds = np.argmax(output, axis=1)
                acc = np.mean(preds == y_batch)
                
                batch_loss += loss * len(X_batch)
                batch_acc += acc * len(X_batch)
                batch_count += len(X_batch)
                
                # Backward pass
                self.loss.backward(output, y_batch)
                self.backward(self.loss.dinputs)
                
                # Update parameters
                for layer in self.layers:
                    if hasattr(layer, 'weights'):
                        self.optimizer.update_params(layer)
            
            # Store metrics
            epoch_loss = batch_loss / batch_count
            epoch_acc = batch_acc / batch_count
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            
            # Validation
            if validation_data:
                val_loss, val_acc = self.evaluate(*validation_data, batch_size)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
            
            # Print progress
            if verbose and (epoch % verbose == 0 or epoch == 1 or epoch == epochs):
                msg = f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}"
                if validation_data:
                    msg += f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
                print(msg)
        
        return history
    
    def evaluate(self, X, y, batch_size=32):
        total_loss = 0
        total_acc = 0
        total_samples = 0
        
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            
            output = self.forward(X_batch, training=False)
            loss = self.loss.calculate(output, y_batch)
            preds = np.argmax(output, axis=1)
            acc = np.mean(preds == y_batch)
            
            total_loss += loss * len(X_batch)
            total_acc += acc * len(X_batch)
            total_samples += len(X_batch)
        
        return total_loss / total_samples, total_acc / total_samples

# Create and split dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False,parser='auto')


X, y = mnist.data, mnist.target.astype(np.int8)  # Convert labels to int

X = X.reshape(X.shape[0], -1) 
X = X / 255.0  


scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=40)

# Initialize components
dense1 = Layer_Dense(2, 128)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(128, 3)
activation2 = Activation_Softmax()
loss = Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(learning_rate=1)
# optimizer = Optimizer_Adam(learning_rate=0.005)


# Create model
model = NeuralNetwork(
    layers=[dense1, activation1, dense2, activation2],
    loss=loss,
    optimizer=optimizer
)

# Train the model
history = model.train(
    X_train, y_train,
    epochs=1000,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=100
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# Predict
sample_input = np.array([[0.5, -0.2], [-1.0, 1.0]])
predictions = np.argmax(model.forward(sample_input), axis=1)
print(f"Sample predictions: {predictions}")