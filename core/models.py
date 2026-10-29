"""
MEC-AI HetFL: Adaptive Federated Learning for Resource-Constrained IoT Devices
Core Mathematical & Neural Network Model
Paper: Mughal et al., Scientific Reports (Nature Portfolio), 2024
"""

import numpy as np

class HeteroEdgeMLP:
    """
    Lightweight, high-performance Neural Network for resource-constrained IoT devices.
    Architecture: Input -> Dense(Hidden, ReLU) -> Dense(Output, Softmax)
    Supports vectorized forward pass, cross-entropy loss, backprop gradient computation,
    and parameter serialization for federated weight transmission.
    """
    def __init__(self, input_dim=12, hidden_dim=24, output_dim=4, seed=42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rng = np.random.RandomState(seed)
        
        # Xavier / He Initialization
        self.W1 = self.rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = self.rng.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    def get_weights(self):
        """Returns deep copy of model parameters (theta)."""
        return {
            'W1': np.copy(self.W1),
            'b1': np.copy(self.b1),
            'W2': np.copy(self.W2),
            'b2': np.copy(self.b2)
        }

    def set_weights(self, weights):
        """Sets model parameters (theta)."""
        self.W1 = np.copy(weights['W1'])
        self.b1 = np.copy(weights['b1'])
        self.W2 = np.copy(weights['W2'])
        self.b2 = np.copy(weights['b2'])

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_deriv(x):
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, X):
        """Computes activations: Z1 -> A1 -> Z2 -> Probabilities"""
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        probs = self.softmax(z2)
        cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'probs': probs}
        return probs, cache

    def compute_loss(self, probs, y_true):
        """Cross-Entropy Loss L(theta, D)"""
        m = y_true.shape[0]
        # Clip to prevent log(0)
        clipped_probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
        loss = -np.sum(np.log(clipped_probs[np.arange(m), y_true])) / m
        return float(loss)

    def compute_gradients(self, cache, y_true):
        """
        Calculates gradients nabla L(theta, D) via backpropagation
        as required by Eq. (1) and Eq. (2) in paper.
        """
        X = cache['X']
        a1 = cache['a1']
        z1 = cache['z1']
        probs = cache['probs']
        m = y_true.shape[0]

        # Softmax cross-entropy gradient
        dZ2 = np.copy(probs)
        dZ2[np.arange(m), y_true] -= 1.0
        dZ2 /= m

        dW2 = np.dot(a1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        da1 = np.dot(dZ2, self.W2.T)
        dZ1 = da1 * self.relu_deriv(z1)

        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def train_on_batch(self, X, y, lr=0.01):
        """
        Gradient Descent Update Rule:
        theta(t) = theta(t-1) - eta * nabla L(theta(t-1), D)  [Eq. 1]
        """
        probs, cache = self.forward(X)
        loss = self.compute_loss(probs, y)
        grads = self.compute_gradients(cache, y)
        
        self.W1 -= lr * grads['W1']
        self.b1 -= lr * grads['b1']
        self.W2 -= lr * grads['W2']
        self.b2 -= lr * grads['b2']
        
        preds = np.argmax(probs, axis=1)
        acc = float(np.mean(preds == y))
        return loss, acc

    def evaluate(self, X, y):
        """Evaluates model and computes standard metrics (Eqs. 5-8)."""
        probs, _ = self.forward(X)
        loss = self.compute_loss(probs, y)
        preds = np.argmax(probs, axis=1)

        acc = float(np.mean(preds == y))

        # Precision, Recall, F1 for multi-class (macro-averaged)
        classes = np.unique(np.concatenate([y, preds]))
        recalls, precisions, f1s = [], [], []

        for c in classes:
            tp = float(np.sum((preds == c) & (y == c)))
            fp = float(np.sum((preds == c) & (y != c)))
            fn = float(np.sum((preds != c) & (y == c)))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        macro_precision = float(np.mean(precisions)) if precisions else 0.0
        macro_recall = float(np.mean(recalls)) if recalls else 0.0
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0

        return {
            'loss': loss,
            'accuracy': acc,
            'precision': macro_precision,
            'recall': macro_recall,
            'f1_score': macro_f1
        }
