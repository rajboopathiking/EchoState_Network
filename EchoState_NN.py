import numpy as np
from sklearn.linear_model import Ridge

class EchoStateNetwork:
    def __init__(self, n_inputs, n_reservoir, n_outputs, spectral_radius=0.95, input_scaling=1.0, leakage_rate=1.0, random_state=None):
        # Hyperparameters
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leakage_rate = leakage_rate
        self.random_state = random_state

        # Initialize the weights
        np.random.seed(self.random_state)
        self.W_in = np.random.rand(self.n_reservoir, self.n_inputs) * 2 - 1  # Input-to-reservoir weights
        self.W_reservoir = np.random.rand(self.n_reservoir, self.n_reservoir) * 2 - 1  # Reservoir-to-reservoir weights
        self.W_out = np.zeros((self.n_outputs, self.n_reservoir))  # Output weights (to be learned)

        # Scale the reservoir weights to ensure the desired spectral radius
        radius = np.max(np.abs(np.linalg.eigvals(self.W_reservoir)))
        self.W_reservoir *= self.spectral_radius / radius

        # Initialize reservoir state
        self.state = np.zeros(self.n_reservoir)

    def _update_state(self, input_data):
        # Compute the next reservoir state using the input and previous state
        pre_activation = np.dot(self.W_in, input_data) + np.dot(self.W_reservoir, self.state)
        self.state = (1 - self.leakage_rate) * self.state + self.leakage_rate * np.tanh(pre_activation)
        return self.state

    def fit(self, X, y):
        # Train the ESN
        reservoir_states = []
        for t in range(X.shape[0]):
            self.state = self._update_state(X[t])
            reservoir_states.append(self.state)

        # Convert the list of reservoir states to a numpy array
        reservoir_states = np.array(reservoir_states)

        # Train the output layer using Ridge regression (Least Squares with L2 regularization)
        self.W_out = Ridge(alpha=1e-6).fit(reservoir_states, y).coef_

    def predict(self, X):
        # Predict using the trained output layer
        predictions = []
        for t in range(X.shape[0]):
            self.state = self._update_state(X[t])
            predictions.append(np.dot(self.W_out, self.state))
        return np.array(predictions)