import numpy as np

# Activation functions
def sigmoid(x): 
     # Clip values to prevent numerical overflow/underflow 
     x_clipped = np.clip(x, -500, 500)
     return 1 / (1 + np.exp(-x_clipped))
def relu(x): return np.maximum(0, x)
def tanh(x): return np.tanh(x)
def linear(x): return x

#Retrieves an activation function by name
def get_fun(name):
    funcs = {'sigmoid': sigmoid, 'relu': relu, 'tanh': tanh, 'linear': linear}
    try:
        return funcs[name]
    except KeyError:
        raise ValueError(f"Unknown activation '{name}'. Valid: {list(funcs.keys())}")

# Generate random weights for each layer
def get_random_ws(nodes):
    return [np.random.uniform(-1, 1, (nodes[i], nodes[i + 1])) for i in range(len(nodes) - 1)]

# Generate zero biases for each layer
def get_random_bs(nodes):
    return [np.zeros((1, nodes[i+1])) for i in range(len(nodes)-1)]


"""
ANN class
Implements a feedforward neural network with configurable architecture
and activation functions.

Parameters:
- num_layers: Total number of layers (input + hidden + output)
- nodes: List of neurons in each layer
- functions: List of activation function names for each layer transition
- check_validity: Validates the architecture parameters
"""
class ANN:
    def __init__(self, num_layers, nodes, functions):
        self.check_validity(num_layers, nodes, functions)
        self.num_layers, self.nodes = num_layers, nodes
        self.ws, self.bs = get_random_ws(nodes), get_random_bs(nodes)
        self.functions = [get_fun(f) for f in functions]
        self.check_validity(num_layers, nodes, functions)


    def feedforward(self, start):
        start = np.array(start)       
        # Handles 1D array input 
        if start.ndim == 1:
             # Check if input size matches the input layer
            if start.shape[0] != self.nodes[0]:
                raise ValueError("Input size mismatch")
            # Reshape to 2D [1, input_size] for matrix operations
            cur = start.reshape(1, -1)
        # Handle 2D array input
        elif start.ndim == 2:
            if start.shape[1] != self.nodes[0]:
                raise ValueError("Batch input dimension mismatch")
            cur = start 
        else:
            raise ValueError("Input must be 1D or 2D")
        # Feedforward through layers
        for i in range(1, self.num_layers):
            cur = self.get_next(cur, i)
        return cur

    #Computes the output of layer i given input x
    def get_next(self, x, i):
        # Get weights, bias, and activation function for this layer
        w, b, f = self.ws[i-1], self.bs[i-1], self.functions[i-1]
        # Return weighted sum + bias, then apply activation function
        return f(np.dot(x, w) + b)
    
    #Set weights and biases from a flattened parameter vector and update activation functions
    def set_weights_bias(self, flat_params):
        idx = 0
        for i in range(len(self.ws)):
            # Get original shapes
            ws_shape = self.ws[i].shape
            bs_shape = self.bs[i].shape
             # Extract and reshape weights
            self.ws[i] = flat_params[idx:idx + np.prod(ws_shape)].reshape(ws_shape)
            idx += np.prod(ws_shape)
            # Extract and reshape bias 
            self.bs[i] = flat_params[idx:idx + np.prod(bs_shape)].reshape(bs_shape)
            idx += np.prod(bs_shape)
        # Update activation functions based on remaining parameters
        if idx < len(flat_params):
            func_codes = flat_params[idx:]
            # Convert codes to activation functions
            # Clip and round to ensure valid indices 
            self.functions = [
                get_fun(['sigmoid', 'relu', 'tanh', 'linear'][int(np.clip(round(c), 0, 3))])
                for c in func_codes
                ]
    #Validate network architecture parameters
    def check_validity(self, num_layers, nodes, functions):
        if num_layers < 2:
            raise ValueError("At least 2 layers needed")
        if len(nodes) != num_layers:
            raise ValueError("Nodes list must match number of layers")
        if len(functions) != num_layers - 1:
            raise ValueError("Functions list must be one less than layers")
