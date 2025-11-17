import numpy as np

def sigmoid(x): 
     x_clipped = np.clip(x, -500, 500)
     return 1 / (1 + np.exp(-x_clipped))
def relu(x): return np.maximum(0, x)
def tanh(x): return np.tanh(x)

def get_fun(name):
    funcs = {'sigmoid': sigmoid, 'relu': relu, 'tanh': tanh}
    try:
        return funcs[name]
    except KeyError:
        raise ValueError(f"Unknown activation '{name}'. Valid: {list(funcs.keys())}")

def get_random_ws(nodes):
    return [np.random.uniform(-1, 1, (nodes[i], nodes[i + 1])) for i in range(len(nodes) - 1)]

def get_random_bs(nodes):
    return [np.zeros((1, nodes[i+1])) for i in range(len(nodes)-1)]

class ANN:
    def __init__(self, num_layers, nodes, functions):
        self.check_validity(num_layers, nodes, functions)
        self.num_layers, self.nodes = num_layers, nodes
        self.ws, self.bs = get_random_ws(nodes), get_random_bs(nodes)
        self.functions = [get_fun(f) for f in functions]
        self.check_validity(num_layers, nodes, functions)

    def feedforward(self, start):
        start = np.array(start)

        if start.ndim == 1:
            if start.shape[0] != self.nodes[0]:
                raise ValueError("Input size mismatch")
            cur = start.reshape(1, -1)

        
        elif start.ndim == 2:
            if start.shape[1] != self.nodes[0]:
                raise ValueError("Batch input dimension mismatch")
            cur = start

        else:
            raise ValueError("Input must be 1D or 2D")

        # Forward pass
        for i in range(1, self.num_layers):
            cur = self.get_next(cur, i)

        return cur

    def get_next(self, x, i):
        w, b, f = self.ws[i-1], self.bs[i-1], self.functions[i-1]
        return f(np.dot(x, w) + b)
    
    def set_weights_bias(self, flat_params):
        idx = 0
        for i in range(len(self.ws)):
            ws_shape = self.ws[i].shape
            bs_shape = self.bs[i].shape

            self.ws[i] = flat_params[idx:idx + np.prod(ws_shape)].reshape(ws_shape)
            idx += np.prod(ws_shape)
    
            self.bs[i] = flat_params[idx:idx + np.prod(bs_shape)].reshape(bs_shape)
            idx += np.prod(bs_shape)

        if idx < len(flat_params):
            func_codes = flat_params[idx:]
            self.functions = [
                get_fun(['sigmoid', 'relu', 'tanh'][int(np.clip(round(c), 0, 2))])
                for c in func_codes
                ]

    def check_validity(self, num_layers, nodes, functions):
        if num_layers < 2:
            raise ValueError("At least 2 layers needed")
        if len(nodes) != num_layers:
            raise ValueError("Nodes list must match number of layers")
        if len(functions) != num_layers - 1:
            raise ValueError("Functions list must be one less than layers")
