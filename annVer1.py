import math, random

def rand(): return random.uniform(-1, 1)

class Matrix:
    def __init__(self, rows, cols, randomize=True):
        if rows <= 0 or cols <= 0:
            raise ValueError("Matrix cannot be empty")
        self.rows, self.columns = rows, cols
        self.value = [[rand() if randomize else 0 for _ in range(cols)] for _ in range(rows)]

    def __mul__(self, other):
        if self.columns != other.rows:
            raise ValueError("Incompatible matrix dimensions for multiplication")
        result = Matrix(self.rows, other.columns, randomize=False)
        for i in range(self.rows):
            for j in range(other.columns):
                result.value[i][j] = sum(self.value[i][k] * other.value[k][j] for k in range(self.columns))
        return result

    def __add__(self, other):
        if self.rows != other.rows or self.columns != other.columns:
            raise ValueError("Incompatible matrix dimensions for addition")
        result = Matrix(self.rows, self.columns, randomize=False)
        for i in range(self.rows):
            for j in range(self.columns):
                result.value[i][j] = self.value[i][j] + other.value[i][j]
        return result

    def to_list(self):
        return self.value[0] if self.rows == 1 else self.value

def sigmoid(x): return 1 / (1 + math.exp(-x))
def relu(x): return max(0, x)
def tanh(x): return math.tanh(x)

def get_fun(name):
    funcs = {'sigmoid': sigmoid, 'relu': relu, 'tanh': tanh}
    try:
        return funcs[name]
    except KeyError:
        raise ValueError(f"Unknown activation '{name}'. Valid: {list(funcs.keys())}")

def fun_on_matrix(f, m):
    result = Matrix(m.rows, m.columns, randomize=False)
    for i in range(m.rows):
        for j in range(m.columns):
            result.value[i][j] = f(m.value[i][j])
    return result

def get_random_ws(nodes):
    return [Matrix(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]

def get_random_bs(nodes):
    return [Matrix(1, nodes[i+1]) for i in range(len(nodes)-1)]

class ANN:
    def __init__(self, num_layers, nodes, functions):
        self.check_validity(num_layers, nodes, functions)
        self.num_layers, self.nodes = num_layers, nodes
        self.ws, self.bs = get_random_ws(nodes), get_random_bs(nodes)
        self.functions = [get_fun(f) for f in functions]
        self.check_validity(num_layers, nodes, functions)

    def feedforward(self, start):
        if len(start) != self.nodes[0]:
            raise ValueError("Input size mismatch")
        cur = Matrix(1, self.nodes[0], randomize=False)
        cur.value[0] = start
        for i in range(1, self.num_layers):
            cur = self.get_next(cur, i)
        return cur.to_list()

    def get_next(self, x, i):
        w, b, f = self.ws[i-1], self.bs[i-1], self.functions[i-1]
        return fun_on_matrix(f, x * w + b)

    def check_validity(self, num_layers, nodes, functions):
        if num_layers < 2:
            raise ValueError("At least 2 layers needed")
        if len(nodes) != num_layers:
            raise ValueError("Nodes list must match number of layers")
        if len(functions) != num_layers - 1:
            raise ValueError("Functions list must be one less than layers")

ann = ANN(3, [2, 4, 1], ['relu', 'sigmoid'])
output = ann.feedforward([0.5, -0.2])
print(output)