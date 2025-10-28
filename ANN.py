class Matrix: #checking git
      def __init__(self, value):
            self.value = value
            self.rows = self.get_rows(value)
            self.columns = self.get_cols(value)
      def __mul__(self, other):
        if self.columns != other.rows:
         raise ValueError("Incompatible matrix dimensions for multiplication")
          
        results = []
        for i in range(self.rows):
            results_row = []
            for j in range(other.columns):
                sum = 0
                for k in range(self.columns):
                    sum += self.value[i][k] * other.value[k][j]
                results_row.append(sum)
            results.append(results_row)
        return Matrix(results)
          
      def __add__(self, other):
          if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Incompatible matrix dimensions for addition")
          
          results = []
          for i in range(self.rows):
                results_row = []
                for j in range(self.columns):
                    results_row.append(self.value[i][j] + other.value[i][j])
                results.append(results_row)
          return Matrix(results)
        
      def get_rows(self, value):
        return len(value)
      
      def get_cols(self, value):
        return len(value[0]) if value else 0
          
       
    
#Ann architecture    
class ANN:
      def init(self, num_of_layers, nodes, functions):
        # check validity and throw error if needed
        # check at least 2 layers (including input)
        # check length of nodes and functions match num of layers
        # check functions are valid strings
        self.nodes = nodes
        self.num_of_layers = num_of_layers 

        #create w based on nodes
        self.ws = _get_random_w(nodes)
        self.bs = _get_random_b(nodes)

        def feedforward(self, start):
            cur = Matrix(start)
            for i in range(1, self.num_of_layers):
                cur = self.get_next(cur, i)
            return cur.value

        def get_next(self, x, i):
            w = self.ws[i - 1]
            b = self.bs[i - 1]
            return f(w * x  + b)

        def get_fun(name):
            if name == 'a':
                def a(num):
                    return numnum
                return a 
            if name == 'b':
                def a(num):
                    return numnum + 1
                return b
        # return error otherwise

        self.functions =  [get_fun(name) for name in functions]
        self.check_validity(num_of_layers, nodes, functions)

