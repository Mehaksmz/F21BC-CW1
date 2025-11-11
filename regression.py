import pandas as pd

class Regression:
    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
    
    def load_data(self, data):
        df = pd.read_csv('file_name.csv')
        