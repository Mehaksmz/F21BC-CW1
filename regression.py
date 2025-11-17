import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ANN import ANN
from PSO import PSO


def load_concrete_data(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_ann_with_pso():
    X_train, X_test, y_train, y_test = load_concrete_data("data/concrete_data.csv")

    input_size = X_train.shape[1]  
    hidden_size = 10             
    output_size = y_train.shape[1]  

    nodes = [input_size, hidden_size, output_size]
    functions = ["relu", "sigmoid"]
    num_layers = len(nodes)

    ann = ANN(num_layers, nodes, functions)

    y_train_pred_initial = ann.feedforward(X_train)
    initial_mae = np.mean(np.abs(y_train_pred_initial - y_train))
    initial_mse = np.mean((y_train_pred_initial - y_train) ** 2)
    initial_rmse = np.sqrt(initial_mse)
    
    print("Before Training (Initial Random Weights):")
    print(f"Initial MAE: {initial_mae:.4f}")
    print(f"Initial MSE: {initial_mse:.4f}")
    print(f"Initial RMSE: {initial_rmse:.4f}\n")

  
    weight_count = sum(nodes[i] * nodes[i+1] for i in range(len(nodes)-1))
    bias_count = sum(nodes[i+1] for i in range(len(nodes)-1))
    num_functions = len(nodes) - 1
    dimensions = weight_count + bias_count + num_functions

    # Initialize PSO
    pso = PSO(
        swarm_size=30,
        alpha=0.5,
        beta=1.8,
        gamma=1.8,
        delta=0.4,
        epsilon=0.8,
        num_dimensions=dimensions,
        obj_func=None,
        max_iter=50,
        num_informants=5
    )

    print("Training PSO...")
    best_position, best_fitness = pso.move_particle(ann, X_train, y_train)

    print("\nTraining complete.")
    print("Best Training MAE:", best_fitness)

    ann.set_weights_bias(best_position)


    # y_train_pred = ann.feedforward(X_train)

    # train_mae = np.mean(np.abs(y_train_pred - y_train))
    # train_mse = np.mean((y_train_pred - y_train) ** 2)
    # train_rmse = np.sqrt(train_mse)

    # print(f"Training MAE: {train_mae:.4f}")
    # print(f"Training MSE: {train_mse:.4f}")
    # print(f"Training RMSE: {train_rmse:.4f}")

    # Evaluate on the test set
    y_test_pred = ann.feedforward(X_test)

    mse_test = np.mean((y_test - y_test_pred) ** 2)
    rmse_test = np.sqrt(mse_test)
    mae_test = np.mean(np.abs(y_test - y_test_pred))

    print(f"MSE (Test): {mse_test:.4f}")
    print(f"RMSE (Test): {rmse_test:.4f}")
    print(f"MAE (Test): {mae_test:.4f}")


if __name__ == "__main__":
    train_ann_with_pso()
      
            

    